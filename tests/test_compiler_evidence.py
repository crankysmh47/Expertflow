from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import pytest

from expertflow.compiler.evidence import EvidenceArtifact, EvidenceStore, MeasurementKey, MeasurementRecord
from expertflow.compiler.plan import CandidatePlan, CandidateStatus, PlanIdentities, RuntimeSettings, seal_candidate
from expertflow.compiler.preflight import file_sha256
from expertflow.compiler.schema import ArtifactIdentity, WorkloadIR, canonical_payload, canonical_sha256


def record_fixture(tmp_path, name='run', **changes):
    w = WorkloadIR('hello', 4096, 3)
    from expertflow.compiler.runner import RuntimeBinding
    executable = tmp_path / 'runtime'
    executable.write_bytes(b'pinned fixture executable')
    binding = RuntimeBinding(ArtifactIdentity(str(executable.resolve()), executable.stat().st_size, file_sha256(executable)),
                             '{"patches":[]}', (), None)
    identities = PlanIdentities('a' * 64, 'b' * 64, canonical_sha256(w), binding.sha256, w)
    candidate = CandidatePlan(identities, RuntimeSettings('auto', True))
    key = MeasurementKey.from_candidate(candidate)
    payloads = {
        'completion': {'tokens': [10, 11, 12], 'timings': {'predicted_n': 3, 'predicted_ms': 100}},
        'tokenize': {'tokens': [1, 2]},
        'request': {'prompt': [1, 2], 'n_predict': 3, 'seed': 42, 'temperature': 0.0,
                    'ignore_eos': True, 'cache_prompt': False, 'return_tokens': True, 'stream': False},
        'memory': {'samples': [{'pid': 123, 'dedicated_bytes': 100, 'device_free_bytes': 512 << 20}]},
        'process': {'pid': 123, 'exited': True, 'exit_code': 0, 'cleanup': True},
        'launch': {'candidate_id': candidate.candidate_id, 'settings_sha256': canonical_sha256(candidate.settings),
                   'runtime_binding': canonical_payload(binding),
                   'runtime_sha256': identities.runtime_sha256, 'model_sha256': identities.model_sha256,
                   'workload_sha256': identities.workload_sha256},
    }
    artifacts = []
    for role, payload in payloads.items():
        p = tmp_path / f'{name}-{role}.json'
        p.write_text(json.dumps(payload), encoding='utf-8')
        artifacts.append(EvidenceArtifact(role, ArtifactIdentity(str(p.resolve()), p.stat().st_size, file_sha256(p))))
    return replace(MeasurementRecord(key, candidate.candidate_id, json.dumps(canonical_payload(identities)),
                    json.dumps(canonical_payload(candidate.settings)), tuple(artifacts),
                    json.dumps({'decode_tps': 30.0}),
                    json.dumps({'exact_tokens': True, 'memory': True, 'cleanup': True}),
                    0, True, 'initial', 'stock_same_runtime'), **changes), candidate


def test_append_only_order_key_isolation_and_transaction(tmp_path):
    store = EvidenceStore(tmp_path / 'store.sqlite3')
    record, _ = record_fixture(tmp_path)
    first = store.append_measurement(record, measurement_id='first')
    second = store.append_measurement(replace(record, candidate_id='other'))
    assert first != second
    assert [r.candidate_id for r in store.measurements_for(record.key)] == [record.candidate_id, 'other']
    assert store.measurements_for(replace(record.key, model_sha256='e' * 64)) == ()
    with pytest.raises(ValueError, match='append-only'):
        store.replace_measurement(first, record)
    with pytest.raises(ValueError, match='duplicate'):
        store.append_measurement(record, measurement_id=first)
    missing = replace(record, artifacts=(replace(record.artifacts[0], identity=ArtifactIdentity('missing', 1, 'a' * 64)),))
    with pytest.raises(ValueError):
        store.append_measurement(missing)
    assert len(store.measurements_for(record.key)) == 2
    with pytest.raises(ValueError, match='boolean'):
        store.append_measurement(replace(record, validation_json='{"memory": 1}'))
    assert len(store.measurements_for(record.key)) == 2
    with sqlite3.connect(store.path) as conn, pytest.raises(sqlite3.IntegrityError, match='append-only'):
        conn.execute('DELETE FROM measurement')


def test_real_store_verification_and_sealing(tmp_path):
    store = EvidenceStore(tmp_path / 'store.sqlite3')
    record, candidate = record_fixture(tmp_path)
    ids = tuple(store.append_measurement(record) for _ in range(3))
    row = store.verify_measurement(ids[0])
    assert row['decode_tps'] == 30
    assert row['generated_tokens_sha256'] == canonical_sha256([10, 11, 12])
    sealed = seal_candidate(replace(candidate, status=CandidateStatus.MEASURED, measurement_ids=ids),
                            store, candidate.identities, None)
    assert sealed.candidate.measurement_ids == ids
    Path(record.artifacts[0].identity.path).write_text('{}')
    with pytest.raises(ValueError, match='artifact'):
        store.verify_artifacts(ids[0])


@pytest.mark.parametrize('role,mutation', [
    ('completion', lambda p: p.update(tokens=[])),
    ('completion', lambda p: p['timings'].update(predicted_n=2)),
    ('completion', lambda p: p['timings'].update(predicted_ms=0)),
    ('request', lambda p: p.update(cache_prompt=True)),
    ('request', lambda p: p.update(prompt=[99])),
    ('memory', lambda p: p.update(samples=[])),
    ('memory', lambda p: p.update(errors=['counter failed'])),
    ('memory', lambda p: p['samples'][0].update(dedicated_bytes=0)),
    ('memory', lambda p: p['samples'][0].update(device_free_bytes=0)),
    ('memory', lambda p: p['samples'][0].update(pid=999)),
    ('process', lambda p: p.update(cleanup=False)),
    ('launch', lambda p: p.update(candidate_id='other')),
    ('launch', lambda p: p.pop('runtime_binding')),
])
def test_caller_pass_flags_cannot_override_raw_failure(tmp_path, role, mutation):
    record, _ = record_fixture(tmp_path)
    artifacts = []
    for artifact in record.artifacts:
        if artifact.role == role:
            p = Path(artifact.identity.path)
            payload = json.loads(p.read_text())
            mutation(payload)
            p.write_text(json.dumps(payload))
            artifact = replace(artifact, identity=ArtifactIdentity(str(p), p.stat().st_size, file_sha256(p)))
        artifacts.append(artifact)
    record = replace(record, artifacts=tuple(artifacts))
    store = EvidenceStore(tmp_path / 'store.sqlite3')
    mid = store.append_measurement(record)
    with pytest.raises(ValueError):
        store.verify_measurement(mid)


def test_missing_ids_and_cross_runtime_comparison_required(tmp_path):
    store = EvidenceStore(tmp_path / 'store.sqlite3')
    with pytest.raises(ValueError, match='missing evidence'):
        store.measurement('invented')
    record, _ = record_fixture(tmp_path)
    mid = store.append_measurement(replace(record, numerical_path='fork_off_vs_pristine'))
    with pytest.raises(ValueError, match='comparison'):
        store.verify_measurement(mid)
