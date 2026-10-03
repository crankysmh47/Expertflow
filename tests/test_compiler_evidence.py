from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import time
import uuid

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
    from test_compiler_schema import model_fixture
    model_path=tmp_path/'model.gguf';model_path.write_bytes(b'GGUF'+bytes(12000))
    model=replace(model_fixture(),identity=ArtifactIdentity(str(model_path.resolve()),model_path.stat().st_size,file_sha256(model_path)))
    identities = PlanIdentities(canonical_sha256(model), 'b' * 64, canonical_sha256(w), binding.sha256, w)
    candidate = CandidatePlan(identities, RuntimeSettings('auto', True))
    key = MeasurementKey.from_candidate(candidate)
    from expertflow.compiler.runner import lower_launch
    launch=lower_launch(candidate,model.identity,binding,12345,tmp_path,inherited={'PATH':'path'})
    run_id=str(uuid.uuid4());creation=time.time_ns()//100
    sample={'pid':123,'dedicated_bytes':100,'device_free_bytes':512 << 20,
            'state':'allocated','counter_available':True,'phase':'measurement'}
    payloads = {
        'completion': {'tokens': [10, 11, 12], 'timings': {'predicted_n': 3, 'predicted_ms': 100}},
        'tokenize': {'tokens': [1, 2]},
        'tokenize-request': {'content':w.prompt,'add_special':True},
        'completion-wall':{'started_monotonic_ns':100000000,'finished_monotonic_ns':200000000,'elapsed_ms':100},
        'request': {'prompt': [1, 2], 'n_predict': 3, 'seed': 42, 'temperature': 0.0,
                    'ignore_eos': True, 'cache_prompt': False, 'return_tokens': True, 'stream': False},
        'memory': {'samples':[sample],'observations':[sample],
                   'teardown_reading':{'pid':123,'state':'absent','counter_available':True,'dedicated_bytes':0}},
        'process': {'pid': 123, 'exited': True, 'exit_code': 0, 'cleanup': True,'memory_settled':True,
                    'run_id':run_id,'creation_time_100ns':creation},
        'run-start':{'pid':123,'run_id':run_id,'creation_time_100ns':creation,
                     'creation_source':'GetProcessTimes','started_monotonic_ns':time.monotonic_ns()},
        'launch': {'candidate_id': candidate.candidate_id, 'settings_sha256': canonical_sha256(candidate.settings),
                   'runtime_binding': canonical_payload(binding),
                   'model_ir':canonical_payload(model),'argv':launch.argv,
                   'environment':launch.environment,
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
    other,_=record_fixture(tmp_path,'other')
    second = store.append_measurement(replace(other, candidate_id='other'))
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
    invalid,_=record_fixture(tmp_path,'invalid')
    with pytest.raises(ValueError, match='boolean'):
        store.append_measurement(replace(invalid, validation_json='{"memory": 1}'))
    assert len(store.measurements_for(record.key)) == 2
    with sqlite3.connect(store.path) as conn, pytest.raises(sqlite3.IntegrityError, match='append-only'):
        conn.execute('DELETE FROM measurement')


def test_real_store_verification_and_sealing(tmp_path):
    store = EvidenceStore(tmp_path / 'store.sqlite3')
    record, candidate = record_fixture(tmp_path)
    ids = tuple(store.append_measurement(record_fixture(tmp_path,f'repeat-{i}')[0]) for i in range(3))
    row = store.verify_measurement(ids[0])
    assert row['decode_tps'] == 30
    assert row['generated_tokens_sha256'] == canonical_sha256([10, 11, 12])
    sealed = seal_candidate(replace(candidate, status=CandidateStatus.MEASURED, measurement_ids=ids),
                            store, candidate.identities, None)
    assert sealed.candidate.measurement_ids == ids
    Path(store.measurement(ids[0]).artifacts[0].identity.path).write_text('{}')
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
    ('launch', lambda p: p.pop('argv')),
    ('launch', lambda p: p['argv'].__setitem__(p['argv'].index('-c')+1,'8192')),
    ('launch', lambda p: p['argv'].__setitem__(p['argv'].index('-m')+1,'other.gguf')),
    ('launch', lambda p: p['environment'].update(GGML_CUDA_DISABLE_GRAPHS='1')),
    ('tokenize-request', lambda p: p.update(content='a different prompt')),
    ('tokenize-request', lambda p: p.update(add_special=False)),
    ('memory', lambda p: p['observations'].append({'pid':123,'phase':'measurement','state':'unavailable'})),
    ('memory', lambda p: p['teardown_reading'].update(state='unavailable',counter_available=False)),
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


def test_one_owned_run_cannot_be_reused_as_multiple_measurements_or_confirmation(tmp_path):
    store=EvidenceStore(tmp_path/'store.sqlite3')
    record,_=record_fixture(tmp_path)
    store.append_measurement(record)
    with pytest.raises(ValueError,match='owned run'):
        store.append_measurement(replace(record,stage='confirmation'))
