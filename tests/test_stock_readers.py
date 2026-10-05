"""Reader lifetime controls; no fixture is a native performance result."""
from hashlib import sha256
import os
from types import SimpleNamespace

import pytest

from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.schema import ArtifactIdentity
from test_compiler_stock_repeatability import completed


def model_at(path, data=b'weights'):
    path.write_bytes(data)
    return SimpleNamespace(identity=ArtifactIdentity(str(path), len(data), sha256(data).hexdigest()))


def test_repeated_reader_requests_hash_unchanged_weights_once(tmp_path, monkeypatch):
    from expertflow.stock.readers import EvidenceReaders
    from expertflow.compiler import evidence
    model = model_at(tmp_path/'model.gguf')
    calls = []
    original = evidence.file_sha256
    monkeypatch.setattr(evidence, 'file_sha256', lambda p: (calls.append(p), original(p))[1])
    readers = EvidenceReaders()
    for _ in range(20):
        readers(tmp_path/'evidence.sqlite3').prime_model(model)
    assert len(calls) == 1
    assert readers.database_count == readers.model_verification_count == 1
    assert readers(tmp_path/'evidence.sqlite3') is readers(tmp_path/'.'/'evidence.sqlite3')
    fresh = EvidenceReaders()
    fresh(tmp_path/'evidence.sqlite3').prime_model(model)
    assert len(calls) == 2  # No cache crosses public invocations.


def test_cached_model_replacement_is_detected_even_with_restored_mtime(tmp_path):
    from expertflow.stock.readers import EvidenceReaders
    model = model_at(tmp_path/'model.gguf')
    store = EvidenceReaders()(tmp_path/'evidence.sqlite3')
    store.prime_model(model)
    before = (tmp_path/'model.gguf').stat()
    replacement = tmp_path/'replacement.gguf'
    replacement.write_bytes(b'changed')
    os.replace(replacement, tmp_path/'model.gguf')
    os.utime(tmp_path/'model.gguf', ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='model artifact hash mismatch'):
        store.prime_model(model)


def test_scoped_aliases_restore_after_an_exception(tmp_path):
    from expertflow.stock.readers import reuse_readers
    first = SimpleNamespace(EvidenceStore=EvidenceStore)
    second = SimpleNamespace(EvidenceStore=EvidenceStore)
    with pytest.raises(RuntimeError, match='fixture'):
        with reuse_readers(first, second) as readers:
            assert first.EvidenceStore(tmp_path/'db') is second.EvidenceStore(tmp_path/'db')
            assert readers.database_count == 1
            raise RuntimeError('fixture')
    assert first.EvidenceStore is second.EvidenceStore is EvidenceStore


def test_reader_rechecks_artifacts_after_success(tmp_path):
    from expertflow.stock.readers import EvidenceReaders
    from test_compiler_refinement import setup_execution
    _, source, _, _, plan_path = setup_execution(tmp_path)
    from expertflow.compiler.plan import load_execution_plan
    store = EvidenceReaders()(source.path)
    mid = load_execution_plan(plan_path).candidate.measurement_ids[0]
    store.verify_measurement(mid)
    completion = next(a.identity.path for a in store.measurement(mid).artifacts if a.role == 'completion')
    from pathlib import Path
    Path(completion).write_text('{"tampered":true}')
    with pytest.raises(ValueError, match='artifact identity mismatch'):
        store.verify_measurement(mid)


def test_completed_repeatability_reconstruction_reuses_each_database(completed, monkeypatch):
    from expertflow.stock.readers import reuse_readers
    from test_compiler_stock_repeatability import api, patch_validation, HOST
    from expertflow.compiler import evidence
    _, _, (inp, source, plan, prior, _, report) = completed
    patch_validation(monkeypatch)
    model_reads = []
    original = evidence.file_sha256
    def counted(path):
        if str(path) == inp.model.identity.path:
            model_reads.append(path)
        return original(path)
    monkeypatch.setattr(evidence, 'file_sha256', counted)
    module = api()
    sources = module.sources()
    with reuse_readers(module, module.utility) as readers:
        pooled_source = readers(source.path)
        result = module.validate_followup(report, inp, inp, plan, pooled_source, prior,
            source_repository=prior.parent, host_environment=HOST)
        assert result['status'] == 'PASS-STOCK-REPEATABILITY-TRANSFER'
        assert len(model_reads) == readers.model_verification_count == readers.database_count == 5
    assert module.sources() == sources
