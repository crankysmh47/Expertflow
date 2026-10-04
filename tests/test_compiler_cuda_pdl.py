from copy import deepcopy

import pytest

from expertflow.compiler.evidence import EvidenceStore
from test_compiler_pipeline import FakeRunner
from test_compiler_stock_discovery import prerequisite, HOST


def collect(prerequisite, tmp_path, monkeypatch, *, fail=False, rates=None):
    from expertflow.compiler import cuda_pdl
    inputs, source, accepted = prerequisite
    monkeypatch.setattr(cuda_pdl, 'qualify_control', lambda *args: {'test': 'fixture-only'})
    target = EvidenceStore(tmp_path / 'pdl.sqlite3')
    runner = FakeRunner(target, fail=fail, rates=rates)
    report = cuda_pdl.execute(inputs, accepted, source, target, runner, tmp_path / 'pdl',
        source_repository=tmp_path, host_capture=lambda: deepcopy(HOST))
    return cuda_pdl, target, runner, report


def test_twenty_owned_pairs_and_negative_verdict(prerequisite, tmp_path, monkeypatch):
    api, store, runner, report = collect(prerequisite, tmp_path, monkeypatch)
    assert len(runner.calls) == len(report['rows']) == 20
    assert report['status'] == 'NO-GO'
    assert report['product_accepted'] is False
    from expertflow.compiler.schema import canonical_payload
    assert canonical_payload(api.audit(report, store, host_environment=HOST)) == report['statistics']


def test_gain_requires_full_exact_confirmation(prerequisite, tmp_path, monkeypatch):
    _, _, _, report = collect(prerequisite, tmp_path, monkeypatch,
        rates=lambda candidate, stage: 33 if candidate.settings.cuda_pdl == 'off' else 30)
    assert report['status'] == 'PASS-DIAGNOSTIC'
    assert report['statistics']['geometric_change_pct'] == pytest.approx(10)
    assert report['product_accepted'] is False


def test_failed_run_is_retained_without_retry(prerequisite, tmp_path, monkeypatch):
    _, _, runner, report = collect(prerequisite, tmp_path, monkeypatch, fail=True)
    assert len(runner.calls) == len(report['outcomes']) == 1
    assert not report['rows']
    assert report['status'] == 'ENVIRONMENT-BLOCKED'


@pytest.mark.parametrize('mutation', ['duplicate', 'partial', 'tokens', 'candidate', 'source', 'host', 'reference', 'unfinished', 'statistics'])
def test_audit_rejects_tampering(prerequisite, tmp_path, monkeypatch, mutation):
    api, store, _, report = collect(prerequisite, tmp_path, monkeypatch)
    bad = deepcopy(report)
    if mutation == 'duplicate': bad['rows'][1] = deepcopy(bad['rows'][0])
    if mutation == 'partial': bad['rows'].pop()
    if mutation == 'tokens': bad['rows'][0]['generated_tokens_sha256'] = 'f' * 64
    if mutation == 'candidate': bad['rows'][0]['candidate_id'] = 'f' * 64
    if mutation == 'source': bad['frozen']['source_files'] = {}
    if mutation == 'host': bad['frozen']['host_environment'] = {}
    if mutation == 'unfinished': bad['status'] = 'RUNNING'
    if mutation == 'statistics': bad.pop('statistics')
    if mutation == 'reference':
        from expertflow.compiler.schema import canonical_sha256
        bad['frozen']['reference_tokens']['generated_tokens_sha256'] = 'f' * 64
        bad['frozen'].pop('manifest_sha256')
        bad['frozen']['manifest_sha256'] = canonical_sha256(bad['frozen'])
    with pytest.raises(ValueError):
        api.audit(bad, store, host_environment=HOST)


def test_correctness_failure_is_validation_stop(prerequisite, tmp_path, monkeypatch):
    from expertflow.compiler import cuda_pdl
    from expertflow.compiler.runner import MeasurementOutcome
    inputs, source, accepted = prerequisite
    monkeypatch.setattr(cuda_pdl, 'qualify_control', lambda *args: {'test': 'fixture-only'})
    target = EvidenceStore(tmp_path / 'pdl.sqlite3')
    class InvalidRunner:
        def run_once(self, *args, **kwargs):
            return MeasurementOutcome('validation_stop', None, 'invalid tokens', None, str(kwargs['output_dir']))
    report = cuda_pdl.execute(inputs, accepted, source, target, InvalidRunner(), tmp_path / 'pdl',
        source_repository=tmp_path, host_capture=lambda: deepcopy(HOST))
    assert report['status'] == 'VALIDATION-STOP'
    assert len(report['outcomes']) == 1
