"""Fresh-reference collector contracts; fixtures are not live model evidence.

Prepared while Q6 timing is live. Run RED only after that experiment is terminal.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.preflight import file_sha256
from expertflow.compiler.schema import canonical_sha256
from test_compiler_pipeline import FakeRunner, inputs
from test_compiler_stock_discovery import FixtureEligibility
from test_compiler_stock_search import host


HOST = {**host(), 'architecture': 'AMD64'}


def api():
    from expertflow.compiler import stock_reference
    return stock_reference


def collect(tmp_path, *, runner_type=FakeRunner, capture=None, rates=None):
    inp = inputs(tmp_path)
    store = EvidenceStore(tmp_path / 'reference.sqlite3')
    runner = runner_type(store, rates=rates)
    report = api().execute_stock_reference(inp, store, runner, tmp_path / 'reference',
        host_capture=capture or (lambda: deepcopy(HOST)), registry=FixtureEligibility(),
        source_repository=tmp_path)
    return inp, store, runner, report


def test_reference_freezes_before_ten_fresh_processes_and_reconstructs(tmp_path):
    inp, store, runner, report = collect(tmp_path)
    assert report['status'] == 'REFERENCE-STABLE'
    assert len(runner.calls) == len(report['rows']) == 10
    assert all(call[1] == 'confirmation' for call in runner.calls)
    assert report['mean_tps'] == pytest.approx(30)
    assert report['cv_pct'] == pytest.approx(0)
    assert report['product_accepted'] is False
    manifest = report['manifest']
    claimed = manifest['manifest_sha256']
    body = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    assert claimed == canonical_sha256(body)
    assert manifest['maximum_native_processes'] == 10
    assert len({r['owned_run_sha256'] for r in report['rows']}) == 10
    for row in report['rows']:
        record = store.measurement(row['measurement_id'])
        launch_path = next(a.identity.path for a in record.artifacts if a.role == 'launch')
        with open(launch_path, encoding='utf-8') as stream:
            launch = json.load(stream)
        assert launch['experiment_context'] == {'manifest_sha256': claimed}
        assert launch['host_environment'] == HOST
    plan = api().load_reference_plan(tmp_path / 'reference/diagnostic', store,
        identities=inp.identities(inp.stock), host_environment=HOST, registry=FixtureEligibility())
    assert len(plan.candidate.measurement_ids) == 10
    assert plan.candidate.settings.static is None
    assert not (tmp_path / 'reference/accepted').exists()


def test_reference_partial_failure_retains_rows_and_never_retries(tmp_path):
    class FailingRunner(FakeRunner):
        def run_once(self, *args, **kwargs):
            self.fail = len(self.calls) >= 3
            return super().run_once(*args, **kwargs)
    _, _, runner, report = collect(tmp_path, runner_type=FailingRunner)
    assert report['status'] == 'ENVIRONMENT-BLOCKED'
    assert len(runner.calls) == len(report['outcomes']) == 4
    assert len(report['rows']) == 3
    assert not (tmp_path / 'reference/diagnostic').exists()


def test_reference_variance_failure_is_terminal_with_all_samples(tmp_path):
    state = {'calls': 0}
    def rates(candidate, stage):
        state['calls'] += 1
        return 20 if state['calls'] % 2 else 40
    _, _, runner, report = collect(tmp_path, rates=rates)
    assert report['status'] == 'INCONCLUSIVE'
    assert len(runner.calls) == len(report['rows']) == 10
    assert report['cv_pct'] > 10
    assert not (tmp_path / 'reference/diagnostic').exists()


def test_reference_requires_measured_native_records(tmp_path):
    class WarmupRunner(FakeRunner):
        def run_once(self, *args, **kwargs):
            return super().run_once(*args, **{**kwargs, 'measured': False})
    _, _, runner, report = collect(tmp_path, runner_type=WarmupRunner)
    assert report['status'] == 'VALIDATION-STOP'
    assert len(runner.calls) == 1
    assert not (tmp_path / 'reference/diagnostic').exists()


def test_reference_reused_owned_process_cannot_fill_another_slot(tmp_path):
    class ReusingRunner(FakeRunner):
        def run_once(self, candidate, model, binding, **kwargs):
            if self.calls:
                self.calls.append((candidate.candidate_id, kwargs['stage'], None))
                return self.first
            self.first = super().run_once(candidate, model, binding, **kwargs)
            return self.first
    _, _, runner, report = collect(tmp_path, runner_type=ReusingRunner)
    assert report['status'] == 'VALIDATION-STOP' and len(runner.calls) == 2
    assert 'reused' in report['reason']
    assert len(report['rows']) == 1
    assert not (tmp_path / 'reference/diagnostic').exists()


def test_reference_source_change_stops_before_another_process(tmp_path, monkeypatch):
    inp = inputs(tmp_path)
    store = EvidenceStore(tmp_path / 'reference.sqlite3')
    runner = FakeRunner(store)
    original = api()._sources
    def changed_sources():
        actual = original()
        return actual if not runner.calls else {**actual, 'changed-source.py': 'f' * 64}
    monkeypatch.setattr(api(), '_sources', changed_sources)
    report = api().execute_stock_reference(inp, store, runner, tmp_path / 'reference',
        host_capture=lambda: HOST, registry=FixtureEligibility(), source_repository=tmp_path)
    assert report['status'] == 'VALIDATION-STOP' and len(runner.calls) == 1
    assert 'source' in report['reason']
    assert not (tmp_path / 'reference/diagnostic').exists()


def test_reference_host_change_stops_before_second_child(tmp_path):
    inp = inputs(tmp_path)
    store = EvidenceStore(tmp_path / 'reference.sqlite3')
    runner = FakeRunner(store)
    def capture():
        return deepcopy(HOST) if not runner.calls else {**HOST, 'os': 'changed'}
    report = api().execute_stock_reference(inp, store, runner, tmp_path / 'reference',
        host_capture=capture, registry=FixtureEligibility(), source_repository=tmp_path)
    assert report['status'] == 'VALIDATION-STOP' and len(runner.calls) == 1
    assert not (tmp_path / 'reference/diagnostic').exists()


def test_reference_changed_native_tokens_stop_collection(tmp_path):
    inp = inputs(tmp_path)
    store = EvidenceStore(tmp_path / 'reference.sqlite3')
    append = store.append_measurement
    state = {'rows': 0}
    def changed_completion(record, **kwargs):
        state['rows'] += 1
        if state['rows'] == 2:
            completion = next(a for a in record.artifacts if a.role == 'completion')
            path = Path(completion.identity.path)
            payload = json.loads(path.read_text())
            payload['tokens'][-1] += 1
            path.write_text(json.dumps(payload))
            artifacts = tuple(replace(a, identity=replace(a.identity,
                size_bytes=path.stat().st_size, sha256=file_sha256(path))) if a.role == 'completion' else a
                for a in record.artifacts)
            record = replace(record, artifacts=artifacts)
        return append(record, **kwargs)
    store.append_measurement = changed_completion
    runner = FakeRunner(store)
    report = api().execute_stock_reference(inp, store, runner, tmp_path / 'reference',
        host_capture=lambda: HOST, registry=FixtureEligibility(), source_repository=tmp_path)
    assert report['status'] == 'VALIDATION-STOP' and len(runner.calls) == 2
    assert len(report['outcomes']) == 2 and len(report['rows']) == 1
    assert not (tmp_path / 'reference/diagnostic').exists()


@pytest.mark.parametrize('corruption', ['statistics', 'missing_row', 'row_order', 'host', 'freeze_time', 'root', 'budget'])
def test_rehashed_reference_claims_cannot_replace_native_evidence(tmp_path, corruption):
    inp, store, _, _ = collect(tmp_path)
    root = tmp_path / 'reference/diagnostic'
    path = root / 'reference-receipt.json'
    receipt = json.loads(path.read_text())
    report = receipt['experiment']
    if corruption == 'statistics':
        report['mean_tps'] = 999
    elif corruption == 'missing_row':
        report['rows'].pop()
    elif corruption == 'row_order':
        report['rows'][0], report['rows'][1] = report['rows'][1], report['rows'][0]
    else:
        manifest = report['manifest']
        if corruption == 'host':
            manifest['host_environment']['power_policy'] = {'settings_sha256': 'f' * 64}
        elif corruption == 'freeze_time':
            manifest['frozen_monotonic_ns'] = 10**30
        elif corruption == 'root':
            manifest['experiment_root'] = str(tmp_path / 'old-reference')
        else:
            manifest['maximum_native_processes'] = 11
        manifest.pop('manifest_sha256')
        manifest['manifest_sha256'] = canonical_sha256(manifest)
    receipt.pop('receipt_sha256')
    receipt['receipt_sha256'] = canonical_sha256(receipt)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        api().load_reference_plan(root, store, identities=inp.identities(inp.stock),
            host_environment=HOST, registry=FixtureEligibility())


def test_reference_requires_fresh_output_and_empty_database(tmp_path):
    inp, store, _, _ = collect(tmp_path)
    runner = FakeRunner(store)
    with pytest.raises(ValueError):
        api().execute_stock_reference(inp, store, runner, tmp_path / 'another-reference',
            host_capture=lambda: HOST, registry=FixtureEligibility(), source_repository=tmp_path)
    assert runner.calls == []


def test_existing_reference_output_is_rejected_before_launch(tmp_path):
    inp = inputs(tmp_path)
    store = EvidenceStore(tmp_path / 'empty.sqlite3')
    runner = FakeRunner(store)
    output = tmp_path / 'reference'
    output.mkdir()
    with pytest.raises(ValueError):
        api().execute_stock_reference(inp, store, runner, output,
            host_capture=lambda: HOST, registry=FixtureEligibility(), source_repository=tmp_path)
    assert runner.calls == []
