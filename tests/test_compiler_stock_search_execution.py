"""Accepted search execution consumes tested settings without retuning its verdict."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from expertflow.compiler.preflight import file_sha256
from expertflow.compiler.runner import MeasurementOutcome
from test_compiler_pipeline import FakeRunner
from test_compiler_stock_discovery import (
    FixtureEligibility, HOST, execute, fast_challenger, prerequisite, trusted_fixture_provider,
)


def api():
    from expertflow.compiler import stock_discovery
    return stock_discovery


@pytest.fixture(scope='module')
def recommendation(prerequisite, tmp_path_factory):
    root = tmp_path_factory.mktemp('accepted-search')
    inp, store, _, report = execute(prerequisite, root, rates=fast_challenger)
    assert report['status'] == 'RECOMMENDED-CHALLENGER'
    return inp, store, root / 'search/recommended'


def test_accepted_execution_uses_challenger_settings_and_preserves_search_verdict(recommendation, tmp_path):
    inp, store, directory = recommendation
    before = {path: file_sha256(path) for path in (
        directory / 'execution-plan.json', directory / 'search-receipt.json', directory.parent / 'report.json')}
    runner = FakeRunner(store)
    status, result = api().run_search_recommendation(directory, store, inp, tmp_path / 'execution',
        runner=runner, host_capture=lambda: HOST, registry=FixtureEligibility())
    assert status == 'MEASURED-ACCEPTED-STOCK-SEARCH' and len(runner.calls) == 1
    assert result['decode_tps'] == pytest.approx(30)
    native = store.verify_measurement(result['measurement_id'])
    assert native['identities']['workload']['threads'] == 16
    assert native['identities']['workload']['cuda_graphs'] == 'off'
    assert all(file_sha256(path) == digest for path, digest in before.items())
    assert result['acceptance_recomputed_from_single_run'] is False
    record = store.measurement(result['measurement_id'])
    launch = json.loads(Path(next(a.identity.path for a in record.artifacts if a.role == 'launch')).read_text())
    receipt = json.loads((directory / 'search-receipt.json').read_text())
    assert launch['experiment_context'] == {
        'accepted_search_receipt_sha256': receipt['receipt_sha256'],
        'accepted_plan_sha256': result['plan_sha256']}


@pytest.mark.parametrize('changed', ['model', 'hardware', 'runtime', 'prompt', 'context', 'kv'])
def test_actual_input_changes_stop_before_new_execution(recommendation, tmp_path, changed):
    inp, store, directory = recommendation
    if changed == 'model':
        inp = replace(inp, model=replace(inp.model, quantization='Q4_0'))
    elif changed == 'hardware':
        inp = replace(inp, hardware=replace(inp.hardware, driver_version='different'))
    elif changed == 'runtime':
        inp = replace(inp, stock=replace(inp.stock, manifest_json='{"patches":[],"different":true}'))
    else:
        updates = {'prompt': {'prompt': 'different'}, 'context': {'context_size': 8192},
            'kv': {'policy':'approximate','approximate_quality_budget':0.1,'kv_type_k':'q8_0'}}[changed]
        inp = replace(inp, workload=replace(inp.workload, **updates))
    runner = FakeRunner(store)
    with pytest.raises(ValueError, match='exact F16' if changed == 'kv' else 'actual inputs'):
        api().run_search_recommendation(directory, store, inp, tmp_path / 'execution',
            runner=runner, host_capture=lambda: HOST, registry=FixtureEligibility())
    assert runner.calls == [] and not (tmp_path / 'execution').exists()


def test_execution_refuses_existing_output_before_native_child(recommendation, tmp_path):
    inp, store, directory = recommendation
    runner = FakeRunner(store)
    output = tmp_path / 'execution'
    output.mkdir()
    with pytest.raises(ValueError, match='fresh'):
        api().run_search_recommendation(directory, store, inp, output,
            runner=runner, host_capture=lambda: HOST, registry=FixtureEligibility())
    assert runner.calls == []


def test_input_tuning_knobs_do_not_override_the_tested_recommendation(recommendation, tmp_path):
    inp, store, directory = recommendation
    inp = replace(inp, workload=replace(inp.workload, threads=8, cuda_graphs='off'))
    runner = FakeRunner(store)
    status, result = api().run_search_recommendation(directory, store, inp, tmp_path/'execution',
        runner=runner, host_capture=lambda: HOST, registry=FixtureEligibility())
    assert status == 'MEASURED-ACCEPTED-STOCK-SEARCH'
    assert store.verify_measurement(result['measurement_id'])['identities']['workload']['threads'] == 16


def test_execution_cannot_write_into_frozen_search_experiment(recommendation):
    inp, store, directory = recommendation
    runner = FakeRunner(store)
    with pytest.raises(ValueError, match='outside'):
        api().run_search_recommendation(directory, store, inp, directory.parent / 'extra-run',
            runner=runner, host_capture=lambda: HOST, registry=FixtureEligibility())
    assert runner.calls == []


def test_native_failure_retained_without_retry(recommendation, tmp_path):
    inp, store, directory = recommendation
    runner = FakeRunner(store, fail=True)
    status, result = api().run_search_recommendation(directory, store, inp, tmp_path / 'execution',
        runner=runner, host_capture=lambda: HOST, registry=FixtureEligibility())
    assert status == 'ENVIRONMENT-BLOCKED' and len(runner.calls) == 1
    assert result['outcome']['measurement_id'] is None
    assert json.loads((tmp_path / 'execution/accepted-execution.json').read_text())['status'] == status


def test_old_process_cannot_be_reported_as_new_accepted_execution(recommendation, tmp_path):
    inp, store, directory = recommendation
    receipt = json.loads((directory / 'search-receipt.json').read_text())
    mid = receipt['experiment']['screening'][0]['measurement_id']
    class ReuseRunner(FakeRunner):
        def run_once(self, candidate, model, binding, **kwargs):
            self.calls.append((candidate.candidate_id, kwargs['stage'], None))
            return MeasurementOutcome('measured', mid, None, None, str(kwargs['output_dir']))
    runner = ReuseRunner(store)
    status, _ = api().run_search_recommendation(directory, store, inp, tmp_path / 'execution',
        runner=runner, host_capture=lambda: HOST, registry=FixtureEligibility())
    assert status == 'VALIDATION-STOP' and len(runner.calls) == 1


def test_changed_host_after_child_rejects_execution(recommendation, tmp_path):
    inp, store, directory = recommendation
    runner = FakeRunner(store)
    def capture():
        return HOST if not runner.calls else {**HOST, 'power_policy': {'changed': True}}
    status, _ = api().run_search_recommendation(directory, store, inp, tmp_path / 'execution',
        runner=runner, host_capture=capture, registry=FixtureEligibility())
    assert status == 'VALIDATION-STOP' and len(runner.calls) == 1


def test_changed_native_tokens_reject_execution(recommendation, tmp_path, monkeypatch):
    inp, store, directory = recommendation
    append = store.append_measurement
    def changed_completion(record, **kwargs):
        path = Path(next(a.identity.path for a in record.artifacts if a.role == 'completion'))
        payload = json.loads(path.read_text())
        payload['tokens'][-1] += 1
        path.write_text(json.dumps(payload))
        artifacts = tuple(replace(a, identity=replace(a.identity,
            size_bytes=path.stat().st_size,sha256=file_sha256(path))) if a.role == 'completion' else a
            for a in record.artifacts)
        return append(replace(record,artifacts=artifacts),**kwargs)
    monkeypatch.setattr(store,'append_measurement',changed_completion)
    runner = FakeRunner(store)
    status, result = api().run_search_recommendation(directory, store, inp, tmp_path/'execution',
        runner=runner,host_capture=lambda:HOST,registry=FixtureEligibility())
    assert status == 'VALIDATION-STOP' and len(runner.calls) == 1
    assert 'tokens' in result['reason']
