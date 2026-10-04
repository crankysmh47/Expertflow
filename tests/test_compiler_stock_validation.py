from copy import deepcopy
import json

import pytest

from test_compiler_refinement import setup_execution


HOST = {'cpu': {'name': 'fixture CPU', 'cores': 8, 'logical_processors': 16},
        'power_scheme': 'fixture', 'ram_bytes': 32 << 30, 'os': 'fixture'}


def api():
    from expertflow.compiler import stock_validation
    return stock_validation


def execute(tmp_path, **kwargs):
    inp, source, target, runner, plan = setup_execution(tmp_path)
    report = api().execute_stock_product(inp, plan, source, target, runner,
        tmp_path/'product', host_capture=lambda: deepcopy(HOST), **kwargs)
    return inp, source, target, runner, plan, report


def test_product_pairs_publish_and_revalidate_fresh_evidence(tmp_path):
    inp, source, target, runner, pending, report = execute(tmp_path)
    assert report['status'] == 'PASS-STOCK-FALLBACK' and len(runner.calls) == 20
    assert all(call[1].startswith('product-') for call in runner.calls)
    plan = tmp_path/'product/accepted/execution-plan.json'
    receipt = tmp_path/'product/accepted/acceptance-receipt.json'
    validated = api().load_validated_stock_plan(plan, receipt, target,
        identities=inp.identities(inp.stock), host_environment=HOST)
    assert len(validated.candidate.measurement_ids) == 20
    assert validated.plan_sha256 != json.loads(pending.read_text())['plan_sha256']
    assert report['live_validated_product'] is True


def test_partial_failure_has_no_product_and_no_retry(tmp_path):
    inp, source, target, runner, plan = setup_execution(tmp_path)
    runner.fail = True
    report = api().execute_stock_product(inp, plan, source, target, runner,
        tmp_path/'product', host_capture=lambda: deepcopy(HOST))
    assert report['status'] == 'ENVIRONMENT-BLOCKED' and len(runner.calls) == 1
    assert not (tmp_path/'product/accepted').exists()


def test_host_change_stops_before_another_launch(tmp_path):
    inp, source, target, runner, plan = setup_execution(tmp_path)
    def host():
        return {**HOST, 'power_scheme': 'changed' if runner.calls else 'fixture'}
    report = api().execute_stock_product(inp, plan, source, target, runner,
        tmp_path/'product', host_capture=host)
    assert report['status'] == 'VALIDATION-STOP' and len(runner.calls) == 1
    assert 'host' in report['reason'] and not (tmp_path/'product/accepted').exists()


@pytest.mark.parametrize('corruption', ['statistics', 'pair', 'old_stage', 'host', 'missing_row', 'plan_hash',
                                      'budget', 'freeze_time', 'artifact_root'])
def test_receipt_corruption_cannot_pass(tmp_path, corruption):
    inp, _, store, _, _, report = execute(tmp_path)
    root = tmp_path/'product/accepted'
    receipt = json.loads((root/'acceptance-receipt.json').read_text())
    if corruption == 'statistics':
        receipt['statistics']['geometric_change_pct'] = 99
    elif corruption == 'pair':
        receipt['experiment']['rows'][0]['pair'] = 9
    elif corruption == 'old_stage':
        receipt['experiment']['frozen']['protocol_version'] = 'measurement-v1'
    elif corruption == 'host':
        receipt['host_environment'] = {**HOST, 'power_scheme': 'different'}
    elif corruption == 'missing_row':
        receipt['experiment']['rows'].pop()
    elif corruption == 'plan_hash':
        receipt['published_plan_sha256'] = '0'*64
    elif corruption == 'budget':
        receipt['experiment']['frozen']['bootstrap_samples'] = 1
    elif corruption == 'freeze_time':
        receipt['experiment']['frozen']['frozen_monotonic_ns'] = 10**30
    else:
        receipt['experiment']['frozen']['experiment_root'] = str(tmp_path/'historical')
    # Rehashing untrusted claims must not bypass reconstruction from EvidenceStore.
    from expertflow.compiler.schema import canonical_sha256
    receipt.pop('receipt_sha256')
    receipt['receipt_sha256'] = canonical_sha256(receipt)
    (root/'acceptance-receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        api().load_validated_stock_plan(root/'execution-plan.json', root/'acceptance-receipt.json',
            store, identities=inp.identities(inp.stock), host_environment=HOST)


def test_historical_aa_experiment_cannot_publish(tmp_path):
    from expertflow.compiler.refinement import execute_pairs
    inp, source, target, runner, pending = setup_execution(tmp_path)
    report = execute_pairs(inp, pending, source, target, runner, tmp_path/'aa')
    assert report['status'] == 'PASS-MEASUREMENT'
    with pytest.raises(ValueError, match='protocol'):
        api().publish_stock_product(report, target, tmp_path/'aa', host_environment=HOST)
    assert not (tmp_path/'aa/accepted').exists()


def test_rehashed_host_claims_cannot_rebind_native_measurements(tmp_path):
    from expertflow.compiler.schema import canonical_sha256
    inp, _, store, _, _, _ = execute(tmp_path)
    root = tmp_path/'product/accepted'
    path = root/'acceptance-receipt.json'
    receipt = json.loads(path.read_text())
    changed = {**HOST, 'cpu': {'name': 'different CPU'}}
    receipt['host_environment'] = changed
    receipt['experiment']['frozen']['host_environment'] = changed
    receipt.pop('receipt_sha256')
    receipt['receipt_sha256'] = canonical_sha256(receipt)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='host'):
        api().load_validated_stock_plan(root/'execution-plan.json', path, store,
            identities=inp.identities(inp.stock), host_environment=changed)


def test_power_settings_change_with_same_scheme_invalidates_identity(monkeypatch):
    from expertflow.compiler import preflight
    from types import SimpleNamespace
    outputs = iter(['scheme fixture\nAC Power Setting Index: 0x00000064',
                    'scheme fixture\nAC Power Setting Index: 0x00000032'])
    def query(argv, **kwargs):
        assert argv == ['powercfg', '/qh', 'fixture']
        return SimpleNamespace(stdout=next(outputs))
    monkeypatch.setattr(preflight.subprocess, 'run', query)
    assert preflight.capture_power_policy('fixture') != preflight.capture_power_policy('fixture')


def test_native_runner_captures_host_before_launch(tmp_path, monkeypatch):
    from expertflow.compiler import preflight
    from expertflow.compiler.plan import load_execution_plan
    from expertflow.compiler.runner import ServerMeasurementRunner
    inp, source, target, _, pending = setup_execution(tmp_path)
    plan = load_execution_plan(pending, identities=inp.identities(inp.stock), store=source)
    monkeypatch.setattr(preflight, 'capture_host_environment', lambda: {**HOST, 'power_policy': 'changed'})
    def forbidden_launch(*args, **kwargs):
        pytest.fail('host mismatch must fail before creating native process')
    runner = ServerMeasurementRunner(target, process_factory=forbidden_launch)
    with pytest.raises(ValueError, match='host'):
        runner.run_once(plan.candidate, inp.model, inp.stock, output_dir=tmp_path/'native',
            measured=True, host_environment=HOST)


def test_publication_failure_leaves_no_half_published_plan(tmp_path, monkeypatch):
    from expertflow.compiler import stock_validation
    original = stock_validation.os.rename
    def fail_publication(src, dst):
        if str(dst).endswith('accepted'):
            raise OSError('publication rename failed')
        return original(src, dst)
    monkeypatch.setattr(stock_validation.os, 'rename', fail_publication)
    *_, report = execute(tmp_path)
    assert report['status'] == 'VALIDATION-STOP' and report['live_validated_product'] is False
    assert not (tmp_path/'product/accepted').exists()
    assert not list((tmp_path/'product').glob('.pending-acceptance-*'))


def test_accepted_run_verifies_receipt_then_records_fresh_exact_execution(tmp_path):
    from test_compiler_pipeline import FakeRunner
    inp, _, store, _, _, report = execute(tmp_path)
    root = tmp_path/'product/accepted'
    runner = FakeRunner(store)
    status, measured = api().run_accepted_stock_plan(root/'execution-plan.json',
        root/'acceptance-receipt.json', inp, store, tmp_path/'fresh',
        runner=runner, host_capture=lambda: deepcopy(HOST))
    assert status == 'MEASURED-ACCEPTED-STOCK' and len(runner.calls) == 1
    assert measured['measurement_id'] not in [r['measurement_id'] for r in report['rows']]


def test_cli_receipt_validation_and_tamper_failure(tmp_path, monkeypatch, capsys):
    from expertflow.cli.main import main
    from expertflow.compiler import commands, preflight
    inp, _, store, _, _, report = execute(tmp_path)
    root = tmp_path/'product/accepted'
    monkeypatch.setattr(commands, 'load_compiler_inputs', lambda *args, **kwargs: inp)
    monkeypatch.setattr(preflight, 'capture_host_environment', lambda: deepcopy(HOST))
    argv = ['validate','--plan',str(root/'execution-plan.json'),
            '--acceptance',str(root/'acceptance-receipt.json'),'--evidence-db',str(store.path)]
    for name in ('descriptor','inventory','hardware','workload','runtime-identity'):
        argv += ['--'+name,str(tmp_path/name)]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'VALIDATED-STOCK-FALLBACK'
    value = json.loads((root/'acceptance-receipt.json').read_text())
    value['receipt_sha256'] = '0'*64
    (root/'acceptance-receipt.json').write_text(json.dumps(value))
    assert main(argv) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'IDENTITY-STOP'


def test_product_reconstruction_requires_complete_frozen_source_map(tmp_path):
    *_, store, runner, pending, report = execute(tmp_path)
    report['status'] = 'PASS-MEASUREMENT'
    report['frozen']['source_files'].pop(next(iter(report['frozen']['source_files'])))
    with pytest.raises(ValueError, match='source'):
        api().reconstruct_product(report, store, host_environment=HOST)


def test_family_spec_change_stops_product_before_next_launch(tmp_path, monkeypatch):
    from expertflow.compiler import refinement
    inp, source, target, runner, plan = setup_execution(tmp_path)
    from expertflow.compiler import preflight
    digest = preflight.file_sha256
    family = '2026-10-04-granite-generalization.md'
    def hashing(path):
        if str(path).endswith(family) and runner.calls:
            return '0' * 64
        return digest(path)
    monkeypatch.setattr(preflight, 'file_sha256', hashing)
    report = api().execute_stock_product(inp, plan, source, target, runner,
        tmp_path/'product', host_capture=lambda: deepcopy(HOST))
    assert report['status'] == 'VALIDATION-STOP' and len(runner.calls) == 1
    assert 'source' in report['reason'] and not (tmp_path/'product/accepted').exists()
