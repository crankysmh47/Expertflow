"""Public stock CLI boundary tests, including installed-package startup."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_public_help_lists_stock_workflow(capsys):
    from expertflow.cli.main import main
    with pytest.raises(SystemExit) as exit:
        main(['--help'])
    assert exit.value.code == 0
    assert ',stock,' in capsys.readouterr().out.replace(' ', '')


def test_missing_project_fails_without_loading_a_driver(tmp_path, monkeypatch, capsys):
    from expertflow.stock import cli
    monkeypatch.setattr(cli, '_load_driver', lambda *a: pytest.fail('no driver load'))
    assert cli.main(['--project', str(tmp_path), 'search', 'generate']) == 3
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'ENVIRONMENT-BLOCKED'
    assert 'checkout' in result['reason']


@pytest.mark.parametrize('workflow', ['utility', 'repeatability'])
def test_closed_study_collection_is_not_exposed(workflow, capsys):
    from expertflow.stock.cli import main
    assert main([workflow, 'run']) == 2
    assert 'closed' in json.loads(capsys.readouterr().out)['reason']


@pytest.mark.parametrize('override', ['--action', '--experiment', '--act', '--exper', '--action=run'])
def test_action_override_cannot_launch_another_workflow(override, monkeypatch, capsys):
    from expertflow.stock import cli
    monkeypatch.setattr(cli, '_load_driver', lambda *a: pytest.fail('no driver load'))
    assert cli.main(['repeatability', 'validate', override, 'run']) == 2
    assert 'override' in json.loads(capsys.readouterr().out)['reason']


def test_validation_routes_through_pool_and_restores_driver(capsys, monkeypatch, tmp_path):
    from expertflow.stock import cli
    from expertflow.compiler.evidence import EvidenceStore
    observed = []
    driver = SimpleNamespace(EvidenceStore=EvidenceStore)
    def validate(argv):
        observed.extend(argv)
        assert driver.EvidenceStore(tmp_path/'fixture.sqlite3') is driver.EvidenceStore(tmp_path/'fixture.sqlite3')
        print(json.dumps({'status':'PASS-STOCK-REPEATABILITY-TRANSFER'}))
        return 0
    driver.main = validate
    @contextmanager
    def load(*args):
        yield driver
    monkeypatch.setattr(cli, '_load_driver', load)
    monkeypatch.setattr(cli, '_check_project', lambda path: path)
    assert cli.main(['repeatability', 'validate', '--output-dir', str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert observed == ['--action', 'validate', '--output-dir', str(tmp_path)]
    assert result['validation']['database_readers'] == 1
    assert result['validation']['cache_lifetime'] == 'this invocation'
    assert driver.EvidenceStore is EvidenceStore


def test_invalid_result_does_not_promote_report_claims(tmp_path, monkeypatch, capsys):
    from expertflow.stock import cli
    report = tmp_path/'report.json'
    report.write_text(json.dumps({'status':'PASS-STOCK-UTILITY-PRODUCT',
        'statistics':{'gain':{'geometric_change_pct':100}}, 'manifest':{'source_files':{}}}))
    @contextmanager
    def load(*args):
        def invalid(argv):
            print(json.dumps({'status':'IDENTITY-STOP','reason':'source mismatch','report':str(report)}))
            return 2
        from expertflow.compiler.evidence import EvidenceStore
        yield SimpleNamespace(main=invalid,EvidenceStore=EvidenceStore)
    monkeypatch.setattr(cli, '_load_driver', load)
    monkeypatch.setattr(cli, '_check_project', lambda path: path)
    assert cli.main(['utility', 'validate']) == 2
    result = json.loads(capsys.readouterr().out)
    assert result['reason'] == 'source mismatch'
    assert result['decision']['utility_gain_established'] is False
    assert 'statistics' not in result['decision']


def test_reconstructed_neutral_result_remains_verified_without_gain(tmp_path, monkeypatch, capsys):
    from expertflow.stock import cli
    report = tmp_path/'report.json'
    report.write_text(json.dumps({'status':'NO-UTILITY-GAIN', 'statistics':{'status':'NO-UTILITY-GAIN'},
        'manifest':{'protocol_version':'stock-utility-proof-v1','default_id':'same'},
        'automatic_id':'same', 'attempts':[{}]*86}))
    @contextmanager
    def load(*args):
        from expertflow.compiler.evidence import EvidenceStore
        def neutral(argv):
            print(json.dumps({'status':'NO-UTILITY-GAIN','report':str(report)}))
            return 2
        yield SimpleNamespace(main=neutral,EvidenceStore=EvidenceStore)
    monkeypatch.setattr(cli, '_load_driver', load)
    monkeypatch.setattr(cli, '_check_project', lambda path:path)
    assert cli.main(['utility','validate']) == 2
    result = json.loads(capsys.readouterr().out)
    assert result['decision']['evidence_verified'] is True
    assert result['decision']['selected_default'] is True
    assert result['decision']['utility_gain_established'] is False


def test_driver_process_state_restores_after_failure(tmp_path):
    from expertflow.stock.cli import _load_driver
    import os
    import sys
    scripts = tmp_path/'scripts'
    scripts.mkdir()
    (scripts/'fail.py').write_text('raise ValueError("driver fixture")')
    cwd, paths = Path.cwd(), sys.path[:]
    with pytest.raises(ValueError,match='driver fixture'):
        with _load_driver(tmp_path,'fail.py'):
            pytest.fail('failing module must not yield')
    assert Path.cwd() == cwd and sys.path == paths
    assert '_expertflow_stock_driver' not in sys.modules


def test_utility_decision_reads_workload_from_default_candidate(tmp_path):
    from expertflow.stock.cli import _decision
    report = tmp_path/'report.json'
    workload = {'prompt':'fixture only', 'threads':8}
    report.write_text(json.dumps({'statistics':{'status':'NO-UTILITY-GAIN'},'automatic_id':'defaults',
        'manifest':{'default_id':'defaults','inputs':{'model':{'family':'gemma4','quantization':'Q4_0',
            'identity':{'sha256':'a'*64}},'hardware':{},'stock':{}},
            'candidates':{'defaults':{'identities':{'workload':workload}}}}}))
    result = _decision({'status':'NO-UTILITY-GAIN','report':str(report)},2,'validate',reconstructed=True)
    from expertflow.compiler.schema import canonical_sha256
    assert result['coverage']['inputs']['inputs']['workload_sha256'] == canonical_sha256(workload)
    assert result['utility_gain_established'] is False


def test_real_search_missing_database_uses_public_json_stop(tmp_path, capsys):
    from expertflow.cli.main import main
    assert main(['stock', 'search', 'generate', '--source-evidence-db', str(tmp_path/'missing'),
        '--output-dir', str(tmp_path/'future'), '--manifest-output', str(tmp_path/'preview')]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result['workflow'] == 'search' and result['status'] == 'ENVIRONMENT-BLOCKED'
    assert not (tmp_path/'future').exists()


def scoped_fixture():
    from expertflow.compiler.schema import canonical_sha256
    workload = {'prompt':'scope fixture','threads':12,'cuda_graphs':'on'}
    identities = {'model_sha256':'a'*64,'hardware_sha256':'b'*64,'runtime_sha256':'c'*64,
        'workload_sha256':canonical_sha256(workload),'workload':workload}
    candidate = {'candidate_id':'incumbent','identities':identities,'settings':{'cuda_graphs':'on'}}
    inputs = {'model':{'family':'gemma4','quantization':'Q4_0','identity':{'sha256':'d'*64}},
        'hardware':{},'stock':{}}
    return identities, candidate, inputs


@pytest.mark.parametrize('recommended', ['incumbent','alternate'])
def test_search_reports_incumbent_retention_without_a_default_claim(tmp_path, recommended):
    from expertflow.stock.cli import _decision
    _, candidate, inputs = scoped_fixture()
    report = tmp_path/'search.json'
    report.write_text(json.dumps({'status':'RECOMMENDED-INCUMBENT','recommended_id':recommended,
        'statistics':None,'manifest':{'incumbent_id':'incumbent','inputs':inputs,
        'candidates':{'incumbent':candidate,'alternate':candidate}}}))
    decision = _decision({'status':'RECOMMENDED-INCUMBENT','report':str(report)},0,'run')
    assert 'selected_default' not in decision
    assert decision['retained_incumbent'] is (recommended == 'incumbent')
    assert decision['coverage']['inputs']['inputs']['workload_sha256'] == candidate['identities']['workload_sha256']


def test_product_report_scope_uses_its_frozen_plan_identities(tmp_path):
    from expertflow.stock.cli import _decision
    identities, candidate, _ = scoped_fixture()
    report = tmp_path/'product.json'
    report.write_text(json.dumps({'frozen':{'identities':identities,'settings':candidate['settings'],
        'source_plan':{'candidate':candidate}},'geometric_change_pct':0.1,'ci90_pct':[-0.1,0.3],
        'outcomes':[{}]*20}))
    decision = _decision({'status':'PASS-STOCK-FALLBACK','report':str(report)},0,'run')
    assert decision['coverage']['inputs']['source_plan']['model_ir_sha256'] == identities['model_sha256']
    assert decision['coverage']['inputs']['source_plan']['workload_sha256'] == identities['workload_sha256']
    assert decision['statistics']['ci90_pct'] == [-0.1,0.3]


@pytest.mark.parametrize('workflow,flag,receipt_name', [
    ('reference','--reference-dir','reference-receipt.json'),
    ('search','--recommendation','search-receipt.json')])
def test_public_validation_reports_scope_from_the_validated_receipt(tmp_path, monkeypatch, capsys,
        workflow, flag, receipt_name):
    from expertflow.stock import cli
    from expertflow.compiler.evidence import EvidenceStore
    identities, candidate, inputs = scoped_fixture()
    manifest = {'inputs':inputs,'candidate':candidate} if workflow == 'reference' else {
        'inputs':inputs,'incumbent_id':'incumbent','candidates':{'incumbent':candidate}}
    receipt = {'experiment':{'manifest':manifest,'recommended_id':'incumbent','outcomes':[{}]*10}}
    (tmp_path/receipt_name).write_text(json.dumps(receipt))
    @contextmanager
    def load(*args):
        def validate(argv):
            print(json.dumps({'status':'VALIDATED-STOCK-'+workflow.upper(),'plan_sha256':'e'*64}))
            return 0
        yield SimpleNamespace(main=validate,EvidenceStore=EvidenceStore)
    monkeypatch.setattr(cli,'_load_driver',load)
    monkeypatch.setattr(cli,'_check_project',lambda path:path)
    assert cli.main([workflow,'validate',flag,str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['decision']['coverage']['inputs']['inputs']['workload_sha256'] == identities['workload_sha256']
    assert result['decision']['utility_gain_established'] is False


def test_receipt_changed_during_validation_cannot_supply_public_scope(tmp_path, monkeypatch, capsys):
    from expertflow.stock import cli
    from expertflow.compiler.evidence import EvidenceStore
    receipt = tmp_path/'search-receipt.json'
    receipt.write_text('{"experiment":{}}')
    @contextmanager
    def load(*args):
        def validate(argv):
            receipt.write_text('{"changed":true}')
            print(json.dumps({'status':'VALIDATED-STOCK-RECOMMENDATION'}))
            return 0
        yield SimpleNamespace(main=validate,EvidenceStore=EvidenceStore)
    monkeypatch.setattr(cli,'_load_driver',load)
    monkeypatch.setattr(cli,'_check_project',lambda path:path)
    assert cli.main(['search','validate','--recommendation',str(tmp_path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'IDENTITY-STOP'
    assert result['decision']['evidence_verified'] is False
