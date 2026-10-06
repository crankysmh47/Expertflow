"""Sequence contract controls use injected CPU fixtures, not measurements."""
from copy import deepcopy
import json
from pathlib import Path
import time

import pytest


def driver():
    from scripts import benchmark_compiler_stock_coverage
    return benchmark_compiler_stock_coverage


def registration(tmp_path):
    return {'cases':[{'case_id':f'case-{n}','planned_root':str(tmp_path/'sequence'/f'case-{n}')}
                     for n in range(4)],'registration_sha256':'c'*64,
            'source_repository':str(tmp_path),'default_source_proof':{},'host_environment':{}}


def test_statistical_failure_continues_but_environment_failure_stops():
    module=driver()
    assert module.can_continue({'status':'NO-UTILITY-GAIN','all_raw_records_valid':True})
    assert module.can_continue({'status':'PASS-STOCK-UTILITY-PRODUCT','all_raw_records_valid':True})
    assert not module.can_continue({'status':'VARIANCE-STOP','all_raw_records_valid':False})
    assert not module.can_continue({'status':'ENVIRONMENT-BLOCKED','all_raw_records_valid':True})
    assert not module.can_continue({'status':'RESOURCE-BUDGET-STOP','all_raw_records_valid':True})


def test_sequence_freezes_all_cases_before_any_call_and_stops_on_environment(tmp_path,monkeypatch):
    module=driver()
    data=registration(tmp_path)
    loaded=[]
    monkeypatch.setattr(module.wider,'scope_case',lambda *args:(None,None,None))
    monkeypatch.setattr(module.utility,'audit_defaults',lambda *args:{})
    monkeypatch.setattr(module,'require_committed_sources',lambda source_files:None)
    def loader(case,reg):
        loaded.append(case['case_id'])
        return {'fixture':case['case_id']}
    def execute(inputs,case,sequence,**kwargs):
        assert len(loaded)==4
        assert len(sequence['case_inputs'])==4
        assert Path(case['planned_root']).parent.joinpath('frozen-manifest.json').is_file()
        Path(case['planned_root']).mkdir()
        report={'status':'NO-UTILITY-GAIN' if case['case_id']=='case-0' else 'ENVIRONMENT-BLOCKED',
            'manifest':{'input_load_seconds':kwargs['input_load_seconds'],'case_started_monotonic_ns':time.monotonic_ns()}}
        Path(case['planned_root'],'report.json').write_text(json.dumps(report))
        return report
    monkeypatch.setattr(module.wider,'execute_case',execute)
    monkeypatch.setattr(module.wider,'validate_case',lambda r,*a,**kw:{'status':r['status'],
        'all_raw_records_valid':r['status']=='NO-UTILITY-GAIN','attempts':1,'native_processes':0})
    report=module.run_sequence(data,loader=loader,runner_factory=lambda store:None,capture=lambda:{})
    assert report['status']=='SEQUENCE-STOP'
    assert [c['status'] for c in report['cases']]==['NO-UTILITY-GAIN','ENVIRONMENT-BLOCKED','NOT-RUN','NOT-RUN']


def test_sequence_refuses_existing_root_before_loading(tmp_path):
    module=driver()
    data=registration(tmp_path)
    (tmp_path/'sequence').mkdir()
    with pytest.raises(ValueError,match='fresh'):
        module.run_sequence(data,loader=lambda *args:pytest.fail('must not load'),capture=lambda:{})


def test_coverage_public_routing_accepts_run_validate_and_keeps_inspect():
    from expertflow.stock.cli import DRIVERS
    assert DRIVERS['coverage']==('benchmark_compiler_stock_coverage.py',{'run','validate'})


def test_case_walltime_includes_reconstruction():
    assert driver().case_wall_seconds({'manifest':{'input_load_seconds':5,'case_started_monotonic_ns':1_000_000_000}},
        finished_ns=11_000_000_000)==15


@pytest.mark.parametrize('path_spelling',['native','posix'])
def test_live_loader_normalizes_model_path_without_changing_other_inputs(tmp_path,monkeypatch,path_spelling):
    from dataclasses import replace
    from test_compiler_pipeline import inputs as make_inputs
    from expertflow.compiler.schema import canonical_sha256
    module=driver()
    raw=make_inputs(tmp_path)
    canonical_path=Path(raw.model.identity.path).resolve().as_posix()
    if path_spelling=='posix':
        raw=replace(raw,model=replace(raw.model,identity=replace(raw.model.identity,path=canonical_path)))
    calls=[]
    def live_loader(request,*,live):
        calls.append(live)
        return raw
    monkeypatch.setattr(module,'load_compiler_inputs',live_loader)
    case={'planned_root':str(tmp_path/'case'),'descriptor':'descriptor','inventory':'inventory',
        'workload':'workload','runtime_identity':'runtime'}
    loaded=module.load_case_inputs(case,{'hardware':'hardware'})
    expected=replace(raw,model=replace(raw.model,identity=replace(raw.model.identity,path=canonical_path)))
    assert calls==[True] and loaded==expected
    assert loaded.model.identity.sha256==raw.model.identity.sha256
    assert Path(loaded.model.identity.path).samefile(raw.model.identity.path)
    assert canonical_sha256(loaded.model)==canonical_sha256(expected.model)


def test_reconstruction_wall_cap_removes_accepted_gain_claim():
    report={'manifest':{'input_load_seconds':0,'case_started_monotonic_ns':1_000_000_000}}
    raw={'status':'PASS-STOCK-UTILITY-PRODUCT','utility_gain_established':True}
    result=driver().apply_resource_gate(report,raw,start=10_000_000_000,finish=14402_000_000_000,
        sequence_started=1_000_000_000)
    assert result['resource_budget_pass'] is False and result['utility_gain_established'] is False
    assert result['raw_utility_product_gate_pass'] is True


def test_installed_stock_adapter_must_match_declared_source(tmp_path,monkeypatch):
    module=driver()
    import expertflow.stock
    installed=tmp_path/'installed'
    declared=tmp_path/'src/expertflow/stock'
    installed.mkdir()
    declared.mkdir(parents=True)
    (installed/'__init__.py').write_text('')
    (installed/'wider.py').write_text('modified installed behavior')
    (declared/'__init__.py').write_text('')
    (declared/'wider.py').write_text('declared reviewed behavior')
    monkeypatch.setattr(expertflow.stock,'__file__',str(installed/'__init__.py'))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError,match='installed stock'):
        module.require_matching_package()


@pytest.fixture(scope='module')
def completed_sequence(tmp_path_factory):
    from test_stock_wider import fixture_context
    from expertflow.compiler.schema import canonical_payload,canonical_sha256
    root=tmp_path_factory.mktemp('wider-sequence')
    with pytest.MonkeyPatch.context() as patch:
        context=fixture_context(root,patch,rates=lambda c,s:20)
        wider,inputs,case,sequence,clock,factory,runners,host=context
        module=driver()
        data={'cases':[{**case,'case_id':f'case-{n}','planned_root':str(root/'sequence'/f'case-{n}')}
                       for n in range(4)],'source_repository':str(root),'default_source_proof':{'fixture':True},
              'host_environment':canonical_payload(host)}
        from expertflow.compiler.refinement import balanced_schedule
        data['paired_schedule']=canonical_payload(balanced_schedule())
        data['registration_sha256']=canonical_sha256(data)
        patch.setattr(module,'require_committed_sources',lambda *args:None)
        report=module.run_sequence(data,loader=lambda *args:inputs,runner_factory=factory,capture=lambda:deepcopy(host))
    return context,data,report


def validate_fixture(completed,monkeypatch,report=None):
    from test_compiler_stock_discovery import FixtureEligibility
    from scripts import benchmark_compiler_stock_utility as utility
    context,data,original=completed
    wider,inputs,case,sequence,clock,factory,runners,host=context
    module=driver()
    monkeypatch.setattr(module,'REGISTERED_SHA',data['registration_sha256'])
    monkeypatch.setattr(module,'verify_registration',lambda d,p:d)
    monkeypatch.setattr(wider.EligibilityRegistry,'with_builtins',classmethod(lambda cls:FixtureEligibility()))
    monkeypatch.setattr(utility,'audit_defaults',lambda *args:{'fixture':True})
    return module.validate_sequence(original if report is None else report,loader=lambda *args:inputs,capture=lambda:deepcopy(host))


def test_four_complete_neutral_cases_reconstruct_without_native_calls(completed_sequence,monkeypatch):
    context,data,report=completed_sequence
    assert report['status']=='COMPLETE-STOCK-COVERAGE',report.get('reason')
    assert report['attempts']==344 and report['native_processes']==344
    before=sum(len(r.calls) for r in context[6])
    result=validate_fixture(completed_sequence,monkeypatch)
    assert result['additional_native_calls']==0 and before==sum(len(r.calls) for r in context[6])
    assert all(c['status']=='NO-UTILITY-GAIN' and c['selected_default'] for c in result['cases'])


def test_independent_raw_auditor_rebuilds_four_neutral_gates(completed_sequence):
    import runpy
    auditor=runpy.run_path('docs/research/evidence/stock-coverage-20261005/independent_audit.py')['audit']
    report=completed_sequence[2]
    result=auditor(Path(report['manifest']['experiment_root'])/'report.json')
    assert result['status']=='RAW-AUDIT-PASS' and result['native_processes']==344
    assert all(c['utility_verdict']=='NO-UTILITY-GAIN' for c in result['cases'])


def test_independent_bootstrap_rejects_incomplete_rates():
    import runpy
    paired=runpy.run_path('docs/research/evidence/stock-coverage-20261005/independent_audit.py')['paired']
    with pytest.raises(ValueError):paired([20]*9,[22]*10)
    result=paired([20]*10,[22]*10)
    assert result['geometric_change_pct']==pytest.approx(10)
    assert result['ci95_pct'][0]>0


def test_public_coverage_validation_uses_one_restored_reader_pool(tmp_path,monkeypatch,capsys):
    from contextlib import contextmanager
    from expertflow.stock import cli,wider,wider_audit
    original_loader=cli._load_driver
    constructors=[wider.EvidenceStore,wider_audit.EvidenceStore,wider_audit.repeatability.EvidenceStore]
    @contextmanager
    def loader(project,filename):
        with original_loader(project,filename) as module:
            def validate(argv):
                first=module.wider.EvidenceStore(tmp_path/'fixture.sqlite3')
                assert first is wider_audit.EvidenceStore(tmp_path/'fixture.sqlite3')
                assert first is module.utility.EvidenceStore(tmp_path/'fixture.sqlite3')
                print(json.dumps({'status':'SEQUENCE-STOP','cases':[],'attempts':0,'native_processes':0}))
                return 2
            module.main=validate
            yield module
    monkeypatch.setattr(cli,'_load_driver',loader)
    assert cli.main(['coverage','validate'])==2
    output=json.loads(capsys.readouterr().out)
    assert output['validation']['database_readers']==1
    assert output['validation']['native_calls']==0 and output['decision']['evidence_verified'] is True
    assert constructors==[wider.EvidenceStore,wider_audit.EvidenceStore,wider_audit.repeatability.EvidenceStore]


@pytest.mark.parametrize('mutation',['stats','cost','resource','unrun','count','sequence-cost','sequence-start'])
def test_sequence_rejects_claimed_result_and_cost_mutations(completed_sequence,monkeypatch,mutation):
    report=deepcopy(completed_sequence[2])
    if mutation=='stats':report['cases'][0]['statistics']['gain']['geometric_change_pct']=100
    elif mutation=='cost':report['cases'][0]['case_wall_seconds']=0
    elif mutation=='resource':report['cases'][0]['resource_budget_pass']=False
    elif mutation=='unrun':report['cases'][1]['status']='NOT-RUN'
    elif mutation=='count':report['native_processes']=0
    elif mutation=='sequence-cost':report['sequence_wall_seconds']=0
    elif mutation=='sequence-start':report['sequence_started_monotonic_ns']+=1
    with pytest.raises(ValueError):validate_fixture(completed_sequence,monkeypatch,report)
