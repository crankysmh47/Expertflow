from dataclasses import replace
import math

import pytest


def api():
    from expertflow.compiler import refinement
    return refinement


def row(pair, arm, rate=25, **changes):
    return {'pair':pair,'arm':arm,'measurement_id':f'{pair}-{arm}',
            'owned_run_sha256':f'{pair}-{arm}', 'candidate_id':'same',
            'identities':{'model_sha256':'model','hardware_sha256':'hardware','runtime_sha256':'runtime','workload_sha256':'workload'},
            'settings_sha256':'settings','generated_tokens_sha256':'tokens','prompt_tokens_sha256':'prompt',
            'validations':{'exact_tokens':True,'memory':True,'cleanup':True},'measured':True,'exit_code':0,
            'decode_tps':rate, **changes}


def rows(ratios):
    return [r for i,ratio in enumerate(ratios) for r in (row(i,'direct',25+i*.1),row(i,'sealed',(25+i*.1)*ratio))]


def test_optional_diagnostics_never_replace_mandatory_memory_state():
    from expertflow.compiler.diagnostics import DiagnosticSampler
    base=lambda pid:{'pid':pid,'state':'allocated','counter_available':True,'dedicated_bytes':100,'device_free_bytes':512 << 20}
    def missing(pid): raise OSError('sensor unavailable')
    sampler=DiagnosticSampler(base,gpu_reader=missing,cpu_reader=lambda pid:{'cpu_performance_pct':120})
    value=sampler(123)
    assert value['state']=='allocated' and value['dedicated_bytes']==100
    assert value['diagnostics']['gpu']['available'] is False
    assert value['diagnostics']['cpu']['cpu_performance_pct']==120
    assert DiagnosticSampler(lambda pid:None,gpu_reader=missing,cpu_reader=missing)(123) is None


def test_optional_factory_failure_is_unavailable_without_closing_live_base(monkeypatch):
    from expertflow.compiler import diagnostics
    def broken(base): raise AttributeError('unsupported diagnostic API')
    monkeypatch.setattr(diagnostics,'_gpu_reader',broken)
    monkeypatch.setattr(diagnostics,'_cpu_reader',broken)
    base=lambda pid:{'pid':pid,'state':'allocated','counter_available':True,'dedicated_bytes':100}
    value=diagnostics.DiagnosticSampler(base)(123)
    assert value['state']=='allocated'
    assert value['diagnostics']['cpu']['available'] is False


def test_optional_runtime_type_failure_is_contained():
    from expertflow.compiler.diagnostics import DiagnosticSampler
    def broken(pid): raise AttributeError('unsupported sample API')
    sample=DiagnosticSampler(lambda pid:{'state':'allocated','counter_available':True},gpu_reader=broken,cpu_reader=lambda pid:None)(1)
    assert sample['state']=='allocated'
    assert sample['diagnostics']['gpu']['available'] is False
    assert sample['diagnostics']['cpu']['available'] is False


def test_balanced_frozen_schedule_and_deterministic_paired_resampling():
    r=api()
    schedule=r.balanced_schedule()
    assert len(schedule)==10 and schedule.count(('direct','sealed'))==5
    assert schedule==r.balanced_schedule()
    outcome=r.evaluate_pairs(rows([1.0]*10))
    assert outcome['status']=='PASS-MEASUREMENT'
    assert outcome['geometric_change_pct']==0 and outcome['ci95_pct']==[0,0]
    assert r.paired_statistics([20]*10,[21.2]*10)['geometric_change_pct']==pytest.approx(6)
    assert r.paired_statistics([20]*10,[21.2]*10)==r.paired_statistics([20]*10,[21.2]*10)


def test_faster_is_not_regression_but_aa_is_not_equivalent():
    outcome=api().evaluate_pairs(rows([1.04]*10))
    assert outcome['noninferior'] is True and outcome['equivalent'] is False
    assert outcome['status']=='INCONCLUSIVE'
    assert api().evaluate_pairs(rows([.95]*10))['status']=='VALIDATION-STOP'


def test_noisy_single_replay_does_not_replace_group_uncertainty():
    result=api().evaluate_pairs(rows([.94,1.06]*5))
    assert result['status']=='INCONCLUSIVE'
    assert result['ci90_pct'][0]<-2 and result['ci90_pct'][1]>2


@pytest.mark.parametrize('change',[
    {'generated_tokens_sha256':'different'}, {'settings_sha256':'different'},
    {'candidate_id':'different'}, {'measured':False}, {'exit_code':1},
    {'decode_tps':float('nan')}, {'validations':{'memory':False}},
    {'owned_run_sha256':'0-direct'}, {'measurement_id':'0-direct'},
])
def test_bad_native_evidence_cannot_pass_statistics(change):
    values=rows([1]*10);values[-1].update(change)
    with pytest.raises(ValueError): api().evaluate_pairs(values)


def test_budget_missing_pair_and_bad_rates_fail_closed():
    with pytest.raises(ValueError): api().evaluate_pairs(rows([1]*9))
    with pytest.raises(ValueError): api().paired_statistics([20]*10,[0]*10)
    with pytest.raises(ValueError): api().paired_statistics([20]*9,[20]*10)


def test_model_priming_does_not_allow_changed_weights(tmp_path):
    from expertflow.compiler.evidence import EvidenceStore
    from test_compiler_runner import model_for_test
    import os
    model=model_for_test(tmp_path);store=EvidenceStore(tmp_path/'store.sqlite3')
    store.prime_model(model)
    path=tmp_path/'model.gguf';stat=path.stat()
    path.write_bytes(b'x'*model.identity.size_bytes)
    os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
    with pytest.raises(ValueError,match='model artifact'): store.prime_model(model)


def setup_execution(tmp_path):
    from expertflow.compiler.evidence import EvidenceStore
    from expertflow.compiler.pipeline import atomic_json
    from expertflow.compiler.plan import seal_candidate,CandidateStatus
    from expertflow.compiler.stock import stock_candidate_matrix
    from test_compiler_pipeline import FakeRunner,inputs
    inp=inputs(tmp_path);source=EvidenceStore(tmp_path/'source.sqlite3');runner=FakeRunner(source)
    candidate=stock_candidate_matrix(inp.identities(inp.stock))[-1]
    ids=[runner.run_once(candidate,inp.model,inp.stock,output_dir=tmp_path/f'source-{i}',measured=True,stage='confirmation').measurement_id for i in range(10)]
    plan=seal_candidate(replace(candidate,status=CandidateStatus.MEASURED,measurement_ids=tuple(ids)),source,candidate.identities,None)
    path=tmp_path/'pending.json';atomic_json(path,plan)
    target=EvidenceStore(tmp_path/'target.sqlite3')
    return inp,source,target,FakeRunner(target),path


def test_execution_uses_exact_budget_and_preserves_source(tmp_path):
    inp,source,target,runner,path=setup_execution(tmp_path)
    before=path.read_bytes()
    report=api().execute_pairs(inp,path,source,target,runner,tmp_path/'aa')
    assert report['status']=='PASS-MEASUREMENT' and len(runner.calls)==20
    assert path.read_bytes()==before and len(report['rows'])==20
    assert all(call[1].startswith('aa-') for call in runner.calls)
    assert not (tmp_path/'aa/execution-plan.json').exists()


def test_execution_stops_on_partial_environment_failure_without_retry(tmp_path):
    inp,source,target,runner,path=setup_execution(tmp_path)
    runner.fail=True
    report=api().execute_pairs(inp,path,source,target,runner,tmp_path/'aa')
    assert report['status']=='ENVIRONMENT-BLOCKED' and len(runner.calls)==1
    assert len(report['rows'])==0 and (tmp_path/'aa/report.json').exists()


def test_source_change_after_first_run_stops_without_collecting_more(tmp_path,monkeypatch):
    inp,source,target,runner,path=setup_execution(tmp_path)
    from expertflow.compiler import preflight
    original=preflight.file_sha256
    def changed(file):
        if str(file).endswith('refinement.py') and runner.calls: return '0'*64
        return original(file)
    monkeypatch.setattr(preflight,'file_sha256',changed)
    report=api().execute_pairs(inp,path,source,target,runner,tmp_path/'aa')
    assert report['status']=='VALIDATION-STOP' and 'source changed' in report['reason']
    assert len(runner.calls)==1 and report['frozen']['source_files']


def setup_thread_execution(tmp_path):
    from expertflow.compiler.evidence import EvidenceStore
    from test_compiler_pipeline import FakeRunner
    inp,source,aa_store,runner,path=setup_execution(tmp_path)
    aa=api().execute_pairs(inp,path,source,aa_store,runner,tmp_path/'aa')
    assert aa['status']=='PASS-MEASUREMENT'
    target=EvidenceStore(tmp_path/'threads.sqlite3')
    return inp,aa_store,target,FakeRunner(target),tmp_path/'aa/report.json'


def test_thread_experiment_binds_distinct_workloads_and_stops_at_budget(tmp_path):
    inp,aa_store,target,runner,path=setup_thread_execution(tmp_path)
    result=api().execute_thread_pairs(inp,path,aa_store,target,runner,tmp_path/'threads')
    assert result['status']=='INCONCLUSIVE' and len(runner.calls)==20
    assert result['geometric_change_pct']==0
    assert result['frozen']['candidates']['threads8']['identities']['workload']['threads']==8
    assert len({row['candidate_id'] for row in result['rows']})==2
    assert not (tmp_path/'threads/execution-plan.json').exists()


def test_thread_experiment_rejects_forged_prerequisite_before_native(tmp_path):
    inp,aa_store,target,runner,path=setup_thread_execution(tmp_path)
    import json
    value=json.loads(path.read_text());value['rows'][0]['decode_tps']=999
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='prerequisite'):
        api().execute_thread_pairs(inp,path,aa_store,target,runner,tmp_path/'threads')
    assert not runner.calls


def test_thread_experiment_preserves_native_failure_without_retry(tmp_path):
    inp,aa_store,target,runner,path=setup_thread_execution(tmp_path)
    runner.fail=True
    result=api().execute_thread_pairs(inp,path,aa_store,target,runner,tmp_path/'threads')
    assert result['status']=='ENVIRONMENT-BLOCKED' and len(runner.calls)==1
