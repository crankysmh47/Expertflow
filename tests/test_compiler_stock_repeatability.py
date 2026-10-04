"""Repeatability orchestration contracts; fixtures are not native speed evidence."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from test_compiler_refinement import setup_execution
from test_compiler_stock_discovery import HOST


def api():
    from scripts import benchmark_compiler_stock_repeatability
    return benchmark_compiler_stock_repeatability


def test_documented_script_entry_point_loads_without_launching_a_model():
    result = subprocess.run([sys.executable, 'scripts/benchmark_compiler_stock_repeatability.py', '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--transfer-workload' in result.stdout


class Clock:
    def __init__(self):
        self.value = time.monotonic_ns()
        self.waits = []

    def now(self):
        self.value += 1_000_000
        return self.value

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.value += int(seconds * 1e9)


def patch_dependencies(patch, clock):
    from scripts import benchmark_compiler_stock_utility as utility
    from test_compiler_stock_discovery import FixtureEligibility
    patch.setattr(api().time, 'sleep', clock.sleep)
    patch.setattr(api().time, 'monotonic_ns', clock.now)
    patch.setattr(api(), 'verify_prerequisite', lambda *args, **kwargs: {'fixture': 'native-artifact fixture'})
    patch.setattr(utility.EligibilityRegistry, 'with_builtins', classmethod(lambda cls: FixtureEligibility()))
    patch.setattr(utility, 'audit_defaults', lambda *args: {'fixture': True})
    patch.setattr(utility, 'audit_protocol_scope', lambda *args: {'fixture': True})


def patch_validation(monkeypatch):
    from scripts import benchmark_compiler_stock_utility as utility
    from test_compiler_stock_discovery import FixtureEligibility
    monkeypatch.setattr(api(), 'verify_prerequisite', lambda *args, **kwargs: {'fixture':'native-artifact fixture'})
    monkeypatch.setattr(utility.EligibilityRegistry,'with_builtins',classmethod(lambda cls:FixtureEligibility()))
    monkeypatch.setattr(utility,'audit_defaults',lambda *args:{'fixture':True})
    monkeypatch.setattr(utility,'audit_protocol_scope',lambda *args:{'fixture':True})


def run_fixture(root, *, rates=None):
    from test_compiler_pipeline import FakeRunner
    inp, source, _, _, plan = setup_execution(root)
    prior = root/'prior.json'
    prior.write_text('{"fixture":true}')
    runners = []

    def factory(store):
        runner = FakeRunner(store, rates=rates or (lambda candidate, stage: 22 if
            (candidate.identities.workload.threads, candidate.settings.cuda_graphs) == (12, 'on') else 20))
        runners.append(runner)
        return runner

    report = api().run_followup(inp, inp, plan, source, prior, root/'study',
        source_repository=root, runner_factory=factory, host_capture=lambda: deepcopy(HOST))
    return inp, source, plan, prior, runners, report


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp('stock-repeatability')
    clock = Clock()
    with pytest.MonkeyPatch.context() as patch:
        patch_dependencies(patch, clock)
        clock.terminal_states=[]
        original=api().validate_followup
        def observed(report,*args,**kwargs):
            persisted=Path(report['manifest']['experiment_root'])/'report.json'
            clock.terminal_states.append(json.loads(persisted.read_text())['status'])
            return original(report,*args,**kwargs)
        patch.setattr(api(),'validate_followup',observed)
        data = run_fixture(root)
    return root, clock, data


def test_two_independent_blocks_consumer_and_transfer_share_one_attempt_budget(completed):
    root, clock, (_, _, _, _, runners, report) = completed
    assert report['status'] == 'PASS-STOCK-REPEATABILITY-TRANSFER'
    assert [block['status'] for block in report['blocks']] == ['PASS-MEASUREMENT'] * 2
    assert len(report['attempts']) == sum(len(r.calls) for r in runners) == 148
    assert clock.waits == [30] * 148
    assert report['consumer']['status'] == 'MEASURED-ACCEPTED-STOCK'
    assert report['transfer']['status'] == 'PASS-STOCK-UTILITY-PRODUCT'
    assert not (root/'study/block-a/accepted').exists()
    assert (root/'study/block-b/accepted/acceptance-receipt.json').is_file()
    assert len({(a['process_identity']['pid'], a['process_identity']['creation_time_100ns'],
                 a['process_identity']['run_id']) for a in report['attempts']}) == 148


@pytest.mark.parametrize('failed_block, expected_attempts', [('block-a', 20), ('block-b', 40)])
def test_inconclusive_block_is_never_pooled_or_retried(tmp_path, monkeypatch, failed_block, expected_attempts):
    clock = Clock()
    patch_dependencies(monkeypatch, clock)
    from test_compiler_pipeline import FakeRunner
    original = FakeRunner.run_once

    def changed(self, candidate, model, binding, **kwargs):
        self.rates = lambda c, stage: 24 if failed_block in str(kwargs['output_dir']) and stage.endswith('sealed') else 22
        return original(self, candidate, model, binding, **kwargs)

    monkeypatch.setattr(FakeRunner, 'run_once', changed)
    _, _, _, _, runners, report = run_fixture(tmp_path)
    assert report['status'] == 'REPEATABILITY-STOP'
    assert len(report['attempts']) == sum(len(r.calls) for r in runners) == expected_attempts
    assert not (tmp_path/'study/block-b/accepted').exists()
    assert not (tmp_path/'study/consumer').exists()
    assert not (tmp_path/'study/transfer').exists()


def test_prerequisite_is_original_failed_study_not_an_arbitrary_pass_boolean(tmp_path):
    inp, source, _, _, plan = setup_execution(tmp_path)
    prior = tmp_path/'forged.json'
    prior.write_text('{"status":"PASS-STOCK-UTILITY","statistics":{"status":"PASS-STOCK-UTILITY"}}')
    with pytest.raises(ValueError, match='original.*report'):
        api().verify_prerequisite(inp, inp, plan, source, prior, tmp_path, HOST)


def test_prerequisite_matches_registered_paths_on_windows(tmp_path, monkeypatch):
    module = api()
    inp, source, _, _, plan_path = setup_execution(tmp_path)
    from expertflow.compiler.plan import load_execution_plan
    plan = load_execution_plan(plan_path,store=source)
    prior = tmp_path/'prior.json'
    prior.write_text(json.dumps({'automatic_id':plan.candidate.candidate_id, 'rows':[
        {'label':f'manual-pair-{i:02}-sealed','measurement_id':mid}
        for i,mid in enumerate(plan.candidate.measurement_ids)]}))
    monkeypatch.setattr(module,'file_sha256',lambda path:module.PRIOR_REPORT_SHA256)
    monkeypatch.setattr(module.utility,'validate_result',lambda *args,**kwargs:{'status':'PRODUCT-VALIDATION-STOP','gain':{'geometric_change_pct':10}})
    registered=iter(str(Path('configs/compiler')/name) for name in
        ('gemma4-q6-single-request.json','gemma4-q6-utility-transfer.json'))
    monkeypatch.setattr(module.utility,'audit_protocol_scope',lambda value:{'registered_workload':next(registered)})
    monkeypatch.setattr(module.runpy,'run_path',lambda path:{'audit':lambda *args:{'actual_native_processes':106,
        'utility_verdict':'PASS-STOCK-UTILITY','terminal_status':'PRODUCT-VALIDATION-STOP'}})
    result=module.verify_prerequisite(inp,inp,plan_path,source,prior,tmp_path,HOST)
    assert result['source_plan_sha256']==plan.plan_sha256


def guard_setup(tmp_path, *, inner=None):
    module = api()
    clock = Clock()
    host = lambda: deepcopy(HOST)
    report = {'manifest': {'host_environment': HOST, 'source_files': {'tracked': 'same'},
        'maximum_native_processes': 148, 'experiment_root': str(tmp_path), 'manifest_sha256': 'a'*64,
        'frozen_monotonic_ns': clock.now()}, 'attempts': [], 'status': 'RUNNING'}
    events = []

    class Inner:
        store = type('Store', (), {'path': tmp_path/'store.sqlite3'})()

        def run_once(self, *args, **kwargs):
            events.append('native')
            persisted = json.loads((tmp_path/'report.json').read_text())
            assert persisted['attempts'][-1]['status'] == 'attempting'
            return type('Outcome', (), {'status': 'environment_blocked', 'measurement_id': None})()

    runner = module.GuardedRunner(inner or Inner(), report, host, lambda: {'tracked':'same'},
        sleep_fn=lambda seconds: (events.append(('wait', seconds)), clock.sleep(seconds)), clock=clock.now)
    return runner, report, events


def test_wait_and_original_guards_precede_every_native_call(tmp_path):
    runner, report, events = guard_setup(tmp_path)
    runner.run_once(output_dir=tmp_path/'raw/a', stage='one')
    assert events == [('wait', 30), 'native']
    assert report['attempts'][0]['wait_elapsed_ns'] >= 30_000_000_000
    assert report['attempts'][0]['native_started'] is False


def test_outer_freeze_includes_all_later_paired_collector_sources():
    from expertflow.compiler.refinement import paired_source_files
    outer = api().sources()
    for name, digest in paired_source_files(product=True).items():
        assert outer.get(str(Path(name).resolve())) == digest, name


def test_last_native_return_cannot_escape_original_source_freeze(tmp_path):
    runner, report, events = guard_setup(tmp_path)
    original = runner.runner.run_once

    def drift(*args, **kwargs):
        outcome = original(*args, **kwargs)
        runner.source_capture = lambda: {'tracked': 'changed'}
        return outcome

    runner.runner.run_once = drift
    with pytest.raises(ValueError, match='source/host changed'):
        runner.run_once(output_dir=tmp_path/'raw/a', stage='last')
    assert events == [('wait', 30), 'native']
    assert len(report['attempts']) == 1
    assert report['attempts'][0]['status'] == 'exception'


@pytest.mark.parametrize('condition', ['budget', 'outside', 'source-before', 'source-during', 'host-during'])
def test_budget_root_and_drift_stop_before_native(tmp_path, condition):
    runner, report, events = guard_setup(tmp_path)
    if condition == 'budget': report['attempts'] = [{}] * 148
    if condition == 'source-before': runner.source_capture = lambda: {'tracked':'changed'}
    if condition == 'source-during':
        runner.sleep_fn = lambda seconds: setattr(runner, 'source_capture', lambda: {'tracked':'changed'})
    if condition == 'host-during':
        runner.sleep_fn = lambda seconds: setattr(runner, 'capture', lambda: {'changed':True})
    output = tmp_path.parent/'foreign' if condition == 'outside' else tmp_path/'raw/a'
    with pytest.raises(ValueError): runner.run_once(output_dir=output, stage='one')
    assert 'native' not in events


def test_post_native_timeout_retains_attempt_and_observed_owner(tmp_path):
    class Broken:
        store = type('Store', (), {'path':tmp_path/'store.sqlite3'})()

        def run_once(self, *args, **kwargs):
            output = kwargs['output_dir']
            output.mkdir(parents=True)
            (output/'run-start.json').write_text(json.dumps({'pid':123,'creation_time_100ns':456,
                'run_id':'observed','started_monotonic_ns':789}))
            raise subprocess.TimeoutExpired('native', 300)

    runner, report, _ = guard_setup(tmp_path, inner=Broken())
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_once(output_dir=tmp_path/'raw/a', stage='one')
    persisted = json.loads((tmp_path/'report.json').read_text())['attempts'][0]
    assert persisted['native_started'] is True
    assert persisted['process_identity']['run_id'] == 'observed'
    assert persisted['exception_type'] == 'TimeoutExpired'


def test_attempt_audit_counts_observed_failed_processes(tmp_path):
    class Broken:
        store = type('Store', (), {'path':tmp_path/'store.sqlite3'})()

        def run_once(self, *args, **kwargs):
            output=kwargs['output_dir']
            output.mkdir(parents=True)
            (output/'run-start.json').write_text(json.dumps({'pid':123,'creation_time_100ns':456,
                'run_id':'observed','creation_source':'GetProcessTimes','started_monotonic_ns':runner.clock()}))
            raise subprocess.TimeoutExpired('native',300)

    runner, report, _ = guard_setup(tmp_path,inner=Broken())
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_once(output_dir=tmp_path/'raw/a',stage='one')
    assert api().audit_attempts(report)==1
    report['attempts'][0]['wait_elapsed_ns']=0
    with pytest.raises(ValueError,match='wait'):
        api().audit_attempts(report)


def test_native_context_cannot_be_rebound_to_another_outer_manifest(tmp_path,monkeypatch):
    from expertflow.compiler.plan import load_execution_plan
    from expertflow.compiler.evidence import EvidenceStore
    from test_compiler_pipeline import FakeRunner
    inp,source,_,_,plan_path=setup_execution(tmp_path)
    study=tmp_path/'study'
    study.mkdir()
    fake=FakeRunner(EvidenceStore(study/'native.sqlite3'))
    clock=Clock()
    monkeypatch.setattr(api().time,'monotonic_ns',clock.now)
    runner,report,_=guard_setup(study,inner=fake)
    runner.clock=clock.now
    runner.sleep_fn=clock.sleep
    report['manifest']['frozen_monotonic_ns']=clock.now()
    plan=load_execution_plan(plan_path,store=source)
    runner.run_once(plan.candidate,inp.model,inp.stock,output_dir=study/'raw/a',measured=True,stage='one')
    assert api().audit_attempts(report)==1
    report['manifest']['manifest_sha256']='b'*64
    with pytest.raises(ValueError,match='context'):
        api().audit_attempts(report)


def test_kernel_owner_reuse_is_rejected_even_with_different_run_uuid(tmp_path):
    class Broken:
        store=type('Store',(),{'path':tmp_path/'store.sqlite3'})()
        def run_once(self,*args,**kwargs):
            output=kwargs['output_dir']
            output.mkdir(parents=True)
            (output/'run-start.json').write_text(json.dumps({'pid':123,'creation_time_100ns':456,
                'creation_source':'GetProcessTimes','run_id':output.name,'started_monotonic_ns':runner.clock()}))
            raise subprocess.TimeoutExpired('native',300)
    runner,report,_=guard_setup(tmp_path,inner=Broken())
    for name in ('a','b'):
        with pytest.raises(subprocess.TimeoutExpired):
            runner.run_once(output_dir=tmp_path/'raw'/name,stage='one')
    with pytest.raises(ValueError,match='owner'):
        api().audit_attempts(report)


def test_partial_environment_stop_binds_prefix_controls_outcomes_and_journal(tmp_path,monkeypatch):
    from test_compiler_pipeline import FakeRunner
    clock=Clock()
    patch_dependencies(monkeypatch,clock)
    original=FakeRunner.run_once
    def failing(self,*args,**kwargs):
        if kwargs['stage'].startswith('product-') and len(self.calls)==2:
            self.fail=True
        return original(self,*args,**kwargs)
    monkeypatch.setattr(FakeRunner,'run_once',failing)
    inp,source,plan,prior,_,report=run_fixture(tmp_path)
    assert report['status']=='REPEATABILITY-STOP' and len(report['attempts'])==3
    assert api().validate_followup(report,inp,inp,plan,source,prior,source_repository=tmp_path,host_environment=HOST)['native_processes']==2
    path=tmp_path/'study/block-a/report.json'
    block=json.loads(path.read_text())
    block['frozen']['seed']=0
    path.write_text(json.dumps(block))
    report['blocks'][0]['report_sha256']=api().file_sha256(path)
    with pytest.raises(ValueError,match='frozen'):
        api().validate_followup(report,inp,inp,plan,source,prior,source_repository=tmp_path,host_environment=HOST)


def test_complete_result_reconstructs_both_blocks_and_nested_transfer(completed, monkeypatch):
    root, _, (inp, source, plan, prior, _, report) = completed
    monkeypatch.setattr(api(), 'verify_prerequisite', lambda *args, **kwargs: {'fixture':'native-artifact fixture'})
    from scripts import benchmark_compiler_stock_utility as utility
    from test_compiler_stock_discovery import FixtureEligibility
    monkeypatch.setattr(utility.EligibilityRegistry,'with_builtins',classmethod(lambda cls:FixtureEligibility()))
    monkeypatch.setattr(utility,'audit_defaults',lambda *args:{'fixture':True})
    monkeypatch.setattr(utility,'audit_protocol_scope',lambda *args:{'fixture':True})
    result = api().validate_followup(report, inp, inp, plan, source, prior, source_repository=root, host_environment=HOST)
    assert result['status'] == 'PASS-STOCK-REPEATABILITY-TRANSFER'
    changed = deepcopy(report)
    changed['attempts'][0]['wait_elapsed_ns'] = 0
    with pytest.raises(ValueError, match='wait'):
        api().validate_followup(changed, inp, inp, plan, source, prior, source_repository=root, host_environment=HOST)
    changed = deepcopy(report)
    changed['blocks'][0]['status'] = 'INCONCLUSIVE'
    with pytest.raises(ValueError):
        api().validate_followup(changed, inp, inp, plan, source, prior, source_repository=root, host_environment=HOST)


@pytest.mark.parametrize('change', ['unknown-status', 'relabel-attempt', 'reorder-attempts', 'collector-cost'])
def test_terminal_state_journal_and_elapsed_cost_cannot_be_rewritten(completed, monkeypatch, change):
    root, _, (inp, source, plan, prior, _, report) = completed
    patch_validation(monkeypatch)
    changed=deepcopy(report)
    if change=='unknown-status': changed['status']='PASS-ARBITRARY'
    elif change=='relabel-attempt': changed['attempts'][0]['status']='exception'
    elif change=='reorder-attempts': changed['attempts'][0],changed['attempts'][1]=changed['attempts'][1],changed['attempts'][0]
    else: changed['collector_elapsed_ns']=-1
    with pytest.raises(ValueError):
        api().validate_followup(changed,inp,inp,plan,source,prior,source_repository=root,host_environment=HOST)


def test_collector_persists_full_elapsed_cost_and_reconstructs_before_success(completed):
    _, clock, (_, _, _, _, _, report)=completed
    assert report['collector_elapsed_ns']==report['collector_finished_monotonic_ns']-report['collector_started_monotonic_ns']
    assert report['collector_elapsed_ns']>=sum(a['wait_elapsed_ns'] for a in report['attempts'])
    assert report['reconstruction']['status']=='PASS-STOCK-REPEATABILITY-TRANSFER'
    assert clock.terminal_states==['PENDING-RECONSTRUCTION']


def test_consumer_reconstruction_requires_accepted_reference_token_parity(completed,monkeypatch):
    from expertflow.compiler.evidence import EvidenceStore
    root, _, (inp,source,plan,prior,_,report)=completed
    patch_validation(monkeypatch)
    original=EvidenceStore.verify_measurement
    def different(self,mid):
        result=original(self,mid)
        if mid==report['consumer']['measurement_id']:
            result={**result,'generated_tokens_sha256':'0'*64}
        return result
    monkeypatch.setattr(EvidenceStore,'verify_measurement',different)
    with pytest.raises(ValueError,match='consumer'):
        api().validate_followup(report,inp,inp,plan,source,prior,source_repository=root,host_environment=HOST)


@pytest.fixture(scope='module')
def stopped(tmp_path_factory):
    root=tmp_path_factory.mktemp('stock-repeatability-stop')
    clock=Clock()
    with pytest.MonkeyPatch.context() as patch:
        patch_dependencies(patch,clock)
        data=run_fixture(root,rates=lambda candidate,stage:24 if stage.endswith('sealed') else 22)
    return root,data


@pytest.mark.parametrize('change',['statistics','schedule','outcome','promote'])
def test_negative_block_reconstructs_full_evidence_and_prevents_promotion(stopped,monkeypatch,change):
    root,(inp,source,plan,prior,_,report)=stopped
    patch_validation(monkeypatch)
    assert api().validate_followup(report,inp,inp,plan,source,prior,source_repository=root,host_environment=HOST)['status']=='REPEATABILITY-STOP'
    path=root/'study/block-a/report.json'
    original=path.read_bytes()
    block=json.loads(original)
    changed=deepcopy(report)
    if change=='statistics': block['ci90_pct']=[-99,99]
    elif change=='schedule': block['rows'][0]['arm']='sealed' if block['rows'][0]['arm']=='direct' else 'direct'
    elif change=='outcome': block['outcomes'][0]['status']='exception'
    else:
        (root/'study/consumer').mkdir()
    try:
        path.write_text(json.dumps(block))
        changed['blocks'][0]['report_sha256']=api().file_sha256(path)
        with pytest.raises(ValueError):
            api().validate_followup(changed,inp,inp,plan,source,prior,source_repository=root,host_environment=HOST)
    finally:
        path.write_bytes(original)
        if change=='promote': (root/'study/consumer').rmdir()
