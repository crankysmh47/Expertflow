"""CPU/native-artifact fixtures, never scientific wider coverage evidence."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time
import subprocess

import pytest


def api():
    from expertflow.stock import wider
    return wider


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


def fixture_context(root, patch, *, rates=None, fail=False):
    from test_compiler_pipeline import FakeRunner, inputs as make_inputs
    from test_compiler_stock_discovery import FixtureEligibility, HOST
    from expertflow.compiler.plan import CandidatePlan, RuntimeSettings
    from expertflow.compiler.schema import canonical_payload, canonical_sha256
    from expertflow.compiler.stock_search import scheduling_space, screening_schedule
    from scripts import benchmark_compiler_stock_utility as utility
    wider = api()
    clock = Clock()
    patch.setattr(wider.time, 'monotonic_ns', clock.now)
    patch.setattr(wider.time, 'sleep', clock.sleep)
    patch.setattr(wider.EligibilityRegistry, 'with_builtins', classmethod(lambda cls: FixtureEligibility()))
    patch.setattr(utility, 'audit_defaults', lambda *args: {'fixture': True})
    inputs = make_inputs(root)
    # This study guards the complete runtime inventory at each spawn.
    from expertflow.compiler.runner import RuntimeBinding
    from expertflow.compiler.schema import ArtifactIdentity
    from expertflow.compiler.preflight import file_sha256
    binary_dir=root/'binaries'
    binary_dir.mkdir()
    server,cli,cuda=binary_dir/'llama-server.exe',binary_dir/'llama-cli.exe',root/'cuda.dll'
    for path in (server,cli,cuda):path.write_bytes(path.name.encode())
    identity=lambda p:ArtifactIdentity(str(p.resolve()),p.stat().st_size,file_sha256(p))
    manifest={'binaries':{p.name:file_sha256(p) for p in (server,cli)},'dependencies':{},
        'cuda_runtime_sha256':file_sha256(cuda),'patches':[]}
    inputs=replace(inputs,stock=RuntimeBinding(identity(server),json.dumps(manifest),(),identity(cuda)))
    inputs = replace(inputs, workload=replace(inputs.workload, threads=8, cuda_graphs='on'))
    default = CandidatePlan(inputs.identities(inputs.stock), RuntimeSettings(99, True))
    candidates = scheduling_space(default, HOST).candidates
    proof = FixtureEligibility().attest(inputs, HOST, root)
    case = {'case_id':'fixture', 'planned_root':str(root/'case'), 'default_id':default.candidate_id,
        'model_artifact':canonical_payload(inputs.model.identity), 'model_ir_sha256':canonical_sha256(inputs.model),
        'workload_sha256':canonical_sha256(inputs.workload), 'runtime_sha256':inputs.stock.sha256,
        'provider_id':proof['provider_id'], 'fixed_settings':canonical_payload(default.settings),
        'candidate_ids':[c.candidate_id for c in candidates],
        'screening_schedule':canonical_payload(screening_schedule(c.candidate_id for c in candidates)),
        'maximum_native_processes':107}
    sequence = {'manifest_sha256':'a'*64,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'source_files':wider.sources(),
        'host_environment':deepcopy(HOST), 'source_repository':str(root),
        'default_source_proof':{'fixture':True}, 'frozen_monotonic_ns':clock.now(),
        'sequence_started_monotonic_ns':clock.now(), 'registration_sha256':'b'*64,
        'registration':{'cases':[case]}}
    runners = []
    def factory(store):
        runner = FakeRunner(store, fail=fail, rates=rates or (lambda c, s: 22 if
            (c.identities.workload.threads,c.settings.cuda_graphs)==(12,'on') else 20))
        runners.append(runner)
        return runner
    return wider, inputs, case, sequence, clock, factory, runners, HOST


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    root=tmp_path_factory.mktemp('wider')
    with pytest.MonkeyPatch.context() as patch:
        context=fixture_context(root,patch)
        wider,inputs,case,sequence,clock,factory,runners,host=context
        report=wider.execute_case(inputs,case,sequence,runner_factory=factory,capture=lambda:deepcopy(host))
    return context,report


def validate(context, report, monkeypatch):
    from test_compiler_stock_discovery import FixtureEligibility
    from scripts import benchmark_compiler_stock_utility as utility
    wider,inputs,case,sequence,clock,factory,runners,host=context
    monkeypatch.setattr(wider.EligibilityRegistry,'with_builtins',classmethod(lambda cls:FixtureEligibility()))
    monkeypatch.setattr(utility,'audit_defaults',lambda *args:{'fixture':True})
    return wider.validate_case(report,inputs,sequence,host_environment=host)


def test_positive_runs_exact_107_independent_calls_with_fixed_wait(completed,monkeypatch):
    context,report=completed
    wider,inputs,case,sequence,clock,factory,runners,host=context
    assert report['status']=='PASS-STOCK-UTILITY-PRODUCT', report.get('reason')
    assert len(report['attempts'])==107==sum(len(r.calls) for r in runners)
    assert clock.waits==[30]*107
    result=validate(context,report,monkeypatch)
    assert result['status']==report['status']
    assert result['selected_default'] is False
    assert result['utility_gain_established'] is True


@pytest.mark.parametrize('mutation',['statistics','cost','selection','prefix','wait','owner','root','scope','collection'])
def test_reconstruction_rejects_tampered_proof(completed,monkeypatch,mutation):
    context,original=completed
    report=deepcopy(original)
    if mutation=='statistics':report['statistics']['gain']['geometric_change_pct']=1000
    elif mutation=='cost':report['statistics']['automatic_evaluations']=1
    elif mutation=='selection':report['automatic_id']=report['manifest']['default_id']
    elif mutation=='prefix':report['rows'].pop()
    elif mutation=='wait':report['attempts'][0]['wait_elapsed_ns']=0
    elif mutation=='owner':report['attempts'][1]['process_identity']=report['attempts'][0]['process_identity']
    elif mutation=='root':report['attempts'][0]['output_dir']=str(Path(report['manifest']['experiment_root'])/'elsewhere')
    elif mutation=='scope':report['manifest']['case']['model_ir_sha256']='f'*64
    elif mutation=='collection':
        report['collection_finished_monotonic_ns']=report['manifest']['case_started_monotonic_ns']
        report['collection_wall_seconds']=0
    with pytest.raises(ValueError):validate(context,report,monkeypatch)


def test_default_optimal_tie_is_neutral_and_has_no_product(tmp_path,monkeypatch):
    context=fixture_context(tmp_path,monkeypatch,rates=lambda c,s:20)
    wider,inputs,case,sequence,clock,factory,runners,host=context
    report=wider.execute_case(inputs,case,sequence,runner_factory=factory,capture=lambda:deepcopy(host))
    assert report['status']=='NO-UTILITY-GAIN'
    assert report['automatic_id']==case['default_id']
    assert len(report['attempts'])==86 and 'consumer' not in report
    result=validate(context,report,monkeypatch)
    assert result['selected_default'] is True and result['utility_gain_established'] is False


def test_failed_start_retained_as_valid_partial_without_gain(tmp_path,monkeypatch):
    context=fixture_context(tmp_path,monkeypatch,fail=True)
    wider,inputs,case,sequence,clock,factory,runners,host=context
    report=wider.execute_case(inputs,case,sequence,runner_factory=factory,capture=lambda:deepcopy(host))
    assert report['status']=='ENVIRONMENT-BLOCKED' and len(report['attempts'])==1
    assert report['statistics'] is None
    result=validate(context,report,monkeypatch)
    assert result['utility_gain_established'] is False and result['complete_utility'] is False
    altered=deepcopy(report)
    altered['status']='PASS-STOCK-UTILITY-PRODUCT'
    with pytest.raises(ValueError):validate(context,altered,monkeypatch)


def test_source_drift_during_wait_prevents_native_launch(tmp_path,monkeypatch):
    context=fixture_context(tmp_path,monkeypatch)
    wider,inputs,case,sequence,clock,factory,runners,host=context
    def drift(seconds):
        clock.sleep(seconds)
        monkeypatch.setattr(wider,'sources',lambda:{'changed':'source'})
    monkeypatch.setattr(wider.time,'sleep',drift)
    report=wider.execute_case(inputs,case,sequence,runner_factory=factory,capture=lambda:deepcopy(host))
    assert report['status']=='VALIDATION-STOP' and sum(len(r.calls) for r in runners)==0
    assert len(report['attempts'])==1 and report['attempts'][0]['native_started'] is False


def test_input_load_cost_can_exhaust_case_before_first_launch(tmp_path,monkeypatch):
    context=fixture_context(tmp_path,monkeypatch)
    wider,inputs,case,sequence,clock,factory,runners,host=context
    report=wider.execute_case(inputs,case,sequence,runner_factory=factory,capture=lambda:deepcopy(host),input_load_seconds=14400)
    assert report['status']=='RESOURCE-BUDGET-STOP'
    assert len(report['attempts'])==0 and sum(len(r.calls) for r in runners)==0


def test_registered_scope_rejects_changed_workload_before_call(tmp_path,monkeypatch):
    context=fixture_context(tmp_path,monkeypatch)
    wider,inputs,case,sequence,clock,factory,runners,host=context
    with pytest.raises(ValueError,match='registered'):
        wider.scope_case(replace(inputs,workload=replace(inputs.workload,seed=43)),case,sequence,host)


def test_owned_spawn_rechecks_wall_budget_after_native_preparation(tmp_path,monkeypatch):
    from expertflow.compiler.evidence import EvidenceStore
    context=fixture_context(tmp_path,monkeypatch)
    wider,inputs,case,sequence,clock,factory,runners,host=context
    root=Path(case['planned_root'])
    root.mkdir()
    m={'experiment_root':str(root),'source_files':wider.sources(),'host_environment':host,
        'case_started_monotonic_ns':clock.now(),'sequence_started_monotonic_ns':clock.now(),
        'input_load_seconds':0,'manifest_sha256':'a'*64}
    report={'manifest':m,'attempts':[]}
    calls=[]
    class PreparingRunner:
        def __init__(self):
            self.store=EvidenceStore(root/'utility.sqlite3')
            self.process_factory=lambda *a,**kw:calls.append('native-spawn')
        def run_once(self,*args,**kwargs):
            clock.value+=14400_000_000_000
            self.process_factory(['fixture'])
            from expertflow.compiler.runner import MeasurementOutcome
            return MeasurementOutcome('environment_blocked',None,'fixture',None,str(kwargs['output_dir']))
    inner=PreparingRunner()
    original=inner.process_factory
    guarded=wider.PacedRunner(inner,report,lambda:deepcopy(host))
    with pytest.raises(wider.ResourceStop):
        guarded.run_once(None,inputs.model,inputs.stock,output_dir=root/'raw/one',stage='fixture',measured=True)
    assert calls==[] and inner.process_factory is original
