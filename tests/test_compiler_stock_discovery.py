"""Complete tiny native artifacts test the collector, not live performance."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.schema import canonical_sha256
from test_compiler_pipeline import FakeRunner
from test_compiler_refinement import setup_execution
from test_compiler_stock_search import host


HOST = {**host(),'architecture':'AMD64'}


def api():
    from expertflow.compiler import stock_discovery
    return stock_discovery


class FixtureEligibility:
    def attest(self, inputs, captured, repository):
        return {'provider_id':'fixture-only','model_ir_sha256':canonical_sha256(inputs.model),
            'runtime_sha256':inputs.stock.sha256,'hardware_sha256':canonical_sha256(inputs.hardware),
            'host_environment_sha256':canonical_sha256(captured),'allowed_controls':['threads','cuda_graphs']}


@pytest.fixture(autouse=True)
def trusted_fixture_provider(monkeypatch):
    # Dependency injection is test-only; CLI production resolves reviewed builtins.
    monkeypatch.setattr(api().EligibilityRegistry,'with_builtins',classmethod(lambda cls:FixtureEligibility()))


@pytest.fixture(scope='module')
def prerequisite(tmp_path_factory):
    from expertflow.compiler.stock_validation import execute_stock_product
    root = tmp_path_factory.mktemp('search-prerequisite')
    inputs,old,source,runner,pending = setup_execution(root)
    result = execute_stock_product(inputs,pending,old,source,runner,root/'product',
        host_capture=lambda:deepcopy(HOST))
    assert result['status'] == 'PASS-STOCK-FALLBACK'
    return inputs,source,root/'product/accepted'


def execute(prerequisite,tmp_path,*,rates=None,fail=False,capture=None,runner_type=FakeRunner):
    inputs,source,accepted = prerequisite
    target = EvidenceStore(tmp_path/'search.sqlite3')
    runner = runner_type(target,fail=fail,rates=rates)
    result = api().execute_stock_search(inputs,accepted/'execution-plan.json',
        accepted/'acceptance-receipt.json',source,target,runner,tmp_path/'search',
        host_capture=capture or (lambda:deepcopy(HOST)),registry=FixtureEligibility(),
        source_repository=tmp_path,excluded_threads={8:'prior rejected hypothesis'})
    return inputs,target,runner,result


def fast_challenger(candidate,stage):
    return 33 if (candidate.identities.workload.threads,candidate.settings.cuda_graphs) == (16,'off') else 30


def test_incumbent_wins_complete_screen_without_self_confirmation(prerequisite,tmp_path):
    inputs,store,runner,report = execute(prerequisite,tmp_path)
    assert report['status'] == 'RECOMMENDED-INCUMBENT' and len(runner.calls) == 12
    assert not report['confirmation']
    result = api().load_search_recommendation(tmp_path/'search/recommended',store,
        host_environment=HOST)
    assert result.candidate.candidate_id == report['manifest']['incumbent_id']


def test_challenger_requires_independent_twenty_run_confirmation(prerequisite,tmp_path):
    inputs,store,runner,report = execute(prerequisite,tmp_path,rates=fast_challenger)
    assert report['status'] == 'RECOMMENDED-CHALLENGER' and len(runner.calls) == 32
    assert len(report['screening']) == 12 and len(report['confirmation']) == 20
    assert report['statistics']['geometric_change_pct'] == pytest.approx(10)
    result = api().load_search_recommendation(tmp_path/'search/recommended',store,host_environment=HOST)
    assert result.candidate.identities.workload.threads == 16 and result.candidate.settings.cuda_graphs == 'off'
    assert len(result.candidate.measurement_ids) == 10
    assert not set(result.candidate.measurement_ids)&{r['measurement_id'] for r in report['screening']}


def test_screening_maximum_cannot_publish_after_negative_confirmation(prerequisite,tmp_path):
    def rates(candidate,stage):
        if 'confirm' in stage and fast_challenger(candidate,stage) == 33:
            return 29
        return fast_challenger(candidate,stage)
    _,store,runner,report = execute(prerequisite,tmp_path,rates=rates)
    assert report['status'] == 'RECOMMENDED-INCUMBENT' and len(runner.calls) == 32
    assert report['confirmation_accepted'] is False
    plan = api().load_search_recommendation(tmp_path/'search/recommended',store,host_environment=HOST)
    assert plan.candidate.candidate_id == report['manifest']['incumbent_id']


def test_partial_failure_retains_outcome_without_retry_or_recommendation(prerequisite,tmp_path):
    _,_,runner,report = execute(prerequisite,tmp_path,fail=True)
    assert report['status'] == 'ENVIRONMENT-BLOCKED' and len(runner.calls) == 1
    assert len(report['outcomes']) == 1 and not (tmp_path/'search/recommended').exists()


def test_real_runner_binds_context_before_process_creation(prerequisite,tmp_path,monkeypatch):
    from expertflow.compiler import preflight
    from expertflow.compiler.plan import load_execution_plan
    from expertflow.compiler.runner import ServerMeasurementRunner
    inputs,source,accepted = prerequisite
    plan = load_execution_plan(accepted/'execution-plan.json')
    monkeypatch.setattr(preflight,'capture_host_environment',lambda:deepcopy(HOST))
    def fail_launch(*args,**kwargs):
        raise OSError('fixture launch failure; no native child')
    runner = ServerMeasurementRunner(source,process_factory=fail_launch)
    context = {'manifest_sha256':'a'*64}
    runner.run_once(plan.candidate,inputs.model,inputs.stock,output_dir=tmp_path/'native',
        measured=True,host_environment=HOST,experiment_context=context)
    assert json.loads((tmp_path/'native/launch.json').read_text())['experiment_context'] == context


def test_warmup_evidence_cannot_satisfy_screening(prerequisite,tmp_path):
    class WarmupRunner(FakeRunner):
        def run_once(self,*args,**kwargs):
            return super().run_once(*args,**{**kwargs,'measured':False})
    _,_,runner,report = execute(prerequisite,tmp_path,runner_type=WarmupRunner)
    assert report['status'] == 'VALIDATION-STOP' and len(runner.calls) == 1
    assert not (tmp_path/'search/recommended').exists()


def test_host_change_stops_before_second_process(prerequisite,tmp_path):
    state = {'calls':0}
    def capture():
        state['calls'] += 1
        return deepcopy(HOST) if state['calls'] <= 2 else {**HOST,'power_policy':{'changed':True}}
    _,_,runner,report = execute(prerequisite,tmp_path,capture=capture)
    assert report['status'] == 'VALIDATION-STOP' and len(runner.calls) == 1
    assert not (tmp_path/'search/recommended').exists()


def prepared(prerequisite,tmp_path,**kwargs):
    inputs,source,accepted = prerequisite
    return api().prepare_search(inputs,accepted/'execution-plan.json',accepted/'acceptance-receipt.json',
        source,tmp_path/'search',host_environment=HOST,source_repository=tmp_path,
        registry=FixtureEligibility(),**kwargs)


@pytest.mark.parametrize('corruption',['provider_id','protocol_version','upstream_commit','source_object_ids','source_files'])
def test_manifest_verifier_requires_trusted_complete_provenance(prerequisite,tmp_path,corruption):
    manifest = prepared(prerequisite,tmp_path,excluded_threads={8:'prior rejection'})
    if corruption == 'source_files':
        manifest['source_files'].pop(next(iter(manifest['source_files'])))
    else:
        manifest['eligibility'][corruption] = 'unreviewed'
    manifest.pop('manifest_sha256')
    manifest['manifest_sha256'] = canonical_sha256(manifest)
    with pytest.raises(ValueError):
        api()._validate_manifest(manifest,HOST)


def test_default_current_space_cannot_reopen_threads8_or_exceed32(prerequisite,tmp_path):
    manifest = prepared(prerequisite,tmp_path)
    assert len(manifest['candidates']) == 4 and manifest['maximum_native_processes'] == 32
    assert {c['identities']['workload']['threads'] for c in manifest['candidates'].values()} == {12,16}


def test_generic_larger_space_requires_explicit_config_and_coherent_budget(prerequisite,tmp_path):
    manifest = prepared(prerequisite,tmp_path,space_config={'policy':'explicit',
        'excluded_threads':[],'maximum_native_processes':38})
    assert len(manifest['candidates']) == 6 and manifest['maximum_native_processes'] == 38


@pytest.mark.parametrize('corruption',['ranking','budget','host','row','statistics','finalist'])
def test_rehashed_claims_cannot_change_recommendation(prerequisite,tmp_path,corruption):
    _,store,_,report = execute(prerequisite,tmp_path,rates=fast_challenger)
    path = tmp_path/'search/recommended/search-receipt.json'
    receipt = json.loads(path.read_text())
    changed_host = deepcopy(HOST)
    experiment = receipt['experiment']
    if corruption == 'ranking':
        experiment['ranking'].reverse()
    elif corruption == 'budget':
        experiment['manifest']['maximum_native_processes'] = 99
        experiment['manifest'].pop('manifest_sha256')
        experiment['manifest']['manifest_sha256'] = canonical_sha256(experiment['manifest'])
    elif corruption == 'host':
        changed_host['power_policy'] = {'settings_sha256':'changed'}
        experiment['manifest']['host_environment'] = changed_host
        experiment['manifest'].pop('manifest_sha256')
        experiment['manifest']['manifest_sha256'] = canonical_sha256(experiment['manifest'])
    elif corruption == 'row':
        experiment['screening'].pop()
    elif corruption == 'statistics':
        experiment['statistics']['geometric_change_pct'] = 99
    else:
        experiment['finalist_id'] = experiment['manifest']['incumbent_id']
    receipt.pop('receipt_sha256')
    receipt['receipt_sha256'] = canonical_sha256(receipt)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        api().load_search_recommendation(path.parent,store,host_environment=changed_host)
