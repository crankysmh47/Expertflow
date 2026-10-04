"""Stock utility proof contracts; fixtures do not establish live speedups."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest


def api():
    from scripts import benchmark_compiler_stock_utility
    return benchmark_compiler_stock_utility


@pytest.fixture(autouse=True)
def trusted_test_dependencies(monkeypatch):
    from test_compiler_stock_discovery import FixtureEligibility
    monkeypatch.setattr(api().EligibilityRegistry,'with_builtins',classmethod(lambda cls:FixtureEligibility()))
    monkeypatch.setattr(api(),'audit_defaults',lambda repository,host:{'scope':'synthetic test fixture'})
    real_scope=getattr(api(),'audit_protocol_scope',None)
    if real_scope:
        monkeypatch.setattr(api(),'audit_protocol_scope',lambda inputs:{'scope':'synthetic test fixture'})
    return real_scope


def test_utility_requires_default_gain_manual_equivalence_and_equal_budget():
    passing = api().evaluate_utility([20]*10, [22]*10, [22]*10, [22]*10,
                                    automatic_evaluations=18, manual_evaluations=18)
    assert passing['status'] == 'PASS-STOCK-UTILITY'
    assert passing['gain']['geometric_change_pct'] == pytest.approx(10)
    assert api().evaluate_utility([22]*10,[22]*10,[22]*10,[22]*10,
        automatic_evaluations=18,manual_evaluations=18)['status'] == 'NO-UTILITY-GAIN'
    assert api().evaluate_utility([20]*10,[22]*10,[24]*10,[22]*10,
        automatic_evaluations=18,manual_evaluations=18)['status'] == 'MANUAL-BASELINE-STOP'
    assert api().evaluate_utility([20]*10,[22]*10,[22]*10,[22]*10,
        automatic_evaluations=19,manual_evaluations=18)['status'] == 'TUNING-COST-STOP'


def test_default_resolution_rejects_partial_affinity_or_unsupported_default_source():
    from test_compiler_stock_search import host
    assert api().resolved_defaults(host()) == (8,'on')
    with pytest.raises(ValueError):
        api().resolved_defaults({**host(),'process_affinity_mask':255})


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    from test_compiler_pipeline import FakeRunner
    from test_compiler_refinement import setup_execution
    from test_compiler_stock_discovery import FixtureEligibility, HOST
    root=tmp_path_factory.mktemp('stock-utility')
    inputs,old,store,runner,pending=setup_execution(root)
    def rates(candidate,stage):
        return 22 if (candidate.identities.workload.threads,candidate.settings.cuda_graphs)==(12,'on') else 20
    runner=FakeRunner(store,rates=rates)
    # Module-scoped fixtures precede function autouse fixtures.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(api().EligibilityRegistry,'with_builtins',classmethod(lambda cls:FixtureEligibility()))
        patch.setattr(api(),'audit_defaults',lambda repository,host:{'scope':'synthetic test fixture'})
        if hasattr(api(),'audit_protocol_scope'):
            patch.setattr(api(),'audit_protocol_scope',lambda inputs:{'scope':'synthetic test fixture'})
        report=api().execute_utility(inputs,store,runner,root/'utility',
            source_repository=root,host_capture=lambda:deepcopy(HOST),registry=FixtureEligibility(),
            default_source_proof={'scope':'synthetic test fixture'},include_product=False)
    return inputs,store,root,report,HOST


def test_complete_utility_uses_independent_search_and_confirmation_owners(completed):
    inputs,store,root,report,host=completed
    assert report['status']=='PASS-STOCK-UTILITY'
    assert len(report['outcomes'])==86
    assert report['manifest']['maximum_native_processes']==107
    assert len({row['measurement_id'] for row in report['rows']})==86
    assert api().reconstruct_utility(report,store,host_environment=host)['status']=='PASS-STOCK-UTILITY'
    assert report['automatic_id']==report['manual_id']


def test_validation_rejects_supplied_workload_or_source_outside_manifest(completed):
    inputs,store,root,report,host=completed
    api().check_live_inputs(inputs,report['manifest'],root)
    with pytest.raises(ValueError,match='supplied'):
        api().check_live_inputs(replace(inputs,workload=replace(inputs.workload,predict_tokens=256)),report['manifest'],root)
    with pytest.raises(ValueError,match='supplied'):
        api().check_live_inputs(inputs,report['manifest'],root/'different-source')


def test_utility_cannot_promote_failed_statistics_to_product_pass(completed):
    inputs,store,root,original,host=completed
    changed=deepcopy(original)
    changed['status']='PASS-STOCK-UTILITY-PRODUCT'
    with pytest.raises((ValueError,FileNotFoundError)):
        api().validate_result(changed,store,inputs,source_repository=root,host_environment=host)


@pytest.mark.parametrize('mutation', ['duplicate','wrong_stage','candidate','rate','source','context','wall'])
def test_receipt_cannot_rewrite_native_proof(completed,mutation):
    inputs,store,root,original,host=completed
    changed=deepcopy(original)
    if mutation=='duplicate': changed['rows'][1]['measurement_id']=changed['rows'][0]['measurement_id']
    elif mutation=='wrong_stage': changed['rows'][0]['label']='auto-screen-0'
    elif mutation=='candidate': changed['rows'][0]['candidate_id']='f'*64
    elif mutation=='rate': changed['rows'][0]['decode_tps']=200
    elif mutation=='source': changed['manifest']['source_files']={}
    elif mutation=='context': changed['manifest']['experiment_root']=str(root/'elsewhere')
    elif mutation=='wall': changed['rows'][0]['run_wall_seconds']=-1
    with pytest.raises(ValueError):
        api().reconstruct_utility(changed,store,host_environment=host)


def test_partial_collection_stops_without_retry_or_publication(tmp_path):
    from test_compiler_pipeline import FakeRunner
    from test_compiler_refinement import setup_execution
    from test_compiler_stock_discovery import FixtureEligibility,HOST
    inputs,old,store,runner,pending=setup_execution(tmp_path)
    runner=FakeRunner(store,fail=True)
    report=api().execute_utility(inputs,store,runner,tmp_path/'utility',
        source_repository=tmp_path,host_capture=lambda:deepcopy(HOST),registry=FixtureEligibility(),
        default_source_proof={'scope':'synthetic test fixture'},include_product=False)
    assert report['status']=='ENVIRONMENT-BLOCKED' and len(runner.calls)==1
    assert len(report['outcomes'])==1 and not (tmp_path/'utility/accepted').exists()


def test_passing_utility_runs_separate_product_database_and_fresh_consumer(tmp_path):
    from test_compiler_pipeline import FakeRunner
    from test_compiler_refinement import setup_execution
    from test_compiler_stock_discovery import FixtureEligibility,HOST
    inputs,old,store,runner,pending=setup_execution(tmp_path)
    def rates(candidate,stage):
        return 22 if (candidate.identities.workload.threads,candidate.settings.cuda_graphs)==(12,'on') else 20
    report=api().execute_utility(inputs,store,FakeRunner(store,rates=rates),tmp_path/'utility',
        source_repository=tmp_path,host_capture=lambda:deepcopy(HOST),registry=FixtureEligibility(),
        default_source_proof={'scope':'synthetic test fixture'},
        product_runner_factory=lambda target:FakeRunner(target,rates=rates))
    assert report['status']=='PASS-STOCK-UTILITY-PRODUCT'
    assert report['product_native_processes']==20
    assert report['consumer']['status']=='MEASURED-ACCEPTED-STOCK'
    assert api().validate_result(report,store,inputs,source_repository=tmp_path,host_environment=HOST)['status']=='PASS-STOCK-UTILITY-PRODUCT'
    changed=deepcopy(report)
    changed['consumer']['measurement_id']=report['rows'][0]['measurement_id']
    with pytest.raises(ValueError):
        api().validate_result(changed,store,inputs,source_repository=tmp_path,host_environment=HOST)


@pytest.mark.parametrize('drift_at,kind',[(0,'source'),(3,'host'),(20,'source')])
def test_original_freeze_guards_every_product_and_consumer_launch(tmp_path,monkeypatch,drift_at,kind):
    from test_compiler_pipeline import FakeRunner
    from test_compiler_refinement import setup_execution
    from test_compiler_stock_discovery import FixtureEligibility,HOST
    inputs,old,store,runner,pending=setup_execution(tmp_path)
    rates=lambda candidate,stage:22 if candidate.identities.workload.threads==12 else 20
    current=deepcopy(HOST)
    product_calls=[]
    class DriftRunner(FakeRunner):
        def run_once(self,*args,**kwargs):
            product_calls.append(kwargs['stage'])
            result=super().run_once(*args,**kwargs)
            if len(product_calls)==drift_at:
                if kind=='source':monkeypatch.setattr(api(),'sources',lambda:{'changed':'source'})
                else:current['process_affinity_mask']=255
            return result
    def factory(target):
        if drift_at==0:
            monkeypatch.setattr(api(),'sources',lambda:{'changed':'source'})
        return DriftRunner(target,rates=rates)
    report=api().execute_utility(inputs,store,FakeRunner(store,rates=rates),tmp_path/'utility',
        source_repository=tmp_path,host_capture=lambda:deepcopy(current),registry=FixtureEligibility(),
        default_source_proof={'scope':'synthetic test fixture'},product_runner_factory=factory)
    assert report['status']!='PASS-STOCK-UTILITY-PRODUCT'
    assert len(product_calls)==drift_at
    assert len(report['attempts'])==86+drift_at
    assert json.loads((tmp_path/'utility/report.json').read_text())['status']==report['status']


@pytest.mark.parametrize('failure',['os','timeout','verification'])
def test_native_exception_retains_attempt_identity_and_terminal_report(tmp_path,failure):
    import subprocess
    from test_compiler_pipeline import FakeRunner
    from test_compiler_refinement import setup_execution
    from test_compiler_stock_discovery import FixtureEligibility,HOST
    inputs,old,store,runner,pending=setup_execution(tmp_path)
    class BrokenRunner(FakeRunner):
        def run_once(self,*args,**kwargs):
            outcome=super().run_once(*args,**kwargs)
            if failure=='timeout':raise subprocess.TimeoutExpired('owned-server',30)
            if failure=='os':raise OSError('post-native artifact error')
            (kwargs['output_dir']/'completion.json').write_text('{}')
            return outcome
    report=api().execute_utility(inputs,store,BrokenRunner(store),tmp_path/'utility',
        source_repository=tmp_path,host_capture=lambda:deepcopy(HOST),registry=FixtureEligibility(),
        default_source_proof={'scope':'synthetic test fixture'},include_product=False)
    assert report['status']=='VALIDATION-STOP'
    assert len(report['attempts'])==1 and len(report['outcomes'])==1
    assert report['attempts'][0]['native_started'] is True
    assert report['attempts'][0]['process_identity']['pid']>0
    assert json.loads((tmp_path/'utility/report.json').read_text())['status']=='VALIDATION-STOP'


def test_live_protocol_accepts_only_registered_q6_workloads(completed,trusted_test_dependencies):
    from expertflow.compiler.reference import load_reference_workload
    from expertflow.compiler.schema import WorkloadIR
    from pathlib import Path
    inputs,store,root,report,host=completed
    production_scope=trusted_test_dependencies
    assert callable(production_scope)
    identity=replace(inputs.model.identity,sha256='089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba')
    model=replace(inputs.model,identity=identity)
    for name in ('gemma4-q6-single-request.json','gemma4-q6-utility-transfer.json'):
        workload=WorkloadIR.from_reference(load_reference_workload(Path.cwd(),Path('configs/compiler/'+name)))
        valid=replace(inputs,model=model,workload=workload)
        assert production_scope(valid)['registered_workload'].endswith(name)
        for changes in ({'predict_tokens':256},{'context_size':8192},{'seed':43},{'prompt':'unregistered'}):
            with pytest.raises(ValueError,match='registered'):
                production_scope(replace(valid,workload=replace(workload,**changes)))
        with pytest.raises(ValueError,match='Q6'):
            production_scope(inputs)


def test_reconstruction_rejects_rehashed_protocol_scope(completed):
    inputs,store,root,original,host=completed
    changed=deepcopy(original)
    changed['manifest']['protocol_scope']={'scope':'different registered workload'}
    payload=dict(changed['manifest'])
    payload.pop('manifest_sha256')
    changed['manifest']['manifest_sha256']=api().canonical_sha256(payload)
    with pytest.raises(ValueError,match='scope'):
        api().reconstruct_utility(changed,store,host_environment=host)
