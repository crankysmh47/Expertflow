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
