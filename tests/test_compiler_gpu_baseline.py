from copy import deepcopy
from dataclasses import replace

import pytest

from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.stock_reference import execute_stock_reference, load_reference_plan
from expertflow.compiler.stock_validation import execute_stock_product
from expertflow.compiler.stock_search import scheduling_space
from test_compiler_pipeline import inputs, FakeRunner
from test_compiler_stock_discovery import FixtureEligibility, HOST


class GPUEligibility(FixtureEligibility):
    def attest(self, inputs, captured, repository):
        return {**super().attest(inputs, captured, repository), 'baseline_cpu_moe':False}


@pytest.fixture(scope='module')
def gpu_reference(tmp_path_factory):
    root = tmp_path_factory.mktemp('gpu-baseline')
    inp = inputs(root)
    store = EvidenceStore(root/'reference.sqlite3')
    runner = FakeRunner(store)
    report = execute_stock_reference(inp, store, runner, root/'reference', source_repository=root,
        host_capture=lambda:deepcopy(HOST), registry=GPUEligibility())
    assert report['status'] == 'REFERENCE-STABLE'
    return inp, store, root/'reference/diagnostic', report


def test_reference_baseline_comes_from_trusted_provider(gpu_reference):
    inp, store, directory, report = gpu_reference
    assert report['manifest']['candidate']['settings']['cpu_moe'] is False
    plan = load_reference_plan(directory, store, identities=inp.identities(inp.stock),
        host_environment=HOST, registry=GPUEligibility())
    assert plan.candidate.settings.cpu_moe is False


def test_product_uses_the_same_gpu_baseline(gpu_reference, tmp_path):
    inp, source, directory, _ = gpu_reference
    target = EvidenceStore(tmp_path/'product.sqlite3')
    report = execute_stock_product(inp, directory/'execution-plan.json', source, target,
        FakeRunner(target), tmp_path/'product', host_capture=lambda:deepcopy(HOST))
    assert report['status'] == 'PASS-STOCK-FALLBACK'
    assert len(report['rows']) == 20
    assert report['frozen']['settings']['cpu_moe'] is False


def test_gpu_scheduling_space_preserves_placement(gpu_reference):
    inp, store, directory, _ = gpu_reference
    plan = load_reference_plan(directory, store, identities=inp.identities(inp.stock),
        host_environment=HOST, registry=GPUEligibility())
    space = scheduling_space(plan.candidate, HOST)
    assert len(space.candidates) == 6
    assert all(c.settings.cpu_moe is False for c in space.candidates)
    assert {c.identities.workload.threads for c in space.candidates} == {8,12,16}


@pytest.mark.parametrize('control', ['on', 'off'])
def test_pdl_cannot_leak_into_stock_search(gpu_reference, control):
    inp, store, directory, _ = gpu_reference
    plan = load_reference_plan(directory, store, identities=inp.identities(inp.stock),
        host_environment=HOST, registry=GPUEligibility())
    changed = replace(plan.candidate, settings=replace(plan.candidate.settings, cuda_pdl=control))
    with pytest.raises(ValueError): scheduling_space(changed, HOST)


@pytest.fixture(scope='module')
def gpu_product(gpu_reference, tmp_path_factory):
    inp, source, directory, _ = gpu_reference
    root = tmp_path_factory.mktemp('gpu-product')
    store = EvidenceStore(root/'product.sqlite3')
    report = execute_stock_product(inp, directory/'execution-plan.json', source, store,
        FakeRunner(store), root/'product', host_capture=lambda:deepcopy(HOST))
    assert report['status'] == 'PASS-STOCK-FALLBACK'
    return inp, store, root/'product/accepted'


def test_published_gpu_search_requires_its_own_baseline_provider(gpu_product, tmp_path):
    from expertflow.compiler.stock_discovery import prepare_search
    inp, store, accepted = gpu_product
    with pytest.raises(ValueError, match='baseline'):
        prepare_search(inp, accepted/'execution-plan.json', accepted/'acceptance-receipt.json',
            store, tmp_path/'search', host_environment=HOST, source_repository=tmp_path,
            registry=FixtureEligibility(), space_config={'policy':'explicit',
            'excluded_threads':[], 'maximum_native_processes':38})


def test_real_gpu_baseline_search_contract_reconstructs(gpu_product, tmp_path):
    from expertflow.compiler.stock_discovery import execute_stock_search, load_search_recommendation
    inp, source, accepted = gpu_product
    target = EvidenceStore(tmp_path/'search.sqlite3')
    report = execute_stock_search(inp, accepted/'execution-plan.json', accepted/'acceptance-receipt.json',
        source, target, FakeRunner(target), tmp_path/'search', host_capture=lambda:deepcopy(HOST),
        registry=GPUEligibility(), source_repository=tmp_path, space_config={
        'policy':'explicit','excluded_threads':[],'maximum_native_processes':38})
    assert report['status'] == 'RECOMMENDED-INCUMBENT' and len(report['screening']) == 18
    plan = load_search_recommendation(tmp_path/'search/recommended', target,
        host_environment=HOST, registry=GPUEligibility())
    assert plan.candidate.settings.cpu_moe is False
