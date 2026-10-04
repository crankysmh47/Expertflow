"""Portable metadata/topology contracts; these are not native model results."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from expertflow.compiler.adapters import AdapterRegistry
from expertflow.compiler.pipeline import CompilationRequest, EnvironmentBlocked, inspect_model, load_compiler_inputs
from expertflow.compiler.schema import ModelIR, MoELayerIR, canonical_sha256
from test_compiler_pipeline import inputs
from test_compiler_stock_discovery import FixtureEligibility, HOST
from test_compiler_stock_search import candidate, host


class ContractAdapter:
    family = 'contract-moe'

    def normalize(self, descriptor, inventory, identity):
        layers = tuple(MoELayerIR(row['id'], descriptor.expert_count, descriptor.expert_top_k,
            row['bytes'] // descriptor.expert_count, row['bytes']) for row in inventory['banks'])
        return ModelIR('1.0.0', identity, descriptor.family, descriptor.architecture,
            descriptor.quantization, descriptor.expert_count, descriptor.expert_top_k,
            layers, descriptor.kv_kind, descriptor.mtp_kind, inventory_sha256=canonical_sha256(inventory))


def metadata(tmp_path, experts=8, bank=1600):
    descriptor = tmp_path/'descriptor.json'
    descriptor.write_text(json.dumps({'family':'contract-moe','architecture':'contract-routing',
        'quantization':'FP16','expert_count':experts,'expert_top_k':2,'kv_kind':'standard','mtp_kind':'none'}))
    inventory = tmp_path/'inventory.json'
    inventory.write_text(json.dumps({'model':{'path':str(tmp_path/'missing.gguf'),'bytes':10,'sha256':'a'*64},
        'banks':[{'id':0,'bytes':bank},{'id':1,'bytes':bank*2}]}))
    registry = AdapterRegistry.with_builtins()
    registry.register(ContractAdapter())
    return descriptor,inventory,registry


@pytest.mark.parametrize('experts,bank', [(8,1600),(16,6400)])
def test_registered_adapter_normalizes_distinct_family_inventory_without_gemma_names(tmp_path,experts,bank):
    descriptor,inventory,registry = metadata(tmp_path,experts,bank)
    model = inspect_model(descriptor,inventory,adapter_registry=registry)
    assert model.family == 'contract-moe' and model.expert_count == experts
    assert tuple(layer.expert_bundle_bytes for layer in model.moe_layers) == (bank//experts,2*bank//experts)
    assert model.inventory_sha256 == canonical_sha256(json.loads(inventory.read_text()))
    with pytest.raises(ValueError,match='unsupported'):
        inspect_model(descriptor,inventory)


def test_generic_input_loader_resolves_registered_adapter_before_missing_weights_gate(tmp_path):
    descriptor,inventory,registry = metadata(tmp_path)
    runtime = tmp_path/'runtime.json'
    runtime.write_text(json.dumps({'schema_version':'1.0.0','model':
        {'path':str(tmp_path/'missing.gguf'),'size_bytes':10,'sha256':'a'*64}}))
    request = CompilationRequest(descriptor,inventory,tmp_path/'hardware',tmp_path/'workload',runtime,
        (),tmp_path/'evidence.sqlite3',tmp_path/'output')
    with pytest.raises(EnvironmentBlocked,match='model artifact unavailable'):
        load_compiler_inputs(request,live=False,adapter_registry=registry)


@pytest.mark.parametrize('cores,logical,incumbent', [(4,8,6),(12,24,18),(16,32,24)])
def test_generic_spaces_derive_distinct_topologies_without_current_q6_exclusions(cores,logical,incumbent):
    from expertflow.compiler.stock_search import scheduling_space, semantic_fingerprint
    base = candidate(incumbent)
    space = scheduling_space(base,host(cores,logical))
    assert space.excluded_threads == ()
    assert {c.identities.workload.threads for c in space.candidates} == {cores,incumbent,logical}
    assert len(space.candidates) == 6
    assert {semantic_fingerprint(c) for c in space.candidates} == {semantic_fingerprint(base)}


@pytest.mark.parametrize('changed', ['cpu','affinity','ram','os','power','environment'])
def test_reference_receipts_invalidate_host_changes_before_any_native_launch(tmp_path,changed):
    from expertflow.compiler.stock_reference import prepare_reference, _validate
    inp = inputs(tmp_path)
    manifest = prepare_reference(inp,tmp_path/'future',host_environment=HOST,
        source_repository=tmp_path,registry=FixtureEligibility())
    current = deepcopy(HOST)
    if changed == 'cpu':
        current['cpu'][0]['name'] = 'different CPU'
    elif changed == 'affinity':
        current['process_affinity_mask'] = 255
    elif changed == 'ram':
        current['ram'] = {'configured_speed_mhz':4800}
    elif changed == 'os':
        current['os'] = 'different build'
    elif changed == 'power':
        current['power_policy'] = {'settings_sha256':'f'*64}
    else:
        current['thread_environment'] = {'OMP_NUM_THREADS':'4'}
    with pytest.raises(ValueError,match='host'):
        _validate(manifest,current,FixtureEligibility())
