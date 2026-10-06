from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from expertflow.compiler.adapters import AdapterRegistry, ModelDescriptor
from expertflow.compiler.schema import ArtifactIdentity, canonical_sha256

INVENTORY = json.loads(Path('docs/research/evidence/compiler-granite-20261004/tensor-inventory.json').read_text())


def normalize(inventory=None, descriptor=None):
    inv = deepcopy(INVENTORY) if inventory is None else inventory
    model = INVENTORY['model']
    identity = ArtifactIdentity(model['path'], model['bytes'], model['sha256'])
    descriptor = descriptor or ModelDescriptor('granitemoe', 'granitemoe', 'Q6_K', 32, 8, 'standard', 'none')
    return AdapterRegistry.with_builtins().resolve('granitemoe').normalize(descriptor, inv, identity)


def test_real_granite_inventory_is_distinct_complete_family():
    model = normalize()
    assert model.family == model.architecture == 'granitemoe'
    assert len(model.moe_layers) == 24
    assert model.expert_count == 32 and model.expert_top_k == 8
    assert model.moe_layers[0].expert_bundle_bytes == 1290240
    assert model.moe_layers[0].component_bank_bytes == (13762560,) * 3
    assert model.inventory_sha256 == canonical_sha256(INVENTORY)


@pytest.mark.parametrize('mutation', ['architecture','count','topk','layers','shape','type','component','identity','axis','slice','scaling','bytes','tensor_layer','tensor_name','dimension'])
def test_granite_rejects_inconsistent_actual_metadata(mutation):
    inv = deepcopy(INVENTORY)
    t = next(t for t in inv['tensors'] if t['routed_expert_tensor'])
    if mutation == 'architecture': inv['gguf_metadata']['general.architecture'] = 'gemma4'
    if mutation == 'count': inv['gguf_metadata']['granitemoe.expert_count'] = 128
    if mutation == 'topk': inv['gguf_metadata']['granitemoe.expert_used_count'] = 4
    if mutation == 'layers': inv['gguf_metadata']['granitemoe.block_count'] = 25
    if mutation == 'shape': t['shape'][0] = 513
    if mutation == 'type': t['type'] = 'Q4_0'
    if mutation == 'component': inv['expert_component_names'].pop()
    if mutation == 'identity': inv['model']['bytes'] += 1
    if mutation == 'axis': t['expert_axis'] = 0
    if mutation == 'slice': t['slice_contiguous'] = False
    if mutation == 'scaling': inv['gguf_metadata']['granitemoe.embedding_scale'] = 0
    if mutation == 'bytes':
        t['bytes'] += 32
        inv['layers'][0]['routed_expert_bank_bytes'] += 32
        inv['layers'][0]['complete_expert_bundle_bytes'] += 1
    if mutation == 'tensor_layer': t['layer_id'] = 24
    if mutation == 'tensor_name': t['name'] = 'blk.24.ffn_down_exps.weight'
    if mutation == 'dimension': inv['gguf_metadata']['granitemoe.embedding_length'] = 1024.0
    with pytest.raises(ValueError): normalize(inv)


def test_wrong_family_descriptor_is_not_reinterpreted():
    descriptor = ModelDescriptor('gemma4', 'gemma4-moe', 'Q6_K', 32, 8, 'standard', 'none')
    with pytest.raises(ValueError): normalize(descriptor=descriptor)


def test_current_gemma_ir_hash_is_unchanged():
    from test_compiler_gemma4_adapter import descriptor_fixture
    inv = json.loads(Path('docs/research/evidence/q6-download/tensor-inventory.json').read_text())
    runtime = json.loads(Path('docs/research/evidence/compiler-phase3/inputs/runtime-identity.json').read_text())
    identity = ArtifactIdentity(**runtime['model'])
    model = AdapterRegistry.with_builtins().resolve('gemma4').normalize(descriptor_fixture(), inv, identity)
    assert canonical_sha256(model) == '3cda0fb02589028b7fce410f7cb3dd42a5322e911e42207be7f157ecb082c424'


@pytest.mark.parametrize('mutation', ['missing_dense', 'dense_shape', 'dense_type', 'dense_bytes',
    'duplicate', 'offset', 'total', 'nonexpert_total', 'tensor_count', 'provenance', 'attention_scale',
    'nonfinite_scale', 'quant_row'])
def test_granite_requires_complete_tensor_accounting(mutation):
    inv = deepcopy(INVENTORY)
    tensor = inv['tensors'][0]
    if mutation == 'missing_dense': inv['tensors'].pop(0)
    if mutation == 'dense_shape': tensor['shape'][0] += 1
    if mutation == 'dense_type': tensor['type'] = 'Q6_K'
    if mutation == 'dense_bytes': tensor['bytes'] += 4
    if mutation == 'duplicate': inv['tensors'][1]['name'] = tensor['name']
    if mutation == 'offset': tensor['data_offset'] = inv['model']['bytes']
    if mutation == 'total': inv['total_tensor_payload_bytes'] += 1
    if mutation == 'nonexpert_total': inv['non_expert_tensor_bytes'] += 1
    if mutation == 'tensor_count': inv['tensor_count'] += 1
    if mutation == 'provenance': del inv['model']
    if mutation == 'attention_scale': inv['gguf_metadata']['granitemoe.attention.scale'] = 0
    if mutation == 'nonfinite_scale': inv['gguf_metadata']['granitemoe.residual_scale'] = float('nan')
    if mutation == 'quant_row': inv['gguf_metadata']['granitemoe.feed_forward_length'] = 513
    with pytest.raises(ValueError): normalize(inv)
