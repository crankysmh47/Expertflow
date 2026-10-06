from copy import deepcopy
import json
from pathlib import Path

import pytest

from expertflow.compiler.adapters import AdapterRegistry, Gemma4Adapter, ModelDescriptor
from expertflow.compiler.schema import ArtifactIdentity, canonical_payload


def inventory_fixture():
    return {'expert_count': 128, 'component_sets_consistent': True,
            'expert_component_names': ['a', 'b'], 'routed_layer_count': 2,
            'routed_layer_ids': [0, 1], 'layers': [
                {'layer_id': layer, 'components': ['a', 'b'],
                 'complete_expert_bundle_bytes': 30, 'routed_expert_bank_bytes': 3840}
                for layer in (0, 1)]}


def descriptor_fixture():
    return ModelDescriptor('gemma4', 'gemma4-moe', 'Q6_K', 128, 8, 'standard', 'none')


def normalize(inventory=None):
    return AdapterRegistry.with_builtins().resolve('gemma4').normalize(
        descriptor_fixture(), inventory or inventory_fixture(), ArtifactIdentity('model.gguf', 10, 'a' * 64))


def test_registry_normalizes_and_ir_has_no_raw_names():
    model = normalize()
    assert len(model.moe_layers) == 2
    assert model.moe_layers[0].expert_bundle_bytes == 30
    assert model.expert_count == 128 and model.expert_top_k == 8
    assert 'components' not in json.dumps(canonical_payload(model))
    registry = AdapterRegistry.with_builtins()
    with pytest.raises(ValueError, match='duplicate'):
        registry.register(Gemma4Adapter())
    with pytest.raises(ValueError, match='unsupported'):
        registry.resolve('unknown')


@pytest.mark.parametrize('mutation', [
    lambda i: i.update(component_sets_consistent=False),
    lambda i: i.update(expert_count=64),
    lambda i: i.update(layers=[]),
    lambda i: i['layers'][1].update(layer_id=0),
    lambda i: i['layers'][1].update(layer_id=3),
    lambda i: i['layers'][0].update(complete_expert_bundle_bytes=31),
    lambda i: i['layers'][0].update(components=['a']),
    lambda i: i.update(routed_layer_ids=[0]),
])
def test_inconsistent_inventory_rejected(mutation):
    inventory = inventory_fixture()
    mutation(inventory)
    with pytest.raises(ValueError):
        normalize(inventory)


def test_family_and_inventory_provenance():
    with pytest.raises(ValueError, match='family'):
        Gemma4Adapter().normalize(ModelDescriptor('other', 'other', 'Q6_K', 128, 8, 'standard', 'none'),
                                 inventory_fixture(), ArtifactIdentity('m', 10, 'a' * 64))
    inventory = inventory_fixture()
    inventory['model'] = {'bytes': 10, 'sha256': 'f' * 64}
    with pytest.raises(ValueError, match='identity'):
        normalize(inventory)


def test_real_inventory_normalizes_thirty_layers():
    inventory = json.loads(Path('docs/research/evidence/q6-download/tensor-inventory.json').read_text())
    identity = ArtifactIdentity(inventory['model']['path'], inventory['model']['bytes'], inventory['model']['sha256'])
    model = Gemma4Adapter().normalize(descriptor_fixture(), inventory, identity)
    assert len(model.moe_layers) == 30
    assert model.moe_layers[0].routed_expert_bank_bytes == 685933056
    assert model.moe_layers[0].expert_bundle_bytes == 5358852
