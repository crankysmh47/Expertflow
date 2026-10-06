"""Family adapters are the only boundary allowed to interpret raw model names."""

from dataclasses import dataclass
from typing import Mapping, Protocol

from ..schema import ArtifactIdentity, ModelIR, MoELayerIR, canonical_sha256, require_int, require_text


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    family: str
    architecture: str
    quantization: str
    expert_count: int
    expert_top_k: int
    kv_kind: str
    mtp_kind: str

    def __post_init__(self):
        for name in ('family', 'architecture', 'quantization', 'kv_kind', 'mtp_kind'):
            require_text(getattr(self, name), name)
        require_int(self.expert_count, 'expert_count')
        require_int(self.expert_top_k, 'expert_top_k')
        if self.expert_top_k > self.expert_count:
            raise ValueError('expert top_k exceeds count')


class ModelAdapter(Protocol):
    family: str

    def normalize(self, descriptor: ModelDescriptor, inventory: Mapping[str, object],
                  identity: ArtifactIdentity) -> ModelIR: ...


class AdapterRegistry:
    def __init__(self):
        self._adapters = {}

    def register(self, adapter: ModelAdapter):
        if adapter.family in self._adapters:
            raise ValueError(f'duplicate adapter: {adapter.family}')
        self._adapters[adapter.family] = adapter

    def resolve(self, family: str):
        try:
            return self._adapters[family]
        except KeyError as error:
            raise ValueError(f'unsupported model family: {family}') from error

    @classmethod
    def with_builtins(cls):
        from .gemma4 import Gemma4Adapter
        from .granitemoe import GraniteMoEAdapter
        registry = cls()
        registry.register(Gemma4Adapter())
        registry.register(GraniteMoEAdapter())
        return registry


def normalize_routed_inventory(descriptor, inventory, identity):
    """Shared complete routed bank accounting; family adapters validate architecture."""
    if inventory.get('component_sets_consistent') is not True:
        raise ValueError('inconsistent component sets')
    if inventory.get('expert_count') != descriptor.expert_count:
        raise ValueError('expert-count mismatch')
    provenance = inventory.get('model')
    if provenance is not None and (provenance.get('sha256'), provenance.get('bytes')) != (
        identity.sha256, identity.size_bytes
    ):
        raise ValueError('inventory/model identity mismatch')
    rows = inventory.get('layers', [])
    if not rows:
        raise ValueError('absent routed layers')
    components = inventory.get('expert_component_names')
    if not isinstance(components, list) or not components or len(set(components)) != len(components):
        raise ValueError('invalid component sets')
    layers = []
    for row in rows:
        if row.get('components') != components:
            raise ValueError('inconsistent component sets')
        component_bytes = tuple(t['bytes'] for t in inventory.get('tensors', [])
                                if t.get('routed_expert_tensor') is True and t.get('layer_id') == row['layer_id'])
        layer = MoELayerIR(row['layer_id'], descriptor.expert_count, descriptor.expert_top_k,
                           row['complete_expert_bundle_bytes'], row['routed_expert_bank_bytes'], component_bytes)
        if layer.expert_bundle_bytes * layer.expert_count != layer.routed_expert_bank_bytes:
            raise ValueError('inconsistent per-layer bundle bytes')
        layers.append(layer)
    ids = sorted(layer.layer_id for layer in layers)
    if ids != list(range(len(layers))):
        raise ValueError('noncontiguous or duplicate routed layer IDs')
    if inventory.get('routed_layer_count') != len(layers) or inventory.get('routed_layer_ids') != ids:
        raise ValueError('routed layer summary mismatch')
    return ModelIR('1.0.0', identity, descriptor.family, descriptor.architecture,
                   descriptor.quantization, descriptor.expert_count, descriptor.expert_top_k,
                   tuple(layers), descriptor.kv_kind, descriptor.mtp_kind,
                   inventory_sha256=canonical_sha256(inventory))
