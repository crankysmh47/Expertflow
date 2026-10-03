"""Gemma 4's normalized metadata accounting, never a full weight load."""

from ..schema import ModelIR, MoELayerIR, canonical_sha256


class Gemma4Adapter:
    family = 'gemma4'

    def normalize(self, descriptor, inventory, identity):
        if descriptor.family != self.family or descriptor.architecture != 'gemma4-moe':
            raise ValueError('wrong family/architecture for Gemma 4 adapter')
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
            layer = MoELayerIR(row['layer_id'], descriptor.expert_count, descriptor.expert_top_k,
                               row['complete_expert_bundle_bytes'], row['routed_expert_bank_bytes'])
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
