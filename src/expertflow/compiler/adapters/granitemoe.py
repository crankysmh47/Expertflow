"""Granite MoE metadata interpretation, separate from scheduling eligibility."""
from .base import normalize_routed_inventory
import math

from ..schema import ArtifactIdentity, require_int, require_number, require_text


def _complete_inventory(inventory, metadata, identity, descriptor):
    if inventory.get('schema_version') != '1.0.0' or inventory.get('measurement_kind') != 'gguf_metadata_derived':
        raise ValueError('Granite requires metadata-derived inventory provenance')
    provenance = inventory.get('model')
    if not isinstance(provenance, dict):
        raise ValueError('Granite inventory model provenance missing')
    declared = ArtifactIdentity(provenance.get('path'), provenance.get('bytes'), provenance.get('sha256'))
    if (declared.sha256, declared.size_bytes) != (identity.sha256, identity.size_bytes):
        raise ValueError('Granite inventory/model identity mismatch')
    for key in ('repository', 'revision'):
        require_text(provenance.get(key), 'Granite model '+key)
    dimensions = {}
    for key in ('embedding_length', 'feed_forward_length', 'block_count', 'vocab_size',
                'context_length', 'attention.head_count', 'attention.head_count_kv', 'rope.dimension_count'):
        value = metadata.get('granitemoe.'+key)
        require_int(value, 'Granite '+key)
        dimensions[key] = value
    for key in ('embedding_scale', 'logit_scale', 'residual_scale', 'attention.scale',
                'attention.layer_norm_rms_epsilon', 'rope.freq_base'):
        require_number(metadata.get('granitemoe.'+key), 'Granite '+key, 1e-30)
    width, hidden = dimensions['embedding_length'], dimensions['feed_forward_length']
    heads, kv_heads = dimensions['attention.head_count'], dimensions['attention.head_count_kv']
    if width % heads or heads % kv_heads or dimensions['rope.dimension_count'] != width//heads:
        raise ValueError('unsupported Granite attention dimensions')
    kv_width = width//heads*kv_heads
    expected = {'output_norm.weight': ([width], 'F32'),
                'token_embd.weight': ([width, dimensions['vocab_size']], 'Q6_K')}
    components = {'attn_k.weight': ([width, kv_width], 'Q6_K'),
        'attn_v.weight': ([width, kv_width], 'Q6_K'), 'attn_q.weight': ([width, width], 'Q6_K'),
        'attn_output.weight': ([width, width], 'Q6_K'), 'attn_norm.weight': ([width], 'F32'),
        'ffn_norm.weight': ([width], 'F32'), 'ffn_gate_inp.weight': ([width, descriptor.expert_count], 'F32'),
        'ffn_down_exps.weight': ([hidden, width, descriptor.expert_count], 'Q6_K'),
        'ffn_gate_exps.weight': ([width, hidden, descriptor.expert_count], 'Q6_K'),
        'ffn_up_exps.weight': ([width, hidden, descriptor.expert_count], 'Q6_K')}
    for layer in range(dimensions['block_count']):
        expected.update({f'blk.{layer}.{name}': value for name, value in components.items()})
    tensors = inventory.get('tensors', [])
    if inventory.get('tensor_count') != len(expected) or len(tensors) != len(expected):
        raise ValueError('incomplete Granite tensor inventory')
    seen, intervals = set(), []
    total = routed_total = 0
    for tensor in tensors:
        name, shape, kind = tensor.get('name'), tensor.get('shape'), tensor.get('type')
        if name in seen or expected.get(name) != (shape, kind):
            raise ValueError('invalid Granite tensor name/shape/type')
        seen.add(name)
        block, size = (256, 210) if kind == 'Q6_K' else (1, 4)
        if shape[0] % block or (tensor.get('quant_block_elements'), tensor.get('quant_block_bytes')) != (block, size):
            raise ValueError('invalid Granite quantized row layout')
        nbytes = math.prod(shape)//block*size
        offset = tensor.get('data_offset')
        require_int(offset, 'Granite tensor offset', 0)
        routed = '_exps.weight' in name
        if (tensor.get('bytes') != nbytes or tensor.get('routed_expert_tensor') is not routed or
                tensor.get('slice_contiguous') is not True or offset % 32 or offset+nbytes > identity.size_bytes):
            raise ValueError('invalid Granite tensor byte accounting')
        if not routed and (tensor.get('expert_axis') is not None or tensor.get('bytes_per_expert_slice') is not None):
            raise ValueError('invalid Granite dense tensor classification')
        intervals.append((offset, offset+nbytes))
        total += nbytes
        routed_total += nbytes if routed else 0
    intervals.sort()
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError('overlapping Granite tensor payloads')
    if (inventory.get('total_tensor_payload_bytes') != total or
            inventory.get('total_routed_expert_bank_bytes') != routed_total or
            inventory.get('non_expert_tensor_bytes') != total-routed_total):
        raise ValueError('Granite inventory payload totals mismatch')


class GraniteMoEAdapter:
    family = 'granitemoe'

    def normalize(self, descriptor, inventory, identity):
        if descriptor.family != self.family or descriptor.architecture != 'granitemoe':
            raise ValueError('wrong family/architecture for Granite MoE adapter')
        if descriptor.quantization != 'Q6_K':
            raise ValueError('unsupported Granite expert quantization layout')
        metadata = inventory.get('gguf_metadata', {})
        if metadata.get('general.architecture') != 'granitemoe':
            raise ValueError('Granite GGUF architecture mismatch')
        if (metadata.get('granitemoe.expert_count'), metadata.get('granitemoe.expert_used_count')) != (
                descriptor.expert_count, descriptor.expert_top_k):
            raise ValueError('Granite routing metadata mismatch')
        if metadata.get('granitemoe.block_count') != inventory.get('routed_layer_count'):
            raise ValueError('Granite block metadata mismatch')
        _complete_inventory(inventory, metadata, identity, descriptor)
        for key in ('embedding_length', 'feed_forward_length', 'embedding_scale', 'logit_scale', 'residual_scale'):
            value = metadata.get('granitemoe.'+key)
            if type(value) not in (int, float) or value <= 0:
                raise ValueError('invalid Granite scaling/dimension metadata')
        width, hidden = metadata['granitemoe.embedding_length'], metadata['granitemoe.feed_forward_length']
        require_int(width, 'Granite embedding length')
        require_int(hidden, 'Granite feed forward length')
        expected = {'ffn_down_exps.weight': [hidden, width, descriptor.expert_count],
                    'ffn_gate_exps.weight': [width, hidden, descriptor.expert_count],
                    'ffn_up_exps.weight': [width, hidden, descriptor.expert_count]}
        if inventory.get('expert_component_names') != sorted(expected):
            raise ValueError('unsupported Granite expert component layout')
        tensors = [t for t in inventory.get('tensors', []) if t.get('routed_expert_tensor') is True]
        if len(tensors) != 3*inventory.get('routed_layer_count', 0):
            raise ValueError('incomplete Granite expert tensors')
        seen = set()
        for tensor in tensors:
            key = (tensor['layer_id'], tensor['component'])
            if (key in seen or tensor.get('shape') != expected.get(tensor['component']) or
                    tensor.get('type') != descriptor.quantization or tensor.get('expert_axis') != 2 or
                    tensor.get('slice_contiguous') is not True or tensor['layer_id'] not in inventory['routed_layer_ids'] or
                    tensor.get('name') != f'blk.{tensor["layer_id"]}.{tensor["component"]}' or
                    tensor.get('quant_block_elements') != 256 or tensor.get('quant_block_bytes') != 210 or
                    tensor.get('bytes') != width*hidden*descriptor.expert_count//256*210):
                raise ValueError('invalid Granite expert tensor layout')
            seen.add(key)
        return normalize_routed_inventory(descriptor, inventory, identity)
