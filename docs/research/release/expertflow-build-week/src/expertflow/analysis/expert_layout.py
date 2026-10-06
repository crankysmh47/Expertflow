"""Exact GGUF expert spans and source-derived CUDA slot projections."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class ExpertTensor:
    """Metadata needed to split one layer-level expert tensor."""

    layer_id: int
    component: str
    tensor_name: str
    tensor_type: str
    shape: tuple[int, ...]
    expert_count: int
    encoded_bytes: int
    data_offset: int
    ne0: int
    quant_block_elements: int | None
    quant_block_bytes: int | None


_EXPERT_TENSOR_NAME = re.compile(
    r"^blk\.(?P<layer>[0-9]+)\."
    r"(?P<name>ffn_down_exps\.weight|ffn_gate_up_exps\.weight|"
    r"ffn_down_exps\.scale)$"
)


def expert_tensor_from_gguf_metadata(
    *,
    name: str,
    shape: tuple[int, ...],
    tensor_type: str,
    encoded_bytes: int,
    data_offset: int,
) -> ExpertTensor | None:
    """Map one pinned GGUF tensor row into the expert layout contract."""

    match = _EXPERT_TENSOR_NAME.fullmatch(name)
    if match is None:
        return None
    suffix = match.group("name")
    if suffix == "ffn_down_exps.scale":
        if tensor_type != "F32" or len(shape) != 1:
            raise ValueError("expert scale tensor must be one-dimensional F32")
        component = "down_scale"
        expert_count = shape[0]
        ne0 = 1
        block_elements = None
        block_bytes = None
    else:
        if tensor_type != "Q4_0" or len(shape) != 3:
            raise ValueError("expert weight tensors must be three-dimensional Q4_0")
        component = (
            "down_weight"
            if suffix == "ffn_down_exps.weight"
            else "gate_up_weight"
        )
        expert_count = shape[-1]
        ne0 = shape[0]
        block_elements = 32
        block_bytes = 18
    return ExpertTensor(
        layer_id=int(match.group("layer")),
        component=component,
        tensor_name=name,
        tensor_type=tensor_type,
        shape=shape,
        expert_count=expert_count,
        encoded_bytes=encoded_bytes,
        data_offset=data_offset,
        ne0=ne0,
        quant_block_elements=block_elements,
        quant_block_bytes=block_bytes,
    )


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _row_padding_bytes(
    tensor: ExpertTensor, *, matrix_row_padding: int
) -> int:
    if tensor.quant_block_elements is None and tensor.quant_block_bytes is None:
        return 0
    if tensor.quant_block_elements is None or tensor.quant_block_bytes is None:
        raise ValueError("quantization block metadata must be complete")
    if tensor.quant_block_elements <= 0 or tensor.quant_block_bytes <= 0:
        raise ValueError("quantization block metadata must be positive")
    remainder = tensor.ne0 % matrix_row_padding
    if remainder == 0:
        return 0
    padding_elements = matrix_row_padding - remainder
    if padding_elements % tensor.quant_block_elements != 0:
        raise ValueError("matrix row padding must align to quantization blocks")
    return (
        padding_elements
        // tensor.quant_block_elements
        * tensor.quant_block_bytes
    )


def build_expert_inventory(
    tensors: list[ExpertTensor] | tuple[ExpertTensor, ...],
    *,
    alignment: int,
    matrix_row_padding: int,
    component_order: tuple[str, ...],
    target_layer_ids: tuple[int, ...],
    capacity_per_layer: int,
) -> dict[str, object]:
    """Enumerate every expert's source spans and conservative CUDA slot."""

    source = tuple(tensors)
    if not source:
        raise ValueError("tensors must not be empty")
    if alignment <= 0 or matrix_row_padding <= 0 or capacity_per_layer <= 0:
        raise ValueError("alignment, row padding, and capacity must be positive")
    if not component_order or len(set(component_order)) != len(component_order):
        raise ValueError("component_order must contain unique values")
    if not target_layer_ids or len(set(target_layer_ids)) != len(target_layer_ids):
        raise ValueError("target_layer_ids must contain unique values")

    by_layer: defaultdict[int, dict[str, ExpertTensor]] = defaultdict(dict)
    for tensor in source:
        if tensor.layer_id < 0 or tensor.expert_count <= 0:
            raise ValueError("layer IDs and expert counts must be valid")
        if tensor.encoded_bytes <= 0 or tensor.data_offset < 0 or tensor.ne0 <= 0:
            raise ValueError("tensor byte ranges and ne0 must be valid")
        if tensor.encoded_bytes % tensor.expert_count != 0:
            raise ValueError("tensor encoded bytes must be divisible by experts")
        if tensor.component in by_layer[tensor.layer_id]:
            raise ValueError("layer component values must be unique")
        by_layer[tensor.layer_id][tensor.component] = tensor

    layer_ids = sorted(by_layer)
    if not set(target_layer_ids).issubset(layer_ids):
        raise ValueError("target layers must exist in the tensor inventory")
    expert_counts = {
        tensor.expert_count for components in by_layer.values() for tensor in components.values()
    }
    if len(expert_counts) != 1:
        raise ValueError("all expert tensors must use one expert count")
    expert_count = next(iter(expert_counts))
    required_components = set(component_order)
    if any(set(components) != required_components for components in by_layer.values()):
        raise ValueError("every layer must contain the declared components")

    objects: list[dict[str, object]] = []
    for layer_id in layer_ids:
        for expert_id in range(expert_count):
            cursor = 0
            encoded_total = 0
            components: list[dict[str, object]] = []
            for component_name in component_order:
                tensor = by_layer[layer_id][component_name]
                encoded_bytes = tensor.encoded_bytes // tensor.expert_count
                row_padding = _row_padding_bytes(
                    tensor, matrix_row_padding=matrix_row_padding
                )
                slot_start = _align(cursor, alignment)
                source_start = tensor.data_offset + expert_id * encoded_bytes
                source_end = source_start + encoded_bytes
                tensor_end = tensor.data_offset + tensor.encoded_bytes
                if source_end > tensor_end:
                    raise ValueError("expert source span exceeds tensor bounds")
                allocation_bytes = encoded_bytes + row_padding
                components.append(
                    {
                        "component": component_name,
                        "tensor_name": tensor.tensor_name,
                        "tensor_type": tensor.tensor_type,
                        "shape": list(tensor.shape),
                        "source_start": source_start,
                        "source_end": source_end,
                        "encoded_bytes": encoded_bytes,
                        "cuda_row_padding_bytes": row_padding,
                        "projected_allocation_bytes": allocation_bytes,
                        "slot_start": slot_start,
                        "slot_start_alignment_padding_bytes": slot_start - cursor,
                    }
                )
                cursor = slot_start + allocation_bytes
                encoded_total += encoded_bytes
            projected_slot_bytes = _align(cursor, alignment)
            objects.append(
                {
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "encoded_packed_bytes": encoded_total,
                    "projected_slot_bytes": projected_slot_bytes,
                    "slot_end_padding_bytes": projected_slot_bytes - cursor,
                    "components": components,
                }
            )

    encoded_sizes = {int(item["encoded_packed_bytes"]) for item in objects}
    slot_sizes = {int(item["projected_slot_bytes"]) for item in objects}
    projection_slot_bytes = max(slot_sizes)
    layer_summaries = [
        {
            "layer_id": layer_id,
            "expert_count": expert_count,
            "encoded_bytes": sum(
                int(item["encoded_packed_bytes"])
                for item in objects
                if item["layer_id"] == layer_id
            ),
            "projected_slot_bytes": projection_slot_bytes,
            "projected_all_experts_bytes": projection_slot_bytes * expert_count,
        }
        for layer_id in layer_ids
    ]
    return {
        "schema_version": "1.0.0",
        "measurement_kind": "measured_encoded_projected_cuda_layout",
        "alignment_bytes": alignment,
        "matrix_row_padding_elements": matrix_row_padding,
        "component_order": list(component_order),
        "layer_ids": layer_ids,
        "layer_count": len(layer_ids),
        "expert_count_per_layer": expert_count,
        "object_count": len(objects),
        "all_objects_same_encoded_bytes": len(encoded_sizes) == 1,
        "all_objects_same_projected_slot_bytes": len(slot_sizes) == 1,
        "encoded_object_bytes_min": min(encoded_sizes),
        "encoded_object_bytes_max": max(encoded_sizes),
        "projected_slot_bytes_min": min(slot_sizes),
        "projected_slot_bytes_max": max(slot_sizes),
        "projection": {
            "target_layer_ids": list(target_layer_ids),
            "target_layer_count": len(target_layer_ids),
            "capacity_per_layer": capacity_per_layer,
            "slot_count": len(target_layer_ids) * capacity_per_layer,
            "slot_bytes": projection_slot_bytes,
        },
        "projected_cache_bytes": (
            projection_slot_bytes
            * len(target_layer_ids)
            * capacity_per_layer
        ),
        "tensor_inventory": [asdict(tensor) for tensor in source],
        "layers": layer_summaries,
        "objects": objects,
    }
