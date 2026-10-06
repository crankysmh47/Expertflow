"""Metadata-derived GGUF tensor and routed-expert accounting."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Iterable


_ROUTED = re.compile(r"^blk\.(?P<layer>\d+)\.(?P<component>[^.]*_exps(?:\..+)?)$")


def summarize_tensor_inventory(
    tensors: Iterable[dict[str, object]], *, expert_count: int
) -> dict[str, object]:
    """Annotate expert axes/slices and aggregate tensor payload bytes."""

    if expert_count <= 0:
        raise ValueError("expert_count must be positive")
    rows: list[dict[str, object]] = []
    by_layer: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    non_expert_bytes = 0

    for tensor in tensors:
        row = dict(tensor)
        name = str(row["name"])
        byte_count = int(row["bytes"])
        shape = [int(value) for value in row["shape"]]  # type: ignore[arg-type]
        match = _ROUTED.fullmatch(name)
        row["routed_expert_tensor"] = match is not None
        if match is None:
            row["expert_axis"] = None
            row["bytes_per_expert_slice"] = None
            non_expert_bytes += byte_count
        else:
            if byte_count % expert_count:
                raise ValueError(f"expert tensor bytes must be divisible by {expert_count}: {name}")
            axes = [index for index, extent in enumerate(shape) if extent == expert_count]
            if not axes:
                raise ValueError(f"expert tensor has no {expert_count}-element axis: {name}")
            axis = axes[-1] if name.endswith(".weight") else axes[0]
            row["expert_axis"] = axis
            row["bytes_per_expert_slice"] = byte_count // expert_count
            row["layer_id"] = int(match.group("layer"))
            row["component"] = match.group("component")
            by_layer[int(match.group("layer"))].append(row)
        rows.append(row)

    layers = []
    for layer_id in sorted(by_layer):
        components = sorted(by_layer[layer_id], key=lambda item: str(item["component"]))
        layers.append(
            {
                "layer_id": layer_id,
                "components": [str(item["component"]) for item in components],
                "complete_expert_bundle_bytes": sum(
                    int(item["bytes_per_expert_slice"]) for item in components
                ),
                "routed_expert_bank_bytes": sum(int(item["bytes"]) for item in components),
            }
        )

    component_sets = {tuple(layer["components"]) for layer in layers}
    return {
        "tensor_count": len(rows),
        "tensors": rows,
        "expert_count": expert_count,
        "routed_layer_ids": [int(layer["layer_id"]) for layer in layers],
        "routed_layer_count": len(layers),
        "expert_component_names": list(next(iter(component_sets))) if len(component_sets) == 1 else None,
        "component_sets_consistent": len(component_sets) == 1,
        "layers": layers,
        "total_tensor_payload_bytes": sum(int(row["bytes"]) for row in rows),
        "non_expert_tensor_bytes": non_expert_bytes,
        "total_routed_expert_bank_bytes": sum(
            int(layer["routed_expert_bank_bytes"]) for layer in layers
        ),
    }
