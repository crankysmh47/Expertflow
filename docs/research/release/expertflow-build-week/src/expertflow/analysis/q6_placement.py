"""Reconcile Q6 tensor inventory with measured llama.cpp placement modes."""

from __future__ import annotations

from collections import defaultdict


def build_placement_manifest(inventory: dict[str, object]) -> dict[str, object]:
    modes: dict[str, list[dict[str, object]]] = {
        "full_cuda_intent": [],
        "cpu_moe": [],
    }
    summaries: dict[str, defaultdict[str, int]] = {
        name: defaultdict(int) for name in modes
    }

    for tensor in inventory["tensors"]:  # type: ignore[index]
        row = dict(tensor)
        name = str(row["name"])
        size = int(row["bytes"])
        routed = bool(row["routed_expert_tensor"])
        for mode in modes:
            backend = (
                "CPU_Mapped"
                if name == "token_embd.weight" or (mode == "cpu_moe" and routed)
                else "CUDA0"
            )
            modes[mode].append(
                {
                    "name": name,
                    "bytes": size,
                    "routed_expert_tensor": routed,
                    "backend": backend,
                }
            )
            summaries[mode][backend] += size

    return {
        "schema_version": "1.0.0",
        "evidence_kind": "per_tensor_backend_reconciled_from_verbose_runtime_and_override_contract",
        "modes": modes,
        "modes_summary": {mode: dict(values) for mode, values in summaries.items()},
        "notes": [
            "token_embd.weight remains CPU-mapped while a tied output allocation contributes to the CUDA model buffer",
            "runtime buffer totals include alignment and tied/shared allocations beyond raw GGUF tensor payload bytes",
        ],
    }
