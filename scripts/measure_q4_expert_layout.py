"""Emit an exhaustive packed-expert inventory from pinned GGUF metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expertflow.analysis.expert_layout import (  # noqa: E402
    build_expert_inventory,
    expert_tensor_from_gguf_metadata,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory every Gemma Q4 layer-expert byte span."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gguf-py", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--llama-revision", required=True)
    parser.add_argument("--capacity", type=int, default=96)
    parser.add_argument("--target-max-layer", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = args.model.resolve()
    gguf_py = args.gguf_py.resolve()
    output = args.output.resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    if not (gguf_py / "gguf").is_dir():
        raise ValueError("gguf-py must contain the gguf package")
    if args.capacity <= 0 or args.target_max_layer < 0:
        raise ValueError("capacity and target layer range must be positive")

    sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader

    reader = GGUFReader(str(model), "r")
    tensors = []
    for tensor in reader.tensors:
        mapped = expert_tensor_from_gguf_metadata(
            name=tensor.name,
            shape=tuple(int(value) for value in tensor.shape),
            tensor_type=tensor.tensor_type.name,
            encoded_bytes=int(tensor.n_bytes),
            data_offset=int(tensor.data_offset),
        )
        if mapped is not None:
            tensors.append(mapped)

    report = build_expert_inventory(
        tensors,
        alignment=128,
        matrix_row_padding=512,
        component_order=("down_weight", "gate_up_weight", "down_scale"),
        target_layer_ids=tuple(range(args.target_max_layer + 1)),
        capacity_per_layer=args.capacity,
    )
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["model"] = {
        "path": str(model),
        "bytes": model.stat().st_size,
        "sha256": sha256_file(model),
        "revision": args.model_revision,
    }
    report["source"] = {
        "gguf_py": str(gguf_py),
        "llama_revision": args.llama_revision,
        "cuda_alignment_source": "ggml/src/ggml-cuda/ggml-cuda.cu",
        "cuda_row_padding_source": "ggml/src/ggml-cuda/common.cuh",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
