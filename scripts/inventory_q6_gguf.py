"""Generate a complete metadata-only GGUF and routed-expert inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expertflow.analysis.q6_inventory import summarize_tensor_inventory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gguf-py", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.gguf_py.resolve()))
    from gguf import GGML_QUANT_SIZES, GGUFReader

    model = args.model.resolve()
    reader = GGUFReader(str(model), "r")
    tensors = []
    for tensor in reader.tensors:
        block_elements, block_bytes = GGML_QUANT_SIZES[tensor.tensor_type]
        tensors.append(
            {
                "name": tensor.name,
                "type": tensor.tensor_type.name,
                "shape": [int(value) for value in tensor.shape],
                "bytes": int(tensor.n_bytes),
                "data_offset": int(tensor.data_offset),
                "quant_block_elements": int(block_elements),
                "quant_block_bytes": int(block_bytes),
                "slice_contiguous": True,
            }
        )

    result = summarize_tensor_inventory(tensors, expert_count=128)
    result.update(
        {
            "schema_version": "1.0.0",
            "measurement_kind": "gguf_metadata_derived",
            "model": {
                "path": str(model),
                "bytes": model.stat().st_size,
                "sha256": args.sha256.lower(),
                "repository": args.repository,
                "revision": args.revision,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
