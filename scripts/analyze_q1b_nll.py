from __future__ import annotations

import argparse
import json
from pathlib import Path

from expertflow.quality.q1b import compare_nll_records, load_nll_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired block-bootstrap analysis for Q1b NLL records")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()

    result = compare_nll_records(
        load_nll_jsonl(args.reference),
        load_nll_jsonl(args.candidate),
        block_size=args.block_size,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        threshold=args.threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
