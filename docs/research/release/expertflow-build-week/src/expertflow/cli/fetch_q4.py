"""Fetch and verify the canonical Gemma 4 Q4 deployment artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from expertflow.artifacts import load_artifact_spec
from expertflow.fetching import fetch_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and verify ExpertFlow's pinned Gemma 4 Q4 artifact."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/model-artifacts.toml"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(r"C:\models\expertflow"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec = load_artifact_spec(args.manifest, "gemma4_q4")
    verified_path = fetch_artifact(spec, args.destination)
    print(f"verified {verified_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
