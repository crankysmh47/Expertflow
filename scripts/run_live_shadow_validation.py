from __future__ import annotations

import argparse
import json
from pathlib import Path

from expertflow.predictor.live_shadow import (
    load_shadow_log,
    summarize_shadow_records,
    validate_offline_equivalence,
    validate_token_and_router_parity,
)
from expertflow.predictor.runtime_artifact import parse_runtime_artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--shadow-log", type=Path, required=True)
    parser.add_argument("--disabled-tokens", type=Path, required=True)
    parser.add_argument("--enabled-tokens", type=Path, required=True)
    parser.add_argument("--disabled-trace", type=Path, required=True)
    parser.add_argument("--enabled-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = parse_runtime_artifact(args.artifact.read_bytes())
    records, runtime_summary = load_shadow_log(args.shadow_log)
    validate_offline_equivalence(artifact, records)
    validate_token_and_router_parity(
        args.disabled_tokens,
        args.enabled_tokens,
        args.disabled_trace,
        args.enabled_trace,
    )
    summary = summarize_shadow_records(records, runtime_summary)
    summary["offline_live_equivalence"] = "exact"
    summary["token_parity"] = "exact"
    summary["router_parity"] = "exact"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
