from __future__ import annotations

import argparse
import json
from pathlib import Path

from expertflow.quality.mmlu import run_mmlu


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Q1b zero-shot MMLU subset")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/completion")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = run_mmlu(manifest["mmlu"]["items"], args.endpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("correct", "total", "accuracy")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
