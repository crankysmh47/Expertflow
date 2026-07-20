from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from expertflow.quality.dataset import export_mmlu_rows, export_wikitext


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export pinned quality datasets into stable inputs")
    parser.add_argument("--wikitext-parquet", type=Path, required=True)
    parser.add_argument("--mmlu-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "wikitext": export_wikitext(args.wikitext_parquet, args.output_dir / "wikitext-test.txt"),
        "mmlu": export_mmlu_rows(args.mmlu_root, args.output_dir / "mmlu-test.json"),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
