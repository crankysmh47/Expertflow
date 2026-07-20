from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from expertflow.quality.manifest import freeze_config_from_json, freeze_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the immutable Option 1 quality manifest")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = freeze_manifest(freeze_config_from_json(args.config))
    print(json.dumps({"manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
