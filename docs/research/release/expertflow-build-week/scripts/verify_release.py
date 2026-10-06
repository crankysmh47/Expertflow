from __future__ import annotations
import hashlib, json
from pathlib import Path

root = Path(__file__).parents[1]
manifest = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
failures = []
for item in manifest["files"]:
    path = root / item["path"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if actual != item["sha256"]:
        failures.append({"path": item["path"], "expected": item["sha256"], "actual": actual})
print(json.dumps({"status": "pass" if not failures else "failure", "verified_files": len(manifest["files"]), "failures": failures}, indent=2))
raise SystemExit(0 if not failures else 2)
