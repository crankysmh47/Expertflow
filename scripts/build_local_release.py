"""Build a private alpha kit; never upload, tag or bundle models/runtimes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["local-quickstart.md", "support-matrix.md", "local-product-pilot.md", "runtime-updates.md", "README.md", "PRODUCT.md", "BENCHMARKING.md", "STATUS.md", "TODO.md"]


def checksums(directory: Path) -> dict:
    return {p.relative_to(directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"}


def audit_payload(directory: Path) -> None:
    def check(name):
        if Path(name).suffix.lower() in {".gguf", ".exe", ".dll", ".db", ".sqlite", ".env"} or "__pycache__" in Path(name).parts:
            raise ValueError(f"Forbidden alpha payload: {name}.")
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        check(path.name)
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    check(name)
        elif path.name.endswith(".tar.gz"):
            with tarfile.open(path) as archive:
                for member in archive:
                    check(member.name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh kit directory; an existing release is never overwritten.")
    args.output.mkdir(parents=True)
    version=tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    env=dict(os.environ)
    env["SOURCE_DATE_EPOCH"]="1791244800"
    subprocess.run(["uv","build","--out-dir",str(args.output.resolve())],cwd=ROOT,env=env,check=True)
    docs=args.output / "docs";docs.mkdir()
    for name in DOCS:
        shutil.copyfile(ROOT / "docs" / name,docs / name)
    for name in ["LICENSE","THIRD_PARTY_NOTICES.md","CHANGELOG.md"]:
        shutil.copyfile(ROOT / name,args.output / name)
    (args.output / "START-HERE.md").write_text(
        "# ExpertFlow private Windows alpha\n\nOpen docs/local-quickstart.md for installation.\n"
        "Use docs/local-product-pilot.md to record your own tasks and failures.\n"
        "Python 3.11+ and your local GGUF/llama.cpp runtime are required.\n"
        "No runtime, model weights, account or telemetry are bundled.\n"
        "This kit is for testing; see docs/support-matrix.md for qualification limits.\n",encoding="utf-8")
    # A standalone kit links to repository material that is outside its allowlist.
    for document in args.output.rglob("*.md"):
        def repository_link(match):
            target = match.group(1)
            if "://" in target or target.startswith("#") or " " in target:
                return match.group(0)
            filename, marker, anchor = target.partition("#")
            if (document.parent / filename).exists():
                return match.group(0)
            original = ROOT / document.relative_to(args.output)
            if document.name == "START-HERE.md":
                return match.group(0)
            resolved = (original.parent / filename).resolve()
            if not resolved.is_relative_to(ROOT):
                return match.group(0)
            relative = resolved.relative_to(ROOT).as_posix()
            return "](https://github.com/crankysmh47/Expertflow/blob/main/" + relative + (marker + anchor if marker else "") + ")"
        document.write_text(re.sub(r"\]\(([^)]+)\)", repository_link,
                           document.read_text(encoding="utf-8")), encoding="utf-8")
    audit_payload(args.output)
    (args.output / "release.json").write_text(json.dumps({"schema_version":1,"version":version,
        "scope":"Private Windows/NVIDIA alpha; one native host tested; human pilot and broader hardware gates open.",
        "native_runtime_bundled":False,"model_weights_bundled":False,"publication":False,
        "source_date_epoch":env["SOURCE_DATE_EPOCH"]},indent=2)+"\n",encoding="utf-8")
    (args.output / "SHA256SUMS.json").write_text(json.dumps(checksums(args.output),indent=2)+"\n",encoding="utf-8")
    print(f"Built private alpha kit {version}: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
