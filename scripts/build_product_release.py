from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


ROOT = Path(__file__).parents[1].resolve()
DEFAULT_OUTPUT = ROOT / "release" / "expertflow-build-week"
ZIP_EPOCH = (2026, 7, 19, 0, 0, 0)
UPSTREAM = "a7312ae94f801fc9c6786dc56e38df57b964f697"
LLAMA_COMMIT = "451224ab4d12a616dc3e16e8c8063f4b331f531c"
EXTERNAL_SHA_SUFFIX = ".zip.sha256"
ALLOWLIST = (
    ("README.md", "README.md"), ("JUDGES.md", "JUDGES.md"), ("LICENSE", "LICENSE"),
    ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
    ("pyproject.toml", "pyproject.toml"), ("uv.lock", "uv.lock"),
    (".env.example", ".env.example"), ("src/expertflow", "src/expertflow"),
    ("deployments", "deployments"), ("examples/openai_client.py", "examples/openai_client.py"),
    ("examples/agentic_session.py", "examples/agentic_session.py"),
    ("scripts/start_expertflow.ps1", "scripts/start_expertflow.ps1"),
    ("scripts/stop_expertflow.ps1", "scripts/stop_expertflow.ps1"),
    ("scripts/setup_release.ps1", "scripts/setup_release.ps1"),
    ("scripts/build_release_runtime.ps1", "scripts/build_release_runtime.ps1"),
    ("scripts/verify_release.py", "scripts/verify_release.py"),
    ("scripts/judge-replay.ps1", "scripts/judge-replay.ps1"),
    ("scripts/judge-replay.sh", "scripts/judge-replay.sh"),
    ("scripts/live-demo.ps1", "scripts/live-demo.ps1"),
    ("scripts/verify-release.ps1", "scripts/verify-release.ps1"),
    ("scripts/verify-release.sh", "scripts/verify-release.sh"),
    ("docs/BENCHMARKING.md", "docs/BENCHMARKING.md"),
    ("docs/assets", "docs/assets"),
    ("docs/product-reproduction.md", "docs/product-reproduction.md"),
    ("docs/product-troubleshooting.md", "docs/product-troubleshooting.md"),
    ("docs/evidence/product-release/release-state.json", "docs/evidence/product-release/release-state.json"),
    ("docs/evidence/product-release/replay-data.json", "docs/evidence/product-release/replay-data.json"),
    ("docs/evidence/product-release/deployment-result.json", "docs/evidence/product-release/deployment-result.json"),
    ("docs/evidence/product-release/throughput-profile.json", "docs/evidence/product-release/throughput-profile.json"),
    ("docs/evidence/product-release/context-profile.json", "docs/evidence/product-release/context-profile.json"),
    ("docs/evidence/product-release/agentic-demo.json", "docs/evidence/product-release/agentic-demo.json"),
    ("docs/evidence/product-release/benchmark-report.md", "docs/evidence/product-release/benchmark-report.md"),
    ("docs/evidence/product-release/release-scorecard.json", "docs/evidence/product-release/release-scorecard.json"),
    ("docs/evidence/product-release/release-scorecard.json", "evidence/release-scorecard.json"),
    ("docs/evidence/product-release/dashboard.html", "dashboard.html"),
    ("submission/final-devpost-draft.md", "submission/final-devpost-draft.md"),
    ("submission/demo-video-script.md", "submission/demo-video-script.md"),
    ("submission/demo-shot-list.md", "submission/demo-shot-list.md"),
    ("submission/judge-test-guide.md", "submission/judge-test-guide.md"),
    ("submission/architecture.md", "submission/architecture.md"),
    ("submission/claims-ledger.md", "submission/claims-ledger.md"),
    ("submission/demo-video-script-final.md", "submission/demo-video-script-final.md"),
    ("submission/demo-video-shot-list-final.md", "submission/demo-video-shot-list-final.md"),
    ("submission/demo-video-fallback-plan.md", "submission/demo-video-fallback-plan.md"),
    ("submission/demo-video-assets", "submission/demo-video-assets"),
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(root: Path) -> list[dict[str, object]]:
    return [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _digest(path)}
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if path.is_file() and path.name != "SHA256SUMS.json"
    ]


def audit_tree(root: Path) -> list[str]:
    errors: list[str] = []
    binary_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".zip", ".dll", ".exe"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in {".gguf", ".safetensors"}:
            errors.append(f"forbidden model file: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"unexpected large file: {relative}")
        if path.suffix.lower() in binary_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "C:\\Users\\Hank47" in text or "C:\\models\\expertflow\\worktrees" in text:
            errors.append(f"private path: {relative}")
        if "sk-test-secret" in text or "BEGIN PRIVATE KEY" in text:
            errors.append(f"credential: {relative}")
    return errors


def _copy_allowlist(output: Path) -> None:
    for source_name, destination_name in ALLOWLIST:
        source, destination = ROOT / source_name, output / destination_name
        if source.is_dir():
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def normalize_text_tree(root: Path) -> None:
    binary_suffixes = {".dll", ".exe", ".jpg", ".jpeg", ".png", ".webp", ".zip"}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() in binary_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        path.write_text(text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8", newline="\n")


def _write_runtime_package(output: Path, llama_repo: Path) -> None:
    patches = output / "patches" / "llama.cpp"
    patches.mkdir(parents=True)
    subprocess.run(["git", "-C", str(llama_repo), "format-patch", "--no-stat", "--full-index", "--binary", "-o", str(patches), f"{UPSTREAM}..{LLAMA_COMMIT}"], check=True, capture_output=True, text=True)
    source = llama_repo / "ggml" / "src" / "ggml-backend.cpp"
    runtime_source = output / "runtime-source"
    runtime_source.mkdir()
    shutil.copy2(source, runtime_source / "ggml-backend.cpp")
    metadata = {
        "schema_version": "1.0.0", "upstream_commit": UPSTREAM,
        "expertflow_llama_commit": LLAMA_COMMIT,
        "ggml_backend_cpp_sha256": _digest(source),
        "build": {"compiler": "MSVC v143 14.39.33519", "cuda": "12.8.93", "generator": "Ninja", "flags": ["GGML_CUDA=ON", "CMAKE_BUILD_TYPE=Release"]},
        "expected_binaries": {"llama_cli_sha256": "5d68046dcd26e2fd018aaeaad5f99cdb7d88eca6fc10935925f1d660f7009407", "llama_server_sha256": "22ecc4f64f91dcbe3a1cfe7d9d4617e43467ea7f3c6fa1ba2c6ad8d07e89334e"},
    }
    (runtime_source / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _write_zip(output: Path) -> Path:
    archive = output.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(item for item in output.rglob("*") if item.is_file()):
            relative = path.relative_to(output).as_posix()
            info = zipfile.ZipInfo(f"expertflow-build-week/{relative}", ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, path.read_bytes())
    return archive


def build(output: Path, llama_repo: Path) -> tuple[Path, Path]:
    output = output.resolve()
    release_root = (ROOT / "release").resolve()
    if output.parent != release_root or output.name != "expertflow-build-week":
        raise ValueError("release output must be release/expertflow-build-week")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    _copy_allowlist(output)
    _write_runtime_package(output, llama_repo.resolve())
    normalize_text_tree(output)
    errors = audit_tree(output)
    if errors:
        raise ValueError("release audit failed:\n" + "\n".join(errors))
    rows = file_manifest(output)
    (output / "SHA256SUMS.json").write_text(json.dumps({"schema_version": "1.0.0", "files": rows}, indent=2) + "\n", encoding="utf-8")
    archive = _write_zip(output)
    archive.with_name(output.name + EXTERNAL_SHA_SUFFIX).write_text(
        f"{_digest(archive)}  {archive.name}\n", encoding="ascii"
    )
    return output, archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--llama-repo", type=Path, required=True)
    args = parser.parse_args()
    output, archive = build(args.output, args.llama_repo)
    print(json.dumps({"status": "pass", "release": str(output), "archive": str(archive), "archive_sha256": _digest(archive)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
