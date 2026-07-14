# ExpertFlow Stage 0 Q4 Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, measured, uninstrumented Gemma 4 Q4 llama.cpp baseline and a source-level routing feasibility decision.

**Architecture:** Model weights are downloaded to an external artifact directory from a pinned Hugging Face revision and verified before use. A pinned llama.cpp dependency runs the Q4 baseline; repository scripts capture commands and memory evidence. Source inspection ends in a written routing-hook map and a pass/fail decision before any runtime patch is planned.

**Tech Stack:** Python 3.11, pytest, huggingface_hub, PowerShell 7/Windows PowerShell compatibility, CMake, Ninja, llama.cpp, GGUF.

## Global Constraints

- Primary model profile: Gemma 4 26B A4B Q4 GGUF, text-only, batch size 1.
- Canonical model revision: `21bfe2a8c89118c9a1a2aa242934fc4d1c0fff15`.
- Canonical model file: `gemma-4-26B_q4_0-it.gguf`.
- Model weights must remain outside Git under `C:\models\expertflow`.
- Q8 is excluded from the critical path.
- No llama.cpp cache or routing mutation is allowed in this plan.
- Every material command, result, failure, and decision must be appended to `PROJECT_LOG.md`.
- Measured and estimated values must never share an unlabeled field.

---

### Task 1: Repository and artifact manifest foundation

**Files:**
- Create: `pyproject.toml`
- Create: `configs/model-artifacts.toml`
- Create: `src/expertflow/__init__.py`
- Create: `src/expertflow/artifacts.py`
- Create: `tests/test_artifacts.py`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: the pinned repository, revision, filename, size, and digest from the approved design.
- Produces: `ArtifactSpec`, `load_artifact_spec(path, name)`, and `verify_artifact(path, spec)`.

- [ ] **Step 1: Write a failing manifest-loading test**

```python
from pathlib import Path

from expertflow.artifacts import load_artifact_spec


def test_loads_pinned_q4_artifact() -> None:
    spec = load_artifact_spec(Path("configs/model-artifacts.toml"), "gemma4_q4")

    assert spec.revision == "21bfe2a8c89118c9a1a2aa242934fc4d1c0fff15"
    assert spec.filename == "gemma-4-26B_q4_0-it.gguf"
    assert spec.size_bytes == 14_439_361_440
```

- [ ] **Step 2: Run the test and verify the import fails**

Run: `python -m pytest tests/test_artifacts.py -v`  
Expected: FAIL because `expertflow.artifacts` does not exist.

- [ ] **Step 3: Implement the typed manifest loader and pinned manifest**

```python
@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    repository: str
    revision: str
    filename: str
    size_bytes: int
    sha256: str


def load_artifact_spec(path: Path, name: str) -> ArtifactSpec:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    record = document["artifacts"][name]
    return ArtifactSpec(**record)
```

- [ ] **Step 4: Run the manifest test**

Run: `python -m pytest tests/test_artifacts.py -v`  
Expected: PASS.

- [ ] **Step 5: Add size and SHA-256 verification tests before implementation**

```python
def test_rejects_wrong_artifact_size(tmp_path: Path) -> None:
    candidate = tmp_path / "model.gguf"
    candidate.write_bytes(b"wrong")
    spec = ArtifactSpec("repo", "rev", "model.gguf", 6, "0" * 64)

    with pytest.raises(ArtifactVerificationError, match="size"):
        verify_artifact(candidate, spec)
```

Run: `python -m pytest tests/test_artifacts.py -v`  
Expected: FAIL because `verify_artifact` is missing.

- [ ] **Step 6: Implement streaming SHA-256 verification and rerun tests**

```python
def verify_artifact(path: Path, spec: ArtifactSpec) -> None:
    actual_size = path.stat().st_size
    if actual_size != spec.size_bytes:
        raise ArtifactVerificationError(
            f"artifact size mismatch: expected {spec.size_bytes}, got {actual_size}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec.sha256:
        raise ArtifactVerificationError("artifact SHA-256 mismatch")
```

Run: `python -m pytest tests/test_artifacts.py -v`  
Expected: PASS.

- [ ] **Step 7: Commit the foundation**

```powershell
git add .gitignore PROJECT_LOG.md expertflow_hackathon_spec_v0_11.md docs configs pyproject.toml src tests
git commit -m "chore: establish ExpertFlow Q4 baseline"
```

### Task 2: Download and verify the pinned Q4 artifact

**Files:**
- Create: `src/expertflow/fetching.py`
- Create: `src/expertflow/cli/fetch_q4.py`
- Create: `tests/test_fetch_q4.py`
- Modify: `pyproject.toml`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: `ArtifactSpec`, `load_artifact_spec`, and `verify_artifact` from Task 1.
- Produces: `fetch_artifact(spec, destination) -> Path` and a CLI that prints the verified absolute path.

- [ ] **Step 1: Write a failing destination-path test**

```python
def test_fetch_uses_repository_scoped_directory(tmp_path: Path) -> None:
    result = artifact_destination(tmp_path, "google/example", "model.gguf")

    assert result == tmp_path / "google--example" / "model.gguf"
```

Run: `python -m pytest tests/test_fetch_q4.py -v`  
Expected: FAIL because `artifact_destination` is missing.

- [ ] **Step 2: Implement destination calculation and fetch orchestration**

```python
def artifact_destination(root: Path, repository: str, filename: str) -> Path:
    return root / repository.replace("/", "--") / filename


def fetch_artifact(spec: ArtifactSpec, root: Path) -> Path:
    destination = artifact_destination(root, spec.repository, spec.filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=spec.repository,
            filename=spec.filename,
            revision=spec.revision,
            local_dir=destination.parent,
        )
    )
    verify_artifact(downloaded, spec)
    return downloaded.resolve()
```

- [ ] **Step 3: Run the unit tests**

Run: `python -m pytest tests/test_fetch_q4.py -v`  
Expected: PASS without downloading the 14 GB file.

- [ ] **Step 4: Download the real artifact**

Run: `expertflow-fetch-q4 --destination C:\models\expertflow`
Expected: the pinned file downloads or resumes and verification prints `verified` plus its absolute path.

- [ ] **Step 5: Record exact file size, SHA-256, elapsed time, and free disk in the project log**

Run: `Get-Item <verified-path> | Select-Object FullName,Length,LastWriteTime`  
Expected: length `14439361440`.

- [ ] **Step 6: Commit the downloader**

```powershell
git add PROJECT_LOG.md pyproject.toml uv.lock src/expertflow/fetching.py src/expertflow/cli tests/test_fetch_q4.py
git commit -m "feat: fetch and verify pinned Gemma Q4 artifact"
```

### Task 3: Pin an unmodified llama.cpp GPU baseline and exact source

**Files:**
- Create: `configs/runtime-artifacts.toml`
- Create: `docs/evidence/llama-baseline.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: the official pinned llama.cpp CUDA 12.4 Windows release assets, the exact pinned source archive, and the target GPU.
- Produces: a verified GPU-capable `llama-cli.exe`, an exact source tree for inspection, and a build-evidence document.

- [ ] **Step 1: Download the pinned official CUDA runtime assets**

Download release `b10002` assets `llama-b10002-bin-win-cuda-12.4-x64.zip` and `cudart-llama-bin-win-cuda-12.4-x64.zip` to `C:\models\expertflow\dependencies`.
Expected: both assets match the sizes and SHA-256 values recorded in `configs/runtime-artifacts.toml`.

- [ ] **Step 2: Download and verify the exact source archive**

Download the GitHub codeload archive for release-matched commit `a7312ae94f801fc9c6786dc56e38df57b964f697`, hash it, and extract it below the external dependency directory.
Expected: the source directory name and archive digest are written into the manifest and evidence document.

- [ ] **Step 3: Extract and identify the runtime**

Extract both official archives into one versioned runtime directory and locate `llama-cli.exe`.

- [ ] **Step 4: Record executable version and GPU visibility**

Run: `<runtime>\llama-cli.exe --version` and `<runtime>\llama-cli.exe --list-devices`
Expected: version `b10002` (or its pinned commit) and the NVIDIA CUDA device are reported.

- [ ] **Step 5: Record the instrumentation build constraint**

Record that the host currently lacks `nvcc` and the Visual Studio C++ toolchain. Use the available MSYS2 UCRT GCC toolchain for a CPU-only telemetry/parity build during the feasibility gate; defer CUDA compiler installation until the gate justifies live runtime work.

- [ ] **Step 6: Commit the pinned runtime manifest and evidence**

```powershell
git add configs/runtime-artifacts.toml docs/evidence/llama-baseline.md PROJECT_LOG.md
git commit -m "build: pin llama.cpp CUDA baseline"
```

### Task 4: Run and measure the unmodified Q4 baseline

**Files:**
- Create: `scripts/run_baseline.ps1`
- Create: `configs/baseline-prompt.txt`
- Create: `docs/evidence/q4-baseline-result.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: verified Q4 path and built `llama-cli.exe`.
- Produces: deterministic output, command manifest, runtime log, and before/after GPU and process-memory measurements.

- [ ] **Step 1: Define the deterministic baseline prompt and command**

```text
Explain in three concise sentences why a cache miss can stall sparse MoE inference.
```

Use temperature `0`, seed `42`, context `2048`, batch size `1`, and initially offload zero layers to protect VRAM while format compatibility is tested.

- [ ] **Step 2: Implement measurement capture around llama-cli**

```powershell
nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv,noheader
$process = Start-Process -FilePath $LlamaCli -ArgumentList $arguments -PassThru -NoNewWindow
while (-not $process.HasExited) {
  $process.Refresh()
  [pscustomobject]@{
    timestamp = [DateTimeOffset]::Now.ToString('o')
    working_set_bytes = $process.WorkingSet64
    private_bytes = $process.PrivateMemorySize64
  } | ConvertTo-Json -Compress
  Start-Sleep -Milliseconds 250
}
nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv,noheader
```

- [ ] **Step 3: Execute the CPU-first compatibility run**

Run: `powershell -ExecutionPolicy Bypass -File scripts/run_baseline.ps1 -GpuLayers 0`  
Expected: successful model load and generated text, or a captured actionable compatibility error.

- [ ] **Step 4: Execute bounded partial-offload probes**

Run sequentially with `-GpuLayers 5`, then `10`, stopping before any run when free VRAM is below 4 GiB.  
Expected: each successful run records elapsed time, peak process memory, and GPU memory.

- [ ] **Step 5: Write the measured result and gate decision**

The evidence document must state `PASS`, `CONDITIONAL`, or `FAIL`, quote the exact command, and separate file size from measured runtime memory.

- [ ] **Step 6: Commit baseline evidence and scripts**

```powershell
git add configs/baseline-prompt.txt scripts/run_baseline.ps1 docs/evidence/q4-baseline-result.md PROJECT_LOG.md
git commit -m "test: measure unmodified Gemma Q4 baseline"
```

### Task 5: Produce the llama.cpp routing source map

**Files:**
- Create: `docs/evidence/gemma4-routing-source-map.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: the exact pinned llama.cpp source extracted below `C:\models\expertflow\dependencies` and successful baseline model load.
- Produces: a source map naming the Gemma 4 model loader, MoE graph construction, top-k routing operation, tensor shapes, and candidate telemetry boundary.

- [ ] **Step 1: Locate Gemma 4 architecture and expert tensors**

Run: `rg -n "GEMMA4|gemma4|expert|top_k|topk|moe|ffn_gate" <pinned-source>/src <pinned-source>/ggml`
Expected: a bounded list of source files and functions copied into the source-map document.

- [ ] **Step 2: Trace the graph from router logits to selected expert IDs**

Read every matched function on that path and record exact file, function, input shape, output shape, and ownership/lifetime of the selected-ID tensor.

- [ ] **Step 3: Identify one telemetry-only boundary**

The boundary passes only when selected expert IDs are materialized or can be copied without changing graph semantics. Record whether routing weights and token indices are available at the same point.

- [ ] **Step 4: Write the 24-hour gate decision**

`PASS` means a telemetry-only patch is bounded to a small set of named functions. `CONDITIONAL` means one isolated GGML callback or graph-output addition is needed. `FAIL` means telemetry requires allocator, scheduler, or model-format redesign.

- [ ] **Step 5: Commit the source map**

```powershell
git add docs/evidence/gemma4-routing-source-map.md PROJECT_LOG.md
git commit -m "docs: map Gemma 4 routing telemetry boundary"
```

## Plan self-review

- Spec coverage: this plan covers only the independently testable Stage 0 baseline and routing-feasibility result; Observatory implementation receives its own plan after the real trace boundary is known.
- Placeholder scan: no deferred implementation markers are present; each action has a concrete command or decision rule.
- Type consistency: `ArtifactSpec`, `load_artifact_spec`, `verify_artifact`, `artifact_destination`, and `fetch_artifact` retain the same names and signatures across tasks.
