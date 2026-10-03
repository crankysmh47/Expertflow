# Compiler execution checkpoint — 2026-10-03

Branch: `ef-v2`. Starting commit: `bb4fef6`.

The repaired plan is `docs/superpowers/plans/2026-08-28-inference-compiler-spine.md`.
The user authorized repairs, execution, a local model search and recovery when
the pinned Q6 weights could not be found.

## Completed work

- Plan/spec repairs committed as `11d6ff5`: historical CLI and product server
  workloads are separate; strict exactness remains; measured stock fallback is
  valid; stock search includes CPU-MoE; runtime caps and selected-plan replay are
  explicit. Historical 28.13 TPS is not a passing exact product target.
- Task 1 committed as `1ccc60f`: Python 3.11 environment, pandas/NumPy repair,
  project-root pytest imports, validated workload/config boundaries and separate
  pristine/fork manifests. Full-suite collection requires the existing predictor
  extra in addition to dev/quality.
- Task 0 implementation: historical evidence audit, model/binary/patch/dependency
  verification and structured atomic preflight reports. Expected DLL and CUDA
  hashes are frozen in the runtime manifests; changed implementation DLLs fail.
- Fresh-context checkpoint review completed. DLL tampering and numeric overflow
  findings were reproduced with failing tests and fixed. Historical launch
  argument/environment consistency checking is deferred: those inputs remain
  report-only and cannot seal or execute a plan.

## Verification

`uv run --extra dev --extra quality --extra predictor pytest -q`:
**355 passed, 6 external-source skips**.

With `EXPERTFLOW_LLAMA_SOURCE` set to the pinned static fork, the two applicable
static/profile source-contract suites passed **6 tests**. Historical temporal
cache tests were not applied to the static fork. Compileall and `git diff --check`
passed. No model measurements or GPU inference processes have run in this checkpoint.

## Current gate and recovery

Task 0's live artifact gate is **ENVIRONMENT-BLOCKED**, solely because the Q6
download is incomplete. It is not a completed task. Tasks 2–11 have not started.
The exact runtime binaries and six patches passed verification; the audit rejects
the historical static result from exact sealing and preserves its quality failure.

The C/D drive search found Q4, Qwen and tiny test weights, but no pinned Q6 GGUF
or complete SHA-named cache object. A stalled zero-byte Hugging Face CLI download
was stopped along with its verified owned children. One native aria2 process
owns the recovery, PID `18900` at this checkpoint. Revision:
`fabed3e586120477355eea23b92644540a79ce2f`.

Destination: `C:/models/gemma-4-26b-a4b-q6/google_gemma-4-26B-A4B-it-Q6_K.gguf`.
Required size: **22,862,575,520 bytes**. Required SHA-256:
`089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba`.
Do not infer completed download bytes from apparent file length: aria2 writes
segments at distant offsets. Its `.aria2` marker and transfer log remain authoritative
until the downloader exits and the full hash is checked.

The hidden verifier in
`.superpowers/sdd/2026-08-28-inference-compiler-spine/finish-model-recovery.ps1`
waits for this owned downloader, then runs the artifact gate once. It writes
`model-recovery-verdict.json` and `model-verification.log` in the same ignored
directory and refreshes `preflight.json`. It launches no model and executes no
compiler tasks. A recovery failure preserves resumable files.

## Resume

Read the ignored `progress.md` ledger and recovery verdict, then rerun:

```powershell
uv run --extra dev --extra quality --extra predictor python scripts/compiler_preflight.py --model C:/models/gemma-4-26b-a4b-q6/google_gemma-4-26B-A4B-it-Q6_K.gguf --stock-dir C:/models/expertflow/builds/llama-a7312ae-cuda128-clean/bin --fork-dir C:/models/expertflow/builds/llama-q6-placement-final/bin --output docs/evidence/compiler-phase3/preflight.json
```

Only exit 0 / `READY` permits marking Task 0 complete and starting Task 2.
`READY` establishes artifact feasibility; it does not establish live token,
performance, GPU memory, numerical-path or selected-plan replay validity.

Keep the pre-existing `PROJECT_LOG.md` edit, R1 report and OpenCode transcript
outside compiler commits. Do not merge, push or begin later research tracks here.
