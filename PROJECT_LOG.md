# ExpertFlow Project Log

This append-oriented log records decisions, commands, evidence, failures, and next actions for the OpenAI Build Week project. Measurements are never rewritten as estimates, and estimates are never presented as measurements.

## 2026-07-14

### 21:25 PKT — Project execution started

- User approved the revised runtime direction with a qualification: llama.cpp instrumentation is a 24-hour feasibility gate, not an assumption that dynamic expert movement is easy.
- Runtime direction: pinned Gemma 4 26B A4B Q4 GGUF for real inference; minimal llama.cpp routing telemetry patch; Python for validation, analysis, simulation, recommendation, and replay.
- Q8 was removed from the critical path.
- Repository began as an unborn `main` branch containing only the untracked `expertflow_hackathon_spec_v0_11.md`.
- Created branch `codex/expertflow-stage0`; no linked worktree was created because the repository had no initial commit.
- Initial repository policy placed weights under `D:\models\expertflow`; this was corrected to `C:\models\expertflow` at 21:31 PKT when the user clarified that `D:` is an HDD.

### 21:25 PKT — Machine and artifact preflight

- GPU after reboot: NVIDIA GeForce RTX 5060 Ti, 2,075 MiB used, 13,976 MiB free.
- System RAM previously measured: 31.1 GiB.
- Free disk at preflight: approximately 137.8 GiB on `C:` and 137.7 GiB on `D:`.
- No Q4 GGUF was found in the existing Hugging Face cache.
- Canonical repository: `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`.
- Pinned repository revision: `21bfe2a8c89118c9a1a2aa242934fc4d1c0fff15`.
- Canonical text-only file: `gemma-4-26B_q4_0-it.gguf`.
- Remote content length: `14,439,361,440` bytes.
- Remote ETag: `21005eb9bd80c75b5236d5b8e9828b5b887609f0cdd9158e86ea3e16044928f4`.
- Next action: commit the approved design and Stage 0 execution plan, then download and verify the pinned Q4 artifact.

### 21:28 PKT — Artifact foundation completed

- Added a clean Python package scaffold, pinned artifact manifest, and model-weight ignore rules.
- TDD red result: `tests/test_artifacts.py` failed during collection with `ModuleNotFoundError: No module named 'expertflow.artifacts'`, confirming the loader did not exist.
- Implemented immutable `ArtifactSpec`, TOML manifest loading, exact byte-length validation, and streaming SHA-256 verification.
- TDD green result: `3 passed in 0.08s` on Python 3.11.9 with pytest 8.4.0.
- No model bytes have been written into the repository.
- Next action: download the pinned Q4 file to `C:\models\expertflow`.

### 21:31 PKT — Model storage corrected to the SSD

- The first download command targeted `D:\models\expertflow` before the user clarified that `D:` is an HDD.
- Stopped the wrapper and its orphaned Python child processes (`2768`, `5036`, and `22976`).
- Inspection showed the lock and incomplete files were both zero bytes; no model payload required migration or redownload.
- Removed only the empty downloader state under the verified path `D:\models\expertflow`.
- Created `C:\models\expertflow` and changed the CLI default, plan, and repository policy to that SSD path.
- Free space on `C:` before restarting the transfer: approximately 137.5 GiB.

### 21:41 PKT — Download transport and runtime toolchain decision

- Hugging Face Xet and `hf_hub_download` transports repeatedly stalled while holding incomplete-file locks; a direct ranged request to the same pinned revision succeeded.
- Preserved the valid partial payload and resumed it with retrying `curl` directly into the canonical path on `C:`. The transfer passed 1 GiB and continued to grow; final size and SHA-256 remain pending and will be recorded only after verification.
- Shallow Git transports for llama.cpp also stalled or left a broken promisor checkout. Switched to GitHub's exact-commit codeload archive for reproducible source inspection, with its local SHA-256 to be pinned after download.
- Host inspection found CMake and Ninja but no `nvcc`, Visual Studio C++ toolchain, or Vulkan SDK. MSYS2 UCRT GCC/G++ are available.
- Chosen bounded path: official llama.cpp `b10002` CUDA 12.4 Windows assets for the real GPU baseline, exact commit `bf2c86ddc0685f580595954056c2e77ebabfab4f` source for inspection, and a CPU-only GCC instrumentation build for trace/parity validation. A CUDA compiler installation is deferred until the 24-hour feasibility gate justifies live runtime modification.
- All model, runtime, and source artifacts remain external under `C:\models\expertflow`; the repository retains only manifests, code, tests, compact evidence, and logs.

### 21:48 PKT — Resumable downloader hardened and Q4 prioritized

- Added a regression test proving that an incomplete canonical model file is resumed through the pinned revision URL instead of being rejected as though it were complete.
- TDD red result: `TypeError: fetch_artifact() got an unexpected keyword argument 'downloader'`.
- Replaced the unreliable Hugging Face client transport with a retrying ranged `curl` downloader, while retaining strict final byte-length and SHA-256 verification. Exact-size corrupt files still fail closed.
- TDD green result: all `6` tests passed in `0.04s` on the project Python 3.12.11 environment.
- Paused the resumable llama.cpp source and binary transfers because three simultaneous connections stalled together. The Q4 transfer was given sole bandwidth and continued growing on `C:`; runtime/source downloads will resume after it completes.

### 22:02 PKT — Gemma 4 routing source boundary located

- Queried GitHub's exact-commit tree for llama.cpp revision `bf2c86ddc0685f580595954056c2e77ebabfab4f` and inspected the authoritative raw source.
- `src/models/gemma4.cpp` builds router logits as `[n_expert, n_tokens]` and passes them to `build_moe_ffn`.
- `src/llama-graph.cpp` materializes `selected_experts` with `ggml_argsort_top_k` as `[n_expert_used, n_tokens]` and already names the tensor `ffn_moe_topk-{layer}` through the graph callback.
- The public scheduler evaluation callback can request only that small tensor and copy it to host after computation. The existing `examples/eval-callback` and `common/debug.cpp` demonstrate this mechanism.
- Source feasibility verdict: `PASS`. No graph, router, allocator, scheduler, or model-format mutation is needed for the first telemetry probe. The overall gate remains pending the real Q4 load, deterministic parity, schema validation, and locality evidence.
- Added `docs/evidence/gemma4-routing-source-map.md` and removed the now-unused empty `third_party/` directory to preserve the external-artifact boundary.

### 22:08 PKT — Trace schema implemented while artifacts transfer

- Installed Scoop-verified `aria2` 1.37.0 outside the repository and resumed the same Q4 partial with eight ranged connections. The target became a sparse file, so apparent length is explicitly not used as completion evidence; the aria2 control map and final cryptographic verification remain authoritative.
- Attempted to resume the exact llama.cpp source archive and CUDA assets alongside the model. A PowerShell argument-interpolation error pointed aria2 at a literal `System.Collections.Hashtable.Dir` path in the repository; the command was stopped, its 2 MB throwaway file was removed after resolving and checking the target path, and no tracked or verified artifact was affected. Runtime downloads remain resumable and will restart with explicit scalar paths.
- TDD red result for the Observatory trace contract: `ModuleNotFoundError: No module named 'expertflow.trace'`.
- Implemented dependency-free, immutable `RouterTraceEvent` parsing with schema-version enforcement, unsigned range checks, unique expert IDs, optional matched weights, finite-value validation, strict fields, and record-numbered failures.
- TDD green result: `8` schema tests passed, and the complete suite passed `14` tests in `0.03s`.

### 22:14 PKT — First Observatory locality profile implemented

- TDD red result: `ModuleNotFoundError: No module named 'expertflow.analysis'`.
- Implemented deterministic per-layer aggregation over canonical router events: selection concentration, stable top-expert ordering, static-residency hit curves for explicit budgets, adjacent-token expert reuse, and mean token reuse distance.
- Request boundaries reset temporal metrics, empty traces remain valid, invalid budgets fail, and backward token indices fail visibly rather than producing misleading locality.
- TDD green result: `3` profile tests passed; the complete suite passed `17` tests in `0.04s`.
- Artifact acquisition remained active at the network's measured approximately 2.6 MiB/s. No completion claim is made while the Q4 aria2 control map exists.

### 22:17 PKT — Measured profile exposed through the public CLI

- TDD red result: `ModuleNotFoundError: No module named 'expertflow.cli.main'`.
- Added streaming JSONL trace input and the public `expertflow profile <trace> --output <report>` command with repeatable static expert budgets.
- Reports are stable JSON, preserve the resolved source-trace path, and explicitly label their outputs as `measured` rather than estimated.
- Verified the installed console entry point with `expertflow --help`.
- TDD green result: the CLI integration test passed and the complete suite passed `18` tests in `0.05s`.

### 22:19 PKT — Judge-facing hardware doctor completed

- TDD red result: `ModuleNotFoundError: No module named 'expertflow.doctor'`.
- Added `expertflow doctor` with machine-readable GPU, system RAM, artifact-disk, Python, platform, and tool-path evidence. The default artifact root is the user-approved `C:\models\expertflow`.
- Real preflight written externally to `C:\models\expertflow\runs\preflight\doctor.json`.
- Measured GPU snapshot: RTX 5060 Ti, driver `591.86`, 16,311 MiB total, 2,258 MiB used, 13,793 MiB free.
- Measured system RAM: `33,396,539,392` bytes. Measured free space on the artifact disk: `132,702,834,688` bytes.
- Confirmed CMake, Ninja, GCC/G++, aria2, curl, and nvidia-smi; `nvcc` remains absent and is reported as `null` rather than hidden.
- TDD green result: `3` doctor tests passed; the complete suite passed `21` tests in `0.05s`.

### 22:23 PKT — llama.cpp release provenance corrected

- Verified `llama-b10002-bin-win-cuda-12.4-x64.zip`: `248,820,066` bytes and SHA-256 `d8fa3634b6a6a2eb64b56d3f4d68b8e71ee6e1ccb980059476dd51787f1d2f3f`, exactly matching the official release asset metadata.
- Extracted the verified archive under `C:\models\expertflow\dependencies\llama-b10002\runtime`.
- `llama-cli --version` reports build `10002`, commit `a7312ae94`, built with Clang 20.1.8 for Windows x86_64.
- This exposed a provenance mismatch: earlier source inspection used then-current HEAD `bf2c86ddc0685f580595954056c2e77ebabfab4f`, not the release's source commit.
- Resolved the runtime's full source commit as `a7312ae94f801fc9c6786dc56e38df57b964f697` through GitHub's commit API and re-inspected the Gemma 4 logits, `build_moe_ffn`, `ffn_moe_topk`, weights, and callback boundaries. The cited paths, line numbers, shapes, and source-feasibility PASS are unchanged.
- Updated the execution plan and source map to the release-matched commit. The obsolete partial source archive will not be used.

### 22:26 PKT — CUDA baseline runtime fully verified

- Verified `cudart-llama-bin-win-cuda-12.4-x64.zip`: `391,443,627` bytes and SHA-256 `8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6`, exactly matching official release metadata.
- Extracted `cublas64_12.dll`, `cublasLt64_12.dll`, and `cudart64_12.dll` into the versioned external runtime directory.
- Measured device check now passes: `CUDA0: NVIDIA GeForce RTX 5060 Ti (16310 MiB, 15158 MiB free)`.
- Added `configs/runtime-artifacts.toml` and `docs/evidence/llama-baseline.md`; binaries and third-party source remain outside Git.
- Resumed the Q4 model through its existing aria2 control map immediately after the runtime check.

### 22:31 PKT — Deterministic baseline invocation frozen

- Inspected `llama-cli b10002 --help` and confirmed the current names and semantics for model, prompt, seed, temperature, context, prediction count, GPU layers, threads, conversation, single-turn, prompt-display, performance, and log-file flags.
- TDD red result: `ModuleNotFoundError: No module named 'expertflow.runtime'`.
- Added immutable `BaselineRunConfig` validation and one reviewable command builder. The first real checks will use fixed seed `42`, temperature `0`, explicit context and token limits, explicit CPU thread counts, a single non-interactive turn, and an explicit GPU-layer policy.
- TDD green result: `2` baseline-command tests passed; the complete suite passed `23` tests in `0.05s`.

### 22:36 PKT — Baseline memory measurement harness completed

- TDD red result: `ModuleNotFoundError: No module named 'expertflow.runtime.measurement'`.
- Added the measured execution wrapper that writes raw stdout, stderr, llama runtime logs, exact command/model/runtime provenance, UTC start/finish, wall time, before/after GPU snapshots, sampled peak global GPU use, process working set, peak working set, private bytes, and peak pagefile bytes.
- A real Windows WDDM check showed `nvidia-smi --query-compute-apps` returns `[N/A]` for per-process GPU bytes. Added a failing regression test, then made the parser skip those unavailable rows. Global GPU used/free sampling remains authoritative and the per-process field stays zero rather than inventing a value.
- Verified the Win32 process-memory sampler against the current Python process after defining 64-bit-safe handle types.
- TDD green result: `4` measurement tests passed; the complete suite passed `27` tests in `0.05s`.

### 22:41 PKT — Minimum cache-policy comparison implemented

- TDD red result: `ModuleNotFoundError: No module named 'expertflow.analysis.cache_sim'`.
- Implemented one capacity-controlled simulator comparing a no-residency reactive baseline, a trace-derived static hotset, and online per-layer LRU over the same canonical event stream.
- Reports separate hits, misses, demand-time loads, and static preloads; reject capacities below router top-k; and are hard-labeled `estimated` so they cannot be confused with live movement telemetry.
- TDD green result: `2` policy tests passed; the complete suite passed `29` tests in `0.05s`.

### 22:43 PKT — Estimated policy simulation exposed through the CLI

- TDD red result: the CLI rejected `simulate` as an invalid command and listed only `doctor` and `profile`.
- Added `expertflow simulate <trace> --capacity-per-layer <slots> --output <report>` over the same strict JSONL and policy engine.
- The CLI integration test confirms reactive, static-hotset, and LRU results and the mandatory top-level `estimated` label.
- TDD green result: the simulation CLI test passed; the complete suite passed `30` tests in `0.07s`.

### 22:45 PKT — One-command measured baseline path completed

- TDD red result: the CLI module had no `run_measured_baseline` integration for the new test to patch.
- Added `expertflow baseline` with required runtime, model, model SHA-256, prompt file, and output directory plus explicit GPU-layer, context, token, and thread controls.
- Added the checked-in deterministic baseline prompt at `configs/baseline-prompt.txt`.
- The command writes `manifest.json`, `stdout.txt`, `stderr.txt`, and `llama.log` into one external run directory and returns the runtime's exit code.
- TDD green result: the baseline CLI integration test passed; the complete suite passed `31` tests in `0.07s`.

### 22:50 PKT — Judge-facing README added and verified

- Added the root `README.md` with reproducible Windows setup, the current Q4-first runtime direction, exact fetch and measured-baseline commands, trace/profile/simulation examples, architecture, repository map, and Build Week integration notes.
- Kept the status section evidence-based: the CUDA runtime and analysis harness are complete, while the Q4 baseline, router trace, and GPT-5.6 integration remain explicitly pending.
- Verified every local README link resolves, both installed console entry points render help, and the full suite passes `31` tests in `0.07s`.
- Model and exact llama.cpp source downloads remained active throughout this documentation work; no download-completion claim is made here.

### 22:58 PKT — Exact token-parity gate implemented

- TDD red results: the token-parity module was initially absent, then the public CLI rejected `parity` as an unknown command.
- Added strict, versioned prompt/generated token artifacts and exact comparison with the first differing generated token reported by index and both token IDs.
- Added `expertflow parity <baseline> <instrumented> --output <report>`; its output is hard-labeled `measured` and never infers equality from rendered text.
- TDD green result: `4` focused tests passed and the complete suite passed `35` tests in `0.09s`.
- A follow-up red test showed parity mismatches still returned process status `0`. The CLI now writes the measured mismatch report and returns `1`, so the parity gate cannot be accidentally ignored; the complete suite passes `36` tests in `0.11s`.
- Decoded the active aria2 control map rather than trusting the sparse file length: `2,392 / 3,443` four-MiB pieces were complete (`69.47%`) at this checkpoint. Paused the independently resumable source archive at `19,509,248` bytes to prioritize the blocking model transfer.
