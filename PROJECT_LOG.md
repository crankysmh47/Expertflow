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

### 23:06 PKT — Baseline warmup made explicit

- Confirmed from the pinned `llama-cli b10002 --help` that warmup is enabled by default and has an explicit `--no-warmup` switch.
- Added a failing command-contract assertion, then included `--no-warmup` in every measured baseline. This avoids a hidden empty warmup pass and keeps the MoE baseline aligned with the callback probe's no-warmup execution.
- The focused command tests pass `2` tests and the complete suite passes `36` tests in `0.11s`.

### 23:07 PKT — Download recovery paths measured and bounded

- A codeload resume attempt failed closed with curl error `33`: the exact source ZIP endpoint does not honor byte ranges. No partial archive is being treated as complete or verified.
- A shallow Git fetch of the same pinned llama.cpp commit established HTTPS but delivered no pack bytes for several minutes, so it was stopped instead of competing indefinitely with the model.
- Restarted only the model's aria2 transfer with `16` ranges. The control map preserved all `2,645 / 3,443` completed pieces across the restart; by this checkpoint it had reached `2,753 / 3,443` (`79.96%`) with about `2,760` MiB remaining.
- Drafted a separate `native/router_probe` executable and external C-drive build script. They remain uncommitted until compiled against the exact pinned source and exercised on the verified model.

### 23:11 PKT — Official-CUDA router probe build passed

- Confirmed the verified b10002 DLLs export every C API symbol required for model loading, greedy decoding, `cb_eval`, tensor contract checks, and host copying.
- Added a separate `native/router_probe` executable that emits strict schema `1.0.0` JSONL from `ffn_moe_topk-{layer}` and versioned prompt/generated token artifacts for exact parity.
- Generated minimal MinGW import libraries from checked-in `.def` files and linked the probe directly to the official verified `llama.dll`, `ggml.dll`, and `ggml-base.dll`; no runtime or graph source was patched.
- Fetched the seven required headers from revision-pinned raw URLs, recorded every size and SHA-256 in `configs/runtime-artifacts.toml`, and kept them under the external C-drive dependency root.
- Build PASS with GCC 15.2.0. `--help` and `--version` both execute. The final build statically links MinGW support libraries, leaving only system DLLs plus the pinned `llama.dll`, `ggml.dll`, and `ggml-base.dll`. Probe result: `2,930,097` bytes, SHA-256 `d791b3835fcebe6e95efcebc7d0ab62967b0aae46916baeaf53acf377845f2d5`.
- Real Q4 trace and token parity remain pending model verification; this build result alone is not recorded as a telemetry PASS.
- A deliberate missing-model smoke advanced through dynamic linking and loaded the official CUDA, RPC, and Zen4 CPU backends before returning the expected model-open exit `1`; this verifies the executable/runtime boundary without claiming real inference.

### 23:27 PKT — Q4 hash provenance corrected and artifact recovered

- aria2 completed all `3,443 / 3,443` pieces and removed its control map. The file has the pinned length `14,439,361,440` bytes.
- The first repository verification failed closed: local SHA-256 `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5` did not match the manifest's `21005eb9...` value.
- Systematic diagnosis found no zero-filled four-MiB pieces. A fresh revision-pinned Hugging Face HEAD response reports `X-Linked-Size: 14439361440` and `X-Linked-ETag: 4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`, exactly matching the independently streamed local digest.
- Root cause: `21005eb9...` is the Xet CAS object identifier embedded in the signed redirect path, not the downloadable file SHA-256. The earlier log entry calling it the remote ETag is superseded by this correction.
- Added a failing pinned-manifest test for the authoritative file digest, then corrected the manifest and all public command examples. No model bytes were changed or redownloaded.

### 23:40 PKT — Real Q4 baseline, trace, and parity gate completed

- Corrected-manifest verification passed in `9.702` seconds: `14,439,361,440` bytes, SHA-256 `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`, no aria2 control map, and `128,417,697,792` free bytes on C. The full suite passed `36` tests in `0.11s`.
- CPU compatibility run passed in `23.683` seconds with peak GPU use `2,935` MiB and peak working set `11,348,004,864` bytes. The model identified itself as text-only `Q4_0` and generated output.
- Bounded 10-layer GPU run passed with peak GPU use `7,437` MiB and cleaned up afterward. A same-limit warm repeat passed in `8.527` seconds with prompt/generation rates `27.0/26.1` t/s; cold and warm state are disclosed, so no speedup claim is made.
- First trace attempt emitted zero records. A fail-closed diagnostic showed the callback saw the correct names but graph-reservation tensors preceded an active token batch. Gating requests to real decode windows exposed a second contract issue: one-token tensors collapse to `dims=1` while retaining `ne0=8, ne1=1`. Sequential one-token submission and extent-based validation resolved both without patching llama.cpp.
- Final parity PASS: all 38 prompt token IDs and 8 generated token IDs match exactly with tracing disabled/enabled.
- Final trace PASS: `1,350` strict events (`45` decoded tokens × `30` MoE layers), `10,800` selected-expert demands, and no schema errors.
- Measured mean adjacent-token reuse is `32.59%`. Estimated 8-slot static-hotset hit rate is `36.37%` versus LRU `31.87%`. Telemetry is PASS; the overall live-cache decision is `CONDITIONAL` pending stratified traces and measured transfer timing.

### 23:49 PKT — Source provenance closed and recommendation layer started

- Clean full source archive completed without resume: `37,514,614` bytes, SHA-256 `b07eaf97a236c7f9d9ec2e919504a40340fb5b090f7463013c2ceb3c1d3004e2`. Archive integrity and the exact commit-named root passed before extraction; the expected `src/models/gemma4.cpp` is present.
- Removed only the known partial archive and stalled `.git`-only shallow-fetch directory after resolving and checking both paths below `C:\models\expertflow\dependencies`. Rebuilt the GPU probe successfully against the full source tree.
- Added the Layer 2 recommendation/replay execution plan.
- Recommendation TDD red results: `expertflow.recommendation` was absent, then the public CLI rejected `recommend`. Implemented strict evidence-kind validation, measured VRAM headroom, policy selection, reason codes, and the public command.
- TDD green result: `3` focused recommendation tests passed and the complete suite passed `39` tests in `0.12s`.
- Real recommendation written externally: `CONDITIONAL`, `live_cache_enabled=false`, static-hotset replay at 8 slots/layer, and `7,234` MiB remaining configurable headroom after the measured `8,053` MiB peak plus a `1,024` MiB safety reserve. Expert bytes, transfer timing, and a stratified trace remain explicit blockers.

### 23:59 PKT — Causal replay and standalone report implemented

- Replay TDD began with missing-module and missing-command failures. Added deterministic per-event `ready` and `blocking` outcomes for static-hotset and LRU while preserving request, phase, forward, token, and layer identity.
- Refactored aggregate simulation and causal replay to consume one shared policy-outcome engine, preventing report totals from drifting from the simulator. Every replay outcome remains explicitly labeled `estimated`.
- Report TDD covered HTML escaping, verdict and memory evidence, reason codes, schema/source provenance, reproduction commands, and absence of scripts or remote assets.
- A follow-up red test rejected the first renderer because it did not accept a recommendation source path. The renderer now embeds resolved trace and recommendation paths plus both schema versions.
- Added `expertflow replay <trace> --recommendation ... --output report.html --max-events 300` and updated the public README.
- Regenerated the real report at `C:\models\expertflow\runs\q4-probe\report.html`: `49,247` bytes, SHA-256 `b0d6f5ea63a4fe7681b74d141b137871b44ad4ed8824c913b1f6f446a12b713a`. It contains 300 bounded timeline rows and records that 1,050 of 1,350 events are omitted from the view, without omitting them from aggregate totals.
- The in-app browser refused the local `file://` URL under its security policy. No workaround was attempted. Automated structure/content checks pass; browser visual inspection remains an explicit pending checklist item.

## 2026-07-15

### 00:03 PKT — Layer 2 reproduction fixture completed

- Added a checked-in, prompt-text-free fixture containing the first eight `layer_id=0` events from the real Q4 trace in source order. The full source trace SHA-256 is `1b40767674870423e50fa4e0422cefe1ac2d17c86d7e363985c00d538668ad22`; the fixture is labeled `previously_measured`.
- TDD red result: the reproduction test failed because `examples/replay/expected.json` did not exist. Added the trace, expected evidence, and a short CPU-only reproduction guide.
- TDD green result: the fixture reproduces 26 static-hotset hits and 19 LRU hits across 64 expert demands at eight slots/layer. The same checked evidence produces `CONDITIONAL`, `live_cache_enabled=false`, static-hotset policy, and `7,234` MiB measured configurable headroom.
- Final Layer 2 gate passed: `44` tests in `0.18s`, Python compilation, both TOML manifests, every local README link, and `git diff --check`.
- Layer 2 verdict remains `CONDITIONAL`. Live caching stays disabled until expert byte size, transfer timing, and a stratified multi-prompt trace are measured. Browser visual inspection of the standalone report is also pending because the in-app browser blocked its local URL. GPT-5.6 explanation is the next product layer and is not yet claimed.

### 00:17 PKT — Stratified Q4 trace collection exposed a GPU parity failure

- Superseded the proposed API-backed GPT-5.6 explanation layer. The user confirmed this Codex/GPT-5.6 build session itself should provide annotations and judge-facing explanations; ExpertFlow will not add an unnecessary API-key dependency.
- Preflight measured the RTX 5060 Ti at `2,212` MiB used, `13,839` MiB free, and 13% utilization, with no llama or probe process active.
- The first collection wrapper stopped before inference because Windows PowerShell promoted normal CUDA initialization text on native stderr into a terminating `NativeCommandError`. Root cause was the wrapper boundary, not the model. A process-level redirection diagnostic passed `--version` with exit `0`; all subsequent runs used `Start-Process`, hidden windows, redirected stdout/stderr, and explicit native exit codes.
- Collected paired tracing-disabled/tracing-enabled Q4 runs for five public synthetic prompt shapes under `C:\models\expertflow\runs\stratified-q4`: code generation, short factual response, constrained planning, incident summarization, and bilingual explanation.
- Collection completed with exit `0` for all ten GPU inference processes: `4,890` strict events and `39,120` selected-expert demands. Demand-weighted estimated hit rates are static-hotset `38.81%` and LRU `33.11%` at eight slots/layer.
- Exact generated-token parity passed for `code-python-lru` and `incident-summary`, but failed for `short-factual-tides` at token 2, `constrained-gpu-plan` at token 2, and `bilingual-cache` at token 1. Prompt token IDs matched in every case.
- Repeated the three failed cases in both modes. Tracing-disabled repeats match tracing-disabled, and tracing-enabled repeats match tracing-enabled, while cross-mode parity still fails. This rules out ordinary run-to-run sampling nondeterminism.
- A CPU-only `short-factual-tides` control passed exact prompt/generated parity across 600 trace events in `8.019` seconds. Current root-cause boundary is therefore the GPU tensor-observation graph path; the lower-level scheduling mechanism is not yet claimed.
- The original one-prompt telemetry PASS is insufficient across the stratified workload. Live caching remains disabled, and the new evidence is recorded in `docs/evidence/stratified-q4-routing.md`.

### 00:23 PKT — Pinned Vulkan backend recovered transparent GPU telemetry

- The machine has CMake and Ninja but no CUDA compiler. Rebuilding the pinned CUDA scheduler would require a large toolkit installation, so the next bounded control used the official b10002 Windows Vulkan artifact from the same llama.cpp revision.
- GitHub release metadata reported `llama-b10002-bin-win-vulkan-x64.zip`, `32,950,223` bytes, SHA-256 `c2b66ab6912e9fad75c7c6d2000f660bb40bc1f063aa62ec671d471e27dd92ea`. The downloaded C-drive archive matched both size and digest before extraction.
- The unchanged probe binary loaded the ABI-compatible Vulkan DLLs and selected the NVIDIA GeForce RTX 5060 Ti as Vulkan device 0. A formerly failing short-factual control passed exact parity across 600 events.
- Repeated the complete five-prompt matrix on Vulkan with the same Q4 model, 10 offloaded layers, 12 threads, and eight generated tokens. All ten processes exited `0`; all five prompt and generated-token comparisons passed exactly.
- Vulkan totals: `4,890` events, `39,120` demands, demand-weighted static-hotset `38.93%`, and LRU `34.63%`. These full-trace policy figures remain estimated.
- Compared only fixed-prompt prefill events across the CUDA and Vulkan observation paths: 3,840 token/layer events and 30,720 expert demands. Ordered top-8 sets matched exactly for `84.74%` of events and individual selected experts overlapped `99.26%`.
- Prefill policy estimates are effectively backend-stable: CUDA/Vulkan static-hotset `40.6868%/40.6803%` and LRU `33.1283%/33.1641%`. Generated continuations were excluded from the cross-backend comparison because three CUDA callback cases diverged.
- Runtime direction is now explicit: verified CUDA for inference and memory baselines; verified Vulkan for parity-safe GPU router telemetry. Live caching remains disabled pending expert byte size and transfer timing.

### 00:27 PKT — Stratified evidence verification checkpoint

- The first GGUF metadata-inspection attempt failed before reading the model because the invoked Python environment did not contain NumPy. The pinned llama.cpp `gguf-py` package declares NumPy, tqdm, PyYAML, and requests; the default Python 3.11 environment already provides a stable NumPy 1.26.4, so inspection will use that interpreter without adding a project dependency.
- An expected `scripts/check_markdown_links.py` helper was not present in this small repository. Replaced that check with a read-only PowerShell validation of all nine local README links; all targets exist.
- Verification passed: `44` tests, runtime-artifact TOML parsing, all nine local README targets, and `git diff --check`.

### 00:38 PKT — Measured real Q4 expert bytes and the first transfer curve

- The first metadata attempt had selected a Python environment without NumPy. Re-ran the pinned llama.cpp `gguf-py` reader with the stable default Python 3.11/NumPy 1.26.4 environment. It memory-mapped the 14.44 GB GGUF and enumerated 658 tensor headers in 14.2 seconds without copying weight arrays.
- All 30 MoE layers contain 128 contiguous Q4_0 experts. Per expert: down projection `1,115,136` bytes, fused gate/up `2,230,272` bytes, and F32 scale `4` bytes; encoded total `3,345,412` bytes / `3.190434` MiB.
- Pinned CUDA source inspection confirmed a direct `cudaMemcpyAsync` tensor-load path, 128-byte tensor alignment, and only matrix-row end padding. A conservative future three-tensor cache-slot projection is `3,346,048` bytes / `3.191040` MiB. This projection is explicitly not labeled as a measured live-cache allocation.
- Added a dependency-free `expertflow transfer-benchmark` command under TDD. Red tests first failed because the module did not exist, then seven focused tests passed after the CUDA Runtime API wrapper, schema, CLI, cleanup, and validation were implemented.
- The first real command wrapper timed out at its five-second tool boundary just as the complete result was written; no benchmark process remained. The JSON parsed successfully. The bounded run was then repeated with a proper long-running process boundary and exited `0` in 11.5 seconds.
- Extended the command under a second red/green TDD cycle to measure the required pageable-to-pinned staging leg. Eight focused tests passed before the final real run.
- Final artifact: `C:\models\expertflow\runs\transfer-q4\transfer.json`, `23,265` bytes, SHA-256 `c2bb5a820e99552cb161aa37216725bb806934637a7deb4a77083b0733539962`. Contract: four payload sizes, 30 batches, 50 copies/batch, and 10 warm-up copies for pageable staging plus pageable/pinned host-to-device paths.
- One expert's two real pinned-to-GPU weight transfers measured `0.2350` ms mean; warm pageable-to-pinned staging adds `0.1038` ms. A contiguous one-expert H2D copy measured `0.2337` ms mean, and an eight-expert layer fill measured `1.8620` ms. Pinned H2D sustained `13.18–13.39` GiB/s across the curve.
- Layer 0 expert-size and first-transfer requirements now pass. The live gate stays `CONDITIONAL / OBSERVATORY-FIRST`: per-layer CUDA deadlines and same-runtime end-to-end cache benefit are not measured, so `live_cache_enabled` remains false.

### 00:44 PKT — Stratified capacity curve superseded the single-prompt policy choice

- Replayed the fixed-prompt prefill portions of all five parity-safe Vulkan traces across capacities 8, 16, 32, 64, 75, 96, and 108. The combined workload has `3,840` token/layer events and `30,720` expert demands.
- Distinguished two analyses that had previously been easy to conflate. Prompt-local hotsets, optimized independently and demand-weighted, produce the earlier static-8 `40.68%` figure. One deployable global static-8 hotset across the ordered workload reaches only `33.93%` over all 30 layers.
- Narrowed the decision to layers 0–20, the 21 repeating layers left on CPU by the measured 10-layer offload. On `2,688` events and `21,504` demands, global static-8 reaches `33.13%` while LRU-8 reaches `35.13%`. The old single-prompt claim that static-8 beats LRU is therefore superseded for the stratified workload.
- Static placement overtakes LRU at 16 slots. At 64 slots it estimates `92.77%` versus LRU `84.55%`, uses a projected `4,288.76` MiB across the 21 target layers, and has `2.85` ms of serialized pinned H2D traffic per token under the measured transfer curve.
- Capacity 96 fits the measured envelope at `6,433.14` MiB and reaches `99.57%` in-sample static hits; capacity 108 requires `7,237.28` MiB and exceeds the `7,234` MiB configurable headroom before cache-specific workspace. No high-capacity value is promoted to a live recommendation because the static hotset is trained and evaluated on the same five prompts.
- Marked the existing static-8 recommendation artifact as superseded and retained the `CONDITIONAL` gate. Regeneration must preserve blockers for held-out policy evidence, per-layer deadlines, and an exact end-to-end live comparison.

### 00:50 PKT — Regenerated the recommendation and aligned causal replay

- Added `expertflow capacity-curve` under a red/green TDD cycle. It consumes multiple canonical traces, filters phase/layers, emits in-sample policy points, projects aligned cache bytes, and attaches transfer-only estimates. Five focused curve tests passed.
- Generated `C:\models\expertflow\runs\stratified-q4-vulkan\capacity-curve-cpu21.json`: `8,268` bytes, SHA-256 `d4922e70c320265b6ddf1e7f0a3ea4a50617bd88921835118c2cd890200b9b4d`, covering `2,688` measured prefill events and `21,504` demands over the 21 target layers.
- Extended the recommendation engine under red/green TDD to accept the curve, choose the highest tested capacity inside measured post-reserve headroom, and replace resolved byte/transfer/stratification blockers with `HELD_OUT_POLICY_REQUIRED`, `PER_LAYER_DEADLINES_NOT_MEASURED`, and `END_TO_END_CACHE_NOT_MEASURED`.
- The regenerated `recommendation-stratified.json` is `1,850` bytes, SHA-256 `f0e275a3f15002b206976d2ec502539eea06122953f994e0f1ac3fef71fdfcab`. It selects static-96 for replay: `6,433.14` MiB projected cache, `800.86` MiB remaining configurable headroom beyond the separate `1,024` MiB reserve, `99.57%` in-sample static hits, and `90.17%` LRU. Verdict remains `CONDITIONAL`; live cache remains disabled.
- Extended replay under red/green TDD to consume and filter multiple trace files, then extended the report to display the projected cache, post-cache headroom, fit scope, and measured transfer input. Generated `report-stratified.html`: `50,184` bytes, SHA-256 `2f20f23c7ffc9e515bbc5334f9dc0a98750e62068b554d876f7dae1f382ecfaf`. Structural checks confirm all five trace sources, `2,688` replay events, `6,433.14` MiB projected cache, `800.86` MiB post-cache headroom, the new blockers, no script tag, and no remote asset URL.
- Verification passed: `60` tests, Python compilation, both TOML manifests, all 11 local README links, CLI help, recommendation/curve reconciliation, and `git diff --check`.

### 01:00 PKT — Held-out Vulkan trials replaced in-sample policy evidence

- Collected five new public synthetic prompt shapes after freezing the original training workload: structured JSON response, cache arithmetic, prefetch/admission comparison, Urdu transfer explanation, and PCIe bottleneck diagnosis.
- Ran ten sequential b10002 Vulkan processes with the unchanged probe, Q4 model, 10 offloaded layers, 12 threads, greedy sampling, and eight generated tokens. All ten exited `0` in 66.7 seconds. All five pairs passed exact prompt and generated-token parity.
- Strict held-out traces contain `5,640` full events and `45,120` expert demands. The target-layer fixed-prefill filter contains `3,213` evaluation events and `25,704` demands; the separately frozen training filter contains `2,688` events and `21,504` demands.
- Added `expertflow heldout-curve` under red/green TDD. Static residents are fit only on training traces; LRU starts empty and adapts online only on held-out evaluation. The structured curve is `8,862` bytes, SHA-256 `3f93aa31897427e50bf0b3c08006176d5041072b62d22a26ba3868f8a48384bf`.
- At 96 slots/layer, frozen static reaches `96.45%` held-out hits versus LRU `91.55%`; the in-sample static estimate had been `99.57%`. Static does not beat LRU on held-out data until 64 slots. Static-96 estimates `1.4026` ms serialized pinned H2D per 21-layer sweep and still uses `6,433.14` MiB projected cache.
- Regenerated `recommendation-heldout.json`: `1,813` bytes, SHA-256 `947f86ee5d0900b1c2493de1b4a53f080f62e278585bafce5b1d4f776bd9155a`. `HELD_OUT_POLICY_REQUIRED` is resolved; only deadline and end-to-end blockers remain. Verdict is still `CONDITIONAL` and live caching is disabled.
- Added frozen-hotset replay and separate training/evaluation provenance under red/green TDD. `report-heldout.html` is `51,181` bytes, SHA-256 `028e0fcb4526d54f57190549c3e391af96d305b556d286c5c77e31584a871840`. It reconciles exactly to `24,791` ready and `913` blocking held-out selections, contains no script tag, and loads no remote asset.
- Verification passed: `63` tests, Python compilation, both TOML manifests, all 12 local README links, CLI help, held-out recommendation reconciliation, and `git diff --check`.

### 01:11 PKT — Decode policy and Vulkan oracle made the remaining deadline gap concrete

- Evaluated the existing held-out decode events separately. A decode-only static-96 cache trained on only 35 complete training forwards reached `87.87%`, demonstrating that the tiny decode training set does not populate enough residents.
- Extended `heldout-curve` under red/green TDD to support separate train/evaluation phases. The deployment-relevant cross-phase trial fits residents on `2,688` original prefill events and evaluates `735` untouched decode events / `5,880` demands over layers 0–20.
- Cross-phase static-96 reaches `93.28%` held-out decode hits versus cold-start LRU `76.99%`, with `2.6526` ms/token serialized pinned H2D. The curve is `8,846` bytes, SHA-256 `6d60685498e31edcb56ff2a90e112f04130c35c189b27bf02a9bfaac7e2c52f1`.
- Measured callback-observed Vulkan target-layer intervals across 35 complete held-out decode forwards. Median adjacent window is `1.4955` ms and p95 is `1.9305` ms. These values are backend/instrumentation-specific and are not relabeled as CUDA deadlines.
- Added `expertflow deadline-eval` under red/green TDD. With perfect one-layer future knowledge, 719/735 events meet the observed window; all 16 late events are layer 0, and estimated residual blocking falls from `2.6526` to `0.2485` ms/token. The oracle timeline is `248,389` bytes, SHA-256 `627de780f3cc501caf6041ae6d029063fe90061be8ba845ae8fe27b8fde65c45`.
- Regenerated the current decode recommendation (`1,824` bytes, SHA-256 `10b4d3bafe1295182915adc5a6415b798742cfeeac73c21c98dca6fbc00162d2`) and frozen-hotset report (`51,048` bytes, SHA-256 `58944c45a978fb16aba777e8453def7b4ce5a2283ae8e54901c66628aa04b16b`). The report reconciles to `5,485` ready and `395` blocking selections.
- The oracle is a product/evidence milestone, not permission to build or claim a live predictor. `live_cache_enabled` remains false because CUDA deadlines/contention and same-runtime end-to-end benefit remain unmeasured.
- Verification passed: `66` tests, Python compilation, both TOML manifests, all 13 local README links, CLI help, decode curve/recommendation/oracle reconciliation, and `git diff --check`.

### 08:55 PKT - Bounded physical-feasibility stage resumed

- GPT-5.6/Codex accepted the user's staged direction without starting the live CUDA cache path. The active branch is `codex/expertflow-stage0` at `32044e8`; the worktree was clean at resumption.
- Preflight commands: `git rev-parse --path-format=absolute --git-dir`, `git rev-parse --path-format=absolute --git-common-dir`, `git branch --show-current`, `git status --short`, `nvidia-smi`, relevant-process inspection, product-spec search with `rg`, and `uv run pytest -q`.
- No interrupted router probe, llama.cpp, Python collector, build, or benchmark process was active. `nvidia-smi` reported `2,357 MiB / 16,311 MiB` with Chrome and ordinary Windows desktop/display clients; no model compute process was listed.
- The fresh baseline remained green at 66 tests. The repository is a normal checkout on the dedicated `codex/expertflow-stage0` branch rather than a linked worktree; this existing user-directed branch remains the continuous shippable workspace.
- Product-spec review locked the dry-run target to the maximum declared `24-40` range: 40 independent conversations. The new frozen split will be 32 train, 4 validation, and 4 test at the conversation level. Existing five-prompt training and five-prompt held-out artifacts remain historical evidence and will not be silently relabeled into the new split.
- The corpus will cover general chat, code, math/reasoning, translation, multilingual/code-switching, long-context summarization/retrieval, structured output, and deliberate topic shifts. Paired trace-off/trace-on runs will use deterministic greedy decoding, 64 generated tokens where feasible, the verified Q4 model, pinned b10002 Vulkan telemetry runtime, 10 GPU layers, and 12 threads.
- Measurement boundary decision: add single-copy CUDA-event p50/p95 and host `cudaMemcpyAsync` enqueue overhead to the independent transfer benchmark. Keep sustained batch bandwidth separate. Any deadline rerun that combines those CUDA copy values with Vulkan callback windows must be labeled `estimated_cross_backend`, never a CUDA deadline or live-runtime result.
- The first combined plan/log patch failed harmlessly because a mojibake-rendered em dash in prior log output did not match the file's actual bytes. No file was partially written; the plan and append-only log entry were then applied separately.
- Added `docs/superpowers/plans/2026-07-15-physical-feasibility.md` as the bounded execution checklist. It explicitly keeps `live_cache_enabled=false`, requires exhaustive 30-layer x 128-expert byte reconciliation, and stops before live cache changes pending a measured go/no-go.

### 09:10 PKT - Frozen corpus and resumable collection checkpoint

- Added the public synthetic corpus `configs/q4-physical-feasibility-corpus.json`: 40 unique complete conversations, exact 32/4/4 train/validation/test split, and domain counts general chat 10, code 8, math/reasoning 6, translation 4, multilingual 4, long-context 4, structured output 2, and topic shift 2. The eight held-out conversations cover all eight domains once each.
- Corpus validation TDD RED failed because `expertflow.collection` did not exist. GREEN passed 3 tests after adding strict schema, unique conversation/source IDs, declared/observed count reconciliation, and held-out domain coverage.
- Collector TDD RED failed at the absent `CollectionConfig` interface. The first GREEN attempt was correctly blocked by an incorrect SHA-256 hard-coded for the test's `b"model"` fixture. A PowerShell `SHA256.HashData` shortcut also failed because this host's .NET API lacks that static method; the compatible `SHA256.Create().ComputeHash(...)` path produced the correct fixture digest `9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4`.
- Added a dependency-free paired process runner and `expertflow collect-pairs`. It records exact argv arrays, process timestamps/durations, native exit codes, stdout/stderr, token files, trace/parity hashes, strict event counts, corpus/probe/model/runtime provenance, and append-only per-shard attempt history. Resume skips only a fully hash-validated exact-parity shard.
- CLI TDD RED failed because `collect_trace_pairs` was not exposed through the public command. GREEN passed after adding the pinned corpus/probe/model/hash/output/predict/GPU-layer/thread interface.
- Focused gate: 5 tests passed. Fresh full gate: 71 tests passed in 0.38 seconds. `expertflow collect-pairs --help` and `git diff --check` passed.
- Re-resolved physical inputs on C: Vulkan probe `C:\models\expertflow\dependencies\llama-b10002-vulkan\runtime\expertflow-router-probe.exe`; CUDA runtime `C:\models\expertflow\dependencies\llama-b10002\runtime\cudart64_12.dll`; Q4 model `C:\models\expertflow\google--gemma-4-26B-A4B-it-qat-q4_0-gguf\gemma-4-26B_q4_0-it.gguf`.
- Rehashed the 14,439,361,440-byte model in 20.9 seconds: SHA-256 `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`, matching the frozen artifact declaration.

### 09:23 PKT - Forty-conversation paired collection completed

- Command: `uv run expertflow collect-pairs --corpus C:\sem4\Expertflow\configs\q4-physical-feasibility-corpus.json --probe C:\models\expertflow\dependencies\llama-b10002-vulkan\runtime\expertflow-router-probe.exe --model C:\models\expertflow\google--gemma-4-26B-A4B-it-qat-q4_0-gguf\gemma-4-26B_q4_0-it.gguf --model-sha256 4c856...054c5 --output-dir C:\models\expertflow\runs\physical-feasibility-q4-vulkan --predict 64 --gpu-layers 10 --threads 12`.
- First pass completed all 40 conversation pairs in 839.8 seconds wall time / 827.886 seconds summed native process time. Latest-attempt traces contain 152,760 events and 1,222,080 expert demands; 39 pairs passed exact prompt/generated parity and one failed.
- `train-translation-02` returned exit 0 in both modes, matched prompt tokens, and diverged at generated token 0: baseline `100`, instrumented `20663`. Its trace contains 3,180 structurally valid events.
- Systematic-debugging review found a collector preservation flaw before retry: later attempts reused shard-root filenames. TDD RED proved the second attempt path overwrote the first. GREEN passed after retries began writing `attempt-0001`, `attempt-0002`, and so on; 6 collector tests passed.
- The explicit retry command revalidated/skipped 39 shards and reran only the failed pair. It reproduced the same mismatch and identical baseline/instrumented token hashes. Original and retry token artifacts were independently rehashed and remain valid. The failure is deterministic and prompt-specific at the Vulkan observation boundary; no lower-level scheduler mechanism is claimed.
- Final collection manifest: 243,939 bytes, SHA-256 `47e2154870c50b2aed9aef148a7b9a6496173d2a1516529c744fa7f5a1981093`. Downstream tools abort on failed shards by default and require `--exclude-failed-shards`, which records every exclusion in output.

### 09:28 PKT - Expanded held-out evaluation completed

- Held-out evaluator TDD RED began with the absent module. GREEN proves static residents use training only, LRU resets for each conversation, prompt/domain/global totals reconcile, incomplete collections fail, and failed-shard exclusion must be explicit.
- Commands: `expertflow heldout-breakdown` over the collection manifest with training phase `prefill`, evaluation phases `prefill` and `decode`, target layers 0-20, capacity 96, slot bytes 3,346,048, explicit failed-shard exclusion, and the pooled two-slice p95 transfer value `0.23628799617290496` ms.
- Training uses 31 parity-safe conversations and 40,740 target-layer prefill events. The failed training translation shard is listed in both outputs. All eight validation/test conversations remain untouched and parity-safe.
- Held-out prefill: 13,503 events / 108,024 demands; static-96 95.8491%, LRU 87.4806%. Artifact `heldout-breakdown-prefill-static96-p95.json`, 19,586 bytes, SHA-256 `24eb45f64a9e4165b1f5ba0e94d91ce7aa1fbb9fde6c7be337aca99e13872b6b`.
- Held-out decode: 10,584 events / 84,672 demands / 504 forwards; static-96 87.5720%, LRU 86.3390%. Artifact `heldout-breakdown-decode-static96-p95.json`, 19,618 bytes, SHA-256 `b887438a9a18e5f730d8d2a5328f9b7dec69417802c7a5ea0ee2489da00be69d`.
- Static-96 loses to LRU on code, structured-output, and topic-shift decode. It has 10,523 misses versus 11,567 for LRU, only 9.0257% fewer cold bytes. This fails the spec's practical-policy gate of 20% fewer cold bytes or 25% less estimated PCIe stall than LRU.

### 09:31 PKT - Exhaustive Q4 expert layout reconciled

- Read pinned CUDA source around `ggml_backend_cuda_buffer_type_get_alloc_size`: quantized row ends pad to 512 elements and CUDA tensor starts align to 128 bytes. For one expert, Q4_0 padding is 180 bytes for down and 144 bytes for gate/up.
- Expert-layout TDD RED began with the absent inventory module, then with the absent GGUF metadata adapter. GREEN covers exact source spans, divisibility, row padding, 128-byte packing, component mapping, and projected target allocation.
- Command: `python scripts\measure_q4_expert_layout.py --model ...gemma-4-26B_q4_0-it.gguf --gguf-py ...\llama.cpp-a7312...\gguf-py --output C:\models\expertflow\runs\expert-layout-q4\inventory.json --model-revision 21bfe...fff15 --llama-revision a731...f697 --capacity 96 --target-max-layer 20`.
- The 22.1-second run enumerated all 30 layers, 128 experts/layer, 3,840 objects, and three source spans/object. Every encoded object is 3,345,412 bytes; every projected slot is 3,346,048 bytes.
- Independent arithmetic matched exactly: 2,016 slots across 21 layers at 96 slots/layer require 6,745,632,768 bytes / 6,433.136719 MiB. Inventory: 6,803,857 bytes, SHA-256 `daf9a54c1d03a933a667644de412038fb1530ee90ef1761f2f74dbdacb5f1b7a`.

### 09:34 PKT - Pooled CUDA single-copy transfer measurement completed

- Transfer TDD RED required explicit p50, single-copy CUDA events, host API-call timing, and CLI sample count. GREEN added an idle-stream `measure_single_copy` boundary and preserved sustained batch averages as a separate metric.
- Preflight: RTX 5060 Ti, 16,311 MiB, 2,406 MiB desktop allocation, 0% GPU utilization, WDDM, and no probe/llama process.
- Ran three sequential `expertflow transfer-benchmark` commands over 1, 1,115,136, 2,230,272, 3,345,412, 3,346,048, 26,763,296, and 26,768,384 bytes; each used 30 batches, 50 copies/batch, 10 warmups, and 200 single-copy samples. All three completed in 69 seconds.
- Aggregation TDD RED began with the absent raw-sample pooling function and command. GREEN pools samples rather than averaging percentiles. Aggregate: 495,063 bytes, SHA-256 `fb90c8820085f80849977cbf7849de2c899d9c1a4dfd20b5e3fe20e63244b94b`.
- Aligned pinned expert: 0.234016 ms p50 / 0.234272 ms p95 and 13.35 GiB/s sustained. Two real weight slices: 0.235808 ms p50 / 0.236288 ms p95. A one-byte pinned `cudaMemcpyAsync` host call is approximately 0.0013 ms p50 / 0.0015 ms p95. These are idle lower bounds without model compute or contention.

### 09:37 PKT - Backend-labeled deadline sensitivity completed

- Deadline TDD RED proved a mixed-backend estimate could be serialized without naming both timing sources. GREEN now requires both labels and emits `estimated_cross_backend`, `contention_measured=false`, and `live_runtime_measurement=false`.
- Ran `expertflow deadline-eval` twice from the collection manifest using training prefill, held-out decode, layers 0-20, static-96, CUDA idle pinned two-slice p50/p95, and Vulkan b10002 callback windows.
- P50 artifact: 3,677,285 bytes, SHA-256 `477c3fcb2a8857e52af25a049495a598a516f751a3a32bc4921c9e16f21d9d79`. No-prefetch 4.9234 ms/token; oracle residual 0.1702 ms/token; 212 late events.
- P95 artifact: 3,679,388 bytes, SHA-256 `4d46952c1eee603ce6bab09201a75c116f1b493aa061283ba8a9b296a320e8ee`. No-prefetch 4.9335 ms/token; oracle residual 0.1705 ms/token; 212 late events. Late layers: 207 at layer 0, 4 at layer 7, 1 at layer 15.
- The p95 oracle reduces transfer-only residual 96.5432%, but it uses perfect future knowledge and cross-backend windows. It does not overcome the weak non-oracle static-vs-LRU result or clear CUDA deadline/contention/live-runtime gates.
- Verification checkpoint: 84 tests passed in 0.43 seconds, Python compilation passed, CLI/help for new commands passed, and `git diff --check` passed. `live_cache_enabled=false` throughout.

### 09:52 PKT - Conversation-reset capacity curve corrected

- Generated an expanded held-out decode capacity curve from all 31 parity-safe training traces and all 8 validation/test traces at capacities 8, 16, 32, 64, 75, 96, and 108. The first output exposed an evaluator defect before it was used by the recommendation: static-96 matched the breakdown at 87.5720%, but LRU incorrectly reached 95.5026% because cache state carried across evaluation files instead of resetting per conversation.
- Systematic-debugging/TDD RED added direct and CLI tests for two evaluation conversations. The first focused run failed with the missing `evaluation_groups` interface and absent `lru_reset_scope`, as intended. GREEN partitions evaluation events by source trace and pools policy counts only after running a fresh LRU per group. Thirteen focused capacity/breakdown tests passed.
- Reran `uv run expertflow heldout-curve` with training prefill, evaluation decode, layers 0-20, the seven capacities, exact 3,346,048-byte slots, and pooled two-slice p95 transfer input. Duration: 1.288 seconds. Corrected artifact: `C:\models\expertflow\runs\physical-feasibility-q4-vulkan\heldout-capacity-decode-p95.json`, 11,979 bytes, SHA-256 `05966196a75f4a0e77b61137e4ab8250cc4f78bc922ebbfcf9c4580d2fcaf678`.
- Capacity 96 now reconciles exactly to the per-prompt breakdown: `lru_reset_scope=conversation`, static 87.5720%, LRU 86.3390%, 10,523 static misses, and 11,567 LRU misses. The cross-conversation artifact is superseded and not used downstream.
- Regenerated `recommendation-physical-feasibility.json` in 0.123 seconds: 1,834 bytes, SHA-256 `cb8c0d33e888a072646c0d0a47a4e5792686b3d717b563e36ca9ff3805955322`. It remains `CONDITIONAL`, selects static-96 for replay, and records `live_cache_enabled=false`.

### 09:59 PKT - Physical Observatory report and no-go memo completed

- Report TDD RED failed on the absent `physical_evidence` renderer interface and unrecognized replay evidence arguments. GREEN added an all-or-none evidence contract for held-out breakdown, expert layout, transfer aggregate, and deadline estimate; it rejects inconsistent slot arithmetic and any physical report whose recommendation does not say `live_cache_enabled=false`.
- The Observatory now shows the per-layer static-96 contract, exact 6,745,632,768-byte projection, per-domain and per-prompt decode tables, pinned/pageable transfer p50/p95, sustained bandwidth, host enqueue timing, explicit cross-backend deadline labels, and the live-runtime boundary. It remains self-contained with no scripts or remote assets.
- Generated `C:\models\expertflow\runs\q4-probe\report-physical-feasibility.html` in 1.072 seconds: 62,669 bytes, SHA-256 `f3dc647d9965d726771632421b8fa5dffddc165d3ebae49f6f10381bbb75a90c`. Replay totals reconcile to 10,584 events, 74,149 ready selections, and 10,523 blocking selections.
- Served the report with `python -m http.server 8765 --bind 127.0.0.1 --directory C:\models\expertflow\runs\q4-probe`. The first browser bootstrap attempt used a stale `createBrowserClient` name and failed without page interaction; inspecting the pinned module showed `setupBrowserRuntime`, which initialized the supported in-app browser path.
- Visual inspection passed at the default 1280 x 720 viewport and an explicit 390 x 844 narrow viewport. All 10 sections and 3 tables rendered, the evidence/warning labels were readable, page-level horizontal overflow was absent, and wide tables used scoped horizontal scrolling. The reproduction command wrapped inside its panel. The temporary viewport was reset, the browser tab was finalized, and HTTP server PID 26912 was stopped; port 8765 no longer listens.
- Added `docs/evidence/q4-live-cache-go-no-go.md`. Decision: no-go for a minimal live-cache spike. Projected fit passes, idle transfer timing is encouraging but incomplete, the practical non-oracle policy fails its 20% cold-byte gate at 9.03%, and an off-by-default cache boundary is not implemented. No live llama.cpp cache code was started.
- Updated README status, current commands, physical evidence inputs, report path, and judge-facing decision. The README-writer/humanizer guidance kept claims direct and preserved the measured/estimated distinction.
- Verification checkpoint: 87 tests passed in 0.42 seconds and `git diff --check` passed. A later artifact-summary PowerShell one-liner initially failed on an empty pipeline after a `foreach`; the corrected `$rows=foreach(...)` form produced the hashes above. `live_cache_enabled=false` throughout.
- Pre-commit verification matrix completed in 2.242 seconds: 87 tests, Python compilation, top-level and seven relevant CLI help paths, three TOML files, all 18 local README links, corrected curve/breakdown hit reconciliation, exact 6,745,632,768-byte layout/recommendation reconciliation, false live-cache assertion, and `git diff --check` all passed.

### 10:04 PKT - Clean judge replay exposed and fixed stale fixture identity

- Committed the shippable physical-evidence checkpoint as `9b1889e` (`feat: publish physical feasibility gate`). Created `C:\models\expertflow\judge-replay-clean-9b1889e.zip` with `git archive`, expanded it on C, and ran `uv sync --frozen --extra dev`, 87 tests, CLI help, and the checked-in simulation. Installation and all commands passed, but the final provenance assertion correctly failed: `expected.json` contained a stale trace SHA-256.
- Systematic-debugging inspection separated two issues. The Git blob/current LF checkout hashes to `245aac7ffa83f464f33f220c2c7cafbf931671884c48fe2f92d48795ef11df8e`; with this machine's global `core.autocrlf=true`, `git archive` materializes the text entry with CRLF and a different raw-byte hash. The historical `1b4076...` value matched neither representation.
- Fixture TDD RED added a cross-platform canonical hash assertion and failed on the absent normalization contract. GREEN records `source_trace_hash_normalization=utf-8-lf`, updates the canonical hash to `245aac...df8e`, and explains the rule in the fixture README. The focused test and fresh full suite passed; 87 tests remain green and `git diff --check` passes.
- The first clean archive remains preserved as failed judge-replay evidence. A new commit and a second unique clean archive are required before the judge path can be called verified.

### 10:08 PKT - Clean judge replay verified

- Committed the normalized fixture contract as `15a945b` (`fix: normalize replay fixture identity`) and created a second fresh setup at `C:\models\expertflow\judge-replay-clean-15a945b` from `git archive`.
- `uv sync --frozen --extra dev` created a new virtual environment and installed the locked package. The clean full suite passed: 87 tests in 0.52 seconds. `expertflow --help` passed and the checked-in `simulate` command wrote `replay-simulation.json` successfully.
- The first post-run checker incorrectly looked for `event_count` and `expert_demand_count` at the simulation root. The CLI schema intentionally stores demand counts inside each policy and does not serialize the source event count. No project command or test failed; the checker was corrected to count JSONL records and compare both policy demand fields.
- Corrected reconciliation passed: canonical trace SHA-256 `245aac7ffa83f464f33f220c2c7cafbf931671884c48fe2f92d48795ef11df8e`, 8 events, 64 demands, static 26 hits / 38 misses, LRU 19 hits / 45 misses. Output SHA-256: `54f46ccbf719b37f5cca55cc87d1625b8e8abdfd88f34faadc13042709010162`.
- The clean judge replay is now verified for exact commit `15a945bde3e4e9f1341cbce01702776381ef8774`. Both clean archives and their virtual environments remain under the C-drive artifact root; no D-drive storage was used.

### 10:12 PKT - Bounded physical-feasibility stage closed

- Fresh final verification completed in 2.232 seconds: 87 tests in 0.45 seconds, Python compilation, CLI startup, three TOML files, 18 local README links, all collection/evaluation/layout/timing/report reconciliations, and `git diff --check` passed.
- Final evidence reconciliation: 40 corpus conversations; 39 parity-passed and 1 explicitly failed pair; 8 held-out conversations; static-96 87.5720% versus conversation-reset LRU 86.3390%; 3,840 packed objects; exact projected cache 6,745,632,768 bytes; 3 transfer trials; deadline label `estimated_cross_backend`; contention measured false; live runtime measured false; live cache enabled false.
- The HTML remained unchanged after visual inspection: SHA-256 `f3dc647d9965d726771632421b8fa5dffddc165d3ebae49f6f10381bbb75a90c`. The HTTP server remained stopped, and no llama/probe/ExpertFlow model process was active. `nvidia-smi` reported 2,422 MiB desktop allocation and 2% utilization, not a model workload.
- Git environment is a normal repository (`.git` equals the common Git directory) on `codex/expertflow-stage0`. The remote advertises no default branch/reference, so no merge, push, or PR action was inferred. The continuously shippable branch is preserved for user review.
- Final decision remains no-go for a minimal live-cache spike on current evidence. No live allocation, blocking slot experiment, asynchronous prefetch, predictor, MTP, KV-cache change, CUDA deadline claim, or runtime speedup claim was added.

### 10:18 PKT - Protected live-cache gates opened

- The user changed the runtime verdict to `CONDITIONAL-GO-FOR-BOUNDED-SPIKE` while keeping the Observatory at `d846bdfcb1980dfc44d9f951e2824f58429f16d7` as the guaranteed submission floor and `live_cache_enabled=false` as the release default.
- Before any installer or native source change, `uv run pytest -q` passed all 87 tests in 0.43 seconds, `git diff --check` passed, and the protected checkout was clean. The annotated tag `observatory-floor-2026-07-15` was created at `d846bdf`.
- Created the isolated branch `codex/live-cache-blocking-spike` at the same commit in `C:\models\expertflow\worktrees\live-cache-blocking`. `uv sync --frozen --extra dev` created its C-drive environment and a second baseline passed all 87 tests in 0.84 seconds. The protected branch was not modified.
- Read-only source inspection confirmed the existing true-router-selected `GGML_OP_MUL_MAT_ID` copy boundary in pinned `ggml/src/ggml-backend.cpp` and the authoritative top-k construction in `src/llama-graph.cpp`. CUDA offload uses `GGML_OP_MUL_MAT_ID` batch size and defaults `GGML_OP_OFFLOAD_MIN_BATCH` to 32.
- Verified the static-96 correction before changing the verdict: the serialized report fits on 31 `train-*` traces and evaluates eight untouched conversations (four validation, four test). `build_held_out_capacity_curve` counts residents only from `training_events` and scores only `evaluation_events`. The 9.0257% cold-byte reduction remains unchanged and is not relabeled.
- Flagged an exact-execution constraint: this model selects eight experts per token, while unchanged `MUL_MAT_ID` requires every selected expert to be simultaneously addressable. The bounded working interpretation is seven fixed slots plus one replaceable slot for a controlled one-layer proof. If a measured sequence cannot exercise repeated one-at-a-time replacement, or if one/two total slots are mandatory, Gate 4 stops rather than introducing a broad graph/kernel rewrite.
- Harmless read-only command failures were preserved: one `rg` invocation named an absent `pytest.ini`; one Git-isolation command called `.Trim()` on empty superproject output; three broad `rg` searches returned exit 1 after producing matches or because a Windows glob was invalid. No repository or runtime file was changed by those failures.
- Added the bounded design at `docs/superpowers/specs/2026-07-15-bounded-live-cache-design.md`. The four-hour bound is 90 minutes for provenance/toolchain/unmodified build and 150 minutes for the blocking proof, with earlier stop on any gate failure.
- Updated `docs/evidence/q4-live-cache-go-no-go.md` and the README to `CONDITIONAL-GO-FOR-BOUNDED-SPIKE`. The 9.0257% static-96 cold-byte improvement is still reported exactly; it permits only the physical proof and remains below the 20% expansion target. `live_cache_enabled=false` remains the default.
- Searched all measured trace transitions for an exact one-replaceable-slot proof. The held-out set contains 2,316 adjacent top-8 transitions with seven-expert overlap, but no held-out prompt was selected. Training trace `train-general-08` has a 50-event layer-1 window (tokens 62-111) with seven fixed experts `{0,13,84,88,92,95,124}` and dynamic expert `{28,40,66}`. This supports repeated one-slot replacement while preserving unchanged top-8 execution.
- Added `docs/superpowers/plans/2026-07-15-bounded-live-cache.md`. It separates protected provenance, supported toolchain, unmodified pinned runtime, TDD slot planning/integration, live proof, and expansion decision into stop-on-failure gates.

### 10:36 PKT - Gate 1 protected floor reproduced and recorded

- Captured immutable pre-install provenance at `C:\models\expertflow\runs\live-cache-spike\gate1\provenance.json`: Windows 11 Pro 10.0.26200, RTX 5060 Ti, driver 591.86, compute capability 12.0, 16,311 MiB VRAM, CMake 4.3.1, Ninja 1.13.2, uv 0.11.1, Git 2.52.0.windows.1, existing VS Community 2026 18.3.1, and no `nvcc` or PATH-visible `cl`. The record hashes 116 relevant executables/DLLs plus the 14,439,361,440-byte Q4 model; the model SHA-256 remains `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`. Relevant non-secret environment variables were captured; every `EXPERTFLOW_LIVE_CACHE*` variable was unset.
- Verified the annotated tag object `2da2dd960dd74aee319e680615e124c68077319c` records exactly protected commit `d846bdfcb1980dfc44d9f951e2824f58429f16d7`. The protected checkout remains clean on `codex/expertflow-stage0`; the isolated live worktree is on `codex/live-cache-blocking-spike`.
- Archived the protected tag to `protected-d846bdf.zip` (SHA-256 `be042bc9578f845a3b4e77fd2a53adfe18fabbb2463810a9348941e6a381c6d2`), expanded it on C, installed the locked environment, passed all 87 tests in 0.52 seconds, and reproduced the judge fixture: canonical LF trace `245aac7ffa83f464f33f220c2c7cafbf931671884c48fe2f92d48795ef11df8e`, 8 events / 64 demands, static 26/38 and LRU 19/45.
- The first post-run evidence checker failed because it looked for policy totals at the JSON root rather than under `simulation`; all protected project commands had already passed. Schema inspection proved the checker defect. Correcting only the checker yielded the expected totals.
- Found and documented a second evidence-identity issue: raw replay JSON hashes depend on the absolute `source_trace` path. The fresh raw hash is `2550bcbb081994d8fe95fab0da5b50ede02f5c81d8b04a65a7b7f020fd94dfff`, while the historical raw hash is `54f46ccbf719b37f5cca55cc87d1625b8e8abdfd88f34faadc13042709010162`. Normalizing only that path field gives matching canonical SHA-256 `6c658457ea3320f065a47b8931bb5992700c2ba075c6b77a2351072a3589358c`; measured values are untouched.
- Re-executed the Observatory's embedded replay command from the protected archive using the same 31 train and eight validation/test inputs. `report-physical-feasibility.html` remained byte-identical at 62,669 bytes and SHA-256 `f3dc647d9965d726771632421b8fa5dffddc165d3ebae49f6f10381bbb75a90c`, with `live_cache_enabled=false`.
- Independently verified static-96 selection uses only 31 parity-safe training conversations and scores eight disjoint held-out conversations; overlap is empty and LRU resets per conversation. Static-96 remains 74,149 hits / 10,523 misses versus LRU's 73,105 / 11,567, a 9.0257% measured cold-byte reduction. This supports only the user-authorized bounded proof and remains below the earlier 20% expansion target.
- A PowerShell tag-peeling check mishandled Git's `^{}` syntax and returned exit 128. The corrected direct `git cat-file` inspection passed and is the tag-object evidence above. No file, tag, or branch changed during either check.
- Added `docs/evidence/live-cache/gate1-protected-floor.md`. No VS/CUDA installer has been downloaded or run and no llama.cpp source has been modified.
- Gate 1 pre-commit verification passed all 87 tests in 0.43 seconds, `git diff --check`, a clean protected checkout at `d846bdf`, and an exact tag-object target check. The finalized 3,371-byte append-only Gate 1 command ledger hashes to `11a6748bf6ef11148124e9e0009987aa07e4e120001b598dcaf06fdbfb598f0c`.

### 10:46 PKT - Gate 2 prepared; supported installation blocked at local UAC

- Preflight found 117.91 GiB free on C, a non-elevated Codex process, no `C:\BuildTools2022`, and no CUDA v12.8 installation. Existing Visual Studio Community 2026 remains present and unchanged.
- Downloaded only the two official installers to `C:\models\expertflow\installers`. VS Build Tools 17.14.37502.11 is 4,458,200 bytes, SHA-256 `c9cc76c0d03cbcb523e18b559a978ce5df11a667ef78e4e4d264331f1227ddd7`, and has a valid Microsoft signature. CUDA 12.8.1 network installer 1.0.14 is 14,404,064 bytes, SHA-256 `779bee8ff557255c1cf5f36e0230f081675b9bb41e44be38839920cd5209bdeb`, and has a valid NVIDIA signature. Download durations were 3.691 and 6.846 seconds respectively.
- Started the exact VS side-by-side command twice with `Microsoft.VisualStudio.Workload.VCTools`, recommended components, `Microsoft.VisualStudio.Component.VC.14.39.17.9.x86.x64`, and `Microsoft.VisualStudio.Component.Windows11SDK.26100`. Both elevation requests waited about 122.4 seconds then returned `The operation was canceled by the user`; no setup process or target directory appeared.
- The user explicitly authorized accepting UAC through Windows Computer Use. The skill's supported `list_apps` and `list_windows` path could not expose the UAC secure desktop, so it could not click consent. No UAC/security setting was changed and no terminal/UI or elevation bypass was attempted. A person at the machine must accept the secure-desktop prompt.
- Continued only non-admin Gate 2 preparation. Cloned official NVIDIA CUDA samples tag `v12.8` to C in 62.565 seconds and verified detached clean commit `db3eea23946bca2e90a75eca2b5b3e07158a9e11`. Added external unbuilt CMake fixtures that enforce MSVC 19.39/CUDA 12.8 and compile the pinned official `deviceQuery` source for SM120 only.
- Added `docs/evidence/live-cache/gate2-toolchain.md` with status `BLOCKED-PENDING-LOCAL-UAC`. Gate 2 is not marked passed; Gate 3, llama.cpp compilation, and live-cache source changes have not begun. `live_cache_enabled=false` remains the only state.
- Partial Gate 2 checkpoint verification passed all 87 tests in 0.43 seconds, `git diff --check`, and a clean protected worktree. The external partial command ledger is 2,844 bytes, SHA-256 `6d3d92781a1009f384f004ebcc99d16b7e7ea39f58e15081de1d765195580f63`; exact fixture and installer-ledger hashes are in the Gate 2 evidence file.

### 16:38 PKT - Gate 2 supported CUDA toolchain passed

- With the user present to accept UAC, the exact VS 2022 Build Tools command completed at `C:\BuildTools2022` with exit 0 in 891.907 seconds. `vswhere` reports Build Tools 17.14.37502.11 and the original VS Community 2026 18.3.11512.155 as complete and launchable. Required-component checks resolve the VCTools workload, VC 14.39.17.9, and Windows SDK 26100 to Build Tools. The original VS 2026 `cl.exe` retained pre/post SHA-256 `a040a369b63177427584253a1a670cebe4a10e770d7c1b5c9d3d568e30433c8e`.
- The developer shell selects `-vcvars_ver=14.39` and resolves `cl.exe` 19.39.33523.0, SHA-256 `dc1ef4e36c7044ae9bd0ce24d27de45f8fe26dc1210897b8717e8ef0232360e8`. A first `cl /Bv` check printed the correct version but returned D8003 because no source was supplied; the corrected command compiled and ran a trivial probe.
- The explicit toolkit-only CUDA 12.8.1 command completed with exit 0 in 516.298 seconds. Installed packages are nvcc, cudart, cuBLAS runtime/dev, nvJitLink, NVRTC runtime/dev, Thrust, and VS integration. No `Display.Driver` package was selected, no reboot is pending, and `nvidia-smi` still reports driver 591.86. `nvcc` reports V12.8.93.
- The installer added machine `CUDA_PATH` and `CUDA_PATH_V12_8` but did not alter machine PATH. Reproduction explicitly prepends CUDA v12.8 `bin` inside the VS 2022 shell. That shell resolves MSVC 19.39, nvcc 12.8.93, CMake 4.3.1, and Ninja 1.13.2.
- The strict SM120 CMake probe configured, built, and ran in 7.438 seconds, identifying MSVC 19.39.33523.0 and NVIDIA CUDA 12.8.93 with that host compiler. The official CUDA samples `v12.8` deviceQuery overlay configured, built, and first ran in 5.911 seconds; it detected one RTX 5060 Ti, runtime 12.8, capability 12.0, WDDM, and `Result = PASS`.
- A consolidated transcript launch initially ran `deviceQuery.exe` outside the prepared CUDA PATH and returned Windows status `0xC0000135`. `dumpbin /dependents` proved the binary dynamically requires `cudart64_12.dll`; running it with CUDA `bin` prepended passed. This is an environment-boundary diagnosis, not a CUDA failure. Two transcript/reporting failures—PowerShell promoting the normal `cl` stderr banner and a direct `foreach` pipeline parser error—were corrected without changing measurements.
- Repeated the pinned CUDA 12.4-runtime transfer benchmark in three independent processes for the 3,346,048-byte aligned slot. Preflight was 1,960 MiB desktop VRAM, 8% instantaneous utilization, and no model/probe process. The 600 pooled pinned samples measured 0.233984 ms p50 and 0.236864 ms p95 versus prior 0.234016/0.234272 ms: -0.0137% and +1.1064%. Sustained bandwidth was 13.319 GiB/s. Aggregate SHA-256 is `477c8659a407e3a7a722834452fd243873aa985e42da60853e5b588286654e97`.
- The compiler probe initially wrote a 1,156-byte object into the repository because `/Fo` was omitted. That known artifact was hashed, removed, and the durable helper now writes it externally. Verification confirmed the repo object is absent and the helper still returns zero.
- Final Gate 2 verification passed all 87 tests in 0.60 seconds, `git diff --check`, a clean protected checkout at `d846bdf`, unchanged driver 591.86, and no pending reboot. `verification.txt` is 12,248 bytes, SHA-256 `cfb4b426c3f1554c438e6c8f33a16028b4ced613d88c48f057b220a1fbe7acc6`; the 11,183-byte append-only command ledger hashes to `9beb0cee23744312c760262879594e361a8400a6b59f678a87ad2bd4fa79fc73`.
- Gate 2 is PASS. Gate 3 may start only from exact unmodified llama.cpp commit `a7312ae94f801fc9c6786dc56e38df57b964f697`. No live-cache source exists and `live_cache_enabled=false` remains mandatory.

### 17:10 PKT - Gate 3 clean CUDA build blocked by upstream Release test

- Created a fresh, detached, clean Git checkout at exact llama.cpp commit `a7312ae94f801fc9c6786dc56e38df57b964f697`. A byte comparison of all 3,167 tracked files against the prior pinned archive found 3,167 matches, zero missing files, and zero differences. The comparison artifact hashes to `46bbe0631254a94f5e3290f6d55c0101c3982deaba6fc3c3bc5fb0c68682240b`.
- Configured an unmodified Release build with VS 2022 MSVC 19.39.33523.0, CUDA 12.8.93, `GGML_CUDA=ON`, `GGML_NATIVE=OFF`, tests/examples enabled, curl disabled, and device architecture `120a-real`. Configure passed in 9.085 seconds; the full 634-target build passed in 300.4 seconds. The source remained clean.
- The initial CTest run failed `test-jinja-py` because the PATH-visible Python lacked upstream's unpinned Jinja dependency. Created a Gate-3-only Python 3.11 environment, installed Jinja2 3.1.6 and MarkupSafe 3.0.3, and reran the focused test successfully in 28.34 seconds. No ExpertFlow lockfile or llama.cpp source changed.
- The final full CTest rerun passed 42/43 in 143.93 seconds. `test-backend-ops` and every test except `test-opt` passed. Durable output is 152,236 bytes, SHA-256 `7abab529beb0d97c5f024907af87223be456ac4845aa1ced15b86f4378b1ce18`.
- `test-opt` fails the same four high-level AdamW forward-result checks on CUDA0 and CPU; weights/backward results pass and all SGD cases pass. Two focused unmodified Release reruns reproduced it. A separate exact unmodified CPU Debug build passed 2/2. A disposable diagnostic worktree showed the failing Release result remained empty (`ndata=0`, `loss=0`, `loss_unc=NaN`); adding logging alone changed behavior and made Release pass. This is an upstream Windows/MSVC optimization-sensitive heisenbug, not a demonstrated CUDA-kernel failure, but it prevents the required full-test pass.
- Queried the official checks for the exact commit. Windows CPU and CUDA build jobs succeeded but did not run CTest. Linux `gpu-cuda` excluded `test-opt` from Debug and passed all Release `main|python` tests. The platform evidence does not waive the required local failure.
- The plan explicitly says any source/toolchain failure stops cache work. Gate 3 is therefore `BLOCKED` at Step 3. Router-probe rebuild, model load, CPU/ten-layer validation, tracing parity, VRAM/throughput measurements, and all live-cache code remain unstarted. `live_cache_enabled=false`; no speedup, CUDA deadline, parity, or live-runtime claim is made.
- Added `configs/llama-a7312ae-cuda128.json` and `docs/evidence/live-cache/gate3-clean-llama.md`. An explicit user exception is required to continue only the remaining unmodified Gate 3 inference/parity checks while carrying the unrelated optimizer-training test as an upstream Windows Release exception.
- Blocker-checkpoint verification passed all 87 ExpertFlow tests in 0.54 seconds, JSON configuration parsing, 28-line JSONL command-ledger parsing, and `git diff --check`. The protected checkout and exact llama.cpp source are clean. The 12,435-byte Gate 3 command ledger hashes to `1eabeee28f325cbed8b6ef8c445f3356aae7b75e0dc3e19c31e730110fb1b27c`.
- The first checkpoint commit attempt was stopped before commit by `git diff --cached --check` because three new Markdown status lines used trailing spaces for hard breaks. Replacing them with blank-line-separated paragraphs corrected formatting only; no evidence or gate decision changed.

### 19:10 PKT - Gate 3 inference continuation reached a parity fail-stop

- The user authorized a narrow exception for the single reproducible Windows/MSVC Release `test-opt` heisenbug. The exception did not waive inference, tracing, router, memory, replay, or cache-runtime gates. Validation used only the exact clean, pinned, unmodified llama.cpp CUDA build; the disposable diagnostic-logging build was not used.
- Preflight found driver 591.86, 16,311 MiB total VRAM, 13,155 MiB free, zero GPU utilization, and no llama/probe/ExpertFlow model process. All `EXPERTFLOW_LIVE_CACHE*` variables remained unset. The 14,439,361,440-byte Q4 model rehashed in 28.679 seconds to `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`.
- Copied all 90 clean build outputs into an external evidence runtime rather than modifying the clean build directory. Every copied file matches its source by byte size and SHA-256. Rebuilt the existing router probe against those libraries in 3.86 seconds; the 2,950,816-byte probe hashes to `99ee90b72052a9601669b73f055a19dca02be7ede685e9a717e034320ecb607c`. The exact llama.cpp checkout remained clean.
- A first runtime-smoke PowerShell wrapper failed because `$ErrorActionPreference=Stop` promoted llama.cpp's normal version banner on stderr to `NativeCommandError`. Direct native execution through `cmd` proved the binaries were healthy and recorded their real exit codes. A separate read-only search also used an invalid Windows wildcard, and the resumption ledger lookup initially omitted the `gate3` directory. All three command-layer failures are preserved; none changed source, binaries, or measurements.
- Added external evidence-only measurement and validation helpers. The bounded eight-token ten-layer GPU smoke passed trace-off/on token parity, strict schema, causal order, and all 1,350 expected routing events.
- Ran the frozen baseline prompt with greedy sampling, 64 generated tokens, single-token decode, ten GPU layers, and 12 threads. The clean trace-off and two trace-on runs produced identical tokens; the repeated traces produced identical ordered router selections; strict validation found all 3,030 expected events across 101 forwards, 30 layers per forward, and eight experts per event.
- The previously verified reference runtime independently reproduced exact trace-off/on token parity. Its trace-off run took 8.279 seconds and peaked at 7,530 MiB device use. The clean trace-off run took 8.648 seconds and peaked at 7,466 MiB, a descriptive 4.4521% duration difference. Clean trace-on runs took 9.354 and 9.009 seconds and peaked at 7,451 and 7,394 MiB. Settled GPU use returned to the desktop range after every run, with no persistent allocation/process growth observed.
- Cross-runtime parity failed. Prompt tokens match, but generated tokens first differ at index 35: reference `171502`, clean `219220`. The first selected-expert-set difference is event 24 at forward 0, prompt token 2, layer 24: clean selects expert `117` where the reference selects `113`. Of 2,190 events before token divergence and therefore directly comparable, 256 expert sets differ. The clean runtime's internal repeatability does not satisfy the required parity with the previously verified runtime.
- This is a hard, non-waived Gate 3 failure. The validation matrix stopped before CPU and `train-general-08`; those runs could not reverse the failed prerequisite. No one-layer recommendation was issued and no cache source, slot planner, allocator, transfer path, async stream, predictor, or MTP work began. `live_cache_enabled=false` remains mandatory.
- Post-stop regression evidence passed all 87 ExpertFlow tests in 0.90 seconds and reproduced the judge fixture at 8 events / 64 demands, static 26/38 and LRU 19/45. An initial replay checker used obsolete root-level fields and was corrected after schema inspection; the project command itself had passed. The protected Observatory remains 62,669 bytes with SHA-256 `f3dc647d9965d726771632421b8fa5dffddc165d3ebae49f6f10381bbb75a90c`. The protected checkout is clean at `d846bdf`, and exact llama.cpp source is clean at `a7312ae`.
- The consolidated measured fail-stop summary is `C:\models\expertflow\runs\live-cache-spike\gate3\gate3-inference-stop-summary.json`, 16,415 bytes, SHA-256 `e6d24d8166d216f5e97e9a8b38771c1f40cc8f5706647ce2704c1c89a39b5411`. Fresh checkpoint verification passed 87/87 ExpertFlow tests in 0.51 seconds, replay reconciliation, zero durable-artifact hash mismatches, configuration and JSONL parsing, protected/source cleanliness, and the no-cache environment/process checks. The final pre-commit 58-entry external command ledger is 27,356 bytes and hashes to `4e206c1b94280747e5423ff143ae81870c9efd9dc14f1ffe106e0ffe107bc6a2`.

### 20:30 PKT - Bounded divergence audit explains cross-build drift but confirms trace-parity fail-stop

- Preserved the protected Observatory, clean pinned llama.cpp checkout, commit `1fc549bd87ed58aea7c603c59ac26037324ec157`, and all earlier Gate 3 evidence. The audit began from 87 passing ExpertFlow tests. No cache code or llama.cpp source modification was made; `live_cache_enabled=false` remained the only state.
- Reconciled the full reference/clean invocation. The 14,439,361,440-byte GGUF, exact prompt bytes and token IDs, raw/no-template tokenization, greedy sampler, context, batch/ubatch, ten-layer offload, thread count, split/device, flash setting, F16 K/V, and mmap/mlock options match. Explicit build differences include official CUDA 12.4 versus local CUDA 12.8, different MSVC/LLVM environments, dynamic backend/CPU-variant selection, CUDA architecture sets, and runtime DLLs.
- Corrected the earlier event coordinate: the first selected-set difference is forward 0, token **index 0**, token ID 2 (BOS), layer 24—not token index 2. Reference rank 8/9 is expert 113/117 with gap `0.0002393993`; clean is 117/113 with gap `0.0000435682`. Maximum absolute probability drift is `0.0004364327`, exceeding both gaps. All 128 scores and host rankings reproduce the runtime selections exactly.
- Bounded backward tracing found byte-identical `inp_scaled` input across both runtimes. Drift first appears in checked layer-0 output at maximum `1.9e-6` / RMS `9.36e-8`, then accumulates before layer 24. A controlled exact-source local Release-versus-Debug CPU capture showed the same top eight but maximum router-probability drift `0.0002273954` / RMS `0.0000309939`. The reference-versus-clean mismatch is therefore classified as benign build-dependent accumulated floating-point drift at a near-tied boundary, not corrupted inference.
- Expanded clean stability to three domains, with three trace-off and three trace-on runs per prompt. Each mode is internally token-deterministic; traced repeats have identical ordered routing, complete 30-layer/eight-expert events, strict causal order, and no persistent model process. General chat passes off/on token parity. Code and translation fail off/on generated-token parity deterministically.
- Source inspection establishes the perturbation boundary: with no callback the pinned scheduler computes a whole backend split asynchronously; with the callback it computes synchronized graph views ending at requested router tensors. The current CUDA trace path is therefore observationally intrusive. Internal repeatability does not satisfy the non-waived trace-off/on parity gate.
- Split verdict: cross-binary bit identity is diagnostic and no longer the blocking reason, but Gate 3 remains **FAIL-STOP** on clean trace safety. The clean Release binary is not promoted as a cache correctness baseline, no one-layer recommendation is issued, and Gate 4 remains closed. Evidence is under `C:\models\expertflow\runs\live-cache-spike\gate3-divergence-audit`; the machine-readable verdict is `audit-summary.json`.
- Final verification passed 87/87 tests in 0.49 seconds; replayed 8 events / 64 demands with static 26 hits and LRU 19; rebuilt/checked the opt-in diagnostic probe contract; parsed all selected JSON and all prior ledger records; reconciled zero artifact-hash mismatches; passed `git diff --check`; confirmed protected commit `d846bdfcb1980dfc44d9f951e2824f58429f16d7` and pinned llama.cpp `a7312ae94f801fc9c6786dc56e38df57b964f697` are clean; found zero live-cache environment variables and zero persistent model processes. A fresh pre-commit rerun passed 87 tests in 0.43 seconds and reproduced the fixture again. `audit-summary.json` is 3,925 bytes, SHA-256 `142f579a466a80b377ded5a99fcb5362d450c87bd1dbc123cf176664cda60cdf`; `verification.json` is 1,477 bytes, SHA-256 `f94314f3967d5cc829105bcd85b4290b96b1145a94c3ec3a18c619198a4226f2`; the 39-record pre-commit command ledger is 10,690 bytes, SHA-256 `ec353b1526f9f97853222bda8add2b3b58864b116f498bd13db3249dc71d511d`.

### 20:55 PKT - Callback-derived real-model evidence quarantined

- The tracing-repair directive treats the cross-build numerical drift as resolved and the callback's observational perturbation as the active blocker. Created branch `codex/trace-observer-repair` from preserved audit commit `c41b9394c0443a66b3b486936d128c034ea3a4d7`; the protected Observatory and pinned llama.cpp checkout remain out of scope for modification.
- TDD RED failed because `configs/trace-evidence-status.json` did not exist. Added the manifest with label `trace_v1_perturbing`, stopped corpus collection, closed Gate 4, and marked all current callback-derived real-model roots and locality/static/LRU/session/oracle/deadline claims ineligible for final use.
- The historical multi-domain `93.28%` result is explicitly withdrawn pending parity-safe recollection. Existing artifacts are preserved and may be used only for audit/diagnosis. The small checked-in replay fixture remains available for offline parser/simulator reproduction and cannot support a real-model cache-policy claim.
- No cache code, replacement trace, MTP, prefetch, or API work began. The next bounded experiment is T0 callback disabled versus T1 callback registered with a completely empty body.

### 21:30 PKT - First perturbing observer behavior isolated

- Added diagnostic trace modes sequentially under red/green contracts. T1 registers a callback whose entire body is `return false`; T2 increments one preallocated counter; T3 records token/layer metadata in fixed storage with overflow failure and sentinels. Each variant passed three disabled-versus-observed repetitions across general, code, and translation with exact prompt/generated token parity and internal determinism.
- T2 observed stable callback ask counts of 140,132 / 150,708 / 142,776. T3 recorded 1,590 / 1,710 / 1,620 selected-tensor metadata events with zero overflow and intact sentinels. These results rule out registration alone, callback invocation volume, layer parsing, token/layer writes, fixed-buffer mutation, allocation outside decode, and deferred status printing.
- T4 requested each selected-ID tensor, copied eight preallocated I32 values after scheduler observation, and failed exact parity deterministically: code output token 0 changed `108→5676`; translation output token 2 changed `45518→676`; general remained equal. All captured IDs were in range, tensor contracts passed, event/value sentinels were intact, and no process persisted.
- Added one boundary-only control to separate scheduler segmentation from host readback. It returns true for `ffn_moe_topk-*` but reads zero tensor bytes and only counts completed observations. It reproduces the exact T4/historical changes with the same stable event counts. Therefore `ggml_backend_tensor_get`, ID copying, file I/O, formatting, allocation, and locking are not required for the perturbation.
- Root cause boundary: returning true for router tensors makes pinned `ggml-backend.cpp` split graph views and synchronize the backend at every selected-ID tensor. Empty/counter/metadata callbacks return false and preserve one graph view per backend split. The observer cannot be repaired inside this callback merely by adding a ring buffer or deferred flush.
- Stopped T5–T7 and all corpus collection at the first failing boundary. Gate 4 remains closed, `live_cache_enabled=false`, the 93.28% result remains withdrawn, and no replacement trace or cache work began. The required next investigation is a post-graph or backend-native deferred boundary that does not select the callback scheduler path.
- Checkpoint verification passed 88 tests in 0.39 seconds, the judge replay at 8 events / 64 demands / static 26 / LRU 19, a fresh probe build and all trace-mode CLI contracts, seven JSON files plus the append-only ledger, zero artifact/source hash mismatches, `git diff --check`, protected/source cleanliness, and zero cache variables or persistent processes. `isolation-summary.json` is 11,577 bytes, SHA-256 `02ffe8fe1eca25fd36d4b73a82ca4364fb0f441d628dee2b1b488cd250f2becc`; `verification.json` is 1,934 bytes, SHA-256 `a96cc5363700af2bcf59aa61aa6341eddd272ef09a0432f449c6697a9fc30bee`; the 38-record pre-commit ledger is 10,608 bytes, SHA-256 `083da05301678feccdaa3bdd5b363a14cffb328251eb27d69be01347a99d9ff8`.

### 22:20 PKT - Observer isolation merged; non-segmenting v2 boundary identified

- Locally merged `codex/trace-observer-repair` into `codex/live-cache-blocking-spike` as `62fee62bd54775a13257ddc4b52aa40e7309a701`. Isolation commit `aec5dd1ae0171f9814a4124827a2c73ba5089aff` remains reachable. No push or pull request was made.
- Post-merge verification passed all 88 tests and reproduced the judge fixture at 64 demands, static-hotset 26 hits, and LRU 19 hits. The first combined PowerShell wrapper failed before testing due to a parser error; a later checker used obsolete replay field names. Both command-layer mistakes were corrected and preserved in the repair ledger. The merged worktree was clean after verification.
- Created `codex/trace-observer-v2` from the verified merge. Read-only inspection used the exact clean pinned llama.cpp checkout at `a7312ae94f801fc9c6786dc56e38df57b964f697`; the checkout remained clean.
- Identified the smallest non-segmenting boundary at the existing `GGML_OP_MUL_MAT_ID` host-weight offload optimization in `ggml/src/ggml-backend.cpp`. That path already materializes and synchronizes the authoritative selected IDs before selecting expert byte ranges. A one-layer record immediately after the existing read would add no callback request, graph view, synchronization, or allocator lifetime change.
- Rejected deferred post-graph tensor retention because gallocr storage is reset/reused. Deferred a CUDA `ARGSORT` ring buffer because it requires materially broader CUDA state, allocation, metadata, and drain plumbing than the already-existing host-ID boundary.
- Wrote `docs/evidence/live-cache/trace-observer-v2-feasibility.md` with verdict `CONDITIONAL-GO-FOR-ONE-LAYER-OBSERVER-PROTOTYPE`. No implementation began. The recommendation is limited to one future llama.cpp file, approximately 90-140 lines, strict one-token microbatches, an observer-only disabled-by-default flag, and immediate stop if the target layer does not traverse the existing boundary.
- Gate 4 remains closed, `live_cache_enabled=false`, corpus collection remains stopped, all callback-derived traces remain quarantined, and the historical `93.28%` claim remains withdrawn.

### 23:00 PKT - One-layer Observer v2 stops because boundary is not traversed

- Created isolated llama.cpp worktree `C:\models\expertflow\worktrees\llama-trace-observer-v2` on `codex/trace-observer-v2-prototype` at exact pin `a7312ae94f801fc9c6786dc56e38df57b964f697`. The pristine pinned checkout remained clean. Committed the execution plan as `f82e9c4` before native implementation.
- TDD RED confirmed the observer contract was absent. Added a disabled-by-default, fixed-capacity prototype in only `ggml/src/ggml-backend.cpp`; the hot capture function contains no file I/O, formatting, allocation, tensor retrieval, or synchronization. The one-file diff is 185 insertions, above the earlier 90-140 estimate but without any second file, graph, CUDA, allocator, or cache change.
- Four configure wrappers failed before source compilation: inherited MSYS compilers, missing `cl` in the first developer-shell wrapper, an MSYS resource compiler, and a backslash-escaped Windows SDK path. Clearing `CC/CXX`, pinning MSVC 19.39/CUDA host compiler, pinning Windows SDK `rc`/`mt`, and using forward-slash CMake paths produced a successful 435-target `llama-cli` build. The existing external router probe rebuilt successfully against it.
- V0 completed nine runs across general/code/translation and matched prior clean-runtime tokens exactly for every repetition. V1 `noop` completed nine runs with exact V0 parity, zero records, intact canaries, and no overflow or contract error.
- V2 metadata stopped on its first general run: tokens still matched V0, but layer 24 produced zero records instead of the expected 53. The valid 40-byte output header proves the observer was configured and emitted at teardown; the target boundary was simply not encountered.
- Applied the mandatory stop rule. V3 was not started. No other layer was tried; no callback, tensor request, synchronization, graph split, CUDA hook, allocator/backend rewrite, cache code, or runtime expansion was attempted. The llama prototype remains uncommitted and isolated for audit.
- Gate 4 remains closed, `live_cache_enabled=false`, all earlier traces remain quarantined, the `93.28%` claim remains withdrawn, and corpus collection remains stopped.
### 23:27 PKT - Canonical graph-segmenting observer accepted

- Preserved the failed one-file Observer v2 llama.cpp prototype as stash `bb8ee4522f28dafd1b819747d58f7198d0a3a038`; protected Observatory `d846bdf`, Observer v2 evidence `8c3cef0`, and clean pinned llama.cpp `a7312ae` were not modified.
- Created isolated branch/worktree `codex/canonical-observer-runtime`, restored the existing Observer v1 callback path, and added only post-generation text serialization needed for objective validation. A rejected batched-prefill attempt was reverted after it produced malformed output; canonical batch and microbatch remain one.
- Final canonical binary SHA-256 is `7ea12e0c44258bb2d75f99b7e180e7ef7cb0c7d3e285ca231fe26ad8c8c4932c`; Gemma Q4 remains `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`.
- Seven-task smoke: Mode N 6/7, Mode O 7/7, retained successful outcomes 6/7. Final Mode O determinism repeated exactly for tokens and 1,980 routing records excluding timestamps. All processes cleaned up and settled GPU memory returned to baseline.
- Accepted `expertflow-canonical-observer-v1`; old `trace_v1_perturbing` evidence remains quarantined and new collection is labeled `trace_v2_canonical_segmented`. Cache remains disabled and no speed claim is made.

### 23:35 PKT - Canonical multi-domain pilot confirms bounded locality

- Froze 14 public synthetic conversations before collection: seven train and seven validation/test, covering general chat, code, math/reasoning, translation, multilingual, structured output, and topic shift. All 14 observer-enabled/cache-disabled shards passed and produced 50,310 routing events.
- Preserved raw callback traces and generated canonical copies by changing only placeholder request/conversation identifiers. Every shard records tokens, duration, memory snapshots, hashes, domain, split, and process result.
- Held-out decode adjacent reuse is 39.44%. At 32 slots/layer, training-only static is 49.34%, reset LRU 72.30%, and causal session frequency 71.83%. At 96 slots/layer, training-only static is 92.80% versus 78.49% LRU/session on this small pilot.
- The 92.80% number is new small-pilot evidence, not restoration of the withdrawn 93.28% result and not a generalization claim. Policy outputs remain estimated over measured routing; cache remains disabled.

### 23:55 PKT - Canonical one-layer blocking-cache stage opened

- Accepted the user-authorized bounded C0-C4 stage from canonical Observer v1 commits `42a6b21` and `56d2ab4`. Protected Observatory `d846bdf` and pinned llama.cpp `a7312ae9` were confirmed clean before modification.
- Canonical preflight passed 89 tests in 0.58 seconds. The preserved probe rehashed to `7ea12e0c44258bb2d75f99b7e180e7ef7cb0c7d3e285ca231fe26ad8c8c4932c`; the 14,439,361,440-byte model rehashed to `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`.
- Created isolated ExpertFlow worktree `C:\models\expertflow\worktrees\one-layer-blocking-cache` on branch `codex/one-layer-blocking-cache` from `56d2ab4`. Its first immediate test invocation reported 87 tests while environment setup completed; a subsequent collection in both same-commit worktrees found the identical 89 tests. No source discrepancy or hidden test file exists.
- Mapped middle MoE layer 24. Canonical `-ngl 10` offloads nine repeating layers plus output; layer 24 expert tensors are normally CUDA-resident. Existing llama.cpp tensor-buffer overrides can keep only `blk.24.ffn_gate_up_exps.weight`, `blk.24.ffn_down_exps.weight`, and `blk.24.ffn_down_exps.scale` on host while retaining CUDA execution, avoiding a model-loader rewrite.
- Confirmed the unchanged Gemma graph binds one logical top-8 ID tensor to fused gate/up and down `MUL_MAT_ID` consumers and the F32 scale lookup. Because the CUDA kernels index one rank-3 expert tensor, one or two total physical slots cannot execute an eight-expert event without a prohibited mixed-storage kernel or graph rewrite. The minimum exact layout is eight coordinated slots: seven stable active experts and one replaceable proof slot.
- Recorded the revised approved design and inline execution plan. No llama.cpp or cache runtime source has been modified yet; `live_cache_enabled=false` remains the only executed state.
- A canonical-pilot scan found the longest layer-24 seven-fixed/one-rotating window is only four decode events: `train-translation-01` indices 84-87, fixed `2,3,5,32,38,58,91`, rotating `33,34,44`. This is sufficient for controlled C2/C3 replacement but not arbitrary C4.
- Applied the stop rule before creating a llama.cpp worktree or writing tests/source. C4 requires eight replaceable slots because arbitrary true-router top-8 sets can change multiple experts and the unchanged CUDA operation cannot mix the original tensor with an external one-slot tensor. The constrained alternative stays one layer, exact, blocking, and predictor-free, but needs explicit user approval because it widens the requested one/two-slot arena.

### 00:10 PKT - Eight-slot minimum approved for C4

- The user approved eight coordinated replaceable slots as the minimum exact Gemma top-8 architecture, not a scope expansion. Scope remains exactly layer 24, blocking true-router loads, no prediction, no asynchronous stream, no MTP/ML, no second layer, and disabled by default.
- Updated the design, plan, mapping verdict, and machine-readable configuration before native source work. C4 must load and verify all eight selected experts, preserve selected order and weights, and record the complete logical-to-physical mapping.

### 01:05 PKT - Exact layer-24 eight-slot C4 passes

- Used test-first planner/config contracts with active assertions. Discovered that the first Release C++ harness used `assert` while `NDEBUG` was defined, so calls were compiled away; preserved the false-green diagnosis, added `#undef NDEBUG` before `<cassert>`, and reran genuine RED/GREEN sequences. The planner now covers eight-slot fill, retention, replacement generations, selected-order mapping, invalid/duplicate IDs, forced eviction, exact tensor layouts, copy bounds, and runtime configuration.
- C0 and C1 passed the seven canonical prompts with exact prompt/generated tokens and exact ordered routing after excluding only observation time. C1 initialized eight-slot infrastructure without redirecting execution.
- The first C2 attempt exited normally but did not exercise CUDA slots: mmap converted overridden tensors from `CUDA_Host` to CPU and single-token `MUL_MAT_ID` stayed on CPU. A narrow blocking-only assignment to an already-supported CUDA operation reached the intended path. The next run stopped explicitly because the scale view was not registered and revealed that temporary graph storage reused component allocations.
- Added one scheduler-owned persistent CUDA arena without changing CUDA kernels, model graph construction, or the general allocator. It redirects only three scheduler-generated layer-24 input copies. Exact measured arena allocation is 26,763,904 bytes (25.523 MiB), 4,480 bytes below the conservative aligned projection. Q4 bytes are copied directly without repacking; allocation occurs once and is freed at teardown.
- Added fixed-capacity deferred event records and explicit overflow failure. Each event records selected experts, physical slots, hits/misses, replacements, packed bytes, host-wall blocking duration, generations, and final residency. File serialization occurs at scheduler teardown, outside expert execution.
- C2 exact smoke passed 1,230/1,230 router events and identical prompt/generated tokens. C3 forced 41 consecutive all-eight replacements; all slot generations reached 41 with exact output/router parity and clean exit.
- C4 ran all seven deterministic prompts three times. All 71,640 router events, prompt tokens, and generated tokens match the canonical runtime; all 2,388 cache mappings are deterministic excluding duration. The runs contain 6,237 hits, 12,867 misses, and 43,045,416,204 packed bytes transferred. Miss-event host-wall duration is p50 3,303 us / p95 7,783 us; eight-miss p50 is 5,089 us / p95 10,738 us. These are blocking host measurements, not CUDA-event or speedup claims.
- System-wide GPU peaks were contaminated by unrelated desktop processes and are retained only as diagnostic evidence. Settled GPU-use delta across 21 focused processes has median -2 MiB; no probe remained. Final feature-disabled seven-prompt restoration is exact.
- Verification passed 89 ExpertFlow tests, the assertion-active probe configuration test, the assertion-active llama planner test, judge replay (SHA-256 `1b1e08dde19b61675deeebad8b6517e06d17fefff5f6809cb615bc4366aaf78`), JSON/JSONL parsing through the comparison scripts, and `git diff --check`. `live_cache_enabled=false` remains the release default. No multilayer, prediction, async stream, MTP, ML, or runtime-speed claim was added.

### 01:46 PKT - C5 architecture specification blocker

- Inspected isolated branches `codex/c5-reactive-cache` at ExpertFlow `9e81124` and llama.cpp `e41f54b0` before modifying runtime code. The worktrees started clean.
- The approved three-track request names a C5 exact live-cache runtime and the canonical pilot reports a simulated 32-slot reset-LRU result, but no checked-in C5/32-slot runtime specification or validation ladder exists. The committed C4 result explicitly leaves the next stage closed.
- C4 hard-codes eight slots in planner state, reduced tensor shapes, copy bounds, and event records. A 32-slot reactive cache could either expose one direct 32-slice packed CUDA operand to unchanged `MUL_MAT_ID`, or use 32 resident slices plus an eight-slice execution staging view. Those alternatives have different allocations, transfers, ID mappings, logs, and correctness risks; selecting one without approval would change architecture.
- Stopped before tests or source changes and reported the blocker. Requested an explicit C5 specification covering the direct/staging choice, deterministic eviction/tie-break semantics, exact allocation and event contracts, and C5 parity/memory progression. No llama.cpp source was modified and the default remains disabled.

### 01:55 PKT - C5 direct 32-slice architecture approved

- Approval resolved the C5 blocker: exactly layer 24 uses one persistent 32-slice packed CUDA operand consumed directly by unchanged `MUL_MAT_ID`; no eight-slice execution staging copy is permitted.
- Frozen deterministic policy: conversation/runtime reset, lowest-ID free-slot allocation, protected current demands, missing experts processed in authoritative order, and least-recently-used eviction with ascending slot-ID tie-break. Transfers remain blocking and true-router-directed; default remains disabled.
- Verified both assigned C5 directories are isolated linked worktrees at ExpertFlow `9e81124` and llama.cpp `e41f54b0`. Baseline ExpertFlow verification passed 89 tests in 0.46 seconds before implementation.
- Added the concise C5 design and TDD execution plan. The conservative 32-slot aligned projection is 107,073,536 bytes (102.113 MiB); it is not a measured allocation.

### 02:36 PKT - Exact layer-24 32-slot reactive C5 passes

- Used assertion-active RED/GREEN cycles to separate eight authoritative demands from 32 physical slots, then added lowest-free allocation, demanded-slot protection, deterministic LRU with ascending-slot tie-break, recency/reset, slot-31 bounds, and stale expert/generation rejection. The manual MSVC Release harness and integrated CMake target both pass.
- The first CMake configure failed before compilation because a Windows SDK short path was serialized with backslashes and CMake rejected `\P`; removed only the verified generated build directory and reconfigured under the VS 2022 developer environment. The first full target wrapper timed out while Ninja continued; no duplicate build was launched, and the completed target was reused. Manual compiler byproducts accidentally created in the llama worktree root were identified and removed before source review.
- The first completed C5-0 suite was rejected: the new build defaulted to `GGML_NATIVE=ON`/AVX-512 instead of the verified C4 `GGML_NATIVE=OFF` AVX2/BMI2/SSE4.2 configuration. With the cache disabled, three generated outputs diverged and near-tied router order changed. Preserved that evidence, corrected only the recorded build flags, and reran into a distinct root. Corrected C5-0 and C5-1 each match all seven prompt/generated token artifacts and 23,880 ordered router events exactly.
- C5-2 executed the unchanged layer-24 CUDA operation directly from the 32-slice packed arena. Exact measured allocation is 107,053,696 bytes (102.094 MiB). The 66-event known-set run matches 1,980 router events and tokens, uses physical slots above seven, and records 355 hits / 173 misses / 578,756,276 transferred bytes.
- C5-3 forced all eight selected experts to reload for 66 consecutive events. All 528 loads advance generation exactly once, transfer 1,766,377,536 bytes, preserve tokens/routing, and exit cleanly.
- C5-4 ran seven prompts three times. All 71,640 ordered router events and prompt/generated tokens are exact; all mappings/generations/recency are deterministic excluding duration. The 2,388 cache events contain 13,353 hits and 5,751 misses over 19,104 demands, transferring 19,239,464,412 packed bytes. Independent replay validated all 2,388 LRU decisions and final mappings with zero failures.
- Miss-bearing host-wall blocking time is p50 1,630 us / p95 7,655 us across 2,007 events, range 444–11,708 us. This is diagnostic blocking time, not CUDA-event or speedup evidence. Settled system GPU deltas across 21 processes range -20 to +5 MiB with median 0 MiB; the process sampler supplied no usable process-specific GPU peak. No allocation-growth pattern or residual probe appeared.
- Final feature-off seven-prompt restoration is exact. Fresh verification passed 89 ExpertFlow tests, both assertion-active native tests, judge replay at 8 events / 64 demands / static 26 / LRU 19, JSON/JSONL parsing, event accounting, and `git diff --check`. Default remains disabled; no prediction, async stream, MTP/ML, multi-layer change, staging copy, repacking, kernel change, graph rewrite, or speed claim was added.
### 14:55 PKT - Performance-first diagnostic benchmark passes multi-layer gate

- Preserved the C4, C5, predictor, expanded-corpus, and protected Observatory milestones. Created and committed the bounded diagnostic benchmark specification before production harness work; no llama.cpp source was modified.
- Added a benchmark-only probe extension under TDD. The parser initially failed with `ModuleNotFoundError`; the probe source contract initially failed because `--performance` was absent; and the runner contract initially failed because its script was absent. The focused suite then passed after minimal implementations.
- The first stock probe link failed because the historical narrow MinGW import libraries omitted `ggml_backend_cpu_buffer_type`, `llama_token_to_piece`, and the newly required `llama_perf_context`. Regenerated import libraries from the checked-in definitions and added only `llama_perf_context`; the unchanged clean, C4, and C5 DLL sets then linked successfully in copied external runtime directories.
- The first stock smoke exited `-1073741515` because CUDA 12.8 runtime DLLs were not on the inherited PATH. Prepended `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin`; the same binary then passed. No driver or runtime file was changed.
- Swept stock `-ngl 0,5,10,15,20,25,30,99` on the fixed general prompt. All were stable; decode TPS rose from 15.98 to 98.69. Froze full offload `-ngl 99` as the strongest no-OOM stock configuration by decode TPS.
- Ran one warmup plus three measured repetitions for general, code, and translation across stock best, matched stock `-ngl 10`, canonical observer/cache-off, C4 eight-slot, and C5 32-slot: 60/60 performance records and 60/60 external measurement records completed in 524.6 seconds.
- C5 exactly matched cache-off tokens and all ordered router selections. It reduced layer-24 misses 57.33% and aggregate blocking host-wall time 45.45% versus C4, while improving mean decode TPS 9.75% and end-to-end time 5.77% versus observer/cache-off. Blocking durations remain host-wall copy-plus-synchronization measurements, not CUDA-event latency or overlap.
- Strongest stock remained 98.53 mean decode TPS versus C5 at 27.71; C5 is 71.88% behind. The multi-layer verdict is therefore PASS for the staged two-layer/five-layer/all-layer 32-slot ramp, not a final throughput claim. Default live caching remains disabled.

### 15:24 PKT - Generic multi-layer C5 design and TDD ramp frozen

- Preserved diagnostic commit `c72f578` and made no llama.cpp source modification. Wrote the approved generic multi-layer design and detailed test-first plan before implementation.
- Confirmed from source and canonical traces that Gemma exposes 30 ordered MoE layers `0..29`, each with top-8 routing. Froze the ramp sets as `[0,24]`, `[0,7,14,21,29]`, then all `0..29`.
- Selected the five representative layers without consulting the expanded test split. Their train+validation decode LRU-32 rates span 57.80% to 80.00% (`0=57.80`, `7=68.22`, `14=80.00`, `21=74.65`, `29=60.26`), while layer 24 preserves the already verified direct-execution boundary.
- Chose fixed per-layer contexts, exact `blk.N` tensor parsing and metadata checks, and one consolidated aligned CUDA allocation with per-layer gate/up, down, and scale regions. Default remains disabled; legacy layer-24 configuration remains a regression path.
- The plan requires separate committed two-layer, five-layer, and all-layer evidence, with exact parity, per-layer event accounting, actual allocation/peak VRAM, blocking time, decode TPS, remaining reserve, KV/state headroom, cleanup, and feature-off restoration at every gate. Prediction, async, MTP/ML, and 64 slots remain out of scope.

### 16:43 PKT - Generic multi-layer runtime infrastructure passes C0/C1

- Implemented the frozen plan under strict RED/GREEN cycles in the isolated worktrees. Generic tensor identity/configuration committed as llama.cpp `a6574e8f` and probe `2fb8dd0`; fixed per-layer contexts and the consolidated arena planner committed as llama.cpp `b3930793`.
- Native tests prove exact layers `0..29`, legacy layer-24 compatibility, ascending explicit lists, independent layer state/counters, duplicate/cross-layer binding rejection, checked arithmetic, non-overlapping aligned regions, and raw packed payloads of 214,106,368 / 535,265,920 / 3,211,595,520 bytes for the frozen 2/5/30-layer ramps.
- The first MSVC configure was rejected because it ran outside the VS 2022 Developer environment and could not find the Windows resource compiler. Reconfigured under `VsDevCmd.bat`; MSVC 19.39 and CUDA 12.8 then rebuilt the scheduler and assertion-active native target successfully.
- A probe link attempt using stale C5 import libraries failed on `llama_perf_context`. Regenerating the import libraries from the current checked-in definitions and verified DLL export set resolved the link without changing llama APIs or source.
- The first C0 launch failed before model load with Windows status `0xC0000135` because CUDA 12.8 runtime DLLs were not on `PATH`; preserved it and reran with the supported CUDA `bin` directory prepended.
- The next C0/C1 matrix was rejected because the fresh build used `GGML_NATIVE=ON`; general/code outputs reproduced the previously documented optimizer-sensitive divergence. Reconfigured only to the frozen accepted `GGML_NATIVE=OFF`, AVX/AVX2/BMI2 enabled, AVX-512 disabled, and CUDA `120a-real`, then reran in a distinct root.
- Corrected C0 disabled and C1 passthrough `[0,24]` match the committed observer/cache-off prompt tokens, generated tokens, and 9,150 ordered router events exactly across general, code, and translation. C1 created no cache log or arena, exited cleanly, and settled GPU usage returned within 0–27 MiB of the pre-run system baseline.
- Fresh verification passed 97 ExpertFlow tests, four judge-replay tests, four multi-layer source-contract tests, the assertion-active native test under both GCC and MSVC, JSON/JSONL parsing, feature-off restoration, and `git diff --check`. Blocking execution and the two-layer ramp remain gated on the next committed milestone.

### 14:57 PKT - Multi-layer blocking ramp stops at CPU-resident layer boundary

- Began the approved `[0,24]` two-layer, 32-slot blocking smoke only after the diagnostic benchmark and generic C0/C1 evidence were committed. The runtime allocated the exact consolidated 214,107,392-byte arena, then aborted on first execution with a CUDA illegal-memory-access error. The full two-layer prompt/repetition matrix was not started.
- Bounded single-layer controls isolated the failure. Layers 21 and 24 pass at `-ngl 10` with an exact 107,053,696-byte arena. Layer 0 fails deterministically with the same arena at `-ngl 10` and `-ngl 30`, but passes at `-ngl 31`, the first setting that reports all 31/31 layers offloaded and makes transformer layer 0 normally GPU-resident.
- The passing layer-0 `-ngl 31` control wrote 42 complete cache records, generated one token after 42 prompt tokens, completed in 606.703 ms, and cleaned up. This is an execution/root-cause control only, not a parity, latency, or throughput claim.
- Tested one narrow destination-retention hypothesis under a source regression. It did not change the layer-0 failure and was removed without commit. No broad graph, scheduler, allocator, kernel, whole-layer-offload, prediction, asynchronous-copy, MTP/ML, or 64-slot change was attempted.
- Root cause: the exact packed-Q4 cache primitive is valid when its target layer is already CUDA-resident, but redirecting a normally CPU-resident layer's expert tensors forces CUDA expert execution inside a CPU-to-CUDA-to-CPU split whose dependent backend contract is not valid. Supporting this case requires graph placement/scheduling or whole-layer offload redesign and therefore triggers the approved stop condition.
- Stopped before the two-layer matrix, five-layer ramp, all-layer ramp, or any further llama.cpp source expansion. Preserved evidence under `C:\models\expertflow\runs\multilayer-cache-ramp` and documented hashes, controls, rejected hypotheses, and the decision in `docs/evidence/live-cache/multilayer-ramp-blocker.md`. Default live caching remains disabled.
- Final verification passed all 104 ExpertFlow tests, including seven replay/fixture tests, plus the four committed llama.cpp multi-layer source-contract tests. `git diff --check` passed and the llama.cpp worktree returned clean at `bb96d708`. One attempted replay-only command named the nonexistent historical path `tests/test_judge_replay.py`; it failed collection without running tests and was corrected to the three current replay test files.

### 15:26 PKT - CUDA-resident eligibility passes correctness and fails broad reactive performance

- Reclassified the layer-0 result as an explicit eligibility constraint. Removed the unconditional cache accelerator selection and added post-natural-assignment classification. Explicit ineligible requests now fail before arena allocation; auto mode alone may filter requested layers.
- The layer-0 `-ngl 10` control rejects with backend CPU and reason `moe_router_not_cuda`. Auto `-ngl 10` discovery validates layers `21..29` on CUDA0 and rejects layers `0..20` on CPU. No graph relocation, whole-layer offload, hybrid expert execution, or backend split work was added.
- The explicit `[21,24]` ramp passed nine measured cache-off/cache-on comparisons with exact prompt/generated tokens, all router selections, ordered cache events, independent mappings, generations, deterministic replay, stable allocation, and cleanup. Exact arena allocation is 214,107,392 bytes. Aggregate hit rate is 73.42%. Decode TPS is -0.42% versus matched cache-off; end-to-end time is +6.06% with 3,381.522 ms aggregate measured blocking across nine runs.
- Expanded only to the auto-discovered eligible set `21..29`. The exact 963,483,264-byte arena passed all exactness, mapping, generation, allocation, and cleanup checks across another nine measured comparisons. Aggregate hit rate is 71.11%, with 63,669,881,184 bytes moved and 16,534.323 ms host-wall blocking.
- The all-eligible performance gate failed: prompt TPS -41.59%, decode TPS -26.84%, and end-to-end time +54.54% versus current matched cache-off. The regression is primarily explained by about 1,837 ms measured blocking per run, not an unexplained metadata cost.
- Applied the approved stop condition before `-ngl 15/20`, 64 slots, or B2 integration. The one-layer C5 result remains the strongest ExpertFlow runtime result. Default live caching remains disabled. Full evidence is in `docs/evidence/live-cache/cuda-resident-eligible-cache.md`; raw roots are under `C:\models\expertflow\runs\cuda-eligible-cache`.
- Final verification passed all 105 ExpertFlow tests, all seven replay/fixture tests, five llama.cpp multi-layer source-contract tests, the assertion-active native cache test, and the assertion-active probe configuration test. `git diff --check` passed in both worktrees, no probe process remained, and every focused benchmark process exited and cleaned up. The implementation remains uncommitted pending the explicit commit authorization required by the llama.cpp repository instructions.

### 16:02 PKT - D0-D3 isolation stops before predictive integration

- Committed the authorized CUDA-resident llama.cpp milestone as `f9231b02` with the required assisted-by trailer and the ExpertFlow evidence milestone as `2e8bb2b`. Neither branch was merged or pushed.
- Proved that the requested D1 warm-static mode cannot remain exact under the simultaneous 32-slot, unchanged-kernel, unchanged-graph, fixed-prompt constraints. The three prompts touch 85-110 unique experts per eligible layer. Training-only top-32 sets cover only 43.85-52.03% and would require 2,901-4,690 exact fallbacks per run.
- Added the bounded D3 aggregate-only diagnostic under TDD. It retains internal mapping/generation validation and aggregate counters but skips all per-event record construction. The first isolation manifest command failed because this Windows PowerShell lacks `ConvertFrom-Json -AsHashtable`; the corrected manifest initially contained a UTF-8 BOM rejected by Python. Both failures occurred before model launch and were corrected without changing the experiment.
- D0 and D3 completed 24/24 processes. All nine measured D3 runs match D0 and committed D2 tokens and every ordered router selection. D3 exactly matches D2 demands, hits, misses, bytes, and 71.11% hit rate.
- D3 improves only +0.32% decode TPS and -0.65% end-to-end time versus D2, rejecting detailed event construction as the main regression source. Measured blocking accounts for 81.39% of the D3 end-to-end increase over D0, but diagnostic subtraction still leaves 416.664 ms/run or 9.96% overhead.
- Applied the user's decision rule: stop before prediction, async transfer, higher `-ngl`, or 64 slots because warm-static execution efficiency was not proven and meaningful residual overhead remains. Full evidence is `docs/evidence/live-cache/performance-isolation-d0-d3.md`. Default remains disabled and no speedup is claimed.
- Added a RED/GREEN benchmark-runner regression test and updated both diagnostic runners to scrub inherited `EXPERTFLOW_LIVE_CACHE_LOG_DETAIL` before applying a mode's explicit environment. This prevents ambient shell state from silently substituting D3 aggregate logging for D2 detailed logging. The isolation manifest SHA-256 is `51f69c79ba75c366801ebda4bb376775fd45f53273eb4b43b592da15d74e8de3`; the D0/D3 root contains 157 files totaling 24,031,403 bytes and the ledger contains 24 process records.
- The user subsequently accepted the D0-D3 evidence as authorization for bounded predictive asynchronous integration and removed the earlier requirement for a 32-slot no-miss warm-static control. Exactness, genuine CUDA-event transfer measurement, stable memory, residual-overhead tracking, and disabled-by-default release behavior remain mandatory. Added the machine-readable isolation summary at `docs/evidence/live-cache/performance-isolation-d0-d3.json`; the historical stop decision above remains preserved as the decision in force when the experiment completed.

### 01:55 PKT - Next-layer shadow predictor pilot

- Created isolated branch/worktree `codex/next-layer-shadow-predictor` from canonical pilot commit `56d2ab4`; baseline passed 89 tests. No llama.cpp, observer runtime, cache branch, GPU residency, transfer, or runtime decision was modified.
- Committed the concise approved specification as `bfde66c` and the TDD execution plan as `b47c561`. Preserved the immutable canonical seven-train/four-validation/three-test split and strict same-conversation/forward/token adjacent-layer joins. Missing, duplicate, ambiguous, non-adjacent, or cross-split data fails closed.
- Genuine RED/GREEN cycles added strict dataset construction, B0 copy, B1 training-only target-layer frequency, B2 training-only transition counts, fixed CPU B3 linear, one fixed shared B4 MLP, metrics, and paired conversation-reset 32-slot shadow LRU. The full suite reached 103 passing tests.
- Canonical traces expose no routing weights; binary source vectors are used. Fixed learned features contain 128 source values, 30 target-layer one-hot values, one phase bit, and 128 uniquely joined causal previous-token target-layer values. Seed is `20260716`; no search was run.
- Validation selected B2 before test access. B2 recall@8/12/16 is 41.66%/52.75%/60.99%, versus B1 23.35%/31.02%/37.59%; B3 is 38.63%/48.85%/56.40% and B4 36.29%/46.04%/53.30%. Width 8 was frozen from useful/wasted/eviction trade-offs. `selection-lock.json` recorded `test_opened=false` before the one test command.
- Sealed test B2 recall@8/12/16 is 42.98%/54.48%/62.84% over 13,427 samples; batch-one CPU latency is 45.9 us p50 / 57.4 us p95. All three conversations remain positive and decode recall@8 is 36.39%.
- A post-lock, no-reconfiguration model comparison confirms test recall@8/12/16 of B1 27.49%/36.03%/42.91%, B2 42.98%/54.48%/62.84%, B3 39.38%/49.76%/57.39%, and B4 37.40%/47.39%/54.72%. No model, feature, seed, or width changed after test access.
- Test shadow width 8 improves modeled ready demands by 3,762 over reactive LRU and reduces uncovered misses from 28,974 to 25,212, but projects 15.92 GB useful versus 34.32 GB wasted bytes, 13,613 speculative evictions, and 1,338 regret events. These results assume predictions are ready and are labeled simulated shadow evidence only.
- The statistical B2 artifact is 870,078 bytes. No runtime integration, transfer overlap, speedup, production generalization, multi-layer lookahead, next-token prediction, MTP, RL, or online adaptation was added.

### 14:00 PKT - Expanded predictor finalized on frozen 60/12/12 corpus

- Preserved the frozen 84-conversation canonical corpus and exact 60 train / 12 validation / 12 test split, with six-domain 10/2/2 balance. Dataset TDD added fail-closed domain counts, unique prompt hashes, and split-selective trace materialization. Focused RED first failed on the absent arguments; GREEN loaded 138,736 train, 27,405 validation, and 28,043 test adjacent-layer samples. Fit validated all identities but opened only train and validation traces.
- B2 TDD RED failed on absent `weighting`, `phase_mode`, and `has_support` APIs. GREEN bounded the search to raw-count/source-normalized scoring and pooled/phase-separated tables. Deterministic fallback remains available for ranking but is excluded from observed-support admission. Per-domain metrics and admission-aware 32-slot shadow accounting were added through separate RED/GREEN cycles.
- Pipeline TDD added the fixed expanded selection rule, exact split contract, selection-payload hash, selected-artifact hash validation, and refusal of a second expanded test command. Before execution, the complete suite passed 112 tests.
- Validation-only fit completed in 295.7 seconds. It evaluated unchanged B0/B1/B3/B4 plus four bounded B2 configurations. Validation selected source-normalized, phase-separated B2, width 12, observed-support admission: recall@8/12/16 55.31%/67.51%/75.03%, CPU p95 78.1 us. B3/B4 did not satisfy the fixed recall override. The verified selection payload hash is `a44de68e4904f65f8cd5e8f4594cd3f35d72f3c070107941573d00da9d4c576b`, and `test_opened=false` was independently confirmed before test access.
- The frozen test was opened once. Over 28,043 samples, selected B2 recall@8/12/16 is 55.1060%/67.3443%/74.8404%, exact top-8 set match is 4.1650%, and batch-one CPU latency is 76.7 us p50 / 83.6 us p95. Recall@12 remains positive across all domains, from 62.57% translation/multilingual to 70.97% code; decode is 63.06% and prefill 70.02%.
- The simulated conversation-reset 32-slot shadow reduces uncovered misses from 60,092 to 36,375, a modeled 39.47% reduction before timing. It records 23,717 ready gain, 6,099 eviction regret, 27,919 useful versus 41,868 wasted insertions, and 93.40 GB useful versus 140.07 GB wasted projected bytes. The model assumes predictions are ready; this is not CUDA latency, transfer overlap, live-cache throughput, or speedup evidence.
- An intentional second test command failed closed and left the test-metrics SHA-256 unchanged. The original lock serialized a null deduplication field because that policy lives in the separately hashed frozen-corpus definition; the authoritative 47,147-byte corpus file hash is `970681c0126cc5400524e5b4328f0ecaf87c72d346a7fd99896a44224720dbab`. Selection configuration was not changed after test access.
- No llama.cpp, runtime cache, transfer path, CUDA stream, MTP, or production integration was modified. Evidence is in `docs/evidence/live-cache/expanded-next-layer-predictor.md` and `C:\models\expertflow\runs\expanded-predictor-final`.
### 17:10 PKT - P1 live-shadow predictor passes the bounded runtime gate

- Preserved frozen P0 commit `6bc8eb68`, source-normalized phase-separated B2
  scoring, width 12, observed-support admission, and the disclosed single
  sealed-test procedural exception. The sealed test was not rerun or retuned.
- Exported the versioned 267,556-byte artifact twice with identical SHA-256
  `54f898ec25fc4b783953f8c98ffb122073e91741b31b94aef2e285d26063409b`;
  payload SHA-256 is
  `8837f31178e1b049f23e6ff2ad1654908055b1d8ae79c19994a33df0a6424f40`.
- Added a fixed native loader/scorer, one-call explicit phase generation, and
  8,192-entry deferred record storage. Missing, invalid, contradictory, stale,
  overflowed, corrupt, mismatched, or incomplete state fails explicitly.
- The first attempted scheduler-side observation stopped with zero records
  because CUDA-resident layers 23/24 do not traverse the selective host-weight
  copy path when cache is disabled. No cache or tensor change occurred. P1 was
  corrected to reuse the canonical full-trace callback's already-materialized
  ID arrays through a narrow runtime API. It adds no callback, tensor request,
  synchronization, cache action, or graph-scheduling decision.
- Ran the frozen general/code/translation suite with one warmup and three
  measured repetitions per mode, followed by seven smoke tasks. All 19 pairs
  passed exact prompt/generated-token parity, exact ordered router parity,
  exact offline/live candidate and float64 score equivalence, and deterministic
  cleanup. The 1,101 transitions had zero support failures.
- Live prefill: 875 transitions, Recall@8 53.77%, Recall@12 66.90%, p50 7.6 us,
  p95 8.5 us. Live decode: 226 transitions, Recall@8 50.77%, Recall@12 63.00%,
  p50 7.5 us, p95 8.375 us. These prompt-specific results do not replace the
  frozen expanded-test metrics.
- Focused shadow overhead versus matched observer baseline: prompt TPS -0.286%,
  decode TPS -0.187%, end-to-end +0.272%, and TTFT +0.293%. No speedup is
  claimed. Peak GPU use was unchanged at 6,680 MiB and settled VRAM delta was
  zero for every process.
- Verification passed 139 ExpertFlow tests, assertion-active native predictor
  and cache tests, and judge replay at 8 events / 64 demands / static 26 hits /
  LRU 19 hits. No ExpertFlow environment variable or model process remained.
- P1 binary hashes, commands, measurements, failures, and exact artifacts are
  recorded under `C:\models\expertflow\runs\p1-live-shadow`. P2 remains design
  only; no prefetch, copy stream, slot reservation, or cache decision was added.

### 18:35 PKT - P2 proves async transfer but stops on zero ready-useful prefetches

- Created isolated P2 ExpertFlow and llama.cpp worktrees from accepted P1
  commits `b50ea6a` and `6e7bdffe`. Verified that passing C5 commits
  `0eb05daf` and `641f5313` are already ancestors, so no merge was required.
- Locked the bounded P2 design around a common-scheduler policy layer and a
  CUDA-backend opaque transfer service resolved through registry function
  pointers. The service uses one dedicated non-blocking stream, fixed
  descriptors, fixed pinned staging, and CUDA timing/completion events.
- The fresh baseline initially hit Python 3.12 1-ULP score-validation drift;
  accepted P1 used Python 3.11. Raw live output was identical and the pinned
  Python 3.11 rerun passed. A separate benchmark attempt omitted CUDA 12.8 from
  `PATH` and failed before model load with `0xC0000135`; the corrected run
  passed. Both attempts are preserved.
- Pre-change gates passed 139 ExpertFlow tests, native cache/predictor tests,
  nine exact P1 focused pairs, and nine exact combined C5 cache-off/on pairs.
- P2.0 transferred and byte-verified one 3,345,412-byte expert into isolated
  slot 31 without exposing it to execution. Measured staging was 1.0254 ms,
  host enqueue 22.2 us, CUDA-event H2D 0.306016 ms, and the later reconciliation
  saw about 70.375 ms of intervening compute with no wait.
- P2.1 recorded 57 deterministic admission plans with exact tokens, routing,
  and offline/live predictions, zero slot mutations, and zero predicted copies.
- P2.2 issued at most one transfer per transition and passed exact
  reactive-versus-predictive tokens and routing across general, code, and
  translation with three measured repetitions per mode.
- Across nine P2.2 runs, 564 transfers moved 1,886,812,368 bytes. Only 228 were
  useful and all 228 were late; zero were ready and useful. Another 336 were
  unused, wasting 1,124,058,432 bytes. Exact fallback waited 147.299 ms total.
- P2.2 reduced mean reactive misses by 12.57% and reactive cache blocking by
  11.17%, but decode TPS fell 0.91% and end-to-end time increased 0.34% versus
  identical reactive C5. Staging plus transfer does not fit the observed live
  layer-23 to layer-24 window.
- Applied the stop rule before P2.3 concurrency, multi-layer prediction,
  higher `-ngl`, 64 slots, MTP, or retraining. No speedup is claimed.
  Experimental runtime source remains isolated and uncommitted; accepted P1
  and C5 remain the floor and live caching remains disabled by default.
- Final verification passed 139 ExpertFlow tests, assertion-active native
  cache and predictor tests, focused runner tests, judge replay at 8 events /
  64 demands / 26 static hits / 19 LRU hits, and `git diff --check`.
  Evidence is in `docs/evidence/live-cache/p2-layer24-async-prefetch-result.md`
  and raw artifacts are under
  `C:\models\expertflow\runs\p2-layer24-async-prefetch`.
