# Bounded Live-Cache Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and measure one exact, blocking, true-router-selected Q4 expert-slot replacement at one Gemma 4 layer without changing the protected Observatory or the default llama.cpp path.

**Architecture:** Protect and reproduce `d846bdf`, install the NVIDIA-supported Windows compiler/toolkit pair, then establish an unmodified CUDA llama.cpp baseline at the exact source pin. Only after that gate passes, add a small pure slot-planning unit beside ggml and integrate it at the existing `MUL_MAT_ID` host-weight transfer boundary. The proof uses layer 1 of training conversation `train-general-08`: seven fixed experts and one replaceable slot across a measured 50-token decode window.

**Tech Stack:** PowerShell 7/Windows PowerShell, Git, uv/Python 3.12, pytest 8.4, Visual Studio 2022 Build Tools, MSVC v143 14.39 (compiler 19.39), Windows 11 SDK, CUDA Toolkit 12.8.1, CMake 4.3.1, Ninja 1.13.2, C++17, pinned llama.cpp.

## Global Constraints

- Protected ref: `observatory-floor-2026-07-15` at `d846bdfcb1980dfc44d9f951e2824f58429f16d7`; never modify `codex/expertflow-stage0`.
- Work only on `codex/live-cache-blocking-spike` and C-drive artifact/source/build paths.
- llama.cpp pin: `a7312ae94f801fc9c6786dc56e38df57b964f697`.
- Q4 model SHA-256: `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`.
- Default and release state stays `live_cache_enabled=false`; an unset feature flag follows the original path.
- Gate 4 is layer 1 only, fixed experts `0,13,84,88,92,95,124`, one replaceable slot for experts `28,40,66`, training trace `train-general-08`, token indices 62 through 111.
- Runtime flags are exact: `EXPERTFLOW_LIVE_CACHE=1`, `EXPERTFLOW_LIVE_CACHE_LAYER=1`, `EXPERTFLOW_LIVE_CACHE_FIXED_IDS=0,13,84,88,92,95,124`, `EXPERTFLOW_LIVE_CACHE_REPLACEABLE_SLOTS=1`, and `EXPERTFLOW_LIVE_CACHE_LOG=<absolute-jsonl-path>`. `EXPERTFLOW_LIVE_CACHE_FORCE_EVICT=1` is test-only and rejected unless the cache flag is enabled.
- Object bytes are 3,345,412 packed and 3,346,048 aligned. No per-load dequantization, conversion, concatenation, or repacking.
- Blocking copies only. No predictor, asynchronous stream/prefetch, MTP, learned policy, broad cache manager, allocator rewrite, full static-96 allocation, KV-cache claim, deadline claim, or speedup claim.
- Stop on any protected reproduction failure, unsupported clean build, parity failure, repacking requirement, broad graph/kernel/allocator change, corruption, synchronization instability, allocation growth, feature-flag leakage, or July 18 cutoff risk.
- Every command, duration, result, failure, artifact/hash, source edit, decision, and commit goes into `PROJECT_LOG.md` and the gate evidence directory.

---

### Task 1: Complete the protected Gate 1 evidence

**Files:**
- Modify: `PROJECT_LOG.md`
- Create: `docs/evidence/live-cache/gate1-protected-floor.md`
- Generate externally: `C:\models\expertflow\runs\live-cache-spike\gate1\provenance.json`
- Generate externally: `C:\models\expertflow\runs\live-cache-spike\gate1\commands.jsonl`

**Interfaces:**
- Consumes: protected tag, existing replay fixture, physical evidence artifacts, existing native runtimes.
- Produces: immutable pre-install hashes and exact replay/report reproduction evidence.

- [ ] **Step 1: Capture pre-install hardware, tools, environment, and binary hashes**

Run from `C:\sem4\Expertflow`. Capture start/end ISO timestamps and exit codes. Hash the model; every `.exe` and `.dll` under the pinned CUDA and Vulkan runtime directories; `cmake.exe`, `ninja.exe`, `uv.exe`, `git.exe`, `nvidia-smi.exe`; the active NVIDIA user-mode CUDA/NVML DLLs; and the VS 2026 `cl.exe` located by `vswhere`. Record only build/runtime environment variables (`PATH`, `CUDA_PATH*`, `VS*`, `VSCMD*`, `CC`, `CXX`, `CMAKE_*`, `GGML_*`, `EXPERTFLOW_*`, `HF_HOME`, `UV_CACHE_DIR`); never serialize tokens or secret-valued variables.

Expected: `nvcc` and a PATH-visible `cl` are absent; CMake is 4.3.1, Ninja is 1.13.2, GPU is RTX 5060 Ti, driver is 591.86, and the tag resolves to `d846bdf`.

- [ ] **Step 2: Reproduce the judge fixture from the protected tag**

```powershell
git archive --format=zip --output C:\models\expertflow\runs\live-cache-spike\gate1\protected-d846bdf.zip observatory-floor-2026-07-15
Expand-Archive C:\models\expertflow\runs\live-cache-spike\gate1\protected-d846bdf.zip C:\models\expertflow\runs\live-cache-spike\gate1\protected-d846bdf
Set-Location C:\models\expertflow\runs\live-cache-spike\gate1\protected-d846bdf
uv sync --frozen --extra dev
uv run pytest -q
uv run expertflow simulate examples\replay\trace.jsonl --capacity-per-layer 8 --output replay-simulation.json
```

Expected: 87 tests pass; the canonical LF trace hash is `245aac7ffa83f464f33f220c2c7cafbf931671884c48fe2f92d48795ef11df8e`; output totals are 8 events/64 demands, static 26/38 and LRU 19/45; output hash is `54f46ccbf719b37f5cca55cc87d1625b8e8abdfd88f34faadc13042709010162`.

- [ ] **Step 3: Regenerate the physical Observatory with its embedded command**

Extract the exact `expertflow replay ...` command from the existing self-contained HTML, prepend `uv run`, and execute it from the protected clean tree using the original output path. Compute SHA-256 afterward.

Expected: `C:\models\expertflow\runs\q4-probe\report-physical-feasibility.html` hashes to `f3dc647d9965d726771632421b8fa5dffddc165d3ebae49f6f10381bbb75a90c` and still says `live_cache_enabled=false`.

- [ ] **Step 4: Verify the held-out selection boundary**

Assert that all 31 fit conversations in the report/manifest have split `train`, all eight evaluated conversations have split `validation` or `test`, the sets are disjoint, and capacity 96 reconciles to 74,149 hits / 10,523 misses.

Expected: no conversation ID overlap and `fit_scope=held_out_conversation_split`.

- [ ] **Step 5: Write the Gate 1 summary and commit**

```powershell
uv run pytest -q
git diff --check
git add PROJECT_LOG.md docs/evidence/live-cache/gate1-protected-floor.md
git commit -m "docs: preserve live-cache gate 1 provenance"
```

Expected: 87 tests pass, worktree clean after commit, protected branch unchanged.

### Task 2: Install and verify the supported CUDA toolchain

**Files:**
- Modify: `PROJECT_LOG.md`
- Create: `docs/evidence/live-cache/gate2-toolchain.md`
- Generate externally: `C:\models\expertflow\runs\live-cache-spike\gate2\installer-hashes.json`
- Generate externally: `C:\models\expertflow\runs\live-cache-spike\gate2\verification.txt`

**Interfaces:**
- Consumes: NVIDIA CUDA 12.8.1 network installer and Microsoft VS 2022 Build Tools bootstrapper.
- Produces: VS 2022 developer shell with MSVC 19.39, CUDA 12.8 `nvcc`, a passing `deviceQuery`, and a repeated transfer curve.

- [ ] **Step 1: Download and hash official installers**

Download `https://aka.ms/vs/17/release/vs_BuildTools.exe` and `https://developer.download.nvidia.com/compute/cuda/12.8.1/network_installers/cuda_12.8.1_windows_network.exe` into `C:\models\expertflow\installers`. Record resolved URLs, byte sizes, SHA-256, Authenticode signer/status, and download duration before execution.

- [ ] **Step 2: Install VS 2022 Build Tools side by side**

```powershell
Start-Process C:\models\expertflow\installers\vs_BuildTools.exe -Wait -ArgumentList @(
  '--quiet','--wait','--norestart','--nocache',
  '--installPath','C:\BuildTools2022',
  '--add','Microsoft.VisualStudio.Workload.VCTools',
  '--includeRecommended',
  '--add','Microsoft.VisualStudio.Component.VC.14.39.17.9.x86.x64',
  '--add','Microsoft.VisualStudio.Component.Windows11SDK.26100'
)
```

Expected: VS 2026 remains installed and unchanged; `vswhere -all -products *` lists both installations.

- [ ] **Step 3: Install CUDA toolkit packages without the display driver**

```powershell
Start-Process C:\models\expertflow\installers\cuda_12.8.1_windows_network.exe -Wait -ArgumentList @(
  '-s','nvcc_12.8','cudart_12.8','cublas_12.8','cublas_dev_12.8',
  'nvjitlink_12.8','nvrtc_12.8','nvrtc_dev_12.8','thrust_12.8',
  'visual_studio_integration_12.8','-n'
)
```

Expected: no `Display.Driver` package is selected; driver remains 591.86; `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe` exists.

- [ ] **Step 4: Verify the exact compiler pair and CMake detection**

```powershell
cmd /d /s /c '"C:\BuildTools2022\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 -vcvars_ver=14.39 && cl && nvcc --version && cmake --version && ninja --version'
```

Expected: `cl` reports 19.39.x, `nvcc` reports release 12.8, and CMake/Ninja match the recorded baseline. Configure a minimal CMake `LANGUAGES CXX CUDA` project and require `CMAKE_CUDA_COMPILER_VERSION` 12.8.

- [ ] **Step 5: Build and pass NVIDIA deviceQuery**

Clone the official `NVIDIA/cuda-samples` source at its CUDA 12.8 tag/commit, configure only `Samples/1_Utilities/deviceQuery` for SM 120, build with the same developer shell, and run it.

Expected: one RTX 5060 Ti is found, compute capability is 12.0, and the final result is `PASS`.

- [ ] **Step 6: Re-run the standalone transfer measurement**

Run the existing `expertflow transfer-benchmark` configuration for 3,346,048 bytes three times with 200 single-copy samples and pool raw samples. Record GPU idle state and compare p50/p95 against 0.234016/0.234272 ms.

Expected: no silent threshold. A difference is accepted only with raw evidence and a written explanation; a gross regression stops before llama.cpp.

- [ ] **Step 7: Commit Gate 2 evidence**

Run 87 ExpertFlow tests and commit only summaries/configuration, not installers or build trees, as `docs: verify supported cuda toolchain`.

### Task 3: Build exact pinned llama.cpp without cache changes

**Files:**
- Modify: `PROJECT_LOG.md`
- Create: `configs/llama-a7312ae-cuda128.json`
- Create: `docs/evidence/live-cache/gate3-clean-llama.md`
- Generate externally: `C:\models\expertflow\dependencies\llama.cpp-a7312ae-git`
- Generate externally: `C:\models\expertflow\builds\llama-a7312ae-cuda128-clean`

**Interfaces:**
- Consumes: verified MSVC/CUDA toolchain and exact upstream commit.
- Produces: clean CUDA binaries, CMake cache, binary hashes, ctest result, and deterministic validation artifacts.

- [ ] **Step 1: Create a verifiable exact source checkout**

```powershell
git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git C:\models\expertflow\dependencies\llama.cpp-a7312ae-git
git -C C:\models\expertflow\dependencies\llama.cpp-a7312ae-git checkout --detach a7312ae94f801fc9c6786dc56e38df57b964f697
git -C C:\models\expertflow\dependencies\llama.cpp-a7312ae-git status --short
```

Expected: detached exact commit, clean status, source tree matches the pinned archive for tracked files.

- [ ] **Step 2: Configure with only SM 120a**

From the VS 2022 14.39 developer shell:

```powershell
cmake -S C:\models\expertflow\dependencies\llama.cpp-a7312ae-git `
  -B C:\models\expertflow\builds\llama-a7312ae-cuda128-clean -G Ninja `
  -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF `
  -DCMAKE_CUDA_ARCHITECTURES=120a-real -DLLAMA_BUILD_TESTS=ON `
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
```

Expected: CMake identifies MSVC 19.39 and CUDA 12.8, and configures without source edits.

- [ ] **Step 3: Build and run the full test set**

```powershell
cmake --build C:\models\expertflow\builds\llama-a7312ae-cuda128-clean --parallel
ctest --test-dir C:\models\expertflow\builds\llama-a7312ae-cuda128-clean --output-on-failure -C Release
```

Expected: build and ctest pass. Any source/toolchain failure stops cache work.

- [ ] **Step 4: Build the existing router probe against clean libraries**

Use `scripts\build_router_probe.ps1` with the clean source/build outputs, writing a distinct `gate3-clean-probe` runtime. This adds no change to the llama.cpp checkout.

Expected: probe reports commit `a7312ae...`, trace-off and trace-on token files are byte-parity, and router traces contain eight experts for every event.

- [ ] **Step 5: Run deterministic CPU and ten-layer CUDA validation**

Run the frozen baseline prompt and `train-general-08` with greedy sampling, 64 generated tokens, one-token decode, and 12 threads. For each CPU and `-ngl 10` case, run trace off/on and record tokens, router events, duration, peak VRAM from sampled `nvidia-smi`, and basic tokens/s.

Expected: prompt/generated parity for trace off/on; router-event parity between repeated instrumented runs; model hash exact; no claim that CPU and CUDA floating-point outputs must match unless measured tokens prove it.

- [ ] **Step 6: Commit the clean build configuration and evidence**

Run llama ctest, 87 ExpertFlow tests, clean replay, `git diff --check`, and commit `docs: verify clean pinned cuda runtime`. No live-cache patch exists at this commit.

### Task 4: Implement the pure slot planner with TDD

**Files:**
- Create in a new llama.cpp linked worktree: `ggml/src/ggml-expertflow-cache.h`
- Create in that worktree: `ggml/src/ggml-expertflow-cache.cpp`
- Create in that worktree: `tests/test-expertflow-cache.cpp`
- Modify in that worktree: `ggml/src/CMakeLists.txt`
- Modify in that worktree: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: logical top-8 IDs and exact fixed/dynamic configuration.
- Produces: `expertflow_cache_state::plan(const std::vector<int32_t> &)` returning physical IDs, misses, reuse, and deterministic eviction records without CUDA dependencies.

- [ ] **Step 1: Write failing disabled-default/config tests**

```cpp
TEST_CASE("expertflow cache is disabled without an explicit flag") {
    CHECK_FALSE(expertflow_cache_config::from_environment().enabled);
}

TEST_CASE("one replaceable slot preserves seven fixed logical experts") {
    expertflow_cache_state state({0, 13, 84, 88, 92, 95, 124}, 1);
    const auto first = state.plan({0, 13, 28, 84, 88, 92, 95, 124});
    const auto second = state.plan({0, 13, 40, 84, 88, 92, 95, 124});
    CHECK(first.misses == std::vector<int32_t>{28});
    CHECK(second.evictions == std::vector<int32_t>{28});
    CHECK(second.misses == std::vector<int32_t>{40});
    CHECK(second.physical_ids.size() == 8);
}
```

- [ ] **Step 2: Run RED**

Build only `test-expertflow-cache`.

Expected: compilation fails because the cache interfaces do not exist.

- [ ] **Step 3: Implement the minimal pure state**

Define `expertflow_cache_config::from_environment()`, `expertflow_cache_state::plan(const std::vector<int32_t> &)`, `expertflow_cache_plan`, and `expertflow_cache_movement`. Reject duplicate/out-of-range IDs, any demand not equal to top-8, more simultaneous dynamic IDs than replaceable slots, and any target other than the configured layer.

- [ ] **Step 4: Run GREEN and edge-case tests**

Add tests for reuse without copy, forced miss, deterministic replacement reason, oversubscription fail-closed, and exact 3,345,412/3,346,048-byte constants. Run the focused test and full llama ctest.

- [ ] **Step 5: Commit the pure planner in the llama worktree**

Commit `test: add exact expert slot planner` before scheduler integration.

### Task 5: Integrate one-layer blocking movement with TDD

**Files:**
- Modify in llama worktree: `ggml/src/ggml-backend.cpp`
- Modify in llama worktree: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Modify in llama worktree: `ggml/include/ggml-cuda.h`
- Modify in llama worktree: `tests/test-expertflow-cache.cpp`

**Interfaces:**
- Consumes: pure plan, existing `MUL_MAT_ID` selected-expert boundary, host Q4 tensor ranges.
- Produces: three coordinated preallocated component slot tensors, timed blocking H2D copies, physical IDs for expert ops, restored logical IDs, and one JSONL movement record per object load/reuse.

- [ ] **Step 1: Write RED tests for tensor eligibility and log schema**

Test exact target names `blk.1.ffn_gate_exps.weight`, `blk.1.ffn_up_exps.weight`, and `blk.1.ffn_down_exps.weight`; reject non-CUDA backends, other layers/types/shapes, adapters, scales/biases, and missing log path. Validate `schema_version`, `run_id`, `request_id`, `token_index`, `layer_id`, `expert_id`, `packed_bytes`, `aligned_bytes`, `copy_bytes`, `source`, `destination`, `queue_host_ns`, `start_host_ns`, `end_host_ns`, `cuda_anchor_elapsed_ms`, `cuda_start_elapsed_ms`, `cuda_end_elapsed_ms`, `blocking_ms`, `reused`, `evicted_expert_id`, `replacement_reason`, `allocation_bytes`, and `synchronizations`; a reuse record has zero copy bytes.

- [ ] **Step 2: Implement compact duplicate tensor creation**

At graph split time, replace only eligible target expert copies with an eight-expert layout and register the three source/destination components in one scheduler-local cache state. Feature-off continues to call `ggml_dup_tensor_layout` exactly as before.

- [ ] **Step 3: Implement blocking direct component loads**

At the first target `MUL_MAT_ID` split, read authoritative logical IDs, plan the one dynamic slot, and copy each missing expert's gate/up/down byte range directly to the corresponding destination slot offset. Do not allocate or repack during load. Synchronize before execution and record actual component bytes plus the packed/aligned object contract.

- [ ] **Step 4: Preserve routing semantics**

Supply remapped physical IDs only to the target expert matrix operations. Preserve logical IDs for weight calculation/callback evidence and restore them before leaving the layer. Fail closed if any unexpected logical-ID consumer exists.

- [ ] **Step 5: Add optional CUDA event timing without a core-to-CUDA dependency**

Expose a CUDA backend proc-address hook that creates an anchor/start/end event set around the blocking copy group and returns elapsed milliseconds relative to the run anchor. CUDA events do not expose wall-clock timestamps, so the record pairs these relative device values with absolute host monotonic nanoseconds. The scheduler resolves the hook only when the feature is enabled; other backends and feature-off never call it.

- [ ] **Step 6: Run focused/full tests and commit**

Run the new tests, full llama ctest, and feature-off clean baseline before committing `feat: add disabled one-layer blocking expert slot`.

### Task 6: Run the exact live proof and stability gates

**Files:**
- Modify: `PROJECT_LOG.md`
- Create: `docs/evidence/live-cache/gate5-blocking-proof.md`
- Create: `patches/llama.cpp/a7312ae94f801fc9c6786dc56e38df57b964f697/0001-expertflow-blocking-slot.patch`
- Generate externally: `C:\models\expertflow\runs\live-cache-spike\gate5\*`

**Interfaces:**
- Consumes: unchanged Gate 3 baseline and feature build.
- Produces: exact parity, replacement, allocation, transfer, and isolation decision.

- [ ] **Step 1: Prove feature-off parity first**

Run the feature binary with all ExpertFlow cache variables unset. Compare tokens, router traces, movement-log absence, VRAM, and normal tests against Gate 3.

Expected: exact tokens/router events and no cache allocation/log.

- [ ] **Step 2: Run the controlled 50-event replacement sequence**

Set `EXPERTFLOW_LIVE_CACHE=1`, `EXPERTFLOW_LIVE_CACHE_LAYER=1`, `EXPERTFLOW_LIVE_CACHE_FIXED_IDS=0,13,84,88,92,95,124`, `EXPERTFLOW_LIVE_CACHE_REPLACEABLE_SLOTS=1`, and an absolute `EXPERTFLOW_LIVE_CACHE_LOG`. Run `train-general-08` through token 111 with one-token decode.

Expected: dynamic IDs are only 28/40/66; every transition is logged; prompt/generated tokens and logical router IDs equal Gate 3.

- [ ] **Step 3: Force misses and repeated swaps**

Set `EXPERTFLOW_LIVE_CACHE_FORCE_EVICT=1` to reload the dynamic selected expert repeatedly. Require direct three-range copies, deterministic `forced_test` or `selected_replacement` reasons, and no stale result.

- [ ] **Step 4: Run allocation and corruption checks**

Run at least 20 same-process iterations and 10 fresh processes. Sample `nvidia-smi`, backend allocation totals, process working set, exit status, and log reconciliation. Run Compute Sanitizer if compatible with the built SM 120 binary.

Expected: allocation reaches a stable plateau; no CUDA/ASAN/Compute Sanitizer error, crash, token drift, or malformed record.

- [ ] **Step 5: Compare live transfer timing to the standalone measurement**

Report live p50/p95 CUDA-event copy-group latency and blocking host duration separately from standalone pageable/pinned values. Explain component count, pageable source, synchronization, and model contention. Do not call it a speedup or deadline result.

- [ ] **Step 6: Run all regression gates**

Run llama ctest, 87 ExpertFlow tests, clean judge replay, Observatory regeneration/hash, feature-off baseline, and `git diff --check`.

- [ ] **Step 7: Export and commit the proof separately**

Export the exact llama commit range with `git format-patch`, hash the patch and all external evidence, update the log, and commit `feat: prove exact blocking q4 expert replacement` only if every Gate 5 condition passes. Otherwise commit a stop report with no expansion.

### Task 7: Decide whether Gate 6 is allowed

**Files:**
- Modify: `PROJECT_LOG.md`
- Create: `docs/evidence/live-cache/gate6-expansion-decision.md`

**Interfaces:**
- Consumes: Gate 5 proof and same-runtime measurements.
- Produces: explicit go/no-go for a separately designed small multi-layer static cache.

- [ ] **Step 1: Evaluate every Gate 5 condition without averaging failures away**

Mark token parity, router parity, direct Q4 execution, no-repack, repeated replacement, stable allocation, transfer consistency, feature isolation, llama tests, ExpertFlow tests, replay, and report reproduction separately.

- [ ] **Step 2: Stop or write the next design**

If any condition fails, stop runtime work and keep the Observatory floor. If all pass, write a new design/spec for a few layers and progressively measured capacity. Do not implement Gate 6 in this plan.
