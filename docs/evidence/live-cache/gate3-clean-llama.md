# Gate 3 Clean Pinned llama.cpp

Status: **BLOCKED at full upstream CTest**

Recorded: 2026-07-15 17:10 PKT

Pinned source: `a7312ae94f801fc9c6786dc56e38df57b964f697`

Default/live state: `live_cache_enabled=false`

No model validation, router-probe rebuild, or ExpertFlow live-cache source change began after this failure.

## Passed boundaries

- A fresh partial Git clone is detached at the exact requested commit and remains clean.
- All 3,167 tracked files byte-match the existing pinned source archive; zero files are missing or different.
- CMake selected VS 2022 MSVC 19.39.33523.0, CUDA 12.8.93, and only `120a-real` for the device architecture.
- The configuration is Release with `GGML_CUDA=ON`, `GGML_NATIVE=OFF`, tests/examples enabled, and curl disabled.
- The completely unmodified source built all 634 targets in 300.4 seconds. `ggml-cuda.dll`, `llama-cli.exe`, `llama-server.exe`, and the test binaries were produced.
- The exact source remained clean after configure, build, and tests.

## Full CTest result

The first full run failed two of 43 tests: `test-jinja-py` and `test-opt`. The Jinja failure was a missing host test dependency, not a native failure: upstream `scripts/jinja/requirements.txt` lists unpinned `jinja2`, while no PATH-visible Python had it.

A dedicated evidence-only Python 3.11 environment was created under the external Gate 3 directory and installed Jinja2 3.1.6 plus MarkupSafe 3.0.3. The focused Jinja test then passed in 28.34 seconds. No project lockfile or llama.cpp source was changed.

The final full clean rerun used that environment:

| Result | Value |
| --- | ---: |
| Tests | 43 |
| Passed | 42 |
| Failed | 1 (`test-opt`) |
| Pass rate | 98% |
| Duration | 143.93 s |
| Durable transcript SHA-256 | `7abab529beb0d97c5f024907af87223be456ac4845aa1ced15b86f4378b1ce18` |

`test-backend-ops` and the other 41 tests passed. The sole failure is not waived.

## `test-opt` diagnosis

The failure is narrow and reproducible:

- AdamW fails only `test_idata_split(high_level=yes, subtest=results_forward)` for epochs 1-4.
- The identical four assertions fail on both CUDA0 and CPU. AdamW weights and backward results pass.
- All SGD cases pass: 46/46 on CUDA0 and 46/46 on CPU.
- Two focused reruns of the unmodified Release binary reproduced the same failure with SHA-256 values `93d5fb60511a93e76246b687ed89c8c7876922a221a6d4d29a9be560610d071c` and `4c65446fee4fa70646a1fb1b5d630c86f3a7d2e096a95e1f4f955b5544773c05`.
- A separate exact, unmodified CPU Debug build with the same MSVC 19.39 compiler passes `test-opt`: 2/2 backend/optimizer combinations.

A disposable diagnostic worktree—not the gated source—added logging around the result check. The failing Release form showed `result2` stayed empty: `ndata=0`, `loss=0`, and `loss_unc=NaN`. Adding loop logging alone changed the Release behavior and made the test pass. This is an optimization-sensitive heisenbug consistent with upstream undefined behavior or an MSVC Release miscompilation. It is not evidence that the CUDA kernel path is wrong, but it also is not safe evidence that the clean Gate 3 baseline passes.

The official checks for this commit report success for Windows CPU, Windows CUDA 12.4, and Linux `gpu-cuda`. The Windows jobs' logs contain build output but no CTest run. The Linux GPU job excludes `test-opt` from Debug and reports all Release `main|python` tests passing. This establishes platform sensitivity; it does not override the local required test.

## Gate decision

The written plan requires full CTest success and states: “Any source/toolchain failure stops cache work.” The clean pinned Release binary does not meet that condition. Therefore:

- Gate 3 is **BLOCKED**, not passed.
- Steps 4-5 (router-probe rebuild and deterministic model validation) have not started.
- Gate 4 live-cache implementation has not started.
- No runtime speedup, CUDA deadline, cache correctness, or parity claim is made.

A narrow user-authorized exception could permit only the remaining unmodified Gate 3 model/parity validation while treating this unrelated optimizer-training test as a documented upstream Windows Release exception. Cache implementation would still remain closed unless every inference/parity/VRAM check passes. Without that explicit exception, the stop condition remains binding.

## Evidence ledger

All generated artifacts are under `C:\models\expertflow\runs\live-cache-spike\gate3`.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `commands.jsonl` | 12,435 | `1eabeee28f325cbed8b6ef8c445f3356aae7b75e0dc3e19c31e730110fb1b27c` |
| `source-comparison.json` | 643 | `46bbe0631254a94f5e3290f6d55c0101c3982deaba6fc3c3bc5fb0c68682240b` |
| `binary-inventory.json` | 3,279 | `806135d40904ea6d4c0b9333dddccd5a806d1ee7b0e56c08c9f5425d7eaf87c4` |
| `upstream-checks-summary.json` | 1,769 | `41ef42e000cf515f0b237237e732909130946d65f668b3bba81b25366ebde261` |
| `upstream-gpu-cuda-ctest-snippet.txt` | 2,567 | `68251bf577d38aa55d81179156dd756487e549af6ef13b25df5a924c4b41202d` |
| `ctest-test-jinja-rerun.txt` | 634 | `bc6a1269f8d6952be71b4aa4fe3fbee05da97a13cb22f1c427489e1f3a5909da` |
| `ctest-test-opt-rerun-1.txt` | 142,844 | `93d5fb60511a93e76246b687ed89c8c7876922a221a6d4d29a9be560610d071c` |
| `ctest-test-opt-rerun-2.txt` | 142,490 | `4c65446fee4fa70646a1fb1b5d630c86f3a7d2e096a95e1f4f955b5544773c05` |
| `clean-cpu-debug-test-opt-run.txt` | 75,778 | `35f498f2efc7752d71e2fbefb80e7576bc73b361704507992ca1290cc439a3eb` |
| `ctest-full-after-jinja.txt` | 152,236 | `7abab529beb0d97c5f024907af87223be456ac4845aa1ced15b86f4378b1ce18` |

The clean build's `CMakeCache.txt` hashes to `13388b9bbb2e626c2abbd300eb704a1bd9584276ba1d34ae2058c4d7c64ccf32`; `build.ninja` hashes to `d9be69efe869a83cd35d3f4fe089c40a8e81cf0856e5dd78b05002095068ee76`. Exact executable and DLL identities are in `binary-inventory.json`.
