# Gate 3 Clean Pinned llama.cpp

Status: **FAIL-STOP; cross-runtime drift explained, clean trace parity fails representative prompts**

Recorded: 2026-07-15 19:10 PKT

Pinned source: `a7312ae94f801fc9c6786dc56e38df57b964f697`

Default/live state: `live_cache_enabled=false`

The user authorized a narrow exception for the single Windows/MSVC Release `test-opt` heisenbug. That exception allowed the exact clean, unmodified build to proceed through Gate 3 inference validation; it did not waive inference, tracing, router, memory, or runtime parity. The clean runtime then failed those non-waived parity requirements. No ExpertFlow live-cache source change began.

## Clean build boundary

- A fresh partial Git clone is detached at the exact requested commit and remains clean.
- All 3,167 tracked files byte-match the existing pinned source archive; zero files are missing or different.
- CMake selected VS 2022 MSVC 19.39.33523.0, CUDA 12.8.93, and only `120a-real` for the device architecture.
- The configuration is Release with `GGML_CUDA=ON`, `GGML_NATIVE=OFF`, tests/examples enabled, and curl disabled.
- The completely unmodified source built all 634 targets in 300.4 seconds. `ggml-cuda.dll`, `llama-cli.exe`, `llama-server.exe`, and the test binaries were produced.
- The exact source remained clean after configure, build, tests, router-probe compilation, and inference validation.

## Full CTest result and authorized exception

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

`test-backend-ops` and the other 41 tests passed. The sole failure is preserved as a user-authorized, narrowly scoped exception for continuing unmodified inference validation only.

The failure is narrow and reproducible:

- AdamW fails only `test_idata_split(high_level=yes, subtest=results_forward)` for epochs 1-4.
- The identical four assertions fail on both CUDA0 and CPU. AdamW weights and backward results pass.
- All SGD cases pass: 46/46 on CUDA0 and 46/46 on CPU.
- Two focused reruns of the unmodified Release binary reproduced the same failure with SHA-256 values `93d5fb60511a93e76246b687ed89c8c7876922a221a6d4d29a9be560610d071c` and `4c65446fee4fa70646a1fb1b5d630c86f3a7d2e096a95e1f4f955b5544773c05`.
- A separate exact, unmodified CPU Debug build with the same MSVC 19.39 compiler passes `test-opt`: 2/2 backend/optimizer combinations.

A disposable diagnostic worktree, not the gated source, added logging around the result check. The failing Release form showed `result2` stayed empty: `ndata=0`, `loss=0`, and `loss_unc=NaN`. Adding loop logging alone changed the Release behavior and made the test pass. This is an optimization-sensitive heisenbug consistent with upstream undefined behavior or an MSVC Release miscompilation. It is not evidence that the CUDA kernel path is wrong. The diagnostic-logging build was never used for validation.

The official checks for this commit report success for Windows CPU, Windows CUDA 12.4, and Linux `gpu-cuda`. The Windows jobs' logs contain build output but no CTest run. The Linux GPU job excludes `test-opt` from Debug and reports all Release `main|python` tests passing. This establishes platform sensitivity; it does not erase the local result.

## Authorized unmodified inference continuation

The exception was authorized with these boundaries:

- exact clean source at `a7312ae94f801fc9c6786dc56e38df57b964f697`;
- exact unmodified Release CUDA build, never the diagnostic-logging build;
- no waiver for model load, prompt/generated tokens, router selections, trace completeness, causal ordering, memory stability, performance, replay, or ExpertFlow tests;
- no cache implementation unless every remaining Gate 3 check passed.

The 90 clean build outputs were copied byte-for-byte into a separate evidence runtime so the original clean build directory remained untouched. All 90 files matched by byte size and SHA-256. The existing router probe was rebuilt against those clean libraries without changing llama.cpp source. The probe is 2,950,816 bytes and hashes to `99ee90b72052a9601669b73f055a19dca02be7ede685e9a717e034320ecb607c`.

The Q4 model loaded successfully. It is 14,439,361,440 bytes and hashes to `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`, matching the frozen model identity.

### Clean-runtime internal validation

A bounded eight-token GPU smoke passed before the full run: trace-off and trace-on tokens matched, all 1,350 expected routing events were present, every forward contained 30 ordered MoE layers and eight experts, and strict schema and causal ordering passed.

The full baseline-prompt, ten-layer GPU matrix used greedy sampling, 64 generated tokens, single-token decode, and 12 threads. Within the clean runtime:

- trace-off and both trace-on runs produced identical prompt and generated tokens;
- the two trace-on runs produced identical ordered router selections;
- both traces contained all 3,030 expected events across 101 forwards, 30 layers per forward, and eight experts per event;
- strict trace schema and causal ordering passed with no validation errors;
- settled GPU use returned to the desktop range after every process, and no persistent llama/probe process remained.

The previously verified reference runtime also reproduced exact trace-off/trace-on token parity, so its reference output was not an instrumentation artifact.

| Runtime/run | Duration (s) | GPU before (MiB) | GPU peak (MiB) | GPU settled (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Previously verified reference, trace off | 8.279 | 2,830 | 7,530 | 2,830 |
| Previously verified reference, trace on | 11.568 | 3,033 | 7,505 | 2,823 |
| Clean CUDA 12.8 build, trace off | 8.648 | 2,823 | 7,466 | 2,782 |
| Clean CUDA 12.8 build, trace on 1 | 9.354 | 2,784 | 7,451 | 2,650 |
| Clean CUDA 12.8 build, trace on 2 | 9.009 | 2,644 | 7,394 | 2,727 |

These are measured process-level wall durations and sampled whole-device VRAM values, not cache-speedup measurements. Clean trace-off duration was 4.4521% above the reference trace-off run; a single-run difference is descriptive only.

### Non-waived parity failure

The clean build does not reproduce the previously verified runtime:

- Prompt token IDs match.
- Generated token IDs first differ at generated index 35: the reference emits `171502`, while the clean build emits `219220`.
- The first ordered router difference occurs at event 23. That event contains the same set in a different order.
- The first actual selected-expert-set difference occurs at event 24: forward 0, token index 0, token ID 2 (BOS), layer 24. The clean set contains expert `117`; the reference set contains expert `113` instead. The earlier wording “prompt token 2” incorrectly treated token ID 2 as token index 2.
- Across the 2,190 events that precede generated-token divergence and are therefore directly comparable, 256 expert sets differ.
- Across all 3,030 events, 1,602 ordered selections and 1,074 selected-expert sets differ, although events after token divergence are not treated as independently causal evidence.

The bounded follow-up audit explains this cross-binary result as build-dependent accumulated floating-point drift at a near-tied rank-8/rank-9 boundary. It is diagnostic rather than evidence that the clean inference runtime is corrupt. The audit nevertheless found a separate non-waived failure: across three representative prompts, clean trace-off/on generated tokens match for general chat but fail for code and translation. See [the divergence audit](gate3-divergence-audit.md).

## Gate decision

Gate 3 is **FAIL-STOP**, not passed. Cross-runtime bit identity is no longer the blocking reason. The blocking reason is the clean runtime's representative trace-off/on token-parity failure: both paths are independently repeatable, but code and translation generate different token arrays when the synchronized router-observation callback path is active. The 87 ExpertFlow tests and replay fixture still pass, the protected Observatory remains byte-identical, and both the protected checkout and exact llama.cpp source remain clean.

- Gate 4 live-cache implementation has not started.
- No one-layer blocking-slot recommendation is issued because the prerequisite Gate 3 pass does not exist.
- `live_cache_enabled=false` remains the only release state.
- No runtime speedup, CUDA deadline, cache correctness, KV-cache, or live-cache parity claim is made.

## Evidence ledger

All generated artifacts are under `C:\models\expertflow\runs\live-cache-spike\gate3`.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `commands.jsonl` | 27,356 | `4e206c1b94280747e5423ff143ae81870c9efd9dc14f1ffe106e0ffe107bc6a2` |
| `source-comparison.json` | 643 | `46bbe0631254a94f5e3290f6d55c0101c3982deaba6fc3c3bc5fb0c68682240b` |
| `binary-inventory.json` | 3,279 | `806135d40904ea6d4c0b9333dddccd5a806d1ee7b0e56c08c9f5425d7eaf87c4` |
| `upstream-checks-summary.json` | 1,769 | `41ef42e000cf515f0b237237e732909130946d65f668b3bba81b25366ebde261` |
| `upstream-gpu-cuda-ctest-snippet.txt` | 2,567 | `68251bf577d38aa55d81179156dd756487e549af6ef13b25df5a924c4b41202d` |
| `ctest-test-jinja-rerun.txt` | 634 | `bc6a1269f8d6952be71b4aa4fe3fbee05da97a13cb22f1c427489e1f3a5909da` |
| `ctest-test-opt-rerun-1.txt` | 142,844 | `93d5fb60511a93e76246b687ed89c8c7876922a221a6d4d29a9be560610d071c` |
| `ctest-test-opt-rerun-2.txt` | 142,490 | `4c65446fee4fa70646a1fb1b5d630c86f3a7d2e096a95e1f4f955b5544773c05` |
| `clean-cpu-debug-test-opt-run.txt` | 75,778 | `35f498f2efc7752d71e2fbefb80e7576bc73b361704507992ca1290cc439a3eb` |
| `ctest-full-after-jinja.txt` | 152,236 | `7abab529beb0d97c5f024907af87223be456ac4845aa1ced15b86f4378b1ce18` |
| `probe-build.txt` | 1,426 | `f8c4ff14d7db5455ae76702316457eee54706a69c4d15487295be5a4bc18cab1` |
| `clean-runtime-inventory.json` | 37,144 | `888bc78e12dc8fd2708fcc782f468db5cc016aacbe69bb1b00bf4a7835304545` |
| `clean-runtime-smoke.txt` | 459 | `567dfa4c9f0ebf12bc9af5536a216924afdc41acd64ae18f3fc1c0defbcda59e` |
| `run_native_measured.py` | 5,931 | `c7fa03a14e76c47db7eb60eb1d448d4a2f2dfac8884e27a6a6c4f9d188509f31` |
| `validate_probe_pair.py` | 9,572 | `056e38abf40ee5815155626437212cb2e79b1a3b29094477f1f4ad06691d6d43` |
| `validation/smoke-gpu10-validation-v2.json` | 1,309 | `10cc7ccd82ac8cb76ea99a3a77ebb87882f30e792a3c12b8c45cc503b8a560b6` |
| `validation/baseline-gpu10/validation-v2.json` | 4,100 | `bbbf0f20196d3be0c621c0557f98bd64513c6b2742795199078c35e677e8d275` |
| `validation/baseline-gpu10/reference-off/trace-parity.json` | 472 | `739b713a6ce179c58f7a96d91feeb66c3fbec7f13d00a95b45a4a529ffa7f714` |
| `gate3-inference-stop-summary.json` | 16,415 | `e6d24d8166d216f5e97e9a8b38771c1f40cc8f5706647ce2704c1c89a39b5411` |
| `replay-simulation.json` | 900 | `4a00708e102e34bbc8feb979f9afd1e9f01815336828761cbbe27e4ab05034ae` |

The clean build's `CMakeCache.txt` hashes to `13388b9bbb2e626c2abbd300eb704a1bd9584276ba1d34ae2058c4d7c64ccf32`; `build.ninja` hashes to `d9be69efe869a83cd35d3f4fe089c40a8e81cf0856e5dd78b05002095068ee76`. Exact executable and DLL identities are in `binary-inventory.json`.
