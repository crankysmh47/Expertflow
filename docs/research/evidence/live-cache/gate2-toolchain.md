# Gate 2: Supported CUDA toolchain

**Result:** PASS. Visual Studio 2022 Build Tools, MSVC 19.39, and CUDA Toolkit 12.8 are installed side by side. The NVIDIA display driver and Visual Studio 2026 remain unchanged. Gate 2 contains standalone CUDA evidence only; no llama.cpp cache source was changed.

## Installer identity and execution

The signed installer files are outside the repository at `C:\models\expertflow\installers`. Machine-readable identity evidence is `C:\models\expertflow\runs\live-cache-spike\gate2\installer-hashes.json`.

| Installer | Version / size | SHA-256 | Signature | Result |
| --- | --- | --- | --- | --- |
| VS 2022 Build Tools | 17.14.37502.11 / 4,458,200 bytes | `c9cc76c0d03cbcb523e18b559a978ce5df11a667ef78e4e4d264331f1227ddd7` | Valid Microsoft signature | Exit 0 in 891.907 s |
| CUDA 12.8.1 network installer | 1.0.14 / 14,404,064 bytes | `779bee8ff557255c1cf5f36e0230f081675b9bb41e44be38839920cd5209bdeb` | Valid NVIDIA signature | Exit 0 in 516.298 s |

The VS install target is `C:\BuildTools2022`. `vswhere -requires` independently resolves the following components to that installation:

- `Microsoft.VisualStudio.Workload.VCTools`
- `Microsoft.VisualStudio.Component.VC.14.39.17.9.x86.x64`
- `Microsoft.VisualStudio.Component.Windows11SDK.26100`

Build Tools contains toolsets 14.39.33519 and 14.44.35207 because the workload's recommended current components were included. Every ExpertFlow build command selects `-vcvars_ver=14.39`; the verified compiler is `C:\BuildTools2022\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64\cl.exe`, version 19.39.33523.0, SHA-256 `dc1ef4e36c7044ae9bd0ce24d27de45f8fe26dc1210897b8717e8ef0232360e8`.

The CUDA command selected only `nvcc_12.8`, `cudart_12.8`, `cublas_12.8`, `cublas_dev_12.8`, `nvjitlink_12.8`, `nvrtc_12.8`, `nvrtc_dev_12.8`, `thrust_12.8`, and `visual_studio_integration_12.8`, with no reboot. `Display.Driver` was not selected. NVIDIA's installer inventory contains those toolkit packages, and the CUDA 12.8 MSBuild integration files exist under Build Tools.

## Side-by-side and environment verification

| Check | Result |
| --- | --- |
| VS 2022 Build Tools | 17.14.37502.11, complete and launchable at `C:\BuildTools2022` |
| Existing VS 2026 | 18.3.11512.155, complete and launchable at its original path |
| VS 2026 `cl.exe` identity | Pre/post SHA-256 both `a040a369b63177427584253a1a670cebe4a10e770d7c1b5c9d3d568e30433c8e` |
| Windows SDK | component `Windows11SDK.26100` resolves; SDK 10.0.26100.0 present |
| `cl` in prepared developer shell | 19.39.33523.0 |
| `nvcc --version` | CUDA 12.8, V12.8.93 |
| CMake / Ninja | 4.3.1 / 1.13.2 |
| GPU / driver after install | RTX 5060 Ti, 16,311 MiB, capability 12.0 / driver 591.86 |
| Reboot pending | false |

The CUDA installer added machine variables `CUDA_PATH` and `CUDA_PATH_V12_8`, both set to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`; it did not add CUDA `bin` to machine `PATH`. Reproduction therefore prepends `%CUDA_PATH%\bin` inside the VS 2022 developer shell. This makes both `cl` and `nvcc` resolve without changing global PATH.

## CMake and deviceQuery

The strict external CMake probe targets SM120 and rejects any compiler outside MSVC 19.39 or CUDA 12.8.x. CMake identified:

- C++ compiler: MSVC 19.39.33523.0
- CUDA compiler: NVIDIA 12.8.93
- CUDA host compiler: MSVC 19.39.33523.0
- CUDA device count: one

Configure, build, and run completed in 7.438 seconds. The probe binary hashes to `ab27cb1dcd6b49251b68c6b0742eaffd40f6f0d4bdc3f583c2adc84e6dc5031e`.

The official `NVIDIA/cuda-samples` repository is pinned at clean detached tag `v12.8`, commit `db3eea23946bca2e90a75eca2b5b3e07158a9e11`. An external SM120-only CMake overlay builds its unmodified `deviceQuery.cpp`. Configure, build, and first run completed in 5.911 seconds. It detected one RTX 5060 Ti, runtime 12.8, compute capability 12.0, 16,311 MiB, WDDM, pageable-memory support, five copy engines, and `Result = PASS`. The binary hashes to `0c4797f7cf7b55078a7ed453d57c1f37fd1ac2a9730464220bb02074c670f8f6`.

`deviceQuery.exe` dynamically depends on `cudart64_12.dll` despite its inherited “static linking” banner. A direct launch outside the prepared CUDA PATH correctly failed with Windows status `0xC0000135`; `dumpbin /dependents` identified the missing runtime boundary. Re-running in the documented CUDA-prepared environment passed. This failure and diagnosis remain in the command ledger.

## Standalone transfer reproduction

The transfer rerun deliberately used the same pinned CUDA 12.4 runtime DLL as the earlier evidence, isolating transfer variance from the newly installed toolkit. Preflight reported 1,960 MiB desktop VRAM, 8% instantaneous GPU utilization, and no llama/router-probe/model process. Three independent processes each measured the 3,346,048-byte aligned expert slot with 30 batches, 50 copies per batch, 10 warmups, and 200 single-copy samples. The aggregate pools 600 raw single-copy samples.

| Pinned transfer metric | Prior | Gate 2 | Change |
| --- | ---: | ---: | ---: |
| p50 CUDA-event latency | 0.234016 ms | 0.233984 ms | -0.0137% |
| p95 CUDA-event latency | 0.234272 ms | 0.236864 ms | +1.1064% |
| sustained bandwidth | 13.35 GiB/s | 13.319 GiB/s | -0.23% |

The change is reasonable idle-desktop variance, not a regression. The result remains a standalone pinned-copy lower bound with no model compute or live cache.

## Failures preserved

- Two earlier UAC attempts timed out while the user was away; neither created an install process.
- `cl /Bv` without a source printed the correct version then returned D8003. The corrected check compiles and runs a trivial external probe.
- Strict PowerShell transcript capture promoted `cl`'s normal stderr banner to `NativeCommandError`; the durable batch helper captures native output and checks exit codes explicitly.
- A PowerShell `foreach`-to-pipeline summary one-liner produced a parser error; assigning the rows before formatting fixed only the report command.
- The first direct `deviceQuery` transcript launch omitted CUDA `bin` from PATH and exposed the dynamic `cudart64_12.dll` dependency described above.
- The first compiler probe omitted `/Fo` and left a known 1,156-byte object in the repository worktree. It was hashed, removed, and the helper now writes the object to the external evidence directory. The repository is clean.

## Artifact ledger

All generated evidence is under `C:\models\expertflow\runs\live-cache-spike\gate2`.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `installer-hashes.json` | 2,509 | `075bafb9a25081468f4951427e9d93cc3031f220ceb65f033360f0d314b51eef` |
| `commands.jsonl` | 11,183 | `9beb0cee23744312c760262879594e361a8400a6b59f678a87ad2bd4fa79fc73` |
| `verification.txt` | 12,248 | `cfb4b426c3f1554c438e6c8f33a16028b4ced613d88c48f057b220a1fbe7acc6` |
| `developer-shell-verification.txt` | 1,317 | `7b18466fcd6d72726c21cd97549cf097416a4a0e7625e9874b9d9495af6fb1c1` |
| `devicequery-verification.txt` | 2,678 | `74a7573f1eac50755a1c5b74426c02f9588fa405d7fb2ce3c6f7147070a42197` |
| `transfer/aggregate.json` | 71,606 | `477c8659a407e3a7a722834452fd243873aa985e42da60853e5b588286654e97` |
| `transfer/trial-1.json` | 26,695 | `bffaee0e851bf73548e763324dab98e309c84089808cb57530fe83802256821f` |
| `transfer/trial-2.json` | 26,593 | `61e97c63ae165be4c829c3820cd5143ca298fcb04ee46b43393ddf03b83657a3` |
| `transfer/trial-3.json` | 26,643 | `42d74193be5f73ce9819166a6f77f7eeb286f49aab8fced8c1059a2a0fdb40a5` |

The final repository checkpoint passed all 87 ExpertFlow tests in 0.60 seconds, `git diff --check`, a clean protected checkout at `d846bdf`, and driver 591.86. Gate 2 is passed. Gate 3 may begin with an entirely unmodified pinned llama.cpp checkout. `live_cache_enabled=false` remains mandatory.
