# Gate 2: Supported CUDA toolchain

**Current result:** BLOCKED-PENDING-LOCAL-UAC. Gate 2 has not passed, no CUDA/llama.cpp build has started, and the live-cache experiment remains disabled.

## Completed preflight

- C drive free before installation: 117.91 GiB.
- The Codex process is not elevated.
- Existing Visual Studio Community 2026 remains installed at `C:\Program Files\Microsoft Visual Studio\18\Community`.
- `C:\BuildTools2022` did not exist before either install attempt and still does not exist.
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8` did not exist before either install attempt and still does not exist.
- No VS/CUDA installer process appeared after either elevation request.

## Installer identities

The installer files are intentionally outside the repository at `C:\models\expertflow\installers`. Machine-readable identity evidence is at `C:\models\expertflow\runs\live-cache-spike\gate2\installer-hashes.json`.

| Installer | Version / size | SHA-256 | Signature |
| --- | --- | --- | --- |
| VS 2022 Build Tools | 17.14.37502.11 / 4,458,200 bytes | `c9cc76c0d03cbcb523e18b559a978ce5df11a667ef78e4e4d264331f1227ddd7` | Valid; Microsoft Corporation; thumbprint `AB172913A2960A224809EE8A0C371CD47A079B72` |
| CUDA 12.8.1 network installer | 1.0.14 / 14,404,064 bytes | `779bee8ff557255c1cf5f36e0230f081675b9bb41e44be38839920cd5209bdeb` | Valid; NVIDIA Corporation; thumbprint `15F760D82C79D22446CC7D4806540BF632B1E104` |

Requested and resolved download locations:

- `https://aka.ms/vs/17/release/vs_BuildTools.exe` resolved to Microsoft's content-addressed download whose path includes the matching SHA-256.
- `https://developer.download.nvidia.com/compute/cuda/12.8.1/network_installers/cuda_12.8.1_windows_network.exe` resolved without redirection.

## Elevation blocker

Two silent side-by-side VS 2022 installation attempts requested the exact approved workload, recommended components, VC 14.39 component, Windows 11 SDK 26100, and `C:\BuildTools2022` target. Each remained at UAC for about 122.4 seconds and then returned `The operation was canceled by the user`. No install process or target directory appeared.

At the user's request, the supported Windows Computer Use client enumerated the desktop to accept consent. Windows intentionally hosts UAC on the secure desktop, which was absent from both the targetable app and window lists. The tool cannot act on that security desktop. UAC was not weakened, no security setting was changed, and no alternate elevation bypass was attempted.

## Prepared official source and strict build fixtures

The official `NVIDIA/cuda-samples` repository is cloned outside the repo at `C:\models\expertflow\dependencies\cuda-samples-v12.8`:

- exact tag: `v12.8`
- detached clean commit: `db3eea23946bca2e90a75eca2b5b3e07158a9e11`
- commit subject: `Update CUDA Samples for CTK 12.8 release and migrate build system to CMake`

External fixtures under `C:\models\expertflow\runs\live-cache-spike\gate2` are ready but unbuilt:

- `cmake-probe` fails configuration unless MSVC is exactly the 19.39 line and CUDA is 12.8.x, then targets SM120.
- `devicequery-cmake` builds the official `deviceQuery.cpp` and Common headers at the pinned tag for SM120 only, without editing NVIDIA source.

| External artifact | SHA-256 |
| --- | --- |
| `installer-hashes.json` | `075bafb9a25081468f4951427e9d93cc3031f220ceb65f033360f0d314b51eef` |
| `commands.jsonl` partial-checkpoint ledger | `6d3d92781a1009f384f004ebcc99d16b7e7ea39f58e15081de1d765195580f63` |
| `cmake-probe/CMakeLists.txt` | `d99f65bdca0bc212045b3380fab1efcffc82c65e4137524e0d6a984178bbc8b4` |
| `cmake-probe/main.cu` | `cf1346321c048236572318d14231bb81ced87c0cb160cb531c26bb22acb3ea73` |
| `devicequery-cmake/CMakeLists.txt` | `d198a0b1467fc15d70ea23d3e75769f47b2a9657ce3b7fa0e2b8605cfce991d3` |

## Unmet pass conditions

Gate 2 remains closed until all of these are directly verified:

- VS 2022 Build Tools/v143 14.39 and Windows SDK installed side by side, with VS 2026 unchanged.
- CUDA Toolkit 12.8 installed without a display-driver package; driver remains 591.86.
- VS developer shell resolves `cl` 19.39 and `nvcc --version` reports 12.8.
- strict CMake C++/CUDA probe configures and runs.
- official `deviceQuery` builds and reports RTX 5060 Ti / compute capability 12.0 / PASS.
- the three-trial pinned transfer benchmark reproduces within documented variance.
- all 87 ExpertFlow tests still pass.

No clean llama.cpp build, live-runtime measurement, cache allocation, CUDA deadline claim, KV-cache claim, or speedup claim is made here.
