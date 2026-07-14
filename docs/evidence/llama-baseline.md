# Pinned llama.cpp baseline

- **Release:** `b10002`
- **Source revision:** `a7312ae94f801fc9c6786dc56e38df57b964f697`
- **Target:** Windows x86_64, CUDA 12.4
- **External root:** `C:\models\expertflow\dependencies`
- **Status:** runtime, full source archive, and Q4 baseline verified

## Verified release artifacts

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `llama-b10002-bin-win-cuda-12.4-x64.zip` | 248,820,066 | `d8fa3634b6a6a2eb64b56d3f4d68b8e71ee6e1ccb980059476dd51787f1d2f3f` |
| `cudart-llama-bin-win-cuda-12.4-x64.zip` | 391,443,627 | `8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6` |

Both sizes and hashes match the official GitHub release metadata. The archives were extracted together into the external `llama-b10002\runtime` directory.

## Runtime checks

`llama-cli --version` reports:

```text
version: 10002 (a7312ae94)
built with Clang 20.1.8 for Windows x86_64
```

After adding the verified CUDA runtime DLLs, `llama-cli --list-devices` reports:

```text
CUDA0: NVIDIA GeForce RTX 5060 Ti (16310 MiB, 15158 MiB free)
```

The exact source commit was resolved from the reported short hash through GitHub's commit API. The Gemma 4 routing source map was rechecked at this release-matched revision.

The clean codeload archive for that commit is `37,514,614` bytes with SHA-256 `b07eaf97a236c7f9d9ec2e919504a40340fb5b090f7463013c2ceb3c1d3004e2`. Its integrity and commit-named root were checked before extraction under the external dependency directory.

## Toolchain boundary

The machine has CMake, Ninja, and MSYS2 UCRT GCC/G++, but no `nvcc` or Visual Studio C++ toolchain. The official verified CUDA build is therefore the real unmodified GPU baseline. Source-level telemetry validation begins as a CPU build or separate callback probe; installing a CUDA compiler remains deferred until the live-runtime gate warrants it.
