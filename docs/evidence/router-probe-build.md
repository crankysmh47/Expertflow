# Router probe build evidence

- **llama.cpp release:** `b10002`
- **Source/header revision:** `a7312ae94f801fc9c6786dc56e38df57b964f697`
- **Runtime:** verified official CUDA 12.4 DLLs
- **Compiler:** MSYS2 UCRT64 GCC 15.2.0
- **Status:** build PASS; real-model trace and parity pending verified Q4 completion

## Boundary

The probe is a separate executable. It does not patch the router, graph builder, allocator, model file, or official runtime DLLs. It supplies `llama_context_params.cb_eval`, asks the scheduler for tensors named `ffn_moe_topk-{layer}`, asserts the runtime tensor contract, and copies only the selected expert IDs.

The official release does not ship import libraries. The checked-in `.def` files list only the C symbols used by the probe, and `dlltool` generates small MinGW import libraries in the external build directory. The executable then links to the already verified `llama.dll`, `ggml.dll`, and `ggml-base.dll`, preserving the official CUDA runtime for the instrumented run.

## Reproduction

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_router_probe.ps1 `
  -LlamaCppSource C:\models\expertflow\dependencies\llama.cpp-a7312ae94f801fc9c6786dc56e38df57b964f697
```

The first verified local build produced:

```text
C:\models\expertflow\dependencies\llama-b10002\runtime\expertflow-router-probe.exe
2,950,780 bytes
SHA-256 97e0b1ccfa47b2f5285682eac6584e6619acf0f72e29b7c468cae12a9c8fd56b
```

`--version` reports:

```text
expertflow-router-probe schema 1.0.0, llama.cpp b10002 (a7312ae94f801fc9c6786dc56e38df57b964f697)
```

A missing-model link smoke reached the real model loader, loaded the official CUDA, RPC, and Zen4 CPU backends, detected the RTX 5060 Ti, and returned the expected model-open failure. This verifies DLL resolution and backend discovery without treating a nonexistent model as an inference test.

The final probe statically links its MinGW support libraries. Its only non-system runtime dependencies are the pinned `llama.dll`, `ggml.dll`, and `ggml-base.dll`; it does not depend on a developer shell's `libgcc`, `libstdc++`, or `libwinpthread` DLLs.

The seven directly included headers were fetched from revision-pinned raw GitHub URLs and their exact sizes and hashes are recorded in `configs/runtime-artifacts.toml`. Full source-archive verification remains pending and is not implied by the successful header/runtime build.

The finalized callback ignores graph-reservation probes, requests tensors only during an active one-token decode, accepts GGML's collapsed trailing unit dimension (`ne0=8`, `ne1=1`), and fails nonzero if a tracing run emits no events. The real-model trace and exact parity result are recorded in `q4-baseline-result.md`.
