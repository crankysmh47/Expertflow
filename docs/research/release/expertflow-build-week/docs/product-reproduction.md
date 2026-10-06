# Product reproduction

## Evidence-only path

Run `scripts/setup_release.ps1`, then `uv run expertflow demo --replay`. This path verifies the bundled evidence without CUDA or the GGUF.

## Runtime build

Check out llama.cpp at `a7312ae94f801fc9c6786dc56e38df57b964f697`, then run:

```powershell
./scripts/build_release_runtime.ps1 -Source C:\src\llama.cpp -Build C:\build\expertflow-llama
```

The script applies the ordered patch series and builds with Ninja, MSVC v143 14.39.33519, CUDA 12.8.93, `GGML_CUDA=ON`, and Release mode. Compare the emitted hashes with `runtime-source/manifest.json`.

Set the three path variables from `.env.example`, run `expertflow doctor`, then use the commands in `submission/judge-test-guide.md`.
