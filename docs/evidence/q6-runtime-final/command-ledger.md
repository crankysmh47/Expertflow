# Q6 runtime final command ledger

| Local time | Gate | Command/action | Result |
|---|---|---|---|
| 2026-07-18 | setup | Create `codex/q6-runtime-final` from `codex/q6-download` (`33bc3f5`) | clean isolated worktree |
| 2026-07-18 | setup | `uv sync --frozen --extra dev --extra predictor` | environment created |
| 2026-07-18 | setup | `uv run python -m pytest -q` | 174 passed, 2 expected external-source skips |
| 2026-07-18 | design | Preserve whole-layer `PLACEMENT STOP`; authorize only narrow scheduler island plus one explicit-boundary fallback | accepted scope frozen |
| 2026-07-18 | baseline | Initial patched CMake configure with compiler PATH only | failed before compile: Windows SDK `rc` absent |
| 2026-07-18 | baseline | Reconfigure/build b910bc37 through VS 2022 `VsDevCmd`, CUDA 12.8, `120a-real` | `llama-cli` built successfully |
| 2026-07-18 | baseline | 512-token packaged Clang pristine and patched MSVC comparison | compiler confound found; excluded from exact equivalence |
| 2026-07-18 | baseline | 512-token matched MSVC pristine vs patched-disabled, `-ngl 99` | identical response hash; 6.4 TPS each |
| 2026-07-18 | baseline | Three pristine MSVC `--cpu-moe` cold processes, 512 tokens | 23.3/22.8/22.8 TPS; 3,098.656 MiB peak; identical hash |
| 2026-07-18 | baseline | Reconcile verbose model buffers, GGUF inventory, and scheduler split topology | fair stock baseline frozen at 22.967 mean decode TPS |
