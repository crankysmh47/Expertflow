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
| 2026-07-18 | baseline | `uv run python -m pytest -q` after placement tooling | 176 passed, 2 expected external-source skips |
| 2026-07-18 | baseline | Commit truthful baseline and placement evidence | `34cf11d` |
| 2026-07-18 | narrow-Q4 | Create `codex/q6-runtime-final-llama` at pristine `a7312ae94` | isolated clean llama.cpp worktree |
| 2026-07-18 | narrow-Q4 | Add then run static-island source-contract test | RED: 2 expected failures, then GREEN: 3 passed |
| 2026-07-18 | narrow-Q4 | Implementation 1: pre-scheduling expert-node assignment plus persistent scheduler-owned CUDA shadows | layer-scoped, disabled by default; compile started |
| 2026-07-18 | narrow-Q4 | Initial CMake configure with `GGML_BACKEND_DL=ON` and static libraries | failed: shared libraries required |
| 2026-07-18 | narrow-Q4 | Two background build launch attempts through `VsDevCmd` | failed before compile: `cl.exe` not initialized; no source change |
| 2026-07-18 | narrow-Q4 | Add parameterized VS 2022/CUDA 12.8 build script; run through `vcvarsall -vcvars_ver=14.39` | configure passed; incremental build running |
| 2026-07-18 | narrow-Q4 | Build pristine-based implementation 1 with MSVC 19.39/CUDA 12.8 | PASS; diagnostic binary SHA-256 `36b083b...b4c40` |
| 2026-07-18 | narrow-Q4 | Feature-disabled Q4 layer-0 smoke, `-ngl 10` | PASS |
| 2026-07-18 | narrow-Q4 | Initial full static-shadow layer-0 smoke | illegal CUDA access; no process or allocation persisted |
| 2026-07-18 | narrow-Q4 | `CUDA_LAUNCH_BLOCKING=1`, scheduler assignment dump | isolated failure after 428,212,736-byte preload |
| 2026-07-18 | narrow-Q4 | Dynamic scheduler-copy diagnostic | same illegal access; persistent shadow allocation exonerated |
| 2026-07-18 | narrow-Q4 | Gate/up-only and down-only diagnostics | gate/up passed; down chain failed; down `MUL_MAT_ID` alone passed |
| 2026-07-18 | narrow-Q4 | Leave scale view/reshape dependency assignment to scheduler; retain complete scale shadow | corrected 128-expert static island passed smoke |
| 2026-07-18 | narrow-Q4 | Pristine, 3 disabled, and 3 enabled 16-token full-router probes | 960 complete events/run; disabled exact; enabled deterministic but not exact |
| 2026-07-18 | narrow-Q4 | Reproducible parity analyzer | first router divergence prefill token 0/layer 13; first generated divergence index 1 |
| 2026-07-18 | narrow-Q4 | Audit explicit-boundary fallback against supported scheduler API | rejected: same CUDA numerical path; next change requires forbidden kernel/new op/broad scheduler work |
| 2026-07-18 | terminal | Stop stages 2–7; preserve evidence only | `NARROW PLACEMENT STOP` |
