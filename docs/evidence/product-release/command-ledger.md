# Product release command ledger

| Gate | Command or decision | Result |
|---|---|---|
| Isolation | `git worktree add ... -b codex/product-release b24eb1f` | Created isolated, unmerged, unpushed release branch. |
| Baseline | `py -m pytest -q` | Environment failure: global Python had no pytest. |
| Baseline | `uv sync --all-extras`; applicable pytest run | 233 passed; seven historical T1/T2 source contracts failed because they target a different preserved llama.cpp branch. |
| Identity | SHA-256 model and release binaries | Model, `llama-cli.exe`, and `llama-server.exe` match frozen values. |
| Reconstruction | Committed Q6 result, MMLU, PPL, placement, and cache-simulation evidence | `RELEASE STATE VALID`; no expensive benchmark rerun. |

The release path deliberately uses environment variables and placeholders instead of the private absolute paths preserved in historical raw evidence.
