# Product release command ledger

| Gate | Command or decision | Result |
|---|---|---|
| Isolation | `git worktree add ... -b codex/product-release b24eb1f` | Created isolated, unmerged, unpushed release branch. |
| Baseline | `py -m pytest -q` | Environment failure: global Python had no pytest. |
| Baseline | `uv sync --all-extras`; applicable pytest run | 233 passed; seven historical T1/T2 source contracts failed because they target a different preserved llama.cpp branch. |
| Identity | SHA-256 model and release binaries | Model, `llama-cli.exe`, and `llama-server.exe` match frozen values. |
| Reconstruction | Committed Q6 result, MMLU, PPL, placement, and cache-simulation evidence | `RELEASE STATE VALID`; no expensive benchmark rerun. |

The release path deliberately uses environment variables and placeholders instead of the private absolute paths preserved in historical raw evidence.

| Batching smoke | Stock and ExpertFlow, one slot, one cold server each | Harness passed; result treated only as smoke. |
| Batching sweep | Slots 1, 2, and 4; three cold servers per mode/slot; 128 generated tokens per request | Four slots selected; no request failure or OOM. |
| Batching final | Four slots; five cold servers per mode; four simultaneous 128-token requests | ExpertFlow 35.6699 aggregate TPS versus stock 24.5231 (+45.4546%); 11,996.8125 MiB peak; 20/20 requests; concurrent outputs were not deterministic across repetitions. |
| Context doubling | 8K through 512K allocated context, stock and ExpertFlow | All points allocated and processed 385 prompt plus 32 decode tokens. |
| Context boundary | 1M allocation | Stock passed; ExpertFlow failed explicitly during layer-20 static-shadow CUDA allocation. |
| Context binary search | 768K and 896K passed; 960K failed; 928K passed | Passing/failing bracket is 950,272/983,040 allocated tokens. Product profile capped at the 262,144-token model training context with 675.418 MiB measured reserve. |
| Agentic demo attempt 1 | One coding, two parallel workers, long repository-analysis prompt | Server passed first three requests; example client omitted `reasoning_content`, and the long prompt exceeded the measured 2,048-token slot share. Server stopped cleanly. |
| Agentic demo attempt 2 | Corrected response field and bounded repository prompt | Coding request, two parallel workers, and repository analysis all completed through the local OpenAI-compatible endpoint; server stopped cleanly. |
| Dashboard | Local HTTP server on loopback; DOM and visual inspection | Eleven panels rendered at 1265×720 without horizontal overflow; 30 layer cells present; no browser console warning/error; local server stopped. |
| Packaging audit | First allowlisted build | Correctly failed because copied `__pycache__` bytecode embedded private worktree paths. Added a regression test and excluded Python cache artifacts without weakening the audit. |
| Product CLI smoke | `compare deployments/max-performance.json` | Found and fixed a release-manifest schema mismatch (`measured`/`evidence_files` versus legacy `evidence`); both schemas now have regression coverage. |
| Final tests | Applicable suite against the frozen static-Q6 llama.cpp source | 259 passed; only the two documented historical temporal-branch source-contract modules were excluded. |
| Live doctor | Verified Q6 GGUF, llama CLI/server, NVIDIA GPU/driver, and CUDA runtime | All checks passed; model and both binary hashes matched the frozen release state. |
| Archive | Two consecutive allowlisted builds | Byte-identical ZIPs; SHA-256 `5b1ea945994f4e069b8a6687b71dae8165d3d78be8c567275a5fe1cc260346f7`. |
| Clean archive replay | Extracted ZIP in a new temporary directory; `verify_release.py`; `setup_release.ps1` | 97 files verified and the model-free evidence replay passed from the extracted package. |
| Polish isolation | `git worktree add ... codex/final-polish f8b88bd` | New unmerged, unpushed worktree created from the frozen release commit. |
| Clean-checkout baseline | `uv sync --frozen`; applicable tests | Found two replay failures caused only by CRLF conversion of hashed JSON; canonical JSON hashing fixed the portability contract. |
| Claims validation | Scorecard, README, dashboard, judge guide, Devpost, and claims-ledger tests | Headline metrics, classifications, layer set, model hash, context caveat, quality caveat, and concurrency caveat agree. |
| Doctor | Verified live paths plus mocked replay-only platform | 12/12 live checks passed; unsupported environments return structured exit 10, missing verified-live prerequisites return exit 20, and ordinary failures have corrective commands without tracebacks. |
| Portable scripts | PowerShell and shell wrapper tests from extracted path containing spaces | Both release-verification wrappers passed; repository replay wrappers passed. |
| Visual inspection | Local SVG and packaged dashboard over loopback HTTP | Architecture visual rendered cleanly; dashboard showed 11 panels, 30 layer cells, exact metrics, no overflow, and zero page console errors. |
| Final tests | Full applicable suite | 272 passed; only the two documented historical temporal-branch source-contract modules were excluded. |
| Final archive | Two normalized, sorted, fixed-timestamp builds | Identical SHA-256 `b33a438a3fcde3180f6e699cbf37e587d3022d332bed7152afe16e84f4c6c77f`; 118 files in the internal manifest. |
