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
