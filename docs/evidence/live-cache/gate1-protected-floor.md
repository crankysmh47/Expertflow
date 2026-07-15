# Gate 1: Protected Observatory floor

**Result:** PASS. No installer or llama.cpp source modification occurred before this gate was captured.

## Protected state

- Protected commit: `d846bdfcb1980dfc44d9f951e2824f58429f16d7`
- Protected branch: `codex/expertflow-stage0` at `C:\sem4\Expertflow`
- Annotated tag: `observatory-floor-2026-07-15`
- Tag object: `2da2dd960dd74aee319e680615e124c68077319c`, whose recorded object is exactly `d846bdfcb1980dfc44d9f951e2824f58429f16d7`
- Live branch: `codex/live-cache-blocking-spike` at `C:\models\expertflow\worktrees\live-cache-blocking`
- Both worktrees were clean at capture. The protected worktree remains unmodified.
- Protected and isolated live checkouts each passed all 87 tests before native work began.

## Pre-install provenance

The full machine-readable record is `C:\models\expertflow\runs\live-cache-spike\gate1\provenance.json`. It contains capture timestamps, Windows identity, relevant non-secret build/runtime environment variables, the pinned llama.cpp revision and paths, the model identity, Visual Studio inventory, and SHA-256/signature/version records for 116 executables and DLLs.

| Item | Pre-install result |
| --- | --- |
| OS | Windows 11 Pro 10.0.26200, 64-bit |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16,311 MiB, compute capability 12.0 |
| Driver | 591.86; full `nvidia-smi -q` retained externally |
| Existing Visual Studio | Visual Studio Community 2026 18.3.1 at `C:\Program Files\Microsoft Visual Studio\18\Community` |
| CMake | 4.3.1 |
| Ninja | 1.13.2 |
| uv | 0.11.1 |
| Git | 2.52.0.windows.1 |
| `nvcc` / PATH-visible `cl` | both absent |
| llama.cpp source pin | `a7312ae94f801fc9c6786dc56e38df57b964f697` |
| Q4 model | 14,439,361,440 bytes; SHA-256 `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5` |
| Live-cache environment | all `EXPERTFLOW_LIVE_CACHE*` variables unset |

## Protected replay reproduction

A fresh `git archive` of the annotated tag was extracted to `C:\models\expertflow\runs\live-cache-spike\gate1\protected-d846bdf-replay`. `uv sync --frozen --extra dev`, the full test suite, and the judge fixture all passed.

| Check | Result |
| --- | --- |
| Tests | 87 passed in 0.52 seconds |
| Canonical LF trace SHA-256 | `245aac7ffa83f464f33f220c2c7cafbf931671884c48fe2f92d48795ef11df8e` |
| Replay totals | 8 events / 64 demands; static 26 hits / 38 misses; LRU 19 hits / 45 misses |
| Canonical output SHA-256 | `6c658457ea3320f065a47b8931bb5992700c2ba075c6b77a2351072a3589358c` |
| Protected archive SHA-256 | `be042bc9578f845a3b4e77fd2a53adfe18fabbb2463810a9348941e6a381c6d2` |

The replay's raw JSON SHA-256 is path-dependent because `source_trace` is an absolute path. The protected replay raw hash is `2550bcbb081994d8fe95fab0da5b50ede02f5c81d8b04a65a7b7f020fd94dfff`; the historical raw hash is `54f46ccbf719b37f5cca55cc87d1625b8e8abdfd88f34faadc13042709010162`. Normalizing only `source_trace` to `examples/replay/trace.jsonl` produces the same canonical hash in both runs. No measured result was normalized.

The first evidence checker was wrong: it looked for policy totals at the root rather than under `simulation`. It failed after the archive, environment, tests, and simulator had succeeded. The preserved failure was diagnosed from the serialized schema, the checker alone was corrected, and the protected commands were not rerun or altered to hide the failure.

## Observatory reproduction and held-out boundary

The exact replay command embedded in the self-contained Observatory was executed from the fresh protected tree with the same 31 training traces and eight held-out traces. The output remained byte-identical:

- `C:\models\expertflow\runs\q4-probe\report-physical-feasibility.html`
- 62,669 bytes
- SHA-256 `f3dc647d9965d726771632421b8fa5dffddc165d3ebae49f6f10381bbb75a90c`
- `live_cache_enabled=false`
- 10,584 decode events, 74,149 static-96 hits, and 10,523 static-96 misses

The static-96 fit boundary was independently reconciled. Residents were derived from 31 parity-safe `train-*` conversations only. Evaluation used exactly eight disjoint conversations: four validation and four test, one per held-out domain. LRU state resets per conversation. Static-96 produced 74,149 hits / 10,523 misses against conversation-reset LRU's 73,105 hits / 11,567 misses, a measured 9.0257% cold-byte reduction. This verifies the user's non-oracle-policy qualification for the bounded spike; it does not satisfy or replace the earlier 20% expansion threshold.

## Artifact ledger

All external artifacts are under `C:\models\expertflow\runs\live-cache-spike\gate1`.

| Artifact | SHA-256 |
| --- | --- |
| `commands.jsonl` | `11a6748bf6ef11148124e9e0009987aa07e4e120001b598dcaf06fdbfb598f0c` (3,371 bytes; final Gate 1 append) |
| `provenance.json` | `18091e46ce6b98959d3bb299048962dbf66a9e100d1b826f5e53f319ab48e672` |
| `nvidia-smi-q-preinstall.txt` | `dd89f83d3365aaa52b251e615448e17f2c21e33ac173b4f86f09bc2824479076` |
| `protected-d846bdf.zip` | `be042bc9578f845a3b4e77fd2a53adfe18fabbb2463810a9348941e6a381c6d2` |
| `protected-replay.json` | `d417636e33533031f738a96094a3a3a3af768e1f030bedfff78637bd519b4fd3` |
| `protected-report.json` | `25370883ff01ea0d35416120dc7873c2c26fab49b695b0da345ebac5ec070e7b` |
| `heldout-boundary.json` | `196b1e0c3b3281ee9e1e27219bd4782ad40a50c45b6e2d928f4991d3f1063721` |

Evidence labels remain separate: this gate contains verified artifacts and protected reproduction results. It introduces no live-runtime, CUDA deadline, KV-cache, speedup, or cache-enabled claim.
