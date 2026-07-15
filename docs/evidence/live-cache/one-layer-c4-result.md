# Exact One-Layer Eight-Slot C4 Result

## Verdict

**PASS — FIRST EXACT LIVE-CACHE PRIMITIVE.** The bounded layer-24 implementation passed C0 through C4. The default remains disabled (`live_cache_enabled=false`). This result does not authorize another layer, prediction, asynchronous transfer, MTP, ML, a runtime speedup claim, or a CUDA-event latency claim.

## Scope and identity

- branch: `codex/one-layer-blocking-cache`
- ExpertFlow base: `56d2ab4cf96ce4b036fc9518b63158491551923a`
- llama.cpp base: `a7312ae94f801fc9c6786dc56e38df57b964f697`
- layer: 24 only
- slots: eight coordinated replaceable slots
- transfer: blocking CPU-to-GPU only
- selection: authoritative router IDs only
- probe SHA-256: `bd6f1c8f70a7788e758fc7464ed2412088188b9ef35a8d21fd1a1033a0df2dcf`
- modified `ggml-base.dll` SHA-256: `4f9b1114dd2fed8286ab120e2d1e5afa2c837a3c6673beb5add73a3b8c3c0ede`
- model SHA-256: `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`

## Physical allocation

The persistent CUDA arena is allocated once by the scheduler and freed at scheduler teardown. Its measured allocation is **26,763,904 bytes (25.523 MiB)**. It contains separate packed component regions for eight Q4 gate/up slices, eight Q4 down slices, and eight F32 scale values. The earlier 26,768,384-byte value was a conservative per-object 4 KiB-aligned projection; the backend's actual component-region alignment saves 4,480 bytes.

GGML separately reports a 17.08 MiB temporary CUDA compute buffer. That is graph-allocator workspace and is not counted as slot storage. The cache never repacks Q4 data and never allocates or frees GPU memory per token.

## Correctness ladder

| Gate | Result | Evidence |
|---|---|---|
| C0 compiled/disabled | PASS | final seven-prompt feature-off rerun exactly matched canonical prompt tokens, generated tokens, and routing |
| C1 passthrough | PASS | seven prompts; 50,880 canonical events exact; stable cleanup |
| C2 fixed/initial sets | PASS | initial eight-miss fill executed directly from packed arena with exact token/router parity |
| C3 replacement | PASS | normal retained/missing replacements passed; forced eviction performed 41 consecutive all-eight replacements and every generation reached 41 |
| C4 arbitrary router sets | PASS | three repetitions of seven prompts; exact token and router parity, deterministic mappings, stable cleanup |

The three C4 repetitions cover code, arithmetic/reasoning, structured JSON, translation, and general/reproducibility prompts. Across them:

- router events: 71,640, all exact against the canonical observer runtime after excluding only `observed_at_ns`
- layer-24 cache events: 2,388
- hits: 6,237
- blocking misses: 12,867
- packed bytes transferred: 43,045,416,204
- event mapping and generations: identical across all three repetitions when excluding measured duration
- forced-miss validation: 41 events, 328 forced replacements, exact output/router parity

## Diagnostic timing and memory

Timing is measured host-wall blocking duration around packed copies plus backend synchronization. It is **not CUDA-event timing**:

| Population | Count | p50 | p95 | Range |
|---|---:|---:|---:|---:|
| all events | 2,388 | 3,301 us | 7,783 us | 0–14,747 us |
| events with at least one miss | 2,382 | 3,303 us | 7,783 us | 452–14,747 us |
| eight-miss events | 441 | 5,089 us | 10,738 us | 3,625–14,491 us |

Aggregate diagnostic effective copy rate is 4,768.7 MiB/s. It must not be presented as overlapped bandwidth or a runtime speedup.

The measurement runner's GPU peak is system-wide and ranged from 6,415 to 10,970 MiB because unrelated desktop GPU processes were active; it is retained but not used as a process-specific claim. Settled system GPU-use delta across the 21 focused processes had a median of -2 MiB. No probe process remained after validation, and no persistent allocation-growth pattern appeared.

## Verification

- ExpertFlow: 89 passed
- focused native config test: passed with assertions active
- llama cache planner/layout test: passed with assertions active
- judge fixture: reproduced at eight events / 64 demands; output SHA-256 `1b1e08dde19b61675deeebad8b6517e06d17fefff5f6809cb615bc4366aaf78`
- feature-disabled restoration: all seven prompts exact
- `git diff --check`: passed in both worktrees
- protected Observatory and pristine pinned llama.cpp checkouts: not modified

Raw artifacts are under `C:\models\expertflow\runs\one-layer-blocking-cache`. The three focused roots are `c4-focused-rep1`, `c4-focused-rep2`, and `c4-focused-rep3`; the forced replacement root is `c3-forced-evict`; final disabled evidence is `c0-final-disabled`.

## Boundary decision

The implementation uses the unchanged CUDA `MUL_MAT_ID` operation. A narrowly gated scheduler assignment is necessary because CUDA's normal host-operation heuristic rejects single-token decode. Scheduler-generated layer-24 input copies are redirected to one persistent backend-owned arena; no CUDA kernel, general allocator, model graph construction, prediction path, or second layer is modified.

The next stage remains closed. Any proposal to expand must first review this committed one-layer result and retain exact blocking fallback and disabled-by-default release behavior.
