# R1/R2 Mover Optimization and Concurrent Prefetch Plan

Branch `ef-v2`. This plan resumes runtime work after the 2026-07-19 freeze,
restricted to the two bounded directions that prior evidence leaves open:
transfer-path efficiency (R1) and prefetch in a concurrent regime (R2).
Every live measurement below requires the verified Windows/NVIDIA system and
the Q6 model; all analysis commands run anywhere.

## Prior evidence this plan builds on

- Idle pinned H2D of one packed Q6 slot: 0.234 ms p50 at 13.3 GiB/s sustained
  (`docs/evidence/q4-live-cache-go-no-go.md`, transfer aggregate trials).
- P2 async prefetch moved real bytes but recorded zero ready-useful transfers;
  host queue-to-ready intervals averaged ~36.8 ms, roughly one decode forward
  (`docs/evidence/live-cache/p2-layer24-async-prefetch-result.md`).
- The projected-state causal filter produced ready-useful covers of true
  reactive misses, but net throughput stayed negative at np=1.
- Oracle deadline analysis shows ~96% of residual blocking disappears with
  perfectly timed prefetch, so timing, not information, was the binding limit.
- 13.3 GiB/s is ~20% of PCIe Gen5 x16 theoretical bandwidth; the mover has
  untested headroom through batching, queue depth, and staging policy.

## R1: mover characterization (new `expertflow mover-benchmark`)

New CUDA Runtime API measurements in one measured report:

1. Contiguous batching: one memcpy spanning N slots versus N separate slot
   memcpys, CUDA-event timed, per (slot_bytes, slot_count) point.
2. Queue depth: bursts of D unsynchronized copies with per-copy event markers,
   exposing serialization inside a burst.
3. Ready latency: host wall time from `cudaMemcpyAsync` enqueue until
   `cudaEventQuery` reports completion, once idle and once while a second
   stream saturates the copy engine with large background copies. This is the
   direct P2 failure diagnostic.

Run when the GPU frees (three independent trials for pooling):

```powershell
uv sync --frozen
uv run expertflow mover-benchmark --cudart <cudart64_12.dll> `
  --slot-bytes 3346048 --slot-bytes 26768384 `
  --slot-count 1 --slot-count 2 --slot-count 4 --slot-count 8 --slot-count 32 `
  --queue-depth 1 --queue-depth 2 --queue-depth 4 --queue-depth 8 --queue-depth 16 `
  --batches 30 --warmup-copies 10 --ready-samples 200 `
  --background-bytes 67108864 --background-copies 8 `
  --staging-mode pinned --output runs\ef-v2\mover-pinned-r1.json
```

Repeat with `--staging-mode pinned_wc` to test write-combined staging.
Gate: proceed toward R2 integration only if loaded p95 ready latency falls
below ~0.75 ms for one slot and batched speedup exceeds 2x at 8 slots.

## R2: concurrent-regime prefetch decision instrument

`expertflow prefetch-sim` interleaves one trace per simulated server slot into
shared-cache steps and estimates bounded next-step prefetch:

```powershell
uv run expertflow prefetch-sim conv-a.jsonl conv-b.jsonl conv-c.jsonl conv-d.jsonl `
  --prediction oracle --capacity-per-layer 32 --max-transfers-per-step 8 `
  --expert-transfer-ms <R1 measured value> --slot-bytes 3346048 `
  --output prefetch-oracle.json
```

Run `--prediction none` as the LRU-only baseline; compare `miss_count`,
`ready_useful`, and `wasted_transfers`. Traces must come from concurrent
server collection (np >= 4) because single-stream traces are the regime where
static placement already won.

Go/no-go gates before any llama.cpp change:

1. Concurrency gate: oracle prefetch at np=4 must cut estimated blocking by
   at least 20% versus `--prediction none` on held-out conversations.
2. Timing gate: R1 loaded ready latency must fit inside the observed
   inter-layer window for the predicted transfer count per step.
3. Only after both gates pass: implement async issue in the isolated
   llama.cpp worktree, exactness-gated, disabled by default, following the
   established paired-benchmark protocol.
