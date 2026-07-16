# Multi-layer reactive-cache ramp: CPU-resident layer blocker

**Verdict:** STOP before the two-layer blocking matrix and before further
multi-layer source expansion.

This is a measured diagnostic result, not a throughput result. The committed
generic runtime remains disabled by default. No prediction, asynchronous copy,
MTP, 64-slot expansion, kernel change, or graph/allocator redesign was added.

## Scope and frozen ramp

The approved first blocking ramp was layers `[0,24]` at 32 slots per layer and
`-ngl 10`. The runtime preallocated one consolidated CUDA arena. Its exact
two-layer allocation was 214,107,392 bytes. The first execution failed with a
CUDA illegal-memory-access error, so the full prompt/repetition matrix was not
started.

The investigation was bounded to single-layer controls needed to distinguish a
bad slot primitive from a backend-placement problem:

| Control | Offload report | Arena | Result |
|---|---:|---:|---|
| layer 24, `-ngl 10` | 10/31 layers | 107,053,696 B | PASS |
| layer 21, `-ngl 10` | 10/31 layers | 107,053,696 B | PASS |
| layer 0, `-ngl 10` | 10/31 layers | 107,053,696 B | FAIL: CUDA illegal access |
| layer 0, `-ngl 30` | 30/31 layers | 107,053,696 B | FAIL: CUDA illegal access |
| layer 0, `-ngl 31` | 31/31 layers | 107,053,696 B | PASS |

At `-ngl 30`, transformer layer 0 remains CPU-resident. At `-ngl 31`, all 30
transformer layers plus the output layer are offloaded, and layer 0 becomes
GPU-resident. The layer-0 cache therefore crosses from deterministic failure to
success at the exact backend-residency threshold.

The passing `-ngl 31` control completed 42 prompt tokens and one generated
token in 606.703 ms end-to-end, wrote 42 cache records, and tore down cleanly.
This is only an execution control; it is not a parity or performance claim.

## Root-cause conclusion

The direct packed-Q4 32-slot primitive works for an MoE layer whose normal graph
placement is already CUDA-resident. It does not safely force only the expert
operation of a normally CPU-resident layer onto CUDA inside a CPU-to-CUDA-to-CPU
graph split.

The identical one-layer arena size succeeds for layers 21 and 24 at `-ngl 10`
and for layer 0 at `-ngl 31`, so the evidence rejects these candidate causes:

- incorrect layer-0 packed byte size or metadata;
- consolidated-arena size or alignment;
- generic per-layer state alone;
- layer ID 0 parsing;
- the number of physical slots.

Supporting layer 0 at `-ngl 10` would require changing graph placement,
scheduling, or whole-layer offload behavior so all required operands and
dependent operations share a valid backend contract. That is the broad graph or
scheduler redesign prohibited by the approved stop conditions.

One narrow hypothesis—retaining and redirecting every scheduler-generated
destination for a binding—was tested and rejected because layer 0 still failed.
That change and its source-only regression test were removed without commit.

## Evidence

Evidence root:
`C:\models\expertflow\runs\multilayer-cache-ramp`

- Frozen two-layer failure:
  `blocking-smoke-0-24\combined.log`
- Layer-0 failures:
  `blocking-smoke-layer0\combined.log`,
  `blocking-smoke-layer0-v2\combined.log`,
  `blocking-smoke-layer0-sync\combined.log`,
  `blocking-smoke-layer0-debug\combined.log`,
  `blocking-smoke-layer0-ngl30\combined.log`
- Passing GPU-resident controls:
  `blocking-smoke-layer21`,
  `blocking-smoke-layer24`,
  `blocking-smoke-layer24-debug`,
  `blocking-smoke-layer0-ngl31`

Selected SHA-256 hashes:

| Artifact | SHA-256 |
|---|---|
| layer-0 `-ngl 10` failure log | `f0b9249b45809b45dc5fa9747671221bcf0b0b0f1ad6b827ffec40753bbc9c16` |
| layer-0 `-ngl 30` failure log | `d7dcba661bf003ae90c8014b97eacf14aa5249357c457c5c5106b6dbd0e7ae58` |
| layer-0 `-ngl 31` passing log | `6ee89fd5051f688ccad48bf0823a579993320b1f48776fda26720435eac1272b` |
| layer-0 `-ngl 31` cache events | `dea00b42f2d03cf3dbf92054badb6462eb0842a47a60a6dd3c949f0437806f12` |
| layer-21 `-ngl 10` cache events | `5a235f5762be0eb7d084e37c4cc88696b3b0b0d04993e1026f57f077bde3ddee` |
| layer-24 `-ngl 10` cache events | `574faf006b9257a9673a83f2d891b463be64d200e18a90b2156f755afd375085` |
| probe executable | `caa70a22f27a61bd9abc4324223eed3460513455ccbbeb5a6638b8816964a50a` |
| `ggml-base.dll` | `2c09945cd3484cc24617fecca27fa12b46330dc57117f2cd03de00bd0b1dc6d2` |
| `ggml-cpu.dll` | `e3129c973c4adb4471b6e9b64f66a4a02af6bde151a8de57b78e0ddcc47c30e2` |
| `ggml-cuda.dll` | `fba48f2c70d1107b886c396b1ad2dad600ba3625ab991e1a985dc87d49fa2c4f` |
| `ggml.dll` | `509f43652e7a82566cf947b7ee81f28ab4e768e8f8dc199dec17032b43d0f36c` |
| `llama.dll` | `70bbc2eeb5f2a58764aec2fa6531f24a53998995366b3c28bde34d60a1c9d005` |

The control command used the supported CUDA 12.8 runtime, the committed probe,
the canonical Gemma Q4 model, `EXPERTFLOW_LIVE_CACHE=1`,
`EXPERTFLOW_LIVE_CACHE_MODE=blocking`, a singular layer through
`EXPERTFLOW_LIVE_CACHE_LAYERS`, and `-n 1`. The exact model, probe, DLL, driver,
and commit hashes are recorded in
`configs/canonical-multilayer-cache.json`.

## Decision

Do not run the two-layer, five-layer, or all-layer cache matrices at `-ngl 10`.
Do not modify llama.cpp graph placement to bypass this result. Preserve the C5
layer-24 result and the committed generic infrastructure as disabled-by-default
research milestones.

A future runtime track may be reopened only with a separately approved design:

1. restrict caching to layers already resident on CUDA under the chosen stock
   offload configuration; or
2. explicitly authorize and test whole-layer/backend-placement changes.

Neither alternative is implied by this report. The protected Observatory and
the previously committed diagnostic benchmark remain the submission floor and
performance evidence.
