# Benchmarking ExpertFlow

## Current proof and acceptance rules

Current results and next work are in [STATUS.md](STATUS.md) and the
[proof/fallback plan](superpowers/plans/2026-10-04-placement-proof-and-stock-fallback.md).
Fresh stock reference/product/search acceptance covers Gemma Q6, Gemma Q4 and
Granite Q6 on one pinned host/build; each retained its incumbent. No new accepted
quality-preserving speedup over tuned stock has been demonstrated.

For new placement proof, compare a compiler-selected plan with contemporary
tuned stock using the same model bytes, workload/interface, host, context,
concurrency and numerical/quality policy. Keep discovery and held-out quality/
confirmation separate; freeze all source/data identities and the complete
process budget before collection. Report tuning cost and startup separately
from decode TPS. Exact policies require numerical eligibility; short token
parity does not make a CPU-to-CUDA numerical change exact.

The new plan proposes at least 10% paired geometric decode-TPS gain with a
positive 95% lower confidence bound, CV at most 10%, numerical/quality PASS,
owned memory/reserve/cleanup PASS, and sealed execution acceptance. The stock
fallback separately proposes at least 5% over defaults and equivalence within
2% of a documented manual result at no greater tuning cost. Each future
protocol must freeze these gates and quality tests before running. Existing
experimental thresholds/verdicts remain unchanged.

The [bounded stock repeatability/transfer study](evidence/stock-repeatability-20261004/report.md)
passed 148 calls and final reconstruction under fixed 30-second prelaunch spacing
and diagnostics. Its two main blocks passed independently. Untouched-workload
utility gained 9.38% over resolved 8-thread/graphs-on defaults, CI95 [7.92%,10.61%],
with manual CI90 [-0.39%,+0.65%],18 search evaluations each, fresh product and
consumer acceptance. All samples were retained; no pooling, retry or extra
finalist was used. Defaults scope fixes CPU-MoE, placement, workload and F16
controls; it is not an all-flags out-of-box comparison.

Report the collector interval of 11,717.250 seconds, 4,441.791 seconds of waiting and
5,364.040 seconds of native phases separately. The collector interval excludes
input loading and read-only reconstruction. Automatic/manual native search
phase costs were 685.212/667.924 seconds, so equal evaluation budgets do not
establish lower wall time or human effort. The public CLI now reuses verified
readers within one read-only validation: 207.94 seconds, then 221.15 seconds
after the final reporting fixes, versus the earlier 2,191.48 seconds. Five
reader caches, zero additional native calls, all original source/history pins
unchanged. These descriptive comparisons do not establish a
general validation speedup. Legacy script validation retains its original
repeated reads. Decode TPS is not total
CLI or steady-state serving throughput. Diagnostics do not prove a thermal
cause or repair. Wider controls, workloads and hosts need separately frozen
eligibility, gates and full native budgets; neutral/default-optimal outcomes
must remain explicit.

The [wider coverage registration](superpowers/specs/2026-10-05-stock-coverage.md)
defines four Q4/Granite cases with prose/code prompts and unchanged thread/graph
controls. Each has86 utility calls and conditional product20/consumer1,
107 maximum; total428. Use fixed30s waits,4hours per case, separate18-evaluation
automatic/manual grids, original utility/product gates and independent raw
auditing. Report each case independently, including default-optimal
`NO-UTILITY-GAIN`; no case pooling, retries or new host claim. Registration is
not execution evidence. The separate reviewed collector froze 63 source/input
files and completed all four cases from `78ad5f0` on 2026-10-06 without changing
the closed Q6 protocols:344 retained calls, four NO-UTILITY-GAIN outcomes,
no conditional product/consumer. Q4 changes +1.03%/+0.78% were below 5%; Granite
retained the default. The sequence took 17,741.438 seconds, including 10,325.149
seconds of fixed waits; native phases totaled 4,427.471 seconds. Fresh public
validation passed in 93.40 seconds with zero additional calls. Final independent
raw audit passed. See [per-case intervals, costs and scope](evidence/stock-coverage-20261005/report.md).

The subsequent [control eligibility audit](evidence/stock-control-scope-20261006/report.md)
closed at a scoped no-go with zero native calls. A new experiment first needs
an admitted numerical path and useful mechanism, or a separate quality policy
and acceptance protocol; the closed study's budget cannot be transferred.

The separate public sequence uses `expertflow stock coverage run` and
`expertflow stock coverage validate`. All four live inputs, exact roots,
eligibility/defaults proofs and the reviewed source map freeze before the first
call. The per-case four-hour cap includes input loading and reconstruction;
the overall cap is16hours. Every attempted call is retained before waiting,
including failed starts and conditional product/consumer calls. Source/host
drift during a wait blocks launch. Statistical stops continue only when every
raw record is valid; environment, identity, memory, cleanup or resource stops
end the sequence. Validation starts no new model process and recomputes native
bindings, winners, intervals and costs. The registration inspector remains
metadata-only and continues to report `execution_ready: false`.

The completed registered roots are occupied: use `coverage validate` for
read-only reconstruction. Do not reissue collection or spend the 84 unused
conditional calls on retries, replacements or new candidates. Inspector status
describes the immutable registration rather than current measured progress.

## Historical placement release protocol

The protocol below documents the earlier release. Its terminal placement
verdict was **QUALITY STOP**: the +2.25% upper PPL confidence bound exceeded
the +1% limit. Replay integrity and favorable point estimates cannot waive
that gate. The separate 22.967 TPS reference is not the stock mean of the
matched ten-pair experiment, which measured 22.28/28.13 TPS. Do not treat
historical CLI or aggregate server rates as matched current controls.

The headline comparison uses `google_gemma-4-26B-A4B-it-Q6_K.gguf` (22,862,575,520 bytes, SHA-256 `089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba`) on a Windows 11 x64 machine with an NVIDIA RTX 5060 Ti 16 GB, driver 591.86, CUDA 12.8.93, and MSVC v143 14.39.33519.

The runtime is llama.cpp `451224ab4d12a616dc3e16e8c8063f4b331f531c`, based on upstream `a7312ae94f801fc9c6786dc56e38df57b964f697`. Stock and ExpertFlow used the same binary, Q6 model, prompt, `-ngl 99`, `--cpu-moe`, 12 threads, seed 42, temperature 0, 2,048-token context, and CUDA graph mode. Batch and ubatch were not overridden for the generation comparison, so the pinned runtime defaults applied equally to both modes.

Stock ran without ExpertFlow environment variables. ExpertFlow added:

```text
LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER=0,1,2,3,4,5,6,7,8,9,15,20
LLAMA_EXPERTFLOW_STATIC_PRECOMPUTE=1
```

Ten matched cold-process pairs generated 512 decode tokens per run. Decode TPS is generated tokens divided by the runtime's measured decode duration. Process-owned peak VRAM came from `nvidia-smi` PID sampling, not global device allocation. The strongest fair stock reference was 22.967 TPS; ExpertFlow averaged 28.13 TPS, a 22.48% improvement against that reference.

The four-slot result used one loaded server, four concurrent requests, 512 generated tokens per repetition, and five cold-server repetitions. Its 35.6699 TPS is aggregate throughput and must not be compared with a single-stream number as if the protocols were identical.

The 262,144-token context result is an allocation test. It processed 385 prompt tokens and 32 generated tokens, 417 total. It is not evidence that a 262,144-token prompt was filled and evaluated.

Public Q4, MTP, cloud, prompt-processing, different-backend, and aggregate-throughput results answer different questions. They are useful references, but a direct speed comparison requires the same model quantization, hardware, runtime, workload, token count, concurrency, CUDA settings, and placement policy.

Machine-readable values live in `release/expertflow-build-week/evidence/release-scorecard.json`.
