# Twelve-thread profiling investigation

The bounded investigation completed two controls and three synchronized profiles
on `ef-v2`. All five independent native runs passed prompt/generated token,
owned memory/reserve and cleanup checks. **Decode-phase attribution is blocked
by the current instrumentation.** No optimization was selected or launched,
and no compiler product plan was published.

The stock and fork-off controls used the pinned context4096 / predict512 /
threads12 / seed42 / F16 KV / graphs-on / ngl99 / CPU-MoE server workload. The
stock control matched the previously accepted A/A reference; the unprofiled fork
control matched the new stock control. These two controls establish token
comparability for this request, not a performance result or universal numerical
equivalence. Their native rates were 23.8582 and 22.8331 TPS; one run per runtime
does not support a speedup/slowdown claim.

## What the profiles measured

The patched runtime's existing split profiler accumulated backend graph work
across warmup, prompt processing and generation. Each profile contained 62 split
records, including thirty CPU expert regions. Each expert region recorded 515
calls and zero `decode_calls`, even though the request generated 512 tokens.
The latter field tests a tensor dimension that denotes selected experts in these
regions, so it does not identify decode correctly. A small count of warmup or
prompt calls does not establish that their time is negligible.

| Run | Aggregate graph time | CPU backend compute call | CPU input boundary | CUDA input boundary | CUDA completion wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 27.742 s | 15.313 s | 4.026 s | 3.438 s | 4.380 s |
| 2 | 35.605 s | 16.790 s | 7.800 s | 6.036 s | 4.459 s |
| 3 | 36.440 s | 16.657 s | 8.419 s | 6.381 s | 4.463 s |

The table omits small CPU completion and CUDA host-submission buckets; complete
values and shares are in [analysis.json](analysis.json). These are aggregated
instrumented graph durations, not the native completion wall time.

CPU expert regions dominate CPU backend computation. CPU backend totals account
for approximately 69-70% of synchronized aggregate graph time. During the actual
completion windows, owned CPU time averaged 7.598 busy cores and GPU utilization
averaged 16.321%. This supports investigating CPU expert work, while leaving its
precise share of uninstrumented decode latency unresolved.

`input_boundary_us` covers routing, copies and synchronization before each split;
it is not a direct PCIe bandwidth or pure transfer-time measurement. The profiler
also forces synchronization after every split, changing overlap and wait
attribution. It therefore cannot distinguish a transfer bottleneck from waits
introduced or moved by instrumentation. Profile-run native TPS is diagnostic only.

## Reproduction and evidence

- [Controls](controls-report.json), [profiles](profiles-report.json), and
  [raw split profiles](split-profile-1.json) retain runtime/workload identities,
  native token hashes, diagnostics and measurement IDs.
- [Manifest](measurement-manifest.json) binds all five owned runs to their raw
  artifacts; [verification](verification.json) records hashes and source checks.
  The raw controls database remained unchanged while profiling used a copy.
- Raw files remain under `C:/models/expertflow/runs/compiler-profile-20261004`.
- `probe/` preserves the throwaway investigation scripts, not a supported product
  API. Both scripts require fresh databases; to reproduce, choose a fresh root
  consistently in `controls.py` and `profiles.py`, then run them sequentially
  with `uv run --extra dev --extra quality --extra predictor python <script>`.

The standard compiler runner intentionally strips profiling environment controls.
This probe used an isolated lowering override permitting only the fixed fork's
profile-output path, and every evidence row had `measured=False`. Its launch
environment is deliberately incompatible with ordinary product validation.
The original runner, evidence store, binaries, weights and native source were
unchanged; this diagnostic policy must not enter compiler selection or sealing.

The native profiler writes its file only during scheduler destruction. A tested
hidden private-console shutdown helper sent one CTRL_C event after verifying
the owned PID and OS creation time, allowing all three native servers to exit
with code zero and flush their profiles. No visible window or foreign-process
termination was used. All native children were absent at completion. Applicable
pinned source contracts passed six tests. No new product code or dependency was
introduced, so a repeated full CPU suite was unnecessary for this evidence-only
investigation.

## Concrete next step

Prepare a separate, bounded native instrumentation change before another
performance experiment:

1. Record explicit warmup/prefill/decode phase markers at the request/evaluation
   boundary; do not infer phase from an expert tensor's shape. Maintain separate
   counters and record how many tokens each phase processed.
2. Preserve the current default-off behavior. Distinguish CPU compute wall time,
   host submission, explicit waits and actual copy durations. Use asynchronous
   device timing where possible; label any synchronized mode separately.
3. Preserve the frozen control binaries and create a new diagnostic build with
   source, patch and DLL hashes. Test phase labeling, counter accounting and
   default-off behavior, then check token/memory/cleanup parity against stock.
4. Collect a fixed small profiling budget on this same twelve-thread workload.
   Only then choose one CPU-kernel or transfer hypothesis and freeze its paired
   acceptance experiment and numerical contract.

Until those measurements exist, do not claim a transfer bottleneck, attribute
the aggregate percentages to decode alone, retest eight threads, or reopen
reactive caching. The previous eight-thread rejection and original Phase 3
validation stop remain unchanged. Product-validation integration is separate
from this diagnostic prerequisite; no gate was silently replaced.
