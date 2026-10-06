# Requested Luna walkthrough and local reconstruction

The user requested that the agent run the full check and assign Luna to answer
the five handoff tasks. This is an **agent walkthrough**, not independent-user
acceptance. Luna ran status and read the existing documents and evidence without
editing files. Its reported elapsed time was approximately two minutes; this is
an estimate of agent time, not a human task-success or timing measurement.

## Luna's answers

1. The qualified Gemma Q6 held-out gain is **+9.38%** over resolved defaults of
   eight threads with CUDA graphs on, CI95 **[+7.92%, +10.61%]**. The automatic
   selection used twelve threads with graphs on. The independent manual grid
   was equivalent, CI90 **[-0.39%, +0.65%]**, with eighteen evaluations per grid.
   See the [repeatability result](../stock-repeatability-20261004/report.md).
2. The four wider results are Q4 prose **+1.03%**, Q4 code **+0.78%**, Granite
   Q6 prose **-0.02%** and Granite Q6 code **-0.23%**. All are NO-UTILITY-GAIN.
   The fixed gate requires at least 5% gain and a positive CI95 lower bound,
   together with manual equivalence and numerical/resource checks. Positive
   Q4 intervals do not meet the point-gain threshold; Granite selected defaults.
   See the [wider result](../stock-coverage-20261005/report.md).
3. `status` checks pinned report digests and displays archived outcomes. It
   does not freshly validate weights, raw artifacts, runtime or host identity.
   `verify` delegates reconstruction to the two public validators and checks
   unchanged source/report/native-start inventories. It launches no models.
4. `verification.json` records overall and per-validator results, exit codes,
   costs and output hashes. Each validator that actually starts has a retained
   `<study>-stdout.json` and `<study>-stderr.log`; when Q6 stops first, no wider
   logs exist. A fresh output directory is required for each invocation.
5. The evidence covers pinned Gemma Q6/Q4 and Granite Q6 workloads on one
   Windows/NVIDIA host/build. It does not establish arbitrary-GGUF support,
   other-host utility, global optimality, custom placement acceleration or
   serving throughput. Keep failed diagnostics, resolve the recorded blocker
   and use a fresh directory for read-only reconstruction. Do not replace native
   samples, change scientific thresholds or reuse a closed collection budget.

These answers are grounded in the [handoff](../../stock-product-handoff.md),
[method](../../stock-configuration-method.md) and
[pinned status helper](../../../scripts/stock_product_status.py).

## Environmental attempts and app closure

The user's `stock-product-user-check-01` run passed Q6 in 219.23 seconds and
stopped at wider ENVIRONMENT-BLOCKED after 11.72 seconds. Its saved diagnostic
was "GPU is busy or device-free reserve unavailable"; total 230.97 seconds.
The first agent attempt stopped at the same guard before Q6 reconstruction
completed, after 14.33 seconds. A monitored attempt stopped at Q6 after 27.81
seconds. Neither agent prefix started the wider validator.

The guard requires at least 256 MiB free VRAM and GPU utilization at most 10%.
Independent two-second telemetry during the monitored failed prefix observed
at most 6% utilization and at least 13,637 MiB free. The public error does not
include its actual rejecting sample; these samples cannot establish which
condition failed at that instant or prove a specific app caused the rejection.

At the user's explicit request, Windows GPU-engine counters identified Zen
video decode at 1% and Windows Terminal 3D activity at 4–7%; DWM also showed
1%. The observed Zen app processes and Terminal instance were stopped.
Windows and Codex were retained. Terminal subsequently reopened under a new
PID and was idle in the next sample; Zen was absent. The subsequent device
reading showed about 15 GB free VRAM. No automatic termination policy was
added to the helper. A separate post-closure reconstruction is retained below.

The original failed directories remain intact. Fresh read-only reconstruction
is distinct from prohibited native collection retries or replacement samples.

## Completed post-closure reconstruction

The [fresh receipt](agent-after-close/verification.json) passed
**PASS-LOCAL-STOCK-QUALIFICATION**, exit 0, in **313.51 seconds**. Q6 passed in
203.71 seconds and wider in 109.69 seconds; both public decisions verified
their expected evidence. The [Q6 output](agent-after-close/q6-stdout.json) and
[wider output](agent-after-close/wider-stdout.json) have empty stderr files.
The helper recorded **zero additional native calls**, fresh validation true
and the same source/report/native-start inventory digest as the earlier accepted
qualification. Original 41 files/six history pins/148 starts and wider source
bindings/344 starts remain intact. Closing apps preceded the successful check;
it does not prove which app caused the earlier guard failures.

The measured checkout was `136f4cc7e7f445ad97de0379eeec183d981b3158`, with the
unchanged helper from `c9d2f9700e59d45eac47d92ebd64c1ce98717deb`. No compiler,
provider, helper, test or pinned catalogue bytes changed. Three agent invocations
used **355.65 seconds** of helper wall time in total; the user's prior invocation
used 230.97 seconds separately. GPU prechecks, closure actions and the estimated
Luna reading time are outside these helper costs. These are reconstruction costs,
not inference speedups or new quality/performance measurements.

The [archive manifest](archive.json) binds byte-for-byte copies of all four
attempts and app-closure records to their original files. The
[reconstruction verifier](verify_archive.py) passed
[archive and integrity checks](archive-verification.json), including the earlier
raw helper snapshot and unchanged native/source inventory. Failed receipts keep
their original null native-call fields; they have not been rewritten as passes.
The [Luna findings](agent-findings.json) and [archived status](status.json) preserve
the interpretation separately from fresh acceptance.

## User-evidence boundary

The user did execute the validation task; its environmental stop is retained.
The user then asked for agent assistance with the interpretation tasks. No
five-task unaided human success, human task timings, portability or demand is
established. See the [current state](../stock-followthrough-20261006/usability-state.json).
