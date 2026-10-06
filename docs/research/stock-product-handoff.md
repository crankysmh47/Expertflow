# Bounded stock prototype handoff

This research checkout exposes validated stock selection and reconstruction on
the pinned Gemma Q6/Q4 and Granite Q6 artifacts. It does not optimize arbitrary
GGUFs or establish a global optimum, placement gain or serving throughput.
The accepted Q6 defaults gain and the four negative wider outcomes remain
separate. See [status](STATUS.md) and [the stock method](stock-configuration-method.md).

## Inspect the current results

From the matching checkout and its existing environment:

```powershell
uv run --no-sync python scripts/stock_product_status.py status
uv run --no-sync python scripts/stock_product_status.py status --json
```

The helper checks the pinned report and transfer digests. It displays archived
results; it does not recheck native artifacts, model weights, runtime identity
or the current host. A missing or altered input fails with exit 2. This is a
checkout helper around the existing `expertflow stock` CLI, not another compiler.

## Run fresh local reconstruction

The registered local source checkout, native databases, exact model/runtime
artifacts and original absolute paths must be available. Reconstruction hashes
the weights and artifacts, so it takes substantially longer than status display.

```powershell
uv run --no-sync python scripts/stock_product_status.py verify --output-dir C:/models/expertflow/runs/stock-product-user-check-01
```

Choose a fresh directory outside the study roots. The helper invokes only
`expertflow stock repeatability validate` and `expertflow stock coverage validate`.
It preserves both validators' output, exit codes, wall costs and a
`verification.json`. Fresh qualification requires both public decisions to
verify the expected reports and unchanged report/source/native-start digests.
It launches zero model processes. A failed or timed-out validator leaves an
explicit unverified result and diagnostics; it does not collect replacements.
An occupied output directory is never overwritten.

Native consumer execution and a new model/host/workload study are separate
actions. Existing acceptance plans remain bound to their original identities;
the helper creates no new plan and changes no provider policy.

## Independent-user acceptance procedure

Current state (2026-10-06): the user ran verification. Q6 passed; wider stopped
at the GPU environment guard. The user then requested a Luna agent walkthrough
of the interpretation tasks. See the [answers and retained attempts](evidence/stock-agent-walkthrough-20261006/report.md)
and [recorded state](evidence/stock-followthrough-20261006/usability-state.json).
The earlier agent-run [local qualification](evidence/stock-followthrough-20261006/verification.json)
passed both studies with zero model calls. Agent answers and validation receipts
do not establish five-task unaided human success; human task timings are unavailable.

The agent can demonstrate commands and validate evidence, but independent-user
usability remains unmeasured. Have an actual intended user, without coaching,
perform these five tasks on the prepared matching environment:

1. Run status and identify the qualified Q6 gain and its comparison baseline.
2. Identify the four negative wider cases and explain why positive Q4 point
   estimates do not qualify under the fixed utility gate.
3. Identify which results are archived and which require fresh reconstruction.
4. Run verify into a fresh directory and locate the two retained validator logs.
5. Explain the model/workload/host limits and the next action if validation fails.

Record the user role, environment, task success without help, elapsed time,
confusion, command exit codes and verification artifact digests. Do not collect
personal details or count agent/fake tests as users. The workflow passes this
small acceptance check only if all five tasks succeed unaided; otherwise record
the observed problem and fix it without changing scientific thresholds. This
check addresses usability on the prepared host, not portability or market demand.

On pass, continue with scoped packaging/onboarding based on that evidence.
On fail, fix the observed workflow problem and repeat the usability procedure
with a fresh user session. A new scientific performance claim still requires its
own registered experiment. Publishing or deployment remains a separate action.

## GPU environment stop

`ENVIRONMENT-BLOCKED` with "GPU is busy or device-free reserve unavailable"
means the live guard detected GPU utilization above 10% or less than 256 MiB
free VRAM. Its error does not report which condition or the rejecting values.
Inspect `q6-stdout.json` or `wider-stdout.json` for the validator's reason; the
overall helper reason alone may be less specific. A wider log exists only if
that validator started.

Close or idle GPU-heavy apps, then run read-only verification into a new output
directory. Keep the failed directory; never overwrite it or change the guard
to obtain a pass. This read-only check launches no model processes and does not
reuse native collection budgets. The helper does not close apps automatically.
