# ExpertFlow local-model architecture

ExpertFlow saves and verifies a working llama.cpp setup, measures its behavior,
and starts local chat or a loopback server. The installed package has no mandatory
third-party Python dependencies and needs no research checkout.

## From files to a reusable profile

`local doctor` inspects hardware and probes runtime version/flags without loading
a model. `local setup` parses bounded GGUF metadata, checks context, estimates F16
KV storage, hashes the model/runtime and declared native libraries, then performs
a short real load and generation probe. Successful profiles are `RUNNABLE-UNTUNED`.
Memory estimates include uncertainty; they do not guarantee a model will fit.

Profiles record requested context, GPU layers, threads, CPU MoE and native file
identities. Their settings are stock launch settings, distinct from a research
compiler ExecutionPlan. Run, serve and measurements reverify the content identities.
Changed files require a fresh profile rather than inheriting old measurements.

## Measurements and selection

`local bench` freezes inputs, package source and a wall/process budget, then retains
raw completion/token/timing, sampled memory and owned-process cleanup receipts.
`local verify-job` reconstructs these observations without starting a model.
`local support` exports a small opt-in redacted summary for the user to review.

`local tune` admits only the bundled reviewed scheduling policy. It compares
against the runtime's default thread setting and independently confirms eligible
candidates on held-out prompts. Explicit thread overrides are ineligible baselines;
create a new setup without `--threads` before tuning. Unknown artifacts/builds
remain usable for untuned launch but receive `UNSUPPORTED` tuning with no native
launch. Negative or inconclusive outcomes keep the original profile.

## Run and serve

Both chat and serving use one owned llama-server process with loopback binding,
one parallel slot and F16 KV. Streaming requests have absolute deadlines. Windows
starts the child suspended, assigns a kill-on-close Job Object, then resumes it;
Linux starts an owned process group. Cleanup is limited to owned processes.
Saved birth/image identity checks support status and stop without trusting a PID
alone. Existing port listeners are preserved.

Native dependencies are explicit. Ambient loader overrides and runtime flag
variables do not silently alter a launch. Models and binaries are never bundled
or deleted by profile management. Local logs may contain paths or prompt text;
share the redacted support export rather than private raw jobs.

See [support](support-matrix.md), [benchmarking](BENCHMARKING.md) and the
[research architecture](research/PRODUCT.md) for each contract's limits.
