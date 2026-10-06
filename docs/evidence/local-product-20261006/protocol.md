# Local product operational smoke registration

This is a new operational load/generation smoke, not a performance experiment
or an extension of any closed stock study. User authorized the product roadmap
on 2026-10-06. Its purpose is proving that portable setup launches and cleans up
a real baseline on this host. It confers no optimization/serving-throughput claim.

Case: pinned local Granite 3.1 1B A400M Q6_K artifact, pristine a7312ae9 CUDA
runtime with its declared CUDA 12.8 dependency directory, Windows/NVIDIA host.
Exact identities and implementation digests are frozen in `registration.json`.

Maximum one new llama-server process. One health/load check (180-second limit),
one raw completion of `Say hello in one short sentence.` with at most eight
generated tokens, seed 42, temperature zero, no prompt cache, context 4096,
F16 K/V, 99 GPU layers, default upstream threads, one slot. No retries or
discarded results. A failed case stays failed; any later case requires a separate
registration with a reason. Hashing/doctor/help launches are not model processes.

Pass requires successful healthy load at requested context, at least one actual
generated token, matching model/runtime files and complete owned-process cleanup.
Retain profile, logs, source/input registration and outcome. Stop on OOM, crash,
identity mismatch, invalid response or timeout. No app closure or unrelated kill.
