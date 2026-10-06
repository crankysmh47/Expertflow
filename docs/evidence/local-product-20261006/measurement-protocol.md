# Fresh local-product diagnostics qualification

Registered after the full historical/model-free suite finished: 920 passed,
seven historical source-environment skips, 1575.67 seconds. No concurrent suite
will run during the performance diagnostic. This is a separate operational
registration; the prior one-process setup smoke remains closed.

Maximum one new model process for `granite-bench-01`, plus one separately named
process for a subsequent user-facing chat/serve operational check. The benchmark
uses the same pinned Granite Q6 artifact, pristine a7312ae9 binaries/dependencies
and requested context 4096 / GPU layers 99 / F16 KV as the setup profile.
Threads remain upstream default. A job manifest freezes source/input identities
before its process starts. Benchmark: one 16-token warmup, three 256-token raw
completions of the shipped training prompt, seed 42, temperature zero, prompt
cache disabled, ignore-EOS enabled, one slot, wall limit 600 seconds.

Retain native responses, stream TTFT, timings, token IDs, process birth/port
ownership, sampled dedicated GPU memory/reserve, source/input manifest, logs and
cleanup receipts. Stop on identity/control drift, invalid tokens/timing, missing
memory telemetry, insufficient reserve, timeout, interruption or crash.
No performance improvement, global optimum or serving-throughput claim follows
from this single diagnostic. A failed result is retained with no retry.

The subsequent chat/serve check uses one owned process and a fixed small chat
request (maximum 32 output tokens), health/status/stop and cleanup. Its exact
launch is registered separately before execution. No additional native call is
authorized by this operational protocol. A tuning job or other model qualification
requires its own fixed manifest; old studies are never reopened.
