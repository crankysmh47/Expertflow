# Q6 predictive follow-up scorecard

| Gate | Requirement | Result | Status |
|---|---:|---:|---|
| Static-state reconstruction | exact | commits, binaries, command and 12 static banks verified | PASS |
| Static MMLU | decline no worse than -1 point | 49/100 OFF, 50/100 ON (+1 point) | SUPPORTIVE |
| Changed-answer determinism | exact repeat | 14/14 exact | PASS |
| Routing coverage | four workloads, all layers | 127 shards, 655,740 events, 30 layers | PASS |
| Cache memory opportunity | >=500 MiB | 654.16 MiB at 8 static + 4 cached/96 | PASS |
| Cache TPS retention | >=26.72 TPS | 26.35 projected TPS | **FAIL** |
| Added-layer alternative | >28.13 TPS | best optimistic 25.83 TPS | **FAIL** |
| Reactive implementation | admitted only after simulation pass | not admitted | NOT RUN |
| Predictor integration | admitted only after reactive utility | not admitted | NOT RUN |
| Strict frozen PPL | upper bound <=+1% | +2.25% upper bound | FAIL, UNCHANGED |

Terminal verdict: `NO CACHE OPPORTUNITY`.
