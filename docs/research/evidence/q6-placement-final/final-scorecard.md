# Final Q6 scorecard

| Gate | Requirement | Result | Status |
|---|---:|---:|---|
| Mean decode TPS | >= 23.5 | 28.13 | PASS |
| Strongest stock | > 22.967 | +22.48% | PASS |
| Paired performance CI | excludes zero | +23.46% to +29.83% | PASS |
| VRAM reserve | >= 256 MiB | 5,344.20 MiB | PASS |
| Determinism | one output per mode | one hash per mode | PASS |
| PPL point change | <= +1.0% | -2.92% | PASS |
| PPL 95% upper bound | <= +1.0% | +2.25% | **FAIL** |
| MMLU decline | <= 1 point | not run after terminal PPL failure | NOT RUN |

The performance configuration is not packaged as a product because the frozen mandatory PPL confidence-bound gate failed. Final verdict: `QUALITY STOP`.
