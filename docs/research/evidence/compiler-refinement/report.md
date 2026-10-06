# Compiler measurement refinement and eight-thread result

The measurement gate passed. The single optimization hypothesis failed.
Keep the twelve-thread stock setting for this workload. No validated product
execution plan was published, and the original Phase 3 **VALIDATION-STOP** remains
unchanged. Both experiments used twenty retained cold processes without retries
or discarded samples, on `ef-v2`.

## Results

| Experiment | Control mean TPS | Candidate mean TPS | Paired geometric change | Paired 95% interval | Verdict |
| --- | ---: | ---: | ---: | --- | --- |
| Direct vs sealed, same twelve-thread settings | 22.9747 | 23.0686 | +0.4009% | [-0.5970%, +1.5072%] | PASS-MEASUREMENT |
| Twelve vs eight threads | 23.6880 | 21.9976 | -7.1788% | [-9.1042%, -4.7441%] | VALIDATION-STOP |

For A/A, the paired 90% interval was [-0.4486%, +1.3379%], wholly inside the
predeclared +/-2% equivalence margin. The one-sided 95% lower bound was -0.4486%,
above the -2% non-inferiority threshold. Control/candidate CVs were 2.4193% and
2.7418%, below 10%. This demonstrates repeatability for this experiment, not an
ExpertFlow speedup, and does not retroactively change the old absolute-2% gate.

Eight threads missed the >=5% gain target and instead showed a paired slowdown.
Control/candidate CVs were 4.6889% and 5.8250%. Exact prompt/generated native
tokens, mandatory owned memory/reserve checks, unique process identities, and
cleanup passed across all forty runs. The negative result ends this hypothesis;
no thread-count sweep or additional architecture track was started.

## Frozen inputs and measured pairs

Both experiments used the pinned Q6 model, pristine runtime, RTX 5060 Ti,
context4096, predict512, seed42, F16 KV, graphs on, ngl99 and CPU-MoE. In the
second experiment only native decode/batch threads changed from twelve to eight;
the two workload identities are explicitly distinct. Ten balanced pairs used
seed20261003. Statistics resampled whole paired log ratios 10,000 times with the
same fixed seed. Source commits were `1192856` for A/A and `922e8b5` for threads.
The frozen source hashes and complete settings are in each report.

| Pair | Twelve-thread TPS | Eight-thread TPS |
| --- | ---: | ---: |
| 1 | 24.2001 | 24.7879 |
| 2 | 24.0915 | 22.3344 |
| 3 | 24.2787 | 22.5296 |
| 4 | 24.3798 | 22.4261 |
| 5 | 24.4241 | 22.4930 |
| 6 | 24.3668 | 21.8220 |
| 7 | 24.2400 | 21.4131 |
| 8 | 23.5024 | 21.4473 |
| 9 | 22.2430 | 20.6225 |
| 10 | 21.1540 | 20.1004 |

Both arms drifted downward late in the second experiment. Balanced paired
ordering reduces order confounding but does not remove temporal dependence;
ten-pair percentile-bootstrap intervals are estimates, not universal guarantees.
The first eight-thread run was favorable and was retained alongside every slower
run. Means from different experiments are not matched performance comparisons.

## Diagnostics and numerical scope

The A/A runs averaged 7.497 busy owned CPU cores and 16.864% GPU utilization,
supporting a bounded CPU-thread hypothesis. In the second experiment twelve
threads averaged 7.940 busy owned CPU cores versus 5.219 for eight. GPU utilization
averaged 18.391% versus 18.118%; GPU temperature averaged 46.464C versus 46.470C.
CPU performance counters averaged 134.376% versus 135.554% of nominal. These are
diagnostics, not proof of the slowdown's mechanism. CPU temperature and the
actual native offloaded-layer count were unavailable and remain explicitly
unknown. Owned allocation bytes do not establish layer deduplication.

The inspected CPU backend has no changes from pinned upstream. Standard x86 Q6
expert kernels partition independent outputs and complete quantization blocks;
the inspected activation/scale kernels partition whole rows. This supported the
experiment without changing kernels or reduction paths. It does not establish a
universal bitwise guarantee for arbitrary operations or models, and finite token
parity alone is not such a proof. Historical synchronized profiles were used only
as mechanism evidence; retained timing runs had no synchronized profiling.

Native decode TPS, completion wall time, startup/health, tokenization, teardown,
and total run wall time are recorded separately. A/A also isolates candidate
preparation time. Missing optional sensors do not weaken memory checks.

## Evidence and reproduction

- [A/A report](aa-report.json) and [thread report](threads8-report.json) retain
  every paired rate, diagnostic summary, frozen schedule and artifact identity.
- [Verification](verification.json) records commands, exit codes, log/database
  hashes and limitations. [Measurement manifest](measurement-manifest.json)
  binds forty independent runs to their raw evidence.
- [Kernel verification](kernel-source-verification.json) records inspected
  source hashes and the empty CPU-backend diff from upstream.
- Protocols: [measurement refinement](../../superpowers/specs/2026-10-03-compiler-measurement-refinement.md)
  and [eight-thread experiment](../../superpowers/specs/2026-10-03-compiler-eight-thread-experiment.md).

Raw files remain under `C:/models/expertflow/runs/compiler-refinement-20261003`
and `C:/models/expertflow/runs/compiler-threads8-20261003`; reports/manifests are
committed here, not the 22.9GB weights. Use the commands in verification.json with
fresh output/database paths; existing outputs deliberately reject retries.

CPU verification passed 502 tests before A/A and 505 before the thread comparison,
with six external-source skips each time. The applicable six pinned native source
contracts passed during live input preflight. Compileall and Git whitespace checks
passed. One fresh final review found two Important sensor/provenance issues and
one Minor timing issue; all were addressed before A/A. Original raw evidence and
the A/A prerequisite database/report remained unchanged.

Recommendation: retain the compiler/evidence infrastructure and this repeatable
measurement protocol, but do not claim a runtime speedup. The next performance
proposal needs a new concrete CPU expert-kernel or transfer hypothesis with its
own numerical contract and bounded comparison. Reactive caching remains a no-go;
historical static-placement speed remains exact-ineligible. No merge or push was
performed.
