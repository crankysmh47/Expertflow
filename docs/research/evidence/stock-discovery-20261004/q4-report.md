# Q4 stock method evidence

Pinned model SHA: 4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5.
Reviewed implementation: 53c346f. Actual reference freeze: 612a0cd51257b762954cfcaf44bec9200b32afdb9d964a659243af64c1be6e5b.

Ten fresh owned stock processes completed with exact prompt/generated tokens,
memory/reserve and cleanup guards. Reference mean36.576202420444865 TPS,
sample CV0.7263824750496013%, below the predeclared10% gate. Independent
artifact/statistical/source audit returned VERIFIED-STOCK-REFERENCE; actual
CLI validation returned VALIDATED-STOCK-REFERENCE. Diagnostic plan SHA:
3872c15dfcc0c67de9d9bd59ceb432fc95500697defdfdffe49f455ba1ecc55e.

The distinct Q4 artifact is not a quality-preserving Q6 speedup.

Fresh paired-stock-product-v1 completed exactly20 native processes with
PASS-STOCK-FALLBACK. Direct36.609465073120944 TPS, sealed36.63854622634969 TPS;
paired geometric change+0.0800610602140323%, CI90[-0.13861372774841826%,
+0.2861447174318017%], CI95[-0.1830811209505298%,+0.3216327650150481%].
CVs0.5317535176497185%/0.38034144111314205%. Independent audit returned
VERIFIED-PASS, rechecking all20 raw artifact sets, unique owners, exact tokens,
memory/reserve, cleanup, frozen host/source and unchanged source plan/database.
Actual CLI returned VALIDATED-STOCK-FALLBACK, plan SHA
277df3aeffb138ad14b2768880a5946db0d27c772b65f9bfaef4bc5a4ec03a0f.

Product raw evidence/database:
C:/models/expertflow/runs/compiler-q4-stock-product-20261004.

## Bounded search: retain incumbent

The search at f02ccbd completed exactly18 native processes, three full screening
blocks across threads8/12/16 and CUDA graphs on/off. The already accepted
incumbent12/on ranked first. Frozen rules consume no self-confirmation, so the
optional20 processes were not launched. RECOMMENDED-INCUMBENT preserves the
accepted plan hash277df3ae. Independent audit returned VERIFIED-STOCK-SEARCH;
actual CLI returned VALIDATED-STOCK-RECOMMENDATION with the same full plan hash.

| Threads | CUDA graphs | Screening mean TPS | Geometric change relative to incumbent |
| --- | --- | --- | --- |
| 12 | on | 36.686660 | 0.000% |
| 8 | on | 36.313004 | -1.019% |
| 16 | on | 35.247414 | -3.923% |
| 12 | off | 33.103872 | -9.766% |
| 8 | off | 33.014120 | -10.012% |
| 16 | off | 31.782298 | -13.370% |

These three-block rates/ratios are descriptive screening results, not confirmed
pairwise speedup estimates or confidence intervals. Complete block-ratio ranges
and CVs remain in q4-search-verification.json. All18 retained processes passed
token, owned memory/reserve, cleanup and complete launch/identity guards. No
retries, discarded samples, alternative finalist or gate changes occurred.

Coverage: six configurations, no thread exclusions; counts1-7,9-11,13-15 and
other controls remain untested. Offload, placement, batching, approximate KV,
CPU SIMD and quantization changes require separate numerical/quality contracts.
This result establishes the strongest validated configuration in the declared
space, not a global optimum. Q4's rates do not establish Q6 quality preservation
or recovery of its historical server throughput.

Total Q4 consumption48 of maximum68: reference10, product20, screening18,
confirmation0. Search evidence/database:
C:/models/expertflow/runs/compiler-q4-stock-search-20261004. Published repository
metadata is q4-recommended/, q4-search-verification.json and this report.

Raw evidence/database: C:/models/expertflow/runs/compiler-q4-stock-reference-20261004.
Repository metadata: q4-reference-verification.json and q4-reference-diagnostic/.
