# CPU expert-row prefetch experiment: not accepted

The single next-row Q6_K cache hint did not meet the frozen performance gate.
Formal result: **INCONCLUSIVE**. Paired geometric change was **-0.5466%**, with
two-sided 95% bootstrap interval **[-1.3692%, +0.1776%]**. The interval includes
zero, so this does not establish a regression; it also excludes the required
5% gain. Retain the pristine twelve-thread runtime. End this hypothesis without
distance tuning, retries, additional samples or product-plan publication.

| Measure | Pristine stock | One-row hint |
|---|---:|---:|
| Retained cold processes | 10 | 10 |
| Mean native decode TPS | 22.6697 | 22.5458 |
| Decode TPS CV | 1.4852% | 1.4752% |
| Exact prompt/generated tokens | pass | pass |
| Owned GPU memory/reserve and cleanup | pass | pass |

Ten balanced pairs used five of each order, seed 20261003, and 10,000 whole-pair
bootstrap draws of mean log TPS ratios. Point gain must be at least 5%, the
two-sided 95% lower bound must exceed zero, and both CVs must be at most 10%.
Both performance requirements failed; variability and correctness passed.
The twenty processes are unique and independent of the accepted A/A runs.
One independently verified historical A/A reference record was imported into
the fresh database for exact-token comparison; it is excluded from these
twenty runs and from the statistics. No native attempt was retried or discarded.

## Implementation and numerical gate

Native source is private commit `2a6d7c0ab9463feab827c193cee817d1297e23a7`, based
on pristine `a7312ae94f801fc9c6786dc56e38df57b964f697`. The entire source diff is
five lines in the CPU expert MUL_MAT_ID caller: for Q6_K only, hint the first
cache line of the next weight row, guarded by `ir0 + 1 < ir0_end`. It remains
inside the same expert and invocation chunk. Dot-product arguments, reduction
width/order, activation conversion, output stores and iteration order are
unchanged. This is a cache hint, not a change to the numerical path.

The new CPU backend was built with the actual pristine stock compile commands:
MSVC 14.39, Release, AVX2, native OFF, identical CPU source set and definitions.
The diagnostic fork's AVX512 flags were not used. CPU exports and imports match
stock. The complete normalized generic and x86 quantization/dot-product object
disassemblies match stock. Forty-eight deterministic Q6 expert graph fixtures
match bitwise at 1/12 threads, 1/3 tokens, widths 256/768 and output row counts
1/15/16/17/33/257, covering expert boundaries and row/chunk tails. Activation
conversion remains F32 to Q8_K. The combined 43,392 output bytes have SHA-256
`1c11eb2a816e37c460a8a644f7151ff7fb229738f75333b3cb9139644c86c8eb`.
Compiled hint presence was checked RED before implementation and GREEN after.

The [runtime manifest](../../../configs/compiler/runtime-cpu-prefetch.json)
pins pristine launchers, scheduler/host libraries, CUDA backend and all other
dependencies. Only `ggml-cpu.dll` differs. Candidate DLL SHA-256 is
`ae299da337e7990d69a703951d1935bfab819cb8ec5c4ca00b049843e8bfb306`.
The isolated native source/build and the original pinned builds remain available;
no original pinned binary was replaced. Profiling synchronization was disabled
for all acceptance timings. Launch settings remain context4096/predict512/
seed42/F16/graphs-on/ngl99/CPU-MoE/twelve threads on the pinned Q6 model.

## Verification and boundaries

- Full CPU suite: **519 passed, 7 source-contract skips**, 44.31 seconds.
- Native numerical/ABI/compile gates passed; the static-island source contracts
  do not apply to pristine CPU-only source. Clean source revision, exact patch,
  compile commands, machine code and DLL identities were verified directly.
- Every retained measurement was independently reconstructed from EvidenceStore
  artifacts after collection; pair/order, candidates/settings, exact tokens,
  unique owned process identities, memory/reserve, clean child exit and cleanup
  passed. All raw hashes and paired statistics were reverified.
- Frozen source/protocol/runtime pins and original A/A database/report remained
  unchanged. No native server remained running after completion.
- One independent implementation review found no material issue. Deferred minor:
  the reproduction hint helper checks caller-name and prefetch presence across
  the whole object independently. Future reproduction should scope that search
  to the caller block. The exact five-line diff, stock absence/new presence,
  and unchanged dot-product disassembly support the current recorded gate.

Preflight initially rejected the patch filename before database/freeze/native
launch because it lacked the manifest's `0001-` prefix. That configuration error
was corrected and the runtime binding revalidated before collection. Its failure
log is retained; it did not consume a native attempt or trigger resampling.

The [phase profile](../compiler-phase-profile-20261004/report.md) remains useful:
CPU expert compute is the largest synchronized decode bucket. This experiment
does not establish cache misses or a memory-bandwidth bottleneck, and does not
invalidate that profiling result. The original compiler VALIDATION-STOP, reactive
cache no-go, eight-thread no-go and numerical-placement exclusions are unchanged.
There is no accepted new runtime or published execution plan.

## Reproduction artifacts

- [Frozen protocol](frozen-protocol.json), [full report](report.json),
  [verification](verification.json), [measurement manifest](measurement-manifest.json).
- [Native patch](0001-cpu-expert-row-prefetch.patch),
  [numerical gate](numerical-gate.json), [runtime identity](runtime-identity.json).
- [Measurement probe](probe.py), [build helper](reproduction/build.cmd),
  [fixture build](reproduction/fixture.cmd), [numerical comparison](reproduction/numerical_gate.py),
  [post-collection verifier](reproduction/verify_results.py).
- Raw database, launches/responses/logs and numerical disassemblies/fixtures:
  `C:/models/expertflow/runs/compiler-cpu-prefetch-20261004`.
  The stored probe intentionally refuses restarting that directory/database.

The local evidence checker additionally rejected the partial RUNNING report
before collection finished. Only the completed twenty-run report was accepted
for outcome reconstruction. The evidence and native checkout are retained so
their pinned paths remain verifiable.
