# Wider collector implementation gate

Reviewed candidate: `6a0eed493be158e3d6575d5759dbb90cb3e7a146`, based on `5da6786`, on `ef-v2`.
The single [independent review](review.md) found five Important issues. Its resource-classification finding was also treated as material because the registered protocol requires distinct stop reasons. All six were addressed in one fix pass; there are no deferred minors or unresolved material findings. No second code review was used.

The fixes preserve interrupted product prefixes and environmental/resource reasons, anchor collection and sequence costs to recorded attempts, recheck the full runtime inventory at launch, bind executing artifact/entry-point sources to the checkout, and require failed-process launch context and owner-bound teardown evidence. The original Q6 compiler/collector sources, registration, candidates and statistical gates remain unchanged.

Fourteen initial review regressions failed against the unfixed candidate. The first/middle/last product and prelaunch resource prefixes reconstructed after the fixes. The additional raw-auditor controls initially had a fixture checksum setup error, corrected before final verification. All controls passed in the final suite: **886 passed, 7 expected historical external-source skips, 1,526.36 seconds**. Six applicable pinned native source checks and seven rebuilt installed-wheel smoke/source-matching checks passed. The two optional phase-profile checks target another historical native source revision; their attempted scope is retained separately.

Evidence: [final suite](full-suite-final.log), [initial regression failures](review-red.log), [fast guards](review-fast-final.log), [native source checks](native-source-checks.log), [installed package checks](packaging-final.json), [execution ledger](execution-ledger.md).

These are implementation and CPU/artifact checks, not wider model-performance evidence. Collection must freeze this committed record, all executing sources, all four live inputs/roots and the unchanged registration before the first native call. The registered maximum remains 428 calls; no retries, replacements or budget reuse.
