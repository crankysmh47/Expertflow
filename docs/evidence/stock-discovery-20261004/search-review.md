# Bounded search review and fix pass

Independent read-only review of `ffff2e2` found no Critical findings, two
Important findings and one Minor finding. No native search had started.

1. Important: reconstruction accepted nonempty provider labels and matching
identity digests without validating the complete reviewed operation-path proof.
It also accepted a self-selected nonempty source-file subset. Five regression
cases reproduced acceptance of changed provider/protocol/upstream/source-object
claims and missing source-file coverage. Reconstruction now decodes identity-
bound model/runtime/hardware snapshots, re-attests through trusted registered
provider code, compares the complete attestation and requires the complete
expected source-file set. Test-only providers use explicit trusted injection;
production CLI resolves reviewed builtins.

2. Important: default CLI coverage could include threads8, yielding six
configurations and38 processes. The regression reproduced that result.
The default now requires the current8/16core,12thread incumbent topology, the
threads8 exclusion, four configurations and32 maximum processes. Generic
alternate spaces require explicit coverage/exclusion/budget configuration and
are not authorized for this current native experiment.

3. Minor: screening had no uncertainty report. It now independently reconstructs
three block ratios, range and coefficient of variation, labeled as descriptive
screening variation rather than confirmation evidence. A regression observed
the missing field before implementation.

Seven provenance/default-space/explicit-space regression tests passed after
the fixes; the descriptive-variation regression and full suite are included
in final verification:596 passed,7 skipped in268.29s. Compileall and diff checks
also passed. Exactly one fresh review/fix pass is used before native
retention. The pre-review `search-preview.json` contains zero native samples and
must not be treated as a retained experiment or used for execution.
