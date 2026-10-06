# Wider stock execution ledger

Plan: [implementation](../../superpowers/plans/2026-10-05-wider-stock-collector.md).
Protocol: [immutable registration](../../superpowers/specs/2026-10-05-stock-coverage.md).
Base: `5da6786`; implementation candidate: `6a0eed4` on `ef-v2`.

## Decisions

- Work in place on the active branch per the established user preference. Cost if wrong: reversible implementation commits are on ef-v2.
- The existing registration and explicit proceed instruction authorize implementation, review, freeze and the fixed native sequence. Cost if wrong: up to428 bounded calls may be spent.
- Exclude the mutable implementation checklist from the scientific source map. Bind the immutable registration/specification, every executing compiler/stock module, collectors, controls and the independent raw auditor. Cost if wrong: planning-only text is not bound to native launch hashes.

## Verification in progress

- Baseline public CLI/registration:33 passed.
- Initial wider API controls: missing-module RED (5failed,9errors).
- Owned-spawn wall-cap control: RED (no ResourceStop), then GREEN.
- Installed-adapter source binding: missing-guard RED, then GREEN.
- Reconstruction budget must remove an accepted-gain claim: missing-helper RED, then GREEN.
- Public coverage reader aliases: AttributeError RED, then37 current fast controls passed after the fix.
- Broad focused run:62 passed; independent auditor exposed a missing paired schedule in the synthetic registration fixture (the real immutable registration includes it). Fixture corrected.
- Candidate full suite passed866 tests with7 expected historical-source skips in1416.77s. The single independent review found five Important issues and one classification issue regraded to Important under the registered stop-reason requirement; no deferred minors.
- Fourteen review regressions failed before the one material fix pass. The interrupted first/middle/last product, prelaunch resource prefix, chronology and public launch/ownership controls passed afterwards. Three additional raw-auditor controls exposed a fixture checksum error, corrected before the final suite; 12 fast controls then passed. Final full suite: 886 passed, 7 expected historical-source skips, 1,526.36 seconds.
- Rebuilt installed wheel passed7 routing/help/source-matching smoke checks; six applicable pinned native source checks passed. Two historical phase-profile checks target a different external source revision and are retained as an environment-scope attempt, not current runtime failures.
- Current host matches the registration; every wider root remains unused. No native freeze or wider native call has started.
- Original study integrity:41 frozen files,6 history pins,148 native starts; original report SHA remains464defd327d43c4f510f3768266d1dacaefc77d25c239afefef5cc4226c1d0c5.
