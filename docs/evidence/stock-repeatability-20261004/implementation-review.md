# Fresh implementation review

Read-only reviewer `repeatability_review` covered `a669f71..4a3d7ef` against the
bounded repeatability protocol. No native inference, checkout edits or additional
reviewer were used. No Critical or Minor findings; five Important findings:

1. Incomplete outer source map and no final-call source guard/reconstruction.
2. Failed-attempt pacing bypass and missing journal/phase/kernel-owner binding.
3. Incomplete negative/partial reconstruction and arbitrary success statuses.
4. Main consumer reconstruction missing accepted-reference token parity.
5. Missing encompassing collector elapsed endpoints/cost.

Verdict: fix before native freeze. Executor reproduced and fixed all five in one
regression-test pass. Thirty focused tests passed; the final persisted-PASS
regression then failed as expected and passed after keeping the report pending
until reconstruction. The full post-review suite passed 797 tests with seven
optional historical-source modules skipped; six pinned native checks passed
separately. No second reviewer or native inference was used during verification.
Reviewing code does not establish repeatability or any native performance result.

| Finding | Regression coverage |
| --- | --- |
| Outer freeze / final call / final success | `outer_freeze_includes_all_later_paired_collector_sources`, `last_native_return_cannot_escape_original_source_freeze`, `collector_persists_full_elapsed_cost_and_reconstructs_before_success` |
| Pacing / journal / kernel owner | `attempt_audit_counts_observed_failed_processes`, `terminal_state_journal_and_elapsed_cost_cannot_be_rewritten`, `kernel_owner_reuse_is_rejected_even_with_different_run_uuid` |
| Negative / partial / state | `negative_block_reconstructs_full_evidence_and_prevents_promotion`, `partial_environment_stop_binds_prefix_controls_outcomes_and_journal`, unknown-status case |
| Consumer reference parity | `consumer_reconstruction_requires_accepted_reference_token_parity` |
| Collector cost | elapsed-cost mutation and collector endpoint arithmetic cases |

An additional valid-negative reconstruction error omitted measurement IDs; it
was fixed before reproducing the deeper statistic/outcome/promotion bypasses.
Initial regression setup had an eligibility fixture error, which was corrected;
the corrected RED results precede the production fixes. No ordinary native data
was collected, discarded or altered during these implementation tests.

The reviewer declined actual repeatability, gains, thermal causation and live
sensor availability, which require the registered native experiment. Deployment
throughput and broader CLI readiness remain outside the protocol. Full-suite
readiness is the executor's post-fix verification responsibility.
