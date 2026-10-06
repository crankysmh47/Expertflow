# Stock research and product follow-through implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution. Preserve the user's instruction to continue down either result path.

**Goal:** Close one candidate's mechanism gate and finish the appropriate next deliverable.

**Architecture:** A source-only gate determines whether to design a new native study or continue the bounded stock prototype. A standalone checkout helper delegates all fresh acceptance decisions to the existing public CLI and preserves measured sources.

**Tech stack:** Python standard library, pytest, pinned Git objects, existing public stock validators.

**Spec:** `docs/superpowers/specs/2026-10-06-stock-research-followthrough.md`

## Global constraints

- Zero new model processes; no changes to the 63 wider or 41 original frozen files.
- No closed budget reuse, new controls, plan publication, merge, push or deployment.
- Archived result display never implies fresh reconstruction or new timing evidence.
- Verification delegates only to the two existing public read-only validators.
- Missing/mismatched evidence fails closed and preserves diagnostics.

## Review focus

Altered catalogue/report/source bytes must stop; successful outer status must not
hide failed utility; archived display must not claim fresh validation; validation
failure or a changed run-start digest must invalidate success; output paths must
not write into a study, including through a symlink.

### Task 1: Candidate mechanism gate

- [x] Bind exact source/default/support-resolution blobs, existing launch hashes and conditions.
- [x] Publish source-feasibility result and zero-call integrity check under `docs/evidence/stock-followthrough-20261006/`.
- [x] Follow pass into quality protocol design; follow fail into Task 2 without a replacement candidate.

### Task 2: Readable status and delegated verification

Files: create `scripts/stock_product_status.py`, `tests/test_stock_product_status.py`, and a pinned evidence catalogue in the new evidence directory.

Interfaces: `snapshot(project: Path, catalogue: Path) -> dict`, `verify(project: Path, catalogue: Path, output: Path) -> dict`, `main(argv=None) -> int`.

- [x] Write failing tests for five separate outcomes, finite intervals, native counts, immutable report bindings and archived-only semantics.
- [x] Run `uv run --no-sync pytest -q tests/test_stock_product_status.py`; confirm missing implementation fails.
- [x] Implement `snapshot` and `status` with only pinned report reads and concise human/JSON output.
- [x] Write failing tests for unsuccessful public verification, wrong status, native-start mutation, protected output path and no alternate action.
- [x] Implement `verify` with an allowlist of read-only commands, fresh retained logs and before/after integrity checks.
- [x] Run focused tests plus existing public CLI/reader guards; commit the helper after review/fixes.

### Task 3: Qualification and next product gate

- [ ] Run one fresh live read-only qualification of both studies; preserve costs, logs and before/after digests, with zero extra native calls.
- [x] Obtain one independent review; fix material findings with reproducing controls.
- [ ] Update README, method, status, TODO and project log with result, scope and path continuation.
- [x] Publish an executable independent-user usability procedure and required evidence; no fabricated user results.
- [ ] Verify links, unchanged measured/history/user files and staged diff; commit on `ef-v2` and leave unrelated files untouched.

## Execution ledger

Ruling: continue inline under the user's explicit instruction to proceed on both
pass and fail paths, without another approval round. Cost if wrong: routine
implementation choices are agent-selected; all scientific admission and native
budget gates remain explicit and preserved.

Task 1: source mechanism gate failed for explicit FA enable; all 492 prior starts,
launch records and 41/63 source bindings preserved. Zero model processes.
Task 2: missing-module RED, identity-binding controls RED then GREEN; one independent
review found two Important defects and one presentation omission. All four added
controls failed before the one fix pass, then passed. Final focused verification:
54 tests passed in 125.98 seconds (28 helper controls plus 26 public CLI/readers).
Status also passed from outside the checkout. Existing compiler/provider sources
were not edited; no new full native/scientific qualification is inferred.
