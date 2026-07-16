# Expanded Next-Layer Predictor Design

## Objective

Finalize the bounded offline next-layer expert predictor on the canonical
84-conversation corpus. Preserve the frozen 60 train / 12 validation / 12 test
conversation split and open the test split exactly once after writing an
immutable validation selection lock.

This work produces offline feasibility evidence only. It does not modify
llama.cpp, move weights, integrate a live predictor, or support a throughput
claim.

## Data and leakage contract

- Input manifest:
  `C:\models\expertflow\runs\trace-v2-canonical-expanded-84\collection-manifest.json`.
- Require canonical runtime `expertflow-canonical-observer-v1` and trace
  generation `trace_v2_canonical_segmented`.
- Require exactly 60 train, 12 validation, and 12 test conversations.
- Require exactly 10/2/2 conversations per domain across general instruction,
  code, math/reasoning, translation/multilingual, structured output, and topic
  shift.
- Preserve conversation-level isolation and strict
  conversation/forward/token adjacent-layer joins.
- Verify unique conversation IDs and prompt hashes. Preserve the collection
  manifest hash and deduplication provenance in the selection lock.
- Training reads train only. Selection reads validation only. Test is evaluated
  by one guarded command after the lock exists.

## Bounded model and policy selection

Retain the existing B0 copy, B1 target-layer frequency, B2 transition, B3 fixed
linear, and B4 fixed shared-MLP implementations. Do not tune B3 or B4.

The only bounded B2 search is the Cartesian product of:

- transition weighting: raw count or source-normalized conditional score;
- phase handling: pooled or separate prefill/decode transition tables.

Validation also evaluates candidate widths 8, 12, and 16 and two admission
rules:

- `all_ranked`: admit every missing expert in the selected prefix;
- `observed_support`: admit a missing expert only when B2 assigns positive
  transition support before deterministic fallback completion.

Select the B2 configuration and admission/width combination by:

1. maximizing reduction in simulated 32-slot blocking misses after eviction
   regret;
2. minimizing wasted predicted bytes;
3. maximizing recall at the selected width;
4. preferring the simpler raw-count, pooled, `observed_support`, narrower
   configuration on ties.

B2 remains selected unless unchanged B3 or B4 exceeds the best B2
recall@8 by at least two percentage points and its p95 batch-one CPU latency is
no more than twice B2 p95. This is a fixed decision rule, not a search.

## Outputs

Write under `C:\models\expertflow\runs\expanded-predictor-final`:

- model artifacts with byte counts and SHA-256 hashes;
- validation metrics for all unchanged B0-B4 models and four bounded B2
  configurations;
- an immutable selection lock initially recording `test_opened=false`;
- exactly one sealed-test metrics artifact;
- append-only command and decision ledger;
- artifact index and verification summary.

Report recall@8/12/16, overlap@8, exact-set match, per-layer, per-phase,
per-conversation, and per-domain results; batch-one CPU p50/p95 latency;
parameter count and artifact size; and 32-slot shadow useful/wasted bytes,
ready gains, uncovered misses, speculative evictions, and eviction regret.

## Failure behavior

Fail closed on split drift, duplicate conversations or prompt hashes, invalid
trace provenance, incomplete layer sequences, ambiguous joins, selection-lock
hash mismatch, a second test invocation, or an artifact hash mismatch.

## Claims

Label routing metrics as measured offline prediction results and cache accounting
as simulated shadow results. State that the timing model assumes predictions are
ready. Make no runtime integration, transfer-overlap, speedup, or broad
generalization claim.
