# Temporal layer-24 next-token predictor design

## Status

Approved bounded offline design. P2 remains preserved at ExpertFlow commit
`0974a32` and llama.cpp commit `d8354b17`. The temporal work starts on
`codex/temporal-layer24-predictor`; neither P2 branch is modified, merged, or
pushed.

The core objective remains end-to-end sparse-MoE throughput. This stage exists
to test whether next-token prediction creates enough lead time to make the
already passing P2 asynchronous transfer primitive useful. It is not a new
general-purpose ML track.

## Fixed data contract

- Canonical manifest:
  `C:\models\expertflow\runs\trace-v2-canonical-expanded-84\collection-manifest.json`
- Frozen identities: 60 train / 12 validation / 12 sealed test conversations
- Domains: general instruction, code, math/reasoning, translation/multilingual,
  structured output, and topic shift
- Target: layer 24 only
- Phase: decode only for the initial experiment
- Source: layer-24 selected experts for decode token `t`
- Target: layer-24 selected experts for decode token `t+1`

A pair is valid only when both records have the same conversation, request,
turn, layer, and decode phase; the target has exactly the next `forward_id`,
`token_index`, and causal `hook_order`; and both records contain exactly eight
unique expert IDs in `0..127`. Missing, duplicate, reset, gapped, or ambiguous
sequences fail closed. Prefill is not silently mixed into decode.

The prior adjacent-layer sealed result is provenance only. It is not an input
to temporal policy selection.

## Bounded policies

### T0.0 copy

Rank the current token's eight experts first, preserving their router order,
then append remaining IDs in ascending order.

### T0.1 causal session frequency

Maintain a per-conversation count using only source observations available up
to and including token `t`. Rank by descending count, retain current experts as
the deterministic secondary preference, then expert ID. Reset at every
conversation boundary.

### T0.2 temporal transition table

Fit source-normalized expert transitions on training conversations only:

`expert(t, layer 24) -> expert(t+1, layer 24)`.

For inference, sum the normalized rows for the eight current experts. Use
training target frequency and then expert ID only as deterministic zero-support
completion.

### T0.3 bounded combined scorer

Combine three normalized terms:

- transition score;
- current-set retention indicator;
- causal session-frequency score.

Only these predeclared mixed weights are evaluated:

- `0.50 / 0.25 / 0.25`
- `0.50 / 0.40 / 0.10`
- `0.60 / 0.20 / 0.20`
- `0.70 / 0.20 / 0.10`

No MLP, broad search, test-driven retuning, or adjacent-layer configuration is
allowed.

## Evaluation and selection

Evaluate widths 8, 12, and 16. Report recall, exact-set match,
per-conversation recall, per-domain recall, batch-one prediction latency, and a
conversation-reset 32-slot temporal shadow:

- reactive misses;
- ready misses covered under the all-ready simulation;
- useful and wasted speculative bytes;
- additional evictions;
- eviction regret.

Validation freezes the policy, combined weights when applicable, and width by
this deterministic key:

1. maximize `ready_improvement_over_reactive - eviction_regret`;
2. minimize wasted speculative bytes;
3. maximize recall at the candidate width;
4. minimize p95 prediction latency;
5. prefer narrower width;
6. prefer the simpler policy order copy, session frequency, transition,
   combined.

The selection lock must contain the manifest hash, frozen identities, fixed
policy grid, selected artifact hash, validation metrics hash, and
`test_opened=false`. The test command materializes only the sealed test traces,
verifies the lock hash, evaluates exactly once, then marks the lock opened.

## Stage boundaries

T0 may produce an offline recommendation only. It does not modify llama.cpp.
T1 shadow integration is permitted only after T0 is sealed and documented.
T2 reuses P2 for at most one next-token layer-24 transfer and remains gated on
exactness, a deterministic ready-useful transfer, reduced blocking time, and
stable cleanup. `live_cache_enabled=false` remains the default.

