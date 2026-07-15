# Next-Layer Shadow Predictor Design

## Scope

Build one offline, one-layer-ahead predictor from the accepted canonical Observer v1 pilot. It consumes layer `L` routing for a token and predicts the eight experts selected by the true router at the immediately following MoE layer. It never modifies llama.cpp, residency, transfers, scheduling, or cache decisions.

The pilot split is immutable: seven train, four validation, three test conversations from `trace_v2_canonical_segmented_pilot`. Test remains sealed until the predictor family, features, seed, and candidate widths are frozen from train/validation evidence. Results are small-pilot feasibility evidence only.

## Deterministic samples

Join strict canonical events by conversation ID, forward ID, token index and token ID, then adjacent observed MoE layer. Reject missing layers, duplicate keys, ambiguous joins, non-adjacent targets, inconsistent phase/token metadata, and split overlap. Each sample stores provenance, phase, source/target layers, ordered source/target IDs, source weights when present, a 128-vector (weights or binary), and a previous-token target-layer vector only when the same-conversation causal join is unique.

## Models and selection

Run B0 copy, B1 training-only target-layer frequency, B2 training-only transition counts, then one deterministic CPU linear multilabel model. After B3 succeeds, run at most one fixed shared CPU PyTorch MLP. No search is permitted. Validation chooses the practical family; B2 remains preferred when it matches or beats learned alternatives. Evaluate the frozen choice once on test at top 8, 12, and 16.

## Metrics and artifacts

Report recall@8/12/16, mean overlap@8, exact-set@8, per-layer, phase, and conversation results, measured batch-1 CPU prediction latency, parameter count, and serialized size. Emit one metrics JSON, one concise report, serialized artifacts, and an append-only command/decision ledger.

## Shadow simulation

For each conversation, compare identical 32-slot per-layer reactive LRU and speculative shadow LRU. Prediction after layer `L` may insert candidates for `L+1` before its demand. Account separately for reactive hits, useful ready predictions, uncovered misses, wasted predictions, extra evictions, eviction regret, and useful/wasted packed bytes. Without runtime timing, predictions are modeled ready and never labeled late. All outcomes are simulated shadow evidence, not latency overlap or speedup.

## Independent tracks

Expanded collection freezes 84 conversations before execution: 60/12/12, with 10/2/2 in each of six fixed domains, prompt-template hashes, exact/normalized/near-duplicate checks, canonical runtime hashes, checkpointed shards, and sealed test. C5 continues from the committed exact C4 branch independently. Neither track consumes predictor test outcomes or modifies this branch.
