# Expanded canonical 84-conversation collection

Status: **collection complete and strictly validated**. This is dataset evidence, not predictor evaluation or a runtime-speed claim.

## Frozen contract

- Dataset: `trace-v2-canonical-expanded-84-v1`
- Conversations: 84 public synthetic prompts
- Split: 60 train / 12 validation / 12 test
- Domains: general instruction, code, math/reasoning, translation/multilingual, structured output, and topic shift
- Per-domain split: exactly 10 train / 2 validation / 2 test
- Frozen manifest SHA-256: `970681c0126cc5400524e5b4328f0ecaf87c72d346a7fd99896a44224720dbab`

Conversation IDs and splits were assigned before collection. Each prompt has a unique template ID, exact UTF-8 SHA-256, normalized NFKC/casefold/token SHA-256, and split-qualified task family. Before and after collection, all 84 exact hashes and all 84 normalized hashes were unique. Pairwise character similarity at the predeclared review threshold of 0.72 flagged zero pairs. No conversation moved between splits.

## Runtime and collection configuration

- Runtime: `expertflow-canonical-observer-v1`
- Runtime SHA-256: `7ea12e0c44258bb2d75f99b7e180e7ef7cb0c7d3e285ca231fe26ad8c8c4932c`
- Model SHA-256: `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`
- Trace generation: `trace_v2_canonical_segmented`
- Cache: disabled
- Decode: greedy, 32 generated tokens, 10 GPU layers, 12 threads, batch 1, microbatch 1
- Prompt template: the pinned Gemma user/model turn template recorded in the machine manifest

The test split remained sealed through all 60 training and 12 validation collections and their complete artifact revalidation. It was opened for collection only after predictor commit `2300c6c` froze `b2_transition`, its feature contract, candidate widths 8/12/16, and seed 20260716. The selection-lock SHA-256 is `19808783fb0fd4e8a7499165e865cff37e07b36b8f1ea12c4aea19793eaeb26d`. Test collection did not train, select, or evaluate a predictor.

## Results

| Split | Conversations | Router events |
|---|---:|---:|
| Train | 60 | 143,520 |
| Validation | 12 | 28,350 |
| Test | 12 | 29,010 |
| **Total** | **84** | **200,880** |

Every shard passed immediately and again in the final whole-corpus audit. Validation requires a matching conversation ID, eight unique expert IDs in `[0, 128)`, strictly increasing hook order, causal observer timestamps, and every forward containing the exact ordered MoE layer sequence 0 through 29. Final artifact and trace errors: zero. Native collection failures: zero. No model process remained after collection.

Measured native-process time totaled 669.995 seconds. Per-split median durations were 7.539 seconds for train, 7.913 seconds for validation, and 10.228 seconds for test. These are collection diagnostics, not comparative performance measurements.

## Reproduction evidence

- External root: `C:\models\expertflow\runs\trace-v2-canonical-expanded-84`
- Checkpoint manifest: `collection-manifest.json`, 127,534 bytes, SHA-256 `b59ab020843a98121fca1f60227d3bf7272fe2a0e8f4b10c53add7182f7437fa`
- Append-only ledger: `command-ledger.jsonl`, 144,080 bytes, 84 records, SHA-256 `c201915e35cc7754d42ae616c08dbc7ced42e353ba39aed9801feceb9e70b11f`
- Raw and canonical trace/token/measurement artifacts: 132,503,298 recorded bytes in aggregate

Bulky run artifacts are intentionally excluded from Git. The checked-in manifest, validator, generator, collection runner, tests, evidence note, and ledger narrative are committed; predictor retraining/evaluation remains a separate track.
