# Option 1 Q1 Quality Gate

## Decision

**OPTION 1 Q1 STOP.** The static layer-0 Q4 CUDA island is physically correct and deterministic, but it missed the predeclared WikiText-2 quality gate. No reactive cache work is authorized from this result.

## Measured result

| Metric | Feature off | Feature on | Gate | Result |
|---|---:|---:|---:|---|
| Final WikiText-2 perplexity | 1176.7406 | 1183.6406 | relative increase <= 0.500% | FAIL |
| Relative change | - | +0.586365% | <= +0.500% | FAIL |
| Absolute change | - | +6.9000 | descriptive | - |
| Candidate ceiling implied by gate | - | 1182.624303 | <= 1182.624303 | FAIL |

The runs used the frozen manifest `294ccc4e6ef9da9d80ee15ac89d989d6d1eaa44e28bc0a043ab219d436a18719`, four complete 2,048-token chunks, the same Q4 model and binary, `-ngl 10`, context 2,048, greedy settings, and the same batch/thread configuration. The only intended mode change was `LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER=0`.

## What passed before the stop

- Feature-disabled output matched the pristine control on the fixed smoke.
- The complete 428,212,736-byte layer-0 bundle was preloaded once: gate/up, down, and scale.
- Three fresh feature-on processes had exact prompt-token, generated-token, and 630-event router parity with one another.
- All runs exited cleanly and GPU memory returned to the pre-run range.

## Scope of the conclusion

This is a narrow empirical failure, not evidence that the static island is corrupt or unstable. The measured relative perplexity increase is 0.586%, only 0.086 percentage points beyond the frozen allowance, but changing the threshold or selecting another corpus after seeing the result would invalidate the experiment.

Per the stop rule, MMLU, long-generation scoring, reactive caching, prediction, and Q6 expansion were not run. No quality-preserving or speedup claim is made. The exactness rollback floor at ExpertFlow `378ce57` and the isolated llama milestone `29857466d39cc532cefc1633ac14e521849541fe` remain preserved.
