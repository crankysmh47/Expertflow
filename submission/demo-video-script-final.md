# ExpertFlow demo script (2:50)

This is a relaxed voiceover over the supplied circuit-board frames, terminal replay, and dashboard. No webcam is required.

## 0:00-0:15 — The problem

On screen: `demo-video-assets/title.svg`.

"I wanted to run the high-quality Q6 version of Gemma 4 26B on my 16-gig GPU. It technically fit, but a lot of the expensive expert computation was still happening on the CPU. So I used Codex to investigate what the runtime was actually doing."

## 0:15-0:40 — The hidden boundary

On screen: `demo-video-assets/architecture.svg`.

"The problem was a little deceptive. llama.cpp could report GPU-offloaded layers while their routed expert matmuls still crossed back to the CPU. ExpertFlow measures that hidden work and places complete packed Q6 expert banks where they remove the most CPU cost."

## 0:40-1:15 — Codex and GPT-5.6

On screen: `demo-video-assets/codex-workflow.svg`, then a short capture of the command ledger and passing tests.

"GPT-5.6 was involved from the first idea through the final product direction. Codex with GPT-5.6-sol managed the engineering workflow: isolated worktrees, llama.cpp instrumentation, tests, traces, benchmarks, implementation changes, and release packaging."

"I set the evidence gates and decided which directions to approve. When an experiment failed, Codex preserved the result, isolated the cause, and helped design the next bounded test instead of forcing the original idea."

## 1:15-1:47 — Replay the result

On screen: a clean terminal.

```console
uv sync --frozen
uv run expertflow demo --replay
```

"This judge replay needs no model or GPU. It verifies the committed evidence. The live result was 28.13 decode TPS versus 22.967 stock: 22.48 percent faster on a 16-gig RTX 5060 Ti."

Hold on `demo-video-assets/result.svg` for three seconds.

## 1:47-2:15 — How the product changed

On screen: `docs/assets/cache-decision.svg`, then `docs/assets/placement-map.svg`.

"The original idea was a predictive expert cache. We built observer paths, reactive caches, and temporal predictor experiments. Some were exact but slower; others reached clear architectural stop conditions. The numbers said prediction would not win on this machine."

"So the product changed. ExpertFlow selected complete 128-expert banks for layers zero through nine, fifteen, and twenty and kept them resident on CUDA."

## 2:15-2:34 — What ships

On screen: the offline dashboard, then `docs/assets/profile-cards.svg`.

"The release now has a small CLI for evidence replay, system inspection, deployment generation, live inference, benchmarking, and an OpenAI-compatible local server. The headline result comes from ten matched 512-token runs."

## 2:34-2:44 — Evidence boundaries

On screen: `demo-video-assets/limitations.svg`.

"The strict perplexity confidence gate was not met, the large context was allocated but not filled, and live acceleration is verified on one Windows and NVIDIA setup. Those limits ship with the result."

## 2:44-2:50 — Close

On screen: `demo-video-assets/final-summary.svg`.

"ExpertFlow puts VRAM where it removes the most CPU work. Replay it in under two minutes."
