# ExpertFlow demo script (2:45)

## 0:00–0:15 — The problem

On screen: `demo-video-assets/title.svg`.

"This is ExpertFlow, a placement compiler for quantized mixture-of-experts models. A model can say its layers are on CUDA while the expensive expert matmuls still run on CPU."

## 0:15–0:40 — The product

On screen: `demo-video-assets/architecture.svg`.

"Stock execution crosses the CPU boundary for routed experts. ExpertFlow profiles that hidden work and keeps the most valuable complete Q6 expert banks on CUDA."

## 0:40–1:05 — Replay

```console
uv sync --frozen
uv run expertflow demo --replay
```

"This model-free replay verifies the evidence hash. The live result was 28.13 TPS versus 22.967 stock, 22.48% faster on a 16 GB RTX 5060 Ti."

## 1:05–1:30 — Placement and novelty

On screen: `demo-video-assets/result.svg`, then the README placement map.

"The optimizer selected complete 128-expert banks at layers 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, and 20. It preserves packed Q6 data and does not ship a reactive cache."

## 1:30–1:50 — Agentic workflow

```console
uv run expertflow optimize <model> --goal agentic --output deployment.json
uv run expertflow serve deployment.json
```

"The measured four-slot server completed 20 out of 20 requests at 35.6699 aggregate TPS through an OpenAI-compatible local endpoint."

## 1:50–2:15 — Codex and GPT-5.6

"Codex built isolated worktrees, instrumentation, tests, placement code, benchmark harnesses, the CLI, and this evidence package. I set the scientific gates and rejected misleading comparisons. GPT-5.6 helped scope bounded experiments, interpret failures, and turn the frozen result into a product workflow."

## 2:15–2:38 — Honest limits

"The strict PPL confidence gate was not met. MMLU moved from 49 to 50. Concurrent outputs were not fully deterministic. The 262,144-token context was allocated, but the bounded run processed 417 tokens. Predictive caching was simulated and rejected, not shipped."

## 2:38–2:45 — Close

On screen: `demo-video-assets/final-summary.svg`.

"ExpertFlow puts VRAM where it removes the most CPU work. Replay it in under two minutes."
