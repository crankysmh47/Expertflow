# ExpertFlow final demo script (2:59)

No webcam is needed. Record the live benchmark and slideshow as separate captures, then edit them together. Speak naturally; the wording below is a guide, not something to rush through mechanically.

## Before recording: prove the live path

Set the exact model and runtime once:

```powershell
$env:EXPERTFLOW_MODEL_PATH = 'C:\models\gemma-4-26b-a4b-q6\google_gemma-4-26B-A4B-it-Q6_K.gguf'
$env:EXPERTFLOW_LLAMA_CLI = 'C:\models\expertflow\builds\llama-q6-placement-final\bin\llama-cli.exe'
```

Run the complete recording rehearsal:

```powershell
.\scripts\live-tps-demo.ps1 -Mode Demo
```

The verified 2026-07-22 rehearsal completed in 146.8 seconds including model hashing. It measured stock at `15.50 TPS`, ExpertFlow at `20.10 TPS`, and a one-pair uplift of `29.68%`. Treat that only as a live rehearsal. The authoritative claim remains ten matched 512-token pairs: `28.13 TPS` versus `22.967 TPS`, or `+22.48%`.

Record the entire terminal run once. In the final edit, keep only four short excerpts:

1. The verified model/runtime preflight.
2. `[RESULT off]` for stock.
3. `[RESULT on]` for ExpertFlow.
4. The final `LIVE RESULT` table and evidence path.

Then serve the deck and open it at 1920×1080:

```powershell
py -m http.server 8767 --bind 127.0.0.1
```

Open `http://127.0.0.1:8767/submission/demo-video-slideshow.html`. Use Arrow Down or Page Down once per scene. Press `REPLAY` before capturing a scene again.

## Final timed narration

### 0:00–0:12 — Opening

On screen: slideshow scene 1.

> I wanted to run the high-quality Q6 version of Gemma 4 26B on a sixteen-gig GPU, without CPU-offloaded experts dragging generation down. ExpertFlow is what came out of investigating that bottleneck with Codex.

### 0:12–0:30 — The hidden boundary

On screen: scene 2.

> Gemma's router was already choosing the right experts. The problem happened afterward: llama.cpp could report GPU-offloaded layers while routed expert matmuls still crossed back to the CPU. This was not an intelligence problem. It was a placement problem.

### 0:30–0:55 — GPT-5.6 and Codex

On screen: scene 3. Briefly cut to the append-only project log or passing test output near the end.

> GPT-5.6 helped frame the hypotheses from the first idea. Codex with GPT-5.6-sol managed the engineering loop: isolated worktrees, llama.cpp instrumentation, traces, cache and predictor prototypes, parity and memory tests, quality gates, documentation, and release packaging. Most importantly, it did not protect the original idea when the measurements disagreed.

### 0:55–1:22 — Live TPS proof

On screen: the four edited terminal excerpts from the real `live-tps-demo.ps1` capture.

> This is the live path on the same sixteen-gig GPU. One command verifies the model and runtime, launches fresh matched stock and ExpertFlow processes with identical settings, and saves the raw evidence. This single rehearsal measured fifteen-point-five stock and twenty-point-one ExpertFlow. That is useful live proof, but it is not the headline benchmark.

### 1:22–1:31 — Authoritative result

On screen: scene 4.

> The authoritative result is ten matched five-hundred-and-twelve-token pairs: twenty-eight-point-one-three TPS versus twenty-two-point-nine-six-seven stock, a twenty-two-point-four-eight percent improvement.

### 1:31–1:52 — The evidence changed the product

On screen: scene 5.

> The original idea was a predictive expert cache. We measured reactive LRU caching, routing prediction, and asynchronous sidecars. Some experiments were exact, but transfer and bookkeeping costs erased the gain. Codex preserved those failures, and the evidence changed the product instead of being hidden.

### 1:52–2:14 — The placement compiler

On screen: scene 6.

> ExpertFlow now treats placement as compilation. It profiles the model and hardware, accounts for complete packed expert banks and a VRAM reserve, scores CPU relief per byte, validates quality, and emits the plan before graph construction. On this machine it selected twelve complete Q6 banks. That is one hardware-specific compiler output, not a universal set of magic layers.

### 2:14–2:34 — What judges can run

On screen: scene 7. Briefly show `uv run expertflow demo --replay` in a terminal overlay.

> Judges can replay the evidence without a model or GPU, inspect a compatible system with doctor, generate a deployment with optimize, run locally, or serve an OpenAI-compatible API. The live command and the full ten-pair reproduction path are both included.

### 2:34–2:47 — Three ways to verify

On screen: scene 8.

> Judges get three proof paths: replay the hashed evidence on any machine, run the live matched TPS command on a compatible GPU, or rebuild the pinned llama.cpp patch series from source. The result is inspectable from the headline number down to the exact command.

### 2:47–2:59 — Close

On screen: scene 9.

> ExpertFlow compiles placement for the machine so you can keep the high-quality model you wanted. And GPT-5.6 with Codex turned every failed hypothesis into the next measured decision.

## Recording rules

- Record at 1920×1080, 30 fps, with terminal text at 22 pt or larger.
- Do not show the 146-second live run uncut. Keep the edited terminal proof to 27 seconds.
- Do not call the live one-pair uplift the benchmark result.
- Do not claim cross-mode token parity, a passed strict PPL gate, or general performance on other GPUs.
- Keep the voice relaxed. Pause briefly after the live table and the `+22.48%` result.
