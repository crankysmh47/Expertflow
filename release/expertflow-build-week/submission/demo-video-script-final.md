# ExpertFlow final demo script (2:59)

No webcam is needed. Record the live benchmark first, then record the slideshow separately and cut the two together.

## Before recording: rehearse the concrete proof

```powershell
cd C:\sem4\Expertflow
$env:EXPERTFLOW_MODEL_PATH = 'C:\models\gemma-4-26b-a4b-q6\google_gemma-4-26B-A4B-it-Q6_K.gguf'
$env:EXPERTFLOW_LLAMA_CLI = 'C:\models\expertflow\builds\llama-q6-placement-final\bin\llama-cli.exe'
.\scripts\live-tps-demo.ps1 -Mode Demo
```

The command verifies both hashes, launches fresh matched stock and ExpertFlow processes, prints each result, and saves the raw evidence. Record the complete run, but retain only the identity check, both result lines, and final table in the edit. Treat it as one live rehearsal. The authoritative ten-pair result remains `28.13 TPS` versus `22.967 TPS`, or `+22.48%`.

## Exact narration and screen direction

### 0:00–0:17 — Personal opening

**Screen:** slideshow scene 1.

> I've always tried to get the most out of the hardware I own. But with local AI, I kept hitting the same annoying choice: run the model I actually wanted and accept slow CPU offloading, or drop to a smaller quant because that was all the usual deployment controls could handle comfortably.

### 0:17–0:34 — The real bottleneck

**Screen:** scene 2.

> Gemma 4 26B made me question that trade-off. It is sparse, and its router already chooses only a few experts. But I found that layers reported as GPU-offloaded could still send the expensive expert work back through the CPU. The problem was not the router. It was where the work lived.

### 0:34–0:57 — Building with Codex

**Screen:** scene 3; briefly cut to `PROJECT_LOG.md` or passing tests.

> I started with a predictive-cache idea, and GPT-5.6 helped turn it into a sequence of experiments. Codex with GPT-5.6-sol handled the engineering loop: isolated branches, llama.cpp instrumentation, traces, cache prototypes, parity checks, VRAM measurements, benchmarks, and all the small fixes between them. When an idea failed, it kept the evidence instead of trying to make the story look cleaner.

### 0:57–1:28 — Live matched TPS run

**Screen:** terminal excerpts from `.\scripts\live-tps-demo.ps1 -Mode Demo`.

> So here is something concrete. This command checks the exact model and runtime, then starts fresh matched stock and ExpertFlow processes with the same prompt and settings. Stock finishes at fifteen-point-five TPS. ExpertFlow reaches twenty-point-one. This is one live rehearsal on my machine, not the headline benchmark, and the raw results are saved automatically.

**Edit:** show the verified identity, `[RESULT off]`, `[RESULT on]`, and `LIVE RESULT` table. Add: `ONE LIVE PAIR · AUTHORITATIVE RESULT USES TEN PAIRS`.

### 1:28–1:39 — Authoritative result

**Screen:** scene 4.

> The proper result comes from ten matched five-hundred-and-twelve-token pairs: twenty-eight-point-one-three TPS against twenty-two-point-nine-six-seven stock. That is a measured twenty-two-point-four-eight percent improvement on the same sixteen-gig GPU.

### 1:39–1:59 — The useful failure

**Screen:** scene 5.

> What surprised me is that the clever cache was not the winner. Reactive loading, prediction, and asynchronous sidecars all taught us something, but transfers and bookkeeping kept eating the benefit. That failure exposed the simpler opportunity: placement itself.

### 1:59–2:20 — What ExpertFlow became

**Screen:** scene 6.

> ExpertFlow became a placement compiler. It profiles the model and machine, measures which complete expert banks remove the most CPU work per byte of VRAM, keeps a safety reserve, and emits the plan before graph construction. These twelve Q6 banks are the plan compiled for this GPU, not a universal list of magic layers.

### 2:20–2:40 — A product judges can run

**Screen:** scene 7; overlay `uv run expertflow demo --replay` completing.

> I also wanted this to be more than a private runtime patch. Judges can replay the hash-checked evidence without the model or a GPU, inspect compatible hardware, generate a deployment, run it locally, or expose it through an OpenAI-compatible server.

### 2:40–2:51 — Three proof paths

**Screen:** scene 8.

> There are three proof paths: replay the committed evidence, run the live matched test, or rebuild the pinned llama.cpp patch series. The headline number connects all the way back to exact commands and raw results.

### 2:51–2:59 — Close

**Screen:** scene 9.

> ExpertFlow lets me keep the higher-quality model I wanted—and use my GPU where it matters most. That is what Codex helped me turn into a real, measured product.

## Recording rules

- Record at 1920×1080 and 30 FPS; use terminal text at 22 pt or larger.
- Do not show the full multi-minute benchmark wait. Compress it with clean cuts, never fake terminal output.
- Keep the live table visible long enough to read.
- Do not call the one-pair result the benchmark result.
- Speak conversationally. Small pauses and natural variation are better than reading quickly.
