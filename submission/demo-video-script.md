# Demo video script

## 0:00–0:20 — The bottleneck

Show the Observatory architecture panel. Explain that stock `--cpu-moe` fits the model but leaves costly routed experts on the CPU. Whole-layer offload does not express the placement we want.

## 0:20–0:55 — The placement

Show the 30-layer grid and the twelve selected layers. State that each selected layer has a complete 128-expert static CUDA bank. There is no cache or prediction in the winning runtime.

## 0:55–1:25 — The result

Show 22.967 stock versus 28.13 ExpertFlow and 10,966.801 MiB peak VRAM. Say "ten matched 512-token runs" before the percentage.

## 1:25–1:50 — Honest quality and cache decision

Show MMLU 49 to 50, PPL point -2.92%, and upper bound +2.25%. State that the strict PPL confidence gate failed. Then show `NO CACHE OPPORTUNITY` and label it simulation.

## 1:50–2:20 — Product workflow

Run `uv run expertflow demo --replay`, then `doctor`, `optimize`, and a dry-run of `serve`. Show the model-free replay hash verification.

## 2:20–2:45 — Live server profile

Show the measured four-slot result: 35.67 aggregate generated TPS, 20/20 requests, and the nondeterministic concurrent-output caveat. Run the example coding request against the local endpoint.

## 2:45–3:00 — Close

Return to the dashboard. Finish with the product definition: profile-guided expert-bank placement under a VRAM budget.

