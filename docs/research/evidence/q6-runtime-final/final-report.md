# ExpertFlow final Q6 runtime sprint

The truthful Stage 0 baseline passed and is frozen at 22.967 mean decode TPS for pristine Q6 `--cpu-moe`, with 3,098.656 MiB peak process-owned VRAM across three cold 512-token runs. The matched patched-disabled runtime produced the same response hash and placement behavior.

Stage 1 proved that a complete 128-expert Q4 bundle can remain authoritative on CPU, occupy a persistent 408.375 MiB CUDA shadow, and execute through a narrow layer-0 CUDA expert island without moving router/top-k or the residual path. The feature-disabled binary remained exactly equivalent to pristine.

The required exactness gate failed. The CUDA island was deterministic, but it changed later router ordering at prefill token 0, layer 13 and changed generated token 1. The alternate explicit-boundary design cannot eliminate this CPU-versus-CUDA numerical result through the existing scheduler API; it requires a forbidden replacement kernel, new operation, or broad scheduler change.

Per the declared gate order, no reactive cache, Q6 kernel generalization, predictive integration, capacity sweep, or product CLI work followed. No speedup claim is made. The protected branches remain unmerged and unpushed.

See `stage1-narrow-placement-stop.md`, `results.json`, `baseline-results.json`, and `placement-manifest.json` for the measured evidence.

NARROW PLACEMENT STOP
