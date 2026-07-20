# Q6 Predictive Final Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate fixed MMLU for the frozen 12-layer Q6 static champion, then admit hybrid caching work only when measured routing locality satisfies the frozen memory and throughput gates.

**Architecture:** Preserve the static champion and its failed strict PPL result. Reuse the established server/MMLU, canonical observer, simulator, and exact cache implementations in sequential stages; every stage emits machine-readable evidence and a commit, and each declared stop condition terminates later runtime work.

**Tech Stack:** Python 3, pytest, PowerShell, llama.cpp/CUDA 12.8, GGUF Q6_K, JSON/CSV evidence.

## Global Constraints

- Use model SHA-256 `089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba` only.
- Preserve 28.13 TPS static configuration and frozen PPL result unchanged.
- Run GPU workloads sequentially; no subagents, merge, push, new kernel, scheduler redesign, or whole-layer placement.
- Require >=500 MiB saving and >=95% of static TPS before reactive implementation.
- Keep every feature disabled by default and stop at the first terminal gate.

---

### Task 1: Freeze state and fixed MMLU

**Files:**
- Create: `docs/evidence/q6-predictive-final/static-state.json`
- Create: `docs/evidence/q6-predictive-final/mmlu-static.json`
- Create: `docs/evidence/q6-predictive-final/reproduction-commands.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: final placement manifest, runtime `451224ab`, fixed quality manifest.
- Produces: validated static identity and deterministic OFF/ON MMLU result.

- [ ] Verify commits, binary/model hashes, commands, environment, static arena bytes, graph mode, and absence of cache/predictor activity.
- [ ] Run the fixed 100-item MMLU manifest once OFF and once ON with identical runtime settings.
- [ ] Diff answers, classify each change, and rerun every changed ON item exactly.
- [ ] Validate cleanup, write evidence, run focused tests, and commit MMLU evidence.

### Task 2: Routing locality gate

**Files:**
- Create: `docs/evidence/q6-predictive-final/routing-locality.json`
- Create: `docs/evidence/q6-predictive-final/cache-simulation.json`
- Test/modify only existing trace-analysis tooling when required.

**Interfaces:**
- Consumes: canonical observer traces from performance, held-out PPL, MMLU, and representative conversations.
- Produces: per-layer 112/96/80/64 LRU metrics and at most three feasible hybrid candidates.

- [ ] Inventory canonical traces and prove their runtime hashes and corpus identities.
- [ ] Add failing analysis tests only if existing tooling lacks required metrics, then implement the minimum analyzer.
- [ ] Compute per-layer frequency, working set, reuse distance, temporal reuse, misses/token, H2D/token, and predictor utility.
- [ ] Simulate 10+2, 8+4, and 6+6 measured placements; stop if no candidate saves >=500 MiB at projected >=26.72 TPS.
- [ ] Commit the locality/simulation result separately.

### Task 3: Reactive hybrid gate

**Files:**
- Modify only the existing exact llama.cpp cache path and its source-contract tests when Task 2 passes.
- Create/update: `docs/evidence/q6-predictive-final/placement-manifest.json`, `run-pairs.csv`.

**Interfaces:**
- Consumes: one selected simulated candidate.
- Produces: exact static-versus-reactive runtime evidence.

- [ ] Write failing source/correctness tests for the chosen static/cached split.
- [ ] Implement the minimum feature-gated reuse of the existing exact Q6 bundle cache.
- [ ] Run correctness/stability smoke, two matched pairs, then five only if >=95% TPS and >=500 MiB saving pass.
- [ ] Commit implementation and evidence or terminate with `REACTIVE CACHE STOP`.

### Task 4: Spend memory and predictive gate

**Files:**
- Modify existing placement/predictor configuration only after reactive pass.
- Create: `docs/evidence/q6-predictive-final/predictor-utility.json`.

**Interfaces:**
- Consumes: passing reactive hybrid and frozen predictor weights.
- Produces: best measured hybrid and optional predictive improvement.

- [ ] Add remaining layers in pairs and retain only measured TPS gains.
- [ ] If reactive is useful, enable the existing predictor without retraining and compare five matched pairs.
- [ ] Require >=15% blocked-time reduction and >=1% total TPS gain; otherwise disable prediction and stop that track.
- [ ] Commit each successful stage separately.

### Task 5: Demonstrations, validation, and packaging

**Files:**
- Create: `docs/evidence/q6-predictive-final/context-concurrency.json`
- Create: `docs/evidence/q6-predictive-final/results.json`
- Create: `docs/evidence/q6-predictive-final/report.md`
- Create: `docs/evidence/q6-predictive-final/final-scorecard.md`
- Add product commands only when a useful cached mode passes.

**Interfaces:**
- Consumes: best passing cached/predictive runtime.
- Produces: final reproducible evidence and optional product profiles.

- [ ] Measure bounded context/concurrency benefits when supported without unrelated harness work.
- [ ] Run ten final cold-process pairs for a passing cached mode and report all required latency/cache/memory fields.
- [ ] Package max-performance, balanced, and memory-efficient profiles only after measured utility exists.
- [ ] Run full tests, validate every JSON/artifact hash, verify clean teardown, commit final evidence, and preserve worktrees unmerged/unpushed.
