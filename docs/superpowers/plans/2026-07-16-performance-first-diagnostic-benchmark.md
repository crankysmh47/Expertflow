# Performance-First Diagnostic Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and commit a reproducible five-mode throughput, latency, memory, parity, and cache-accounting benchmark before any multi-layer llama.cpp source edit.

**Architecture:** Add a benchmark-only C++ probe that emits stable machine-readable timing data while linking unchanged verified runtimes. Add a Python runner/parser that executes the fixed matrix, samples GPU memory, validates parity/accounting, aggregates repetitions, and renders evidence.

**Tech Stack:** C++17, llama.cpp C API, Python 3.11 standard library, pytest, CMake/MinGW runtime-probe build.

## Global Constraints

- Do not modify any llama.cpp source.
- Use one warmup plus three measured repetitions for general, code, and translation.
- Use greedy deterministic sampling, 64 requested decode tokens, 12 threads, and identical matched settings.
- Keep system GPU peak, host-wall latency, llama performance counters, and cache host-wall copy timing explicitly separated.
- Commit benchmark evidence before multi-layer work.

---

### Task 1: Result parser and aggregation

**Files:**
- Create: `tests/test_performance_benchmark.py`
- Create: `src/expertflow/benchmark/performance.py`
- Create: `src/expertflow/benchmark/__init__.py`

**Interfaces:**
- Consumes: probe JSON, token JSON, trace JSONL, cache JSONL.
- Produces: `parse_probe_result`, `summarize_repetitions`, `compare_modes`, and strict reconciliation failures.

- [x] Write parser/aggregation tests for TPS, p50/p95, sample variance, parity, and cache totals.
- [x] Run the focused test and retain the expected missing-module RED result.
- [x] Implement the minimum parser and aggregation functions.
- [x] Run focused and complete ExpertFlow tests.

### Task 2: Instrumented immutable-runtime probe

**Files:**
- Create: `native/performance_probe/main.cpp`
- Create: `native/performance_probe/CMakeLists.txt`
- Test: `tests/test_performance_probe_source.py`

**Interfaces:**
- Consumes: unchanged runtime DLLs and pinned llama.cpp headers.
- Produces: one strict JSON result plus token/trace/cache artifacts per run.

- [x] Write source-contract tests requiring greedy sampling, llama performance counters, first-token wall time, per-token samples, and no llama source mutation.
- [x] Run the source-contract test and retain the expected RED result.
- [x] Implement the minimal probe and build it against clean, observer, C4, and C5 runtimes.
- [x] Run help/smoke verification for every built probe.

### Task 3: Fixed benchmark runner

**Files:**
- Create: `scripts/run_performance_diagnostic.py`
- Test: `tests/test_performance_benchmark_runner.py`

**Interfaces:**
- Consumes: a JSON benchmark manifest listing runtimes, prompts, environment, warmups, and repetitions.
- Produces: raw run directories, append-only ledger, aggregate JSON, and Markdown report inputs.

- [x] Write command-construction and manifest-validation RED tests.
- [x] Implement strict commands, GPU sampling, hashing, cleanup checks, and append-only ledger.
- [x] Empirically scan stock offload and freeze the strongest stable no-OOM value.
- [x] Run the complete one-warmup/three-measured five-mode matrix.

### Task 4: Evidence, gate, and commit

**Files:**
- Create: `docs/evidence/live-cache/performance-first-diagnostic.md`
- Modify: `PROJECT_LOG.md`
- Modify: `docs/superpowers/plans/2026-07-16-performance-first-diagnostic-benchmark.md`

**Interfaces:**
- Consumes: reconciled aggregate JSON and raw artifacts.
- Produces: explicit overhead/gain calculations and multi-layer go/no-go.

- [x] Validate all JSON/JSONL, hashes, counts, cache accounting, parity, and clean process exit.
- [x] Render repetition tables, means, variance, requested comparisons, limitations, and gate verdict.
- [x] Run full tests and `git diff --check`.
- [x] Commit the diagnostic benchmark evidence without any llama.cpp source change.
