# CUDA PDL and generalization implementation plan

> **For agentic workers:** Use superpowers:executing-plans task by task.

**Goal:** Close the qualified Gemma CUDA scheduling hypothesis, then validate
the existing compiler method on a real second MoE family.

**Architecture:** Optional typed scheduling control, existing owned native runner
and EvidenceStore, a bounded diagnostic collector. Reuse existing family adapter,
eligibility, reference, product and search contracts for generalization.

**Tech stack:** Python, pytest, SQLite, pinned llama.cpp/CUDA12.8, native CUPTI probe.

**Spec:** `docs/superpowers/specs/2026-10-04-cuda-pdl-and-generalization.md`.

## Constraints and review focus

Preserve historical plan hashes and immutable inference binaries. No native
retries or changed gates. Reject inherited PDL, unsupported enum values,
changed host/source/protocol, duplicate owners, wrong tokens and partial runs.
Keep all profiler failures distinct from model measurements.

## Task1: qualify the native control

- [x] Inspect immutable launch helper and actualsm120a binary.
- [x] Run18shape/54compute probe; resolve CUPTI compatibility with local tools.
- [x] Save successful counts/output hashes and preserved failures in evidence.

## Task2: bind the scheduling setting

- [x] Add failing tests to `tests/test_compiler_runner.py` for historical payload
  compatibility, explicit identity changes, inherited stripping and invalid values.
- [x] Add optional omission metadata in `schema.py`, typed setting in `plan.py`,
  and environment lowering in `runner.py`; run relevant schema/plan/evidence tests.
- [x] Create `scripts/benchmark_compiler_cuda_pdl.py`: read-only audit/run actions,
  exact accepted source plan verification, source/protocol/host freeze,20owned
  processes, independently reconstructed native statistics and fail-closed report.
- [x] Test tampered identity/environment/token/ownership/partial evidence.
- [x] Run full suite and independent required review; commit measured source.

## Task3: close Gemma scheduling

- [x] Preserve Q4f02ccbd frozen checkout for historical validation.
- [x] Run the frozen20-process experiment once; audit raw native evidence.
- [x] Record terminal verdict and limits, retain accepted plan unless a separate
  product acceptance qualifies a winning control; commit evidence.

## Task4: real-family generalization

- [ ] Inspect official small-MoE metadata and pinned backend support; choose one
  compatible artifact under2GiB and verify downloaded size/hash.
- [ ] Normalize complete real tensor inventory; TDD required family adapter and
  source eligibility extension, with explicit operation/control limits.
- [ ] Freeze and execute its own reference10, product20 and bounded stock search,
  conditional on each gate. Audit and document actual coverage and exclusions.
- [ ] Run required checks and review; checkpoint durable terminal state.
