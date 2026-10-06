# Final Q6 Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:executing-plans`; architectural subagents are prohibited for this sprint.

**Goal:** Build an exact Q6 expert-streaming runtime or stop at the first declared empirical gate.

**Architecture:** Preserve stock router ordering and create only a scheduler-visible CUDA expert-operation island. Generalize the proven complete-bundle cache only after static island parity passes.

**Tech stack:** pinned llama.cpp/C++, GGML scheduler APIs, CUDA 12.8, MSVC/Ninja, Python evidence tooling, pytest.

## Global constraints

- Q6 model SHA-256: `089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba`.
- Minimum reproducible VRAM margin: 256 MiB.
- Exact generated tokens and ordered router IDs are mandatory.
- No whole-layer placement, new CUDA MoE kernel, scheduler replacement, post-allocation mutation, prediction retraining, merge, or push.
- Use test-first source contracts and focused runtime parity at every implementation gate.

### Task 1: Truthful Q6 baseline

- [ ] Hash pristine and patched sources/binaries and clear all ExpertFlow variables.
- [ ] Record per-tensor backend allocations and actual operation backends.
- [ ] Run pristine, patched-disabled, and CPU-authoritative/static-shadow diagnostic modes.
- [ ] Freeze the fastest stable upstream-equivalent stock baseline with 256 MiB margin in `baseline-results.json` and `baseline-audit.md`.

### Task 2: Q4 narrow static island

- [ ] Add a failing source contract requiring the narrow boundary and forbidding whole-layer placement/mutation.
- [ ] Attempt scheduler-supported per-node/tensor assignment for layer 0, `-ngl 10`, 128 identity slots.
- [ ] Build and require exact prefill/decode router order and generated tokens.
- [ ] If it fails, remove the attempt and test the one permitted explicit subgraph boundary.
- [ ] If both fail, discard implementation changes, commit reproducible evidence, and stop `NARROW PLACEMENT STOP`.

### Task 3: Q4 reactive cache

- [ ] Test coherent bundle publication states and logical/physical mappings before implementation.
- [ ] Validate 128, 96, 64, then 32 slots on layer 0; expand only to layers 0-2/0-3 after parity.

### Task 4: Quantization-independent bundles

- [ ] Test metadata-derived tensor members, type traits, strides, offsets, alignment, and bytes.
- [ ] Remove Q4 constants and verify Q4 behavior remains exact.

### Task 5: Q6 static and reactive product

- [ ] Prove one CPU-backed Q6 layer with 128 identity slots, then middle/control layers.
- [ ] Ramp whole-model capacities 88, 80, 72, 64, selecting the largest measured fit.
- [ ] Stop `Q6 KERNEL STOP` for incompatible existing kernels or `REACTIVE STOP` below 90% stock TPS.

### Task 6: Existing prediction and product CLI

- [ ] Compare unchanged reactive mode to the existing temporal async path at identical capacity.
- [ ] Retain prediction only for at least 15% less blocked time and 3% higher decode TPS.
- [ ] Add test-first `expertflow probe`, `run`, and `compare`; produce final 512-token, three-process scorecard and sanitizer evidence.
