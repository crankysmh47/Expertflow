# Granite generalization implementation plan

**Goal:** Validate the stock compiler method on a real second MoE architecture.

**Architecture:** Shared routed-inventory validation with separate family adapters;
reviewed Granite source provider declares a fixed GPU-resident baseline. Existing
reference, paired product and bounded search collectors bind and audit that
baseline without changing old Gemma identities.

**Tech stack:** Python/pytest/SQLite, actual GGUF, pinned pristine llama.cpp/CUDA.

**Spec:** `docs/research/protocols/specs/2026-10-04-granite-generalization.md`.

## Constraints and review focus

No new native build; GPU-resident pristine runtime only. Preserve historical
Q6/Q4/PDL source checkouts and negative evidence. Maximum68native processes, gated
10reference+20product+18screen+optional20confirmation, no retries. New provider
must bind full inventory/source/host/runtime. Fail on changed baseline/proof,
inherited controls, wrong tokens, partial runs, duplicate ownership and altered
source/protocol. PDL/static remain outside stock search. No universal claim.

## Task1: qualify actual artifact and source

- [x] Download immutable public artifact under2GiB and verify actual bytes.
- [x] Save official config, complete GGUF metadata and242tensor inventory.
- [x] Audit immutable Granite model/graph and CPU/CUDA operation paths; save exact
  source object IDs and numerical-control limits in family evidence.

## Task2: implement only missing reuse contracts

- [x] TDD `tests/test_compiler_granite_adapter.py`: real inventory, wrong family,
  count/topk/scaling/shape/components/identity mutations and old Gemma compatibility.
- [x] Factor shared inventory validator in `adapters/base.py`; delegate Gemma
  and add `adapters/granitemoe.py`, register builtin. Preserve old ModelIR hashes.
- [x] TDD separate trusted provider in `stock_eligibility.py` with exact artifact,
  inventory/shape/runtime/host/source guards and `baseline_cpu_moe=False` proof.
- [x] TDD provider-declared reference baseline in `stock_reference.py`, GPU-baseline
  paired product in `refinement.py`, and fixed GPU-placement scheduling space in
  `stock_search.py`. Gemma default/scopes/budgets and unsupported PDL rejection hold.
- [x] Write actual model descriptor/runtime input files and explicit38process
  search-space config; run read-only preview and live pin/source verification.
- [x] Full checks, one required independent review/fix pass, then commit source.

## Task3: gated real-family execution

- [x] Fresh reference10 once; independent audit and actual CLI validation.
- [x] Fresh product20 once; audit/validate own tokens and baseline identity.
- [x] Fresh explicit search18/optional20 once; independently audit ranking,
  confirmation decision, receipt and plan; actual CLI validation.
- [x] Preserve measured source checkout and unchanged recommendation metadata;
  document consumption, exclusions, unsupported cases and result uncertainty.

## Task4: completion checkpoint

- [x] Mark umbrella generalization task with actual terminal evidence.
- [x] Verify all owned model processes cleaned up and unrelated user files intact.
- [x] Checkpoint coherent changes and durable terminal state.

Terminal: own reference and paired product PASS, search RECOMMENDED-INCUMBENT.
All68registered native processes verified;16thread challenger not accepted.
Report: `docs/evidence/compiler-granite-20261004/report.md`. No extension/retry.
