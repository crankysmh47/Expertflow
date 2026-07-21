# Product Architecture Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a canonical product and architecture guide for ExpertFlow and expose it through every judge-facing entry point.

**Architecture:** `docs/PRODUCT.md` becomes the explanatory layer between README and evidence. Existing SVGs provide the visual architecture; source contracts prevent claim drift and guarantee release packaging.

**Tech Stack:** Markdown, HTML, Python/pytest, deterministic Python release builder.

## Global Constraints

- Preserve the measured 28.13 TPS, 22.967 stock TPS, 22.48% improvement, and 10,966.801 MiB peak values.
- Describe `[0,1,2,3,4,5,6,7,8,9,15,20]` as the emitted plan for the measured 16 GB system, not a universal optimum.
- State that the shipped runtime uses full static identity placement with no eviction, reactive loading, prediction, repacking, or per-token transfer.
- Reuse existing project assets and add no runtime dependencies.

---

### Task 1: Architecture guide contract and content

**Files:**
- Create: `tests/test_product_architecture.py`
- Create: `docs/PRODUCT.md`

**Interfaces:**
- Consumes: release scorecard claims and existing files under `docs/assets/`.
- Produces: canonical `docs/PRODUCT.md` architecture reference.

- [ ] Write a failing source-contract test requiring the product pipeline, runtime boundaries, shipped-plan qualification, diagrams, evidence links, and Codex/GPT-5.6 section.
- [ ] Run `uv run pytest tests/test_product_architecture.py -q` and confirm failure because `docs/PRODUCT.md` is absent.
- [ ] Write `docs/PRODUCT.md` with the approved twelve-section judge-to-engineer narrative.
- [ ] Run the focused test and confirm it passes.

### Task 2: Judge-facing integration and release packaging

**Files:**
- Modify: `README.md`
- Modify: `docs/evidence/product-release/dashboard.html`
- Modify: `scripts/build_product_release.py`
- Modify: `tests/test_product_architecture.py`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: `docs/PRODUCT.md`.
- Produces: README/dashboard entry points and packaged `docs/PRODUCT.md`.

- [ ] Extend the test to require README/dashboard links and the release allowlist entry.
- [ ] Run the test and confirm the integration assertions fail.
- [ ] Add prominent architecture-guide links, package the guide, and append the release ledger entry.
- [ ] Run the architecture, dashboard, packaging, and final-polish tests.
- [ ] Rebuild and verify the deterministic release, then commit and push `main`.

