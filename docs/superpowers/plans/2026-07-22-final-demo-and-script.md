# ExpertFlow Final Demo and Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace foreground slideshow flashes with full-background circuit flow, add a judge-readable live TPS rehearsal command, and freeze the final timed recording script.

**Architecture:** Keep visual motion inside the existing offline HTML deck using inline SVG circuit fields. Extend the proven Q6 pair harness with visible progress and one-sample-safe analysis, then wrap it in a small PowerShell recording entry point rather than changing llama.cpp or the Python product runtime.

**Tech Stack:** HTML/CSS/vanilla JavaScript, PowerShell, Python analysis, pytest, llama.cpp CUDA runtime.

## Global Constraints

- Do not change runtime placement, model settings, benchmark claims, or evidence boundaries.
- Do not claim a one-pair rehearsal as the authoritative result.
- Keep the slideshow and judge replay usable without network access.
- Keep the branch clean, release reproducible, and logs outside the repository unless summarized.

---

### Task 1: Background circuit motion

**Files:**
- Modify: `submission/demo-video-slideshow.html`
- Modify: `tests/test_video_slideshow.py`

- [ ] Add a failing contract for `circuit-field`, `trace-flow`, `flow-through`, and `node-glow`, and prohibit foreground scan keyframes and cream sweep colors.
- [ ] Replace foreground scan layers with an inline full-viewport circuit route template, moving luminous dash packets, and softly pulsing nodes.
- [ ] Run focused tests and visually inspect desktop and narrow layouts.

### Task 2: Live TPS recording command

**Files:**
- Create: `scripts/live-tps-demo.ps1`
- Modify: `scripts/run_q6_selected_static_pairs.ps1`
- Modify: `scripts/analyze_q6_selected_static.py`
- Modify: `tests/test_q6_selected_static_analysis.py`
- Modify: `tests/test_q6_selected_static_cli_runner.py`
- Modify: `scripts/build_product_release.py`

- [ ] Add a failing one-pair analysis test requiring `None` sample dispersion.
- [ ] Add failing source contracts for visible stock/ExpertFlow progress, demo/judge modes, 512 tokens, and an explicit replay fallback.
- [ ] Make the analyzer accept one complete pair without changing multi-pair statistics.
- [ ] Print concise per-run progress/results from the existing matched harness.
- [ ] Add `live-tps-demo.ps1` with `Demo` and `Judge` modes and release it through the allowlist.
- [ ] Run unit/source-contract tests.

### Task 3: Rehearsal and final recording script

**Files:**
- Modify: `submission/demo-video-script-final.md`
- Modify: `submission/demo-video-shot-list-final.md`
- Modify: `PROJECT_LOG.md`

- [ ] Run one live 512-token matched demo pair and preserve raw evidence outside the repository.
- [ ] Confirm process cleanup, idle VRAM recovery, both TPS values, and the result path.
- [ ] Rewrite the under-three-minute narration around the live result, compiler narrative, Codex/GPT-5.6 engineering loop, and evidence limitations.
- [ ] Give exact screen actions, scroll points, and voiceover text.
- [ ] Run the applicable full suite, rebuild the release, inspect the deck, commit, and push.
