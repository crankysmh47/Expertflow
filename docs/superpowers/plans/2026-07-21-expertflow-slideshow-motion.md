# ExpertFlow Slideshow Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Animate the nine-slide ExpertFlow recording deck so its copy and motion present an evidence-driven placement compiler and remain reliable offline.

**Architecture:** Keep the deck as one dependency-free HTML file. Embed each existing SVG as a same-origin `object`; the page owns slide activation, navigation, replay, and a lightweight injected SVG animation stylesheet, while each source SVG remains reusable elsewhere.

**Tech Stack:** HTML, CSS, vanilla JavaScript, pytest source-contract tests, local HTTP browser verification.

## Global Constraints

- Preserve the PCB, carbon, signal-green, cream, and gold identity.
- Preserve the measured 28.13 TPS and 22.48% claims and their evidence boundaries.
- Keep all assets local and the release allowlist unchanged.
- Support reduced motion, keyboard navigation, scroll snapping, 720p recording, and narrow viewports.
- Static residency must be presented as a hardware-specific compiler output, not the whole product.

---

### Task 1: Lock the narrative and motion contract

**Files:**
- Modify: `tests/test_video_slideshow.py`
- Test: `tests/test_video_slideshow.py`

**Interfaces:**
- Consumes: `submission/demo-video-slideshow.html` as UTF-8 text.
- Produces: source contracts for object embedding, compiler narrative, active-slide observation, replay, reduced motion, and local-only assets.

- [ ] **Step 1: Write failing source-contract tests**

Add assertions for `<object`, `IntersectionObserver`, `is-active`, `replay-current`, `prefers-reduced-motion`, `placement compiler`, `hardware-specific`, and `aria-current` while preserving the nine-asset and containment checks.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_video_slideshow.py -q`

Expected: FAIL because the current page uses images and has no activation, replay, or compiler-narrative contract.

### Task 2: Implement the cinematic circuit deck

**Files:**
- Modify: `submission/demo-video-slideshow.html`
- Test: `tests/test_video_slideshow.py`

**Interfaces:**
- Consumes: nine local SVG paths and native browser APIs.
- Produces: `setActiveSlide(index)`, `restartSlide(slide)`, `installSvgMotion(object)`, and keyboard/replay controls.

- [ ] **Step 1: Implement minimal HTML, CSS, and JavaScript**

Replace image tags with labelled same-origin objects and textual fallbacks. Add per-slide narrative eyebrows, a compiler-output label, an active progress rail, a replay button, ambient routed traces, staggered slide entrances, SVG path/text/shape reveals, and reduced-motion overrides. Use `IntersectionObserver` to update `.is-active` and `aria-current`; use the object `load` event to inject the SVG motion rules.

- [ ] **Step 2: Verify GREEN**

Run: `uv run pytest tests/test_video_slideshow.py -q`

Expected: PASS.

- [ ] **Step 3: Run browser acceptance checks**

Serve the repository on `127.0.0.1:8767`; inspect all nine slides at 1280 x 720. Require zero horizontal overflow, a contained visual, exactly one active slide and nav dot, working Arrow/Page/Space navigation, replayable current-slide motion, and injected SVG styles.

- [ ] **Step 4: Run release verification**

Run the applicable full pytest suite, `git diff --check`, and `uv run python scripts/build_product_release.py --llama-repo C:\models\expertflow\worktrees\llama-product-release`.

Expected: all applicable tests pass, diff check is clean, and release construction exits zero.

- [ ] **Step 5: Commit and push**

Commit the test, deck, rebuilt release evidence, and plan as `docs: animate ExpertFlow submission deck`, then push `codex/build-week-release`.
