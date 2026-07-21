# Final Video, Dashboard, and Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the final human narration, live proof recording path, narrative dashboard, and public static deployment.

**Architecture:** Keep the video assets and dashboard dependency-free. The source dashboard remains `docs/evidence/product-release/dashboard.html`; release packaging copies it to the archive. Vercel serves that static artifact through root rewrites without adding a frontend build system.

**Tech Stack:** HTML, CSS, vanilla JavaScript, Markdown, Python pytest, PowerShell, Vercel static hosting.

## Global Constraints

- Preserve the authoritative ten-pair result: `28.13` versus `22.967` decode TPS, `+22.48%`.
- Label the one-pair live run as rehearsal evidence.
- Keep all assets local and retain reduced-motion support.
- Do not bundle the GGUF or runtime binaries.

---

### Task 1: Human video and live proof

**Files:** Modify `submission/demo-video-script-final.md`, `submission/demo-video-shot-list-final.md`; test `tests/test_video_slideshow.py`.

- [ ] Add source-contract assertions for the personal opening, live command, matched-process language, and one-pair/ten-pair distinction.
- [ ] Run the focused test and confirm it fails for the missing narration.
- [ ] Rewrite the timed narration and capture instructions in a relaxed first-person voice.
- [ ] Run focused tests and commit the recording package.

### Task 2: Narrative hardware-console dashboard

**Files:** Modify `docs/evidence/product-release/dashboard.html`; test `tests/test_product_dashboard.py`, `tests/test_final_polish.py`.

- [ ] Add failing contracts for the story rail, live/replay actions, GitHub link, placement compiler, circuit substrate, evidence drawer, and reduced motion.
- [ ] Run the focused tests and confirm the required interface is absent.
- [ ] Replace the equal-weight card layout with the self-contained narrative console.
- [ ] Serve locally and verify desktop/mobile layout, console state, actions, and overflow.

### Task 3: Static deployment and local-hardware guide

**Files:** Create `vercel.json`, `DEPLOYMENT.md`; modify `README.md`, `JUDGES.md`, `scripts/build_product_release.py`; test `tests/test_product_packaging.py`.

- [ ] Add failing tests for Vercel routing, GitHub URL, model-free replay, compatible-live commands, model hash, and supported platforms.
- [ ] Add the minimal static hosting configuration and one concise deployment/local-hardware guide.
- [ ] Rebuild and verify the release package.
- [ ] Run Vercel production deployment when existing authentication permits; otherwise report the exact single command remaining.

### Task 4: Final verification and publication

**Files:** Modify `PROJECT_LOG.md` and generated release artifacts.

- [ ] Run focused and complete tests, replay, release verification, and `git diff --check`.
- [ ] Record browser inspection and any deployment URL or blocker.
- [ ] Commit and push `main`; confirm local, remote, and default branch hashes agree.
