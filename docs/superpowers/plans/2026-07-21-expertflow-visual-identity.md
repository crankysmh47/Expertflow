# ExpertFlow Visual Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace generic release visuals with a routed-circuit and carbon-gold identity derived from the supplied ExpertFlow logo, then polish the submission README around the verified result and the full Codex/GPT-5.6 workflow.

**Architecture:** Keep all product claims and runtime artifacts unchanged. Add the canonical raster logo as a repository asset, rebuild diagrams as self-contained GitHub-compatible SVGs, apply the same tokens to the static dashboard, and reference those assets from a reorganized README. Extend release tests to enforce the logo, visual palette, attribution, and evidence boundaries.

**Tech Stack:** Markdown, SVG, static HTML/CSS, Python pytest, PowerShell, existing `uv` environment

## Global Constraints

- Use PCB routing for structure and carbon-black fields with restrained gold accents for hierarchy.
- Preserve `28.13 TPS`, `22.967 stock TPS`, `22.48%`, and `10,966.801 MiB` exactly.
- Keep the four-slot server result separate from single-stream decode TPS.
- State that the strict perplexity confidence gate was not met.
- State that predictive caching was simulated and rejected, not shipped.
- Do not modify runtime code, benchmark evidence, or protected llama.cpp branches.
- Keep public instructions free of private absolute paths.

---

### Task 1: Add the canonical brand asset and visual contracts

**Files:**
- Create: `docs/assets/expertflow-logo.png`
- Modify: `tests/test_final_polish.py`

**Interfaces:**
- Consumes: the user-supplied logo PNG attachment
- Produces: stable repository path `docs/assets/expertflow-logo.png` used by README and release packaging

- [ ] **Step 1: Add failing brand assertions**

Extend the visual asset list in `tests/test_final_polish.py` with `docs/assets/expertflow-logo.png`, then assert that `README.md` references that exact relative path and contains the phrases `GPT-5.6-sol` and `managed the engineering workflow`.

- [ ] **Step 2: Verify the contract fails**

Run: `uv run pytest -q tests/test_final_polish.py`

Expected: failure because the repository logo and revised attribution do not exist yet.

- [ ] **Step 3: Copy the exact supplied logo non-destructively**

Run:

```powershell
Copy-Item -LiteralPath '<attached-logo-path>' -Destination 'docs\assets\expertflow-logo.png'
```

Verify its SHA-256 and dimensions with `Get-FileHash` and the bundled Pillow runtime.

- [ ] **Step 4: Commit the brand foundation with the later README contract once green**

Do not commit while the new README assertions are still failing; Task 3 completes this contract.

### Task 2: Replace generic SVG cards with routed-circuit diagrams

**Files:**
- Modify: `docs/assets/architecture.svg`
- Modify: `docs/assets/result.svg`
- Modify: `docs/assets/placement-map.svg`
- Modify: `docs/assets/cache-decision.svg`
- Modify: `docs/assets/profile-cards.svg`
- Modify: `submission/demo-video-assets/title.svg`
- Modify: `submission/demo-video-assets/architecture.svg`
- Modify: `submission/demo-video-assets/result.svg`
- Modify: `submission/demo-video-assets/final-summary.svg`
- Create: `submission/demo-video-assets/codex-workflow.svg`
- Create: `submission/demo-video-assets/limitations.svg`
- Modify: `tests/test_final_polish.py`

**Interfaces:**
- Consumes: palette and evidence constraints from `docs/superpowers/specs/2026-07-21-expertflow-visual-identity-design.md`
- Produces: standalone SVG assets with `viewBox`, `<title>`, `<desc>`, and no external fonts or scripts

- [ ] **Step 1: Add failing asset and palette assertions**

Require the two new video SVGs. For every SVG under `docs/assets` and `submission/demo-video-assets`, assert the file contains a `viewBox`, a `<title`, and one of the canonical carbon/PCB colors. Assert that all result visuals retain `28.13`, `22.967`, and `22.48%` where applicable.

- [ ] **Step 2: Verify the contract fails**

Run: `uv run pytest -q tests/test_final_polish.py`

Expected: failure for missing assets and the old blue-card palette.

- [ ] **Step 3: Rebuild the documentation SVGs**

Use self-contained SVG primitives only. Each visual starts with a carbon or PCB field, adds thin circuit traces, uses chip-pad groups instead of card grids, and reserves gold for the central decision or measurement. Preserve each existing accessible title and description while making its evidence class visible.

- [ ] **Step 4: Rebuild the video SVGs**

Use a 16:9 `viewBox="0 0 1920 1080"`. Keep all narration-frame text within a 120-pixel safe margin. Add the Codex workflow frame with the sequence `IDEATION -> EXPERIMENT -> VERIFY -> DECIDE -> SHIP`, and a limitations frame containing the exact non-claim boundaries.

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest -q tests/test_final_polish.py`

Expected: SVG contract passes; README assertions remain pending until Task 3.

### Task 3: Rebuild the README and submission narrative

**Files:**
- Modify: `README.md`
- Modify: `submission/final-devpost-draft.md`
- Modify: `submission/demo-video-script-final.md`
- Modify: `submission/demo-video-shot-list-final.md`

**Interfaces:**
- Consumes: new relative visual paths and committed release metrics
- Produces: judge-first README and matching submission/video copy

- [ ] **Step 1: Rewrite the README hero and section order**

Start with the logo, product description, headline measurement, and:

```console
uv sync --frozen
uv run expertflow demo --replay
```

Then answer, in order: why it exists, measurable difference, immediate replay, placement mechanism, Codex/GPT-5.6 workflow, live run, limitations, and reproduction.

- [ ] **Step 2: Expand the Codex/GPT-5.6 section accurately**

State that GPT-5.6 guided the full ideation and project progression and that Codex with GPT-5.6-sol managed the engineering workflow end to end: isolated worktrees, source investigation, instrumentation, testing, benchmark design, measurement collection, failed-path isolation, implementation tweaks, packaging, verification, and submission polish. Preserve the user's role as product and evidence authority.

- [ ] **Step 3: Humanize the copy**

Remove generic promotional language, excessive bolding, repeated three-item lists, and vague claims. Prefer first-person specifics in the build story and measured language everywhere else.

- [ ] **Step 4: Synchronize Devpost and video copy**

Bring `submission/final-devpost-draft.md` and the final video script into line with the README. Add the exact new asset paths to the shot list, including `codex-workflow.svg` and `limitations.svg`.

- [ ] **Step 5: Run focused tests and link checks**

Run: `uv run pytest -q tests/test_final_polish.py`

Expected: all final-polish assertions pass.

### Task 4: Apply the identity to the offline dashboard and release package

**Files:**
- Modify: `docs/evidence/product-release/dashboard.html`
- Regenerate: `release/expertflow-build-week/dashboard.html`
- Regenerate: `release/expertflow-build-week/manifest.sha256`
- Regenerate: `release/expertflow-build-week.zip`

**Interfaces:**
- Consumes: canonical logo path, palette, and unchanged scorecard
- Produces: branded offline judge dashboard and refreshed deterministic release package

- [ ] **Step 1: Add failing dashboard branding assertions**

In `tests/test_product_dashboard.py`, assert the dashboard contains `--pcb:#0b3d20`, `--gold:#d6a84a`, `ExpertFlow`, and the Codex/GPT-5.6 workflow label. Keep existing evidence-class assertions.

- [ ] **Step 2: Verify the dashboard test fails**

Run: `uv run pytest -q tests/test_product_dashboard.py`

Expected: failure on missing palette variables and workflow label.

- [ ] **Step 3: Restyle the source dashboard**

Replace rounded generic cards with circuit-board sections, chip-pad metrics, gold measurement hierarchy, routed separators, and restrained radii. Keep all existing text claims and evidence labels intact.

- [ ] **Step 4: Rebuild the release**

Run: `uv run python scripts/build_product_release.py`

Expected: release directory and ZIP regenerate successfully with the new documentation and visual assets.

- [ ] **Step 5: Run dashboard and packaging tests**

Run:

```powershell
uv run pytest -q tests/test_product_dashboard.py tests/test_product_packaging.py tests/test_final_polish.py
```

Expected: all tests pass.

### Task 5: Render, replay, and final verification

**Files:**
- Modify only if validation finds a presentation defect in an in-scope file

**Interfaces:**
- Consumes: completed README, SVGs, dashboard, and release archive
- Produces: verified presentation milestone with no runtime changes

- [ ] **Step 1: Render every SVG to PNG for inspection**

Use the bundled workspace Python/CairoSVG runtime when available. Inspect the README hero, five documentation diagrams, and six video frames for clipping, contrast, safe margins, and legibility.

- [ ] **Step 2: Serve and inspect the dashboard**

Run `py -m http.server 8765` from the repository root and inspect `http://127.0.0.1:8765/release/expertflow-build-week/dashboard.html` at desktop width.

- [ ] **Step 3: Run replay and applicable tests**

Run:

```powershell
uv run expertflow demo --replay
$env:PYTHONPATH="$PWD;$PWD\src"
uv run pytest -q --ignore=tests/test_t1_temporal_source_contract.py --ignore=tests/test_t2_sidecar_source_contract.py
```

Expected: replay verifies committed evidence; applicable suite passes.

- [ ] **Step 4: Verify repository hygiene**

Run `git diff --check`, scan public docs for `C:\Users\` and `C:\models\`, inspect `git status --short`, and confirm no temporary render files are tracked.

- [ ] **Step 5: Commit the implementation**

```powershell
git add README.md docs/assets docs/evidence/product-release/dashboard.html submission release/expertflow-build-week tests
git commit -m "docs: give ExpertFlow a routed-circuit identity"
```

Do not push or merge.
