# ExpertFlow Layer 2 Recommendation and Replay Plan

> Execute with test-driven development and preserve measured/estimated labels at every boundary.

**Goal:** Turn the real Q4 Observatory artifacts into one machine-specific recommendation and a standalone causal replay report without claiming an unmeasured live-cache speedup.

**Architecture:** A recommendation engine consumes the existing doctor, baseline, profile, and simulation JSON artifacts. It emits a strict JSON recommendation with a verdict, policy, replay capacity, measured VRAM envelope, uncertainty, and explicit blockers. A replay engine applies the selected policy to the same strict router events and renders a self-contained HTML report. Both commands reuse the current artifact schemas and cache-policy implementation.

**Current evidence constraint:** The gate is `CONDITIONAL`. Static hotset beats LRU on one short prompt, but expert byte size, PCIe transfer timing, and a stratified workload are not yet measured. The recommendation must therefore permit replay while keeping live cache enablement false.

---

### Task 1: Recommendation contract and decision engine

**Files:**
- Create: `src/expertflow/recommendation.py`
- Create: `tests/test_recommendation.py`
- Modify: `PROJECT_LOG.md`

- [x] Write failing tests for a conditional recommendation from the measured Q4 evidence.
- [x] Require explicit `measured` versus `estimated` fields and reject missing/mismatched source schemas.
- [x] Report total VRAM, measured peak baseline VRAM, safety reserve, and remaining configurable headroom without converting headroom into context/session claims.
- [x] Select the best replay policy by hit rate, but keep `live_cache_enabled = false` while expert bytes and transfer timing are absent.
- [x] Return stable reason codes so the CLI and HTML do not duplicate decision logic.

### Task 2: Public `recommend` command

**Files:**
- Create: `tests/test_recommend_cli.py`
- Modify: `src/expertflow/cli/main.py`
- Modify: `README.md`
- Modify: `PROJECT_LOG.md`

- [x] Add a failing CLI integration test.
- [x] Implement `expertflow recommend --doctor ... --baseline ... --profile ... --simulation ... --output ...`.
- [x] Return nonzero on invalid provenance and print the resolved output path on success.
- [x] Run it against the real external Q4 artifacts and preserve the generated recommendation outside Git as evidence.

### Task 3: Causal policy replay

**Files:**
- Create: `src/expertflow/analysis/replay.py`
- Create: `tests/test_replay.py`
- Modify: `PROJECT_LOG.md`

- [x] Write failing tests for deterministic per-token/layer `ready` and `blocking` decisions under static-hotset and LRU policies.
- [x] Reuse the existing capacity and trace ordering rules; do not create a second simulator.
- [x] Emit a compact timeline plus aggregate counts, with all policy outcomes labeled `estimated`.
- [x] Preserve request, phase, forward, token, and layer identities so each aggregate is causally inspectable.

### Task 4: Standalone HTML replay report

**Files:**
- Create: `src/expertflow/reporting.py`
- Create: `tests/test_reporting.py`
- Create: `tests/test_replay_cli.py`
- Modify: `src/expertflow/cli/main.py`
- Modify: `README.md`
- Modify: `PROJECT_LOG.md`

- [x] Write failing escaping and report-content tests before rendering code.
- [x] Add `expertflow replay <trace> --recommendation ... --output report.html`.
- [x] Render one self-contained HTML file with hardware readiness, gate verdict, policy comparison, remaining measured headroom, and a bounded causal timeline.
- [x] Embed source paths, schema versions, and a reproduction command; do not require a web server or model weights to open the report.
- [ ] Verify the real report in a browser and capture evidence only after the HTML passes tests and visual inspection.

### Task 5: Reproduction fixture and completion gate

**Files:**
- Create: `examples/replay/README.md`
- Create: `examples/replay/trace.jsonl`
- Create: `examples/replay/expected.json`
- Modify: `README.md`
- Modify: `PROJECT_LOG.md`

- [ ] Derive a small prompt-text-free fixture from the public checked-in baseline prompt trace and label it `previously_measured`.
- [ ] Add a test that reproduces the checked-in expected recommendation and policy totals.
- [ ] Run the complete suite, README link check, TOML validation, and `git diff --check`.
- [ ] Record the final Layer 2 verdict and remaining blockers before committing.

## Stop rules

- Do not add a live cache, transfer scheduler, or predictive runtime in this plan.
- Do not call estimated hits “saved latency” without a measured transfer curve.
- Do not recommend a byte allocation until per-expert byte size is measured.
- Keep the report useful when the correct outcome is `CONDITIONAL` or `DO NOT ENABLE`.
