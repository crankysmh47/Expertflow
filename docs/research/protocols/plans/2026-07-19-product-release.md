# ExpertFlow Product Release Implementation Plan

**Goal:** Ship the verified Q6 static-placement result as a reproducible CLI, offline dashboard, judge workflow, and audited release archive.

**Architecture:** Immutable evidence and deployment JSON drive every workflow. Thin Python modules validate manifests, construct commands, replay evidence, and package an allowlisted offline release; the patched llama.cpp runtime stays unchanged.

**Tech stack:** Python 3.11+, argparse, JSON, PowerShell, static HTML/CSS/JavaScript, pytest.

## Global constraints

- Preserve the frozen 28.13 TPS, 22.48% gain, 10,966.801 MiB peak, MMLU 49/100 to 50/100, PPL upper bound +2.25%, and `NO CACHE OPPORTUNITY` result exactly.
- Do not merge, push, bundle the GGUF, add private paths, or modify llama.cpp runtime architecture.
- Use a 512 MiB live-profile VRAM margin and label measured, replayed, simulated, and projected evidence separately.

### Task 1: Freeze release state

**Files:** Create `docs/evidence/product-release/release-state.json`, `deployment-result.json`, `command-ledger.md`; test `tests/test_product_release_state.py`.

- [ ] Write tests that require exact commits, model/binary hashes, selected layers, metrics, quality wording, evidence hashes, and cache-stop verdict.
- [ ] Run the focused test and confirm it fails because the manifest is absent.
- [ ] Generate immutable manifests from committed evidence, verify all referenced hashes, and rerun to green.
- [ ] Commit the reconstructed release-state gate.

### Task 2: Implement deployment CLI

**Files:** Create `src/expertflow/product/{manifest,commands,doctor,replay}.py`; modify `src/expertflow/cli/main.py`; test `tests/test_product_cli.py`.

- [ ] Write failing parser and JSON-contract tests for `doctor`, `profile`, `optimize`, `run`, `serve`, `compare`, and `demo --replay`.
- [ ] Implement manifest resolution, live inspection, profile generation, safe subprocess command construction, and evidence replay.
- [ ] Verify focused tests and model-free smoke commands, then commit.

### Task 3: Measure batching and context profiles

**Files:** Create `scripts/benchmark_product_server.py`, `scripts/measure_context_frontier.py`; test `tests/test_product_benchmarks.py`; write product-release result JSON/CSV.

- [ ] Write failing parsing, aggregation, stop-rule, and profile-selection tests.
- [ ] Implement the smallest reusable HTTP/server harness and context runner.
- [ ] Run stock and frozen ExpertFlow with identical settings; stop at failure/margin gates.
- [ ] Commit raw measurements and selected profiles; never substitute projections for missing runs.

### Task 4: Create deployments and agentic workflow

**Files:** Create `deployments/*.json`, `examples/openai_client.py`, `examples/agentic_session.py`, `.env.example`, `scripts/start_expertflow.ps1`, `scripts/stop_expertflow.ps1`; test `tests/test_product_deployments.py`.

- [ ] Write failing schema, path-portability, and command tests.
- [ ] Generate latency, measured throughput/context, and agentic manifests.
- [ ] Add health/client examples and run model-free contract tests.
- [ ] Commit deployment profiles.

### Task 5: Build offline judge UI and documentation

**Files:** Modify `README.md`; create `docs/evidence/product-release/dashboard.html`, submission documents, and judge guides; test `tests/test_product_dashboard.py`.

- [ ] Write failing offline-data and forbidden-claim tests.
- [ ] Render the ten required panels from committed JSON and update human-readable setup/reproduction docs.
- [ ] Serve locally, inspect offline load, and commit.

### Task 6: Assemble and verify release

**Files:** Create `scripts/build_product_release.py`, `release/expertflow-build-week/**`; test `tests/test_product_packaging.py`.

- [ ] Write failing allowlist, duplicate, hash, private-path, credential, and forbidden-large-file tests.
- [ ] Assemble the directory and ZIP, write SHA-256 manifest, and audit contents.
- [ ] Run applicable tests, CLI smoke tests, replay, dashboard, git, process, and cleanup checks.
- [ ] Commit final evidence and leave both release worktrees clean, unmerged, and unpushed.
