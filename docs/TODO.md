# ExpertFlow: remaining product roadmap

Updated 2026-10-06 on `ef-v2`. This is the forward product plan requested by the
user; the completed research checklist is retained below. Results and existing
support claims remain governed by [STATUS.md](STATUS.md). The earlier
[placement proof and stock fallback plan](superpowers/plans/2026-10-04-placement-proof-and-stock-fallback.md)
governs its closed studies, not the order of future product work.

**Goal:** someone outside this repository can install ExpertFlow, bring a local
GGUF, understand whether it fits, measure a suitable configuration within a
declared budget, and reliably run it or serve it to their existing client.
Useful setup, reproducibility and diagnosis must stand on their own when tuning
finds no improvement. Stars are an aspiration, not a release acceptance test.

The user authorized implementation on 2026-10-06: follow this list until the
product is ready for users. This supersedes the earlier planning-only state;
publication/outreach and each native experimental registration remain separate.
The user subsequently accepted a private Windows alpha first: complete engineering
and the tested-host artifact checks, then hand it to the first participant. The
five-user pilot, voluntary return use and second GPU/Linux qualification are
public-release gates, not blockers for the private alpha kit. No human results
are implied. After engineering, archive historical research under `docs/research/`,
remove tool attribution/session material, organize docs, and integrate `ef-v2`
into `main` through a reviewed PR; pushing/integration is authorized.
Execute milestones in order; use
`superpowers:executing-plans` for task-by-task work. Each milestone below states
the deliverable, code boundary, checks and stop condition. New experimental
protocols must contain their actual candidate list and fixed process budget
before collection; this roadmap does not substitute for those registrations.

## Scope decisions and product position

The execution request adopts these recommended working defaults; native and
human qualification must still establish the actual released support scope:

- Primary audience: local-model enthusiasts and developers using llama.cpp,
  especially people balancing RAM, VRAM, context and latency on one NVIDIA GPU.
- Windows/NVIDIA private alpha first; Windows and Linux/NVIDIA public beta only
  after native qualification on both. If Linux hardware is unavailable, release
  a clearly Windows-only alpha; never infer Linux GPU support from CI.
- GGUF text generation first. Existing Gemma/Granite MoE paths remain the initial
  tuning contracts; add one dense family for practical breadth after qualification.
- Exact-output tuning by default within a fixed reference execution contract.
  Changing quantization, context, placement, attention or KV policy is never
  silently classified as exact. Quality-bounded optimization is a separate later
  track, conditional on the user's policy decision and actual evidence.
- Local files first; downloading runtimes/models is explicit, resumable and
  checksum-verified. No accounts, cloud service, telemetry or prompt upload is
  required. Do not build a model marketplace or a full chat application.
- A thin optional TUI follows the functioning CLI and pilot feedback. It is not
  a prerequisite for the first useful release and contains no separate tuning logic.

Three approaches considered: continued compiler research offers differentiation
but currently has no accepted new acceleration mechanism; a full chat application
adds substantial UI work in an established category; a measured setup/tuning/run
companion reuses the strongest existing work and can deliver value sooner.
Choose the companion, retaining gated compiler research as a later differentiator.

Upstream already provides important pieces: [llama.cpp server and web UI](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[Ollama's run/API workflow](https://docs.ollama.com/quickstart), and
[LM Studio's per-model load settings](https://lmstudio.ai/docs/app/advanced/per-model).
Reviewed 2026-10-06. ExpertFlow's proposed value is explaining and measuring a
configuration on the user's hardware, preserving the result and launching it.
Compare against the selected runtime's real defaults, including any automatic
fit behavior; a flag wrapper alone is not sufficient differentiation.

## Architecture and public contract

Keep `compiler/` responsible for eligible, evidence-bound optimization and sealed
plans. Add portable orchestration under `product/`, sharing it between CLI and
any later TUI. Retain `stock/` as the historical registered-study interface.
Do not rewrite immutable collectors or weaken their source/identity checks to
make the new product portable. Archive exact old source dependencies where needed.

Proposed new commands are under `expertflow local` to avoid collisions with the
historical `profile`, `optimize`, positional `run`, `serve` and compiler commands:

```text
expertflow local doctor --model <file.gguf> --runtime <directory>
expertflow local setup --model <file.gguf> --runtime <directory> --context 4096
expertflow local bench --profile <profile.json>
expertflow local tune --profile <profile.json> --budget-seconds 600
expertflow local run --profile <profile.json>
expertflow local serve --profile <profile.json> --port 8080
expertflow local report --profile <profile.json> --json
```

These are planned interfaces, not commands that work today. `setup` creates a
launchable baseline profile without requiring tuning. Profiles record schema
version, GGUF/tokenizer identity, runtime/build/backend, host, resolved flags,
context, workload and sampling policy. Mutable file locations are distinct from
content identities. A portable stock launch profile is not an accepted compiler
`ExecutionPlan`; it may reference a validated plan and receipt after qualification.
Unknown families may receive runtime-supported launch/diagnostics only, explicitly
marked untuned; they must not be forced through the current MoE-only ModelIR.

Service results distinguish `RUNNABLE-UNTUNED`, `VERIFIED-IMPROVEMENT`,
`NO-MEASURABLE-GAIN`, `INCONCLUSIVE`, `UNSUPPORTED` and `ENVIRONMENT-BLOCKED`.
These new product labels do not rename historical scientific verdicts. JSON
includes scope, baseline, costs, reasons and next action; human output emphasizes
the usable result. Invalid/blocked actions use documented nonzero exit codes;
successfully completed neutral comparisons remain successful commands.

## P0 — Establish the user contract and preserve evidence

- [x] Record the working audience/platform/quality-policy defaults here. If no
  answer arrives during planning, retain the explicit assumptions above; resolve
  conflicting answers before implementing the affected platform or policy.
- [ ] Complete the existing [five-task handoff](stock-product-handoff.md) with
  an actual intended user. Retain failures and timings. This checks the old
  evidence workflow only; it is not the new product's usability gate.
- [ ] Map every release claim to an acceptance receipt; preserve closed negative
  studies and frozen source snapshots. Update `docs/PRODUCT.md` and `docs/STATUS.md`
  to distinguish the new product direction from verified current capabilities.
- [ ] Freeze a short pilot task sheet: install, inspect own model, create profile,
  run, connect a client, interpret a no-gain result, recover from a failed launch.
  Record current upstream setup effort on the same tasks as the comparison.

**Exit:** user/problem/support/policy contract is explicit; historical claims
remain unchanged. Human scheduling may proceed alongside P1–P4; do not block all
engineering waiting for a participant or mark the human gate complete by proxy.

## P1 — Portable installation and runtime discovery

**Files:** extend `pyproject.toml`, `src/expertflow/doctor.py`, `fetching.py` and
`cli/main.py`; add `product/paths.py`, `product/runtime.py`, `product/local_cli.py`
and `tests/test_local_install.py`, `tests/test_local_runtime.py`.

- [ ] Make the wheel work outside the repository with no `C:/models` paths,
  research databases, source checkout or historical release assets. Use platform
  user-data/cache locations plus explicit overrides; document upgrade/removal.
- [ ] Resolve user-supplied llama CLI/server binaries, version, backend and
  capabilities; pin tested releases in a manifest. Offer an explicit installer
  with digest checks, bounded retries and atomic completion; preserve bring-your-own.
- [ ] Extend doctor to explain RAM/VRAM/disk/runtime/driver failures and actions.
  Report other GPU workloads without closing applications. Separate static
  inspection from measurements that require an idle device.
- [ ] Test installed-wheel help/doctor from an empty directory on Windows/Linux,
  paths with spaces/non-ASCII, missing runtime/driver, offline mode, interrupted
  download, wrong checksum, unwritable cache and unknown runtime version.

**Exit:** a clean environment can install, discover hardware/runtime and produce
actionable diagnostics without development extras, Torch or research files.
Unknown runtimes may be diagnosed but cannot inherit tuning qualification.

## P2 — Bring a model and get a runnable baseline

**Files:** add `product/models.py`, `product/profiles.py`, `product/setup.py`,
`tests/test_local_models.py`, `tests/test_local_profiles.py`; reuse artifact
hashing and doctor primitives without changing existing model hashes or schemas.

- [ ] Inspect GGUF metadata safely, including split-file sets, tokenizer/chat
  template and runtime architecture support. Keep general launch metadata
  separate from the existing MoE adapter contract.
- [ ] Estimate weights, KV and runtime overhead at the requested context; label
  estimates with uncertainty and reserve. Use upstream fitting capabilities
  where supported, then record actual resolved settings and a real load probe.
- [ ] Show an explicit RAM/VRAM/context tradeoff when a load cannot fit. Never
  silently substitute quantization, shorten requested context or alter policy.
- [ ] Write a versioned profile atomically and retain prior revisions. Provide
  list/show/remove operations; deleting a profile never deletes model weights.
  Hash identity on registration and revalidate against mutation before sealed use.
- [ ] Validate malformed/truncated GGUFs, missing shards, unsupported architecture,
  bad templates, OOM, ambiguous binaries, changed weights/runtime and moved files.

**Exit:** a supported local model produces a real baseline response from an
installed wheel with no manual JSON editing. Unknown models fail clearly or run
as explicitly untuned stock; no fabricated compiler support or speed guarantee.

## P3 — Bounded measurement and useful tuning

**Files:** add `product/benchmark.py`, `product/tuning.py`, `product/reports.py`,
`tests/test_local_benchmark.py`, `tests/test_local_tuning.py`; integrate through
`compiler/stock_eligibility.py`, `stock_search.py`, `stock_validation.py`,
`evidence.py` and `runner.py` only with new versioned contracts/source snapshots.

- [ ] Define a per-run manifest before launch: resolved upstream baseline,
  eligible candidates, prompt/token/context/sampling identity, warmups, repeats,
  timeout, wall budget and maximum process count. Count failures against budget;
  a time limit truncates collection, never lowers statistical acceptance gates.
- [ ] Provide quick diagnostics and a separate confirmed tuning mode. Measure
  time to first token, prompt processing, decode speed, end-to-end latency,
  memory and total tuning cost; optimize one declared objective at a time.
  Existing compiler acceptance remains decode-TPS-only until separately extended.
- [ ] Restrict exact search to qualified controls per provider/runtime/host.
  Compare the same artifact/workload to real stock defaults; independently
  confirm winners using held-out inputs. Retain the baseline on no gain,
  instability, exhausted budget or a failed correctness/acceptance gate.
- [ ] Report paired uncertainty, baseline identity, search cost and break-even
  use (tuning cost divided by measured per-use savings where comparable).
  A 600-second budget need not yield a verified improvement; say so explicitly.
- [ ] Cache evidence only for matching identities; changed model, runtime,
  driver/backend, workload, context or numerical settings requires requalification.
  Cancellation leaves a readable partial report and terminates only owned children.
- [ ] Test token mismatch, noisy/inconclusive samples, default-optimal cases,
  partial logs, OOM, timeout, cancellation, stale cache and identity mutation.
  Freeze fresh native protocols for product qualification; do not reopen old grids.

**Exit:** baseline, gain and no-gain paths are equally usable. One independently
reconstructed fresh tuned result must pass before advertising tuning on its
exact support scope. If no new practical gain appears, ship setup/diagnosis
only if the human pilot shows value; do not keep searching to manufacture a win.

## P4 — Reliable local run and serving

**Files:** add `product/session.py`, `product/server.py`,
`tests/test_local_session.py`, `tests/test_local_server.py`; share validated
command construction with compiler runners where contracts permit. Do not reuse
historical `DEFAULT_DEPLOYMENT` as a current serving acceptance receipt.

- [ ] Run profiles with streamed output, resolved-command inspection, readable
  logs and owned process lifecycle. Recheck identities and available resources.
- [ ] Start pinned llama-server on loopback, wait for health, print the local
  endpoint and an example client request, and expose status/stop. Test streaming,
  cancellation, port collision, invalid template, failed health and crash cleanup.
- [ ] Reuse upstream web UI when available; document one existing API client.
  Initially qualify single-user text chat only. Network exposure requires explicit
  configuration and authentication; never silently bind to all interfaces.
- [ ] Qualify serving separately: CLI token/TPS evidence is not a serving claim.
  Run a fixed smoke/soak protocol covering multi-turn context, real prompt lengths,
  disconnects and restart; record TTFT/latency/memory with request counts and load.
- [ ] Verify profiles containing adversarial paths cannot inject shell commands;
  never kill an unrelated process, upload prompts, or include secrets in reports.

**Exit:** a user can chat through a real client, stop/restart safely and recover
from failures. Server tests cannot be satisfied solely with mocked subprocesses.

## P5 — Native coverage and independent pilot: release gate

**Files:** add `docs/local-quickstart.md`, `docs/support-matrix.md`,
`docs/evidence/local-product-pilot/` and `.github/workflows/local-product.yml`;
add `tests/test_local_wheel.py` and scoped native smoke entrypoints.

- [ ] Qualify an installed wheel on clean Windows/NVIDIA and Linux/NVIDIA hosts,
  with at least two distinct GPU capacities overall. Record exact OS, GPU, driver,
  runtime and artifact identities; label untested cells unsupported/experimental.
- [ ] Cover the existing two MoE families plus one explicitly selected dense
  GGUF family; select the dense artifact from pilot demand and runtime support,
  then pin its exact revision/hash before native work. Dense launch support does
  not require pretending it has MoE routing or inheriting an optimization contract.
- [ ] Exercise both fully resident and RAM-offloaded execution, short chat and
  genuinely processed longer prompts (target 4K/8K where supported), load failure,
  busy GPU, insufficient disk and stock-optimal tuning. Avoid an exhaustive grid:
  publish which combinations were actually tested and the resulting limits.
- [ ] Run model-free contract/packaging CI on Windows/Linux, plus separately
  scheduled native checks on qualified hosts. Historical replay stays separate.
- [ ] Recruit five intended users for the pilot (human recruitment, no automatic
  outreach). Target at least four completing install-to-response unaided within
  15 minutes with model/runtime files available; record download/setup time
  separately, all failures and user task timings. At least three must use their
  own setup; at least three should return voluntarily within two weeks.
- [ ] Ask users to identify no-gain vs improvement and one recovery action.
  Compare task time, errors and preference to their current/upstream workflow.
  If users see no practical value, fix the observed problem and repeat the pilot
  before investing in UI breadth or a launch campaign. Do not substitute agents
  for participants. These are proposed product gates, not scientific guarantees.

**Exit:** supported combinations are demonstrated and the small pilot shows
independent utility. Missing hardware/participants remain explicit open gates;
release scope may narrow, but checked boxes cannot stand in for missing evidence.

## P6 — Optional TUI, driven by pilot friction

**Decision:** useful eventually, unnecessary to establish the core product.
Build after P4 and pilot observation if at least two participants struggle with
CLI navigation/report interpretation or explicitly prefer a terminal dashboard.
If no such need appears, defer it and allow P7 to proceed without it.

**Files:** proposed `src/expertflow/tui/`, optional `tui` extra in `pyproject.toml`,
`tests/test_tui_flow.py`; entrypoint `expertflow tui`.

- [ ] Keep scope to model/profile selection, fit summary, budget selection,
  progress/cancel, gain/no-gain report and run/serve controls. Share the P1–P4
  services and structured events; do not shell-parse human CLI output.
- [ ] Use an established terminal framework selected at implementation time;
  keep it optional. Support keyboard-only navigation, narrow terminals, readable
  light/dark themes, no-color/headless fallback and clean interruption.
- [ ] Show requested vs resolved settings, evidence freshness and estimate vs
  measurement. Put detailed provenance behind an expandable view. Include no
  separate chat history, model marketplace, dashboard backend or account system.
- [ ] Test selection-to-run and cancellation with service fakes, then perform a
  real terminal walkthrough on supported platforms. Re-run the failed pilot tasks
  with humans; retain the TUI only if it improves completion or reduces confusion.

## P7 — Release, discovery and maintenance

**Files:** `README.md`, `pyproject.toml`, `docs/local-quickstart.md`, new
`CONTRIBUTING.md`, `CHANGELOG.md`, issue templates and release CI.

- [ ] Rewrite the landing page around one demonstrable promise and the shortest
  working path. Add a short real demo, support matrix, install/run examples,
  neutral-result example and troubleshooting. Move historical research detail
  behind clearly labeled links; keep evidence and attribution available.
- [ ] Build reproducible wheel/sdist artifacts, lock dependencies, publish
  checksums and native-runtime licensing/source notices. Test the exact release
  artifacts from clean environments before tagging or distribution.
- [ ] Document compatibility/versioning, profile migrations, rollback and an
  upstream-runtime update qualification procedure. Never silently upgrade a
  pinned runtime under an accepted profile; keep last-known-working versions.
- [ ] Prepare a release post and reproducible comparison with upstream defaults,
  showing tuning cost and negative cases alongside any gain. Have a contributor
  path for model adapters and opt-in redacted support bundles; exclude prompts,
  access tokens, usernames and local paths by default.
- [ ] After explicit publication/outreach authorization, release and share with
  relevant local-model communities. Track voluntary reports of successful runs,
  repeat use and resolved issues; use stars/downloads as secondary signals.
  Do not add background telemetry to measure adoption.
- [ ] Review pilot/user issues after two weeks; prioritize the three most common
  observed blockers. Assign a maintainer to triage compatibility failures and
  qualify runtime updates before expanding the supported matrix.

**Release definition:** an outsider can install, inspect their model, create a
baseline, optionally tune, run/serve, understand the outcome and recover from
common failures without this checkout or its author. P1–P5 and release-artifact
checks are mandatory for the declared scope; TUI and research breakthroughs are not.

## P8 — Conditional optimization and expansion, after utility is established

- [ ] Quality-bounded optimization: only with an explicit separate policy,
  datasets, quality thresholds, held-out tests, strongest-stock comparison and
  fixed budget. Investigate one mechanism at a time; a changed quantization is
  a separate model artifact, not a quality-preserving speedup.
- [ ] Prioritize KV/memory/context work only when user workloads demonstrate
  that constraint. Compare upstream implementations first; TurboQuant is a
  candidate requiring feasibility/quality/runtime review, not a promised feature.
- [ ] Consider speculative decoding/MTP only where the runtime/model supports it
  and draft memory plus end-to-end benefit pass a bounded protocol. Reopen static
  placement or dynamic residency only with a new mechanism that addresses the
  recorded no-go; do not rerun closed hypotheses with larger budgets.
- [ ] Add Apple Silicon, AMD, CPU-only, multi-GPU or additional model families
  one demand-backed support slice at a time, each with a native owner and gates.
  Ollama/LM Studio integration starts with documented interoperability only;
  controlling their runtimes requires a separate tested adapter.

**Terminal rule:** finish a useful maintained local companion before adding
research breadth. Any failed experiment preserves its verdict and returns work
to demonstrated user needs. No feature above promises stars, universal model
support, global optimality or an automatic speedup.

## Verification and execution discipline

For each implementation milestone: add focused failure/contract tests, implement,
run its named test files, inspect the diff, and record the result before moving
on. At integration boundaries run `uv run --no-sync pytest -q`,
`uv run --no-sync python -m compileall -q src/expertflow` and `git diff --check`.
Build/install the wheel in a fresh environment for P1/P5/P7. Native tests require
separate frozen protocols and owned-process cleanup; model-free tests never
complete native or human gates. Commit coherent changes only after reviewing
scope; preserve unrelated working-tree files and stay on the authorized branch.

After each executed milestone update this file, `STATUS.md`, relevant claims and
`PROJECT_LOG.md`. This planning edit changes no historical verdict or checkbox.

---

## Historical research checklist (retained)

## Delivered

- [x] Compiler spine and owned/evidence-bound runtime validation.
- [x] Fresh accepted stock plans and bounded searches for Gemma Q6 and Q4.
- [x] Real Granite MoE adapter, numerical scope, reference, acceptance and search.
- [x] Close thread/prefetch/PDL/cache experiments with their original verdicts.
- [x] Align current documentation with accepted results and proof-first priorities.

## Closed placement feasibility; inactive experiment branch

- [x] Audit numerical-path differences and reconstruct the failed held-out quality gate.
- [x] Document scoped placement no-go (plan step 1): no new exact mechanism under pinned kernels.
- [ ] Freeze matched controls, numerical/quality policy, datasets, all native
  process budgets and acceptance gates before a new experiment (step 2).
- [ ] Run one bounded experiment and independent raw-evidence audit; retain failures.
- [ ] If it passes, demonstrate automatic compiler selection, sealed execution,
  and a separately budgeted held-out transfer test (step 3).

## Executed stock-autotuning fallback

- [x] Activate after scoped placement no-go; current claim remains validated selection/reproduction.
- [x] Freeze defaults/manual-tuning comparisons, search costs, eligible controls
  and unseen test inputs (step 4).
- [x] Implement and review the collector; reproduce/fix all three review
  findings; verify 767 tests and six applicable native source checks.
- [x] Execute and independently audit the frozen main comparison: +12.70%
  over resolved defaults, manual equivalence, equal 18-evaluation grids.
- [x] Attempt fresh paired product acceptance and retain its inconclusive
  result; close collection at 106/107 processes without retries/discards.
- [x] Choose the narrower validated selection/reproduction product scope
  pending end-to-end acceptance; report the narrow default-tuning gain.
- [x] Register and implement repeatability/acceptance under a separately reviewed
  protocol with a new justification and fixed budget before any further native run.
  [Follow-up protocol](superpowers/specs/2026-10-04-stock-repeatability.md) is
  authorized; all five review findings fixed, 797 tests/seven optional source skips
  and six pinned native source checks verified. Native collection completed
  all 148 calls from `86388e7`; [execution state](evidence/stock-repeatability-20261004/execution-state.md).
- [x] Qualify two independent paced acceptance blocks; stop on the first
  failed/inconclusive gate and independently reconstruct all retained evidence.
- [x] Collect both fresh consumers and the untouched transfer's full 107-call
  utility/product sequence; native gates passed at +9.38% held-out defaults gain.
- [x] Finish frozen outer reconstruction and final independent raw audit: PASS,
  148 native calls, 41 frozen files and six historical pins intact.
- [x] Finish fresh read-only CLI validation and verified evidence/documentation;
  exit0, no extra native calls, all source/history pins unchanged.
- [x] Consolidate the validated workflow into public CLI/decision reports.

Placement steps 2–3 are inactive after the feasibility rejection. The original
fallback remains **PRODUCT-VALIDATION-STOP**, with product CI90 [+0.13%, +2.34%]
outside the fixed ±2% margin. The separate follow-up has completed native gates
and independent raw reconstruction. Outer source/phase/receipt reconstruction
passed; the bounded method is qualified under fixed spacing/diagnostics.
See [follow-up results](evidence/stock-repeatability-20261004/report.md)
and [original terminal evidence](evidence/stock-utility-20261004/report.md).
Never reuse either study's budget for additional samples or candidates.

## Historical stock product work

- [x] Reuse verified readers per database in public read-only validation;
  preserve measured source snapshots and all artifact/model checks.
- [x] Publish coverage, defaults scope, tuning costs and invalidation reasons
  with the CLI workflow. Keep neutral/default-optimal outcomes explicit.
- [x] Register broader utility coverage before new workload/model/host runs;
  Q4 and Granite compatibility does not establish defaults gain on those inputs.
- [x] Implement/review the separate wider collector and freeze its source,
  live identities and all four case roots before native collection.
  [Implementation plan](superpowers/plans/2026-10-05-wider-stock-collector.md):
  guarded per-case journal, complete/prefix reconstruction, public sequence
  commands and one independent implementation review.
- [x] Execute the [four registered cases](superpowers/specs/2026-10-05-stock-coverage.md)
  from `78ad5f0`; retain all 344 calls and four NO-UTILITY-GAIN outcomes.
  No conditional product/consumer ran; 84 unspent calls cannot fund retries.
- [x] Run fresh public read-only reconstruction: PASS, 93.40 seconds, zero new
  calls; all 63 source files and original 41 files/six history pins/148 records unchanged.
- [x] Finish independent raw audit: PASS, all 344 records/3,440 artifact hashes,
  no material discrepancies; publish the final wider evidence/docs.

Public CLI and reader-reuse verification are recorded in
[the implementation report](evidence/stock-cli-20261005/report.md). Wider native
results are [recorded separately](evidence/stock-coverage-20261005/report.md);
none met the practical utility gate. Serving and other hosts remain unverified.

## Historical research gates and deferred extensions

- [x] Audit wider offload/attention/batch operation paths and numerical
  eligibility: [scoped no-go](evidence/stock-control-scope-20261006/report.md),
  16 pinned upstream objects, zero native calls, 55 contract tests passed.
- [ ] Blocked native extension: establish complete exact arithmetic equivalence
  and a useful mechanism for a new control, or define a separate quality
  policy/provider, datasets and held-out gates. Register a new fixed budget
  before collection; do not retune or reuse the completed studies.
- [x] Test the explicit-enable attention mechanism prerequisite from pinned
  source: [no new supported fused-decode mechanism](evidence/stock-followthrough-20261006/report.md),
  492 existing launch records preserved and zero new model processes.
- [x] Continue the fail path into readable archived stock outcomes and delegated
  public verification; 55 focused helper/CLI/reader tests passed after one review,
  its fix pass and a live argument-binding correction. No compiler/provider changes.
- [x] Prepare executable status/verification commands and a five-task
  [independent-user handoff](stock-product-handoff.md).
- [x] Qualify both studies through fresh public read-only validation:
  [PASS-LOCAL-STOCK-QUALIFICATION](evidence/stock-followthrough-20261006/verification.json),
  331.65 seconds, zero extra native calls; failed initial attempt preserved.
- [ ] Obtain actual independent-user usability results; continue onboarding
  on pass, or fix the observed workflow problem on fail. Agent checks do not
  complete this product gate. The user attempted verification, then requested
  agent help with the questions; [state](evidence/stock-followthrough-20261006/usability-state.json)
  records no five-task unaided success or human timings.
- [x] Complete the requested Luna walkthrough and full agent-run reconstruction
  after authorized GPU-app closure: [PASS in 313.51 seconds](evidence/stock-agent-walkthrough-20261006/report.md),
  zero extra model calls, earlier blocked attempts retained. Add GPU-stop guidance
  without changing native guards or scientific thresholds.
- [ ] KV compression/TurboQuant and shared memory-budget optimization (conditional P8).
- [ ] MTP/speculation, two-table dynamic residency and joint search (conditional P8).
- [ ] Broad family/hardware coverage, serving integration and presentation polish
  (split into scoped P4/P5/P7 release work and conditional P8 expansion above).

After every decision, append the result to `PROJECT_LOG.md` and update this list,
`STATUS.md` and the relevant claims. Checkboxes indicate completed work only;
writing a protocol or passing CPU tests does not complete a native proof gate.
