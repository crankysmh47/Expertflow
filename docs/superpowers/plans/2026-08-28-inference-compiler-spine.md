# ExpertFlow Inference Compiler Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 0–3 ExpertFlow compiler spine that inspects a Gemma 4 GGUF inventory, models the machine and workload with typed IRs, searches and records a strongest-stock floor, compiles measured static-MoE placement candidates, and emits a sealed, explainable execution plan for the pinned ExpertFlow llama.cpp runtime.

**Architecture:** Add a focused `expertflow.compiler` package around existing artifact, baseline, benchmark, Q6 inventory, and placement modules. The package uses immutable typed IRs, a dependency-checked pass manager, append-only evidence, measured candidate selection, and a strict plan validator; existing research and product commands remain compatible. This plan stops after the required static-placement compiler product—CUDA autotuning, KV/TurboQuant, extra model adapters, MTP, and dynamic caching remain independent later plans.

**Tech Stack:** Python 3.11+, standard-library dataclasses/enums/protocols/JSON/hashlib/sqlite3, pytest 8.4, uv, GGUF metadata exported by the pinned llama.cpp `gguf-py`, Windows 11 x64, one NVIDIA RTX 5060 Ti, CUDA 12.8, pinned ExpertFlow llama.cpp fork.

**Spec:** `docs/superpowers/specs/2026-08-28-expertflow-inference-compiler-design.md`

## Global Constraints

- Target only one NVIDIA CUDA GPU; CPU and Vulkan are verification/fallback mechanisms, not optimization targets.
- Target llama.cpp/GGUF only through a maintained pinned ExpertFlow fork.
- Optimize maximum single-request decode TPS at concurrency one.
- Gemma 4 26B A4B is the first validated adapter; generic passes may not contain Gemma/Qwen tensor-name checks.
- Exact is the default policy. Q8/Q4 KV and every other numerically lossy change require an explicitly separate approximate plan.
- Strongest compatible measured stock llama.cpp is the comparison floor.
- Static placement precedes dynamic residency; this plan contains no dynamic cache, predictor, prefetch, KV-format search, MTP, or custom CUDA optimization.
- Estimated candidates may be pruned or rejected, but only measured candidates may be sealed.
- Every model, runtime, workload, hardware, evidence, and plan identity is hashed and fail-closed.
- Preserve all unrelated working-tree changes and historical evidence. Stage only files named by the active task.
- Heavy model processes run sequentially with hidden windows, file-only logs, explicit exit codes, cleanup checks, and no ordinary-run exclusion.
- Stop at the first task exit-gate failure; record the failure rather than weakening a gate.

---

## File and module map

Create the following focused modules:

- `src/expertflow/compiler/schema.py`: immutable IR and identity types plus canonical serialization.
- `src/expertflow/compiler/adapters/base.py`: model descriptor, adapter protocol, and registry.
- `src/expertflow/compiler/adapters/gemma4.py`: Gemma 4 inventory normalization only.
- `src/expertflow/compiler/plan.py`: candidate/execution-plan types, validation, sealing, and loading.
- `src/expertflow/compiler/passes/base.py`: compiler state, pass protocol, dependency resolution, and execution.
- `src/expertflow/compiler/evidence.py`: append-only SQLite measurement records and artifact hashes.
- `src/expertflow/compiler/cost_model.py`: calibrated estimates, residuals, uncertainty, and exploration selection.
- `src/expertflow/compiler/stock.py`: stock candidate matrix, measurement import, and strongest-floor selection.
- `src/expertflow/compiler/static_placement.py`: generic layer ranking, bounded placement candidates, and lowering fields.
- `src/expertflow/compiler/pipeline.py`: Phase 0–3 orchestration and diagnostic report.
- `src/expertflow/compiler/commands.py`: command handlers kept out of the already-large CLI parser.
- `src/expertflow/compiler/__init__.py`, `adapters/__init__.py`, `passes/__init__.py`: public package exports.

Modify:

- `src/expertflow/cli/main.py`: parser/dispatch wiring only.
- `pyproject.toml` and `uv.lock`: reproducible quality-test environment.
- `PROJECT_LOG.md`: append-only final implementation evidence only in Task 11.

Do not move or rewrite existing `expertflow.analysis`, `expertflow.runtime`, or `expertflow.product` modules in this plan.

---

### Task 1: Reproducible Phase 0 environment and reference workload

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `configs/compiler/gemma4-q6-single-request.json`
- Create: `configs/compiler/runtime-fork.json`
- Create: `tests/test_compiler_reference_config.py`

**Interfaces:**
- Consumes: `configs/baseline-prompt.txt`, existing release evidence, Python 3.11.
- Produces: committed compiler workload schema `1.0.0` and a quality extra that cannot import an incompatible global pandas build.

- [ ] **Step 1: Write the failing reference-config and environment tests**

```python
import json
from pathlib import Path


def test_reference_workload_is_single_request_exact_q6() -> None:
    value = json.loads(
        Path("configs/compiler/gemma4-q6-single-request.json").read_text(
            encoding="utf-8"
        )
    )
    assert value["schema_version"] == "1.0.0"
    assert value["objective"] == "decode_tps"
    assert value["concurrency"] == 1
    assert value["policy"] == "exact"
    assert value["predict_tokens"] == 512
    assert value["temperature"] == 0.0
    assert value["prompt_file"] == "configs/baseline-prompt.txt"


def test_quality_extra_pins_a_local_pandas_build() -> None:
    document = Path("pyproject.toml").read_text(encoding="utf-8")
    quality = document.split("quality = [", 1)[1].split("]", 1)[0]
    assert 'pandas>=' in quality


def test_runtime_fork_manifest_pins_reproducible_patch_stack() -> None:
    value = json.loads(
        Path("configs/compiler/runtime-fork.json").read_text(encoding="utf-8")
    )
    assert value["schema_version"] == "1.0.0"
    assert value["execution_plan_abi"] == "1.0.0"
    assert value["upstream_commit"] == "a7312ae94f801fc9c6786dc56e38df57b964f697"
    assert value["expertflow_commit"] == "451224ab4d12a616dc3e16e8c8063f4b331f531c"
    assert [patch["path"] for patch in value["patches"]] == [
        f"release/expertflow-build-week/patches/llama.cpp/000{i}-{name}.patch"
        for i, name in enumerate(
            (
                "feat-restore-disabled-static-CUDA-expert-island",
                "feat-capture-exact-Q1b-token-NLL",
                "feat-bound-static-islands-to-four-layers",
                "feat-add-bounded-Q6-split-profiler",
                "feat-expand-bounded-static-island-capacity",
                "perf-precompute-static-layer-membership",
            ),
            start=1,
        )
    ]
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `uv run pytest -q tests/test_compiler_reference_config.py`

Expected: FAIL because the config is absent and `quality` does not declare pandas.

- [ ] **Step 3: Add the exact reference configuration and dependency**

Create `configs/compiler/gemma4-q6-single-request.json` exactly:

```json
{
  "schema_version": "1.0.0",
  "objective": "decode_tps",
  "concurrency": 1,
  "policy": "exact",
  "prompt_file": "configs/baseline-prompt.txt",
  "context_size": 4096,
  "predict_tokens": 512,
  "threads": 12,
  "seed": 42,
  "temperature": 0.0,
  "warmup_runs": 1,
  "measured_runs": 3,
  "minimum_vram_reserve_mib": 256
}
```

Create `configs/compiler/runtime-fork.json` exactly. This is the compiler/runtime
compatibility boundary: changing the ABI, upstream base, patch order, build
identity, or binary hashes requires an explicit manifest change and rerunning the
Phase 3 gate.

```json
{
  "schema_version": "1.0.0",
  "execution_plan_abi": "1.0.0",
  "upstream_commit": "a7312ae94f801fc9c6786dc56e38df57b964f697",
  "expertflow_commit": "451224ab4d12a616dc3e16e8c8063f4b331f531c",
  "patches": [
    {"path": "release/expertflow-build-week/patches/llama.cpp/0001-feat-restore-disabled-static-CUDA-expert-island.patch", "sha256": "14b3042c8626a1a99e9d63b2dcc3ea88a92168e505672fa56ee3f987215f87f8"},
    {"path": "release/expertflow-build-week/patches/llama.cpp/0002-feat-capture-exact-Q1b-token-NLL.patch", "sha256": "38c690098d907b7228e3180516c2f541dd07944ece452824b02964ff09711ecb"},
    {"path": "release/expertflow-build-week/patches/llama.cpp/0003-feat-bound-static-islands-to-four-layers.patch", "sha256": "3278c45582df1cd6af4c3dac41756484269619a639903d78dd7b926dd3610e7c"},
    {"path": "release/expertflow-build-week/patches/llama.cpp/0004-feat-add-bounded-Q6-split-profiler.patch", "sha256": "58211a1e6bc82c76705b1f3af6628f98fdd0c2eea7e7278915ae222da1bfe461"},
    {"path": "release/expertflow-build-week/patches/llama.cpp/0005-feat-expand-bounded-static-island-capacity.patch", "sha256": "27f31cbce0369847dc616f6c5ca9ad96ed488a7014300a4b54825d59f20bf52d"},
    {"path": "release/expertflow-build-week/patches/llama.cpp/0006-perf-precompute-static-layer-membership.patch", "sha256": "b95870aa8937506b6476af7291eceec7987cb59c3c54671ddc358ffaf87a1c4e"}
  ],
  "build": {
    "generator": "Ninja",
    "configuration": "Release",
    "compiler": "MSVC v143 14.39.33519",
    "cuda": "12.8.93"
  },
  "binaries": {
    "llama-cli.exe": "5d68046dcd26e2fd018aaeaad5f99cdb7d88eca6fc10935925f1d660f7009407",
    "llama-server.exe": "22ecc4f64f91dcbe3a1cfe7d9d4617e43467ea7f3c6fa1ba2c6ad8d07e89334e"
  }
}
```

Add `"pandas>=2.2,<3"` to `[project.optional-dependencies].quality`, then run `uv lock`.

- [ ] **Step 4: Verify the isolated quality environment and tests**

Run: `uv sync --frozen --extra quality`

Run: `uv run python -c "import numpy,pandas,pyarrow; print(numpy.__version__, pandas.__version__, pyarrow.__version__)"`

Run: `uv run pytest -q tests/test_quality_dataset.py tests/test_compiler_reference_config.py`

Expected: imports succeed and both test files pass.

- [ ] **Step 5: Commit only Phase 0 environment files**

```powershell
git add pyproject.toml uv.lock configs/compiler/gemma4-q6-single-request.json configs/compiler/runtime-fork.json tests/test_compiler_reference_config.py
git commit -m "build: stabilize compiler reference environment"
```

---

### Task 2: Typed identities and compiler IRs

**Files:**
- Create: `src/expertflow/compiler/__init__.py`
- Create: `src/expertflow/compiler/schema.py`
- Create: `tests/test_compiler_schema.py`

**Interfaces:**
- Consumes: plain Python primitives and resolved artifact identities.
- Produces: `ArtifactIdentity`, `MoELayerIR`, `ModelIR`, `HardwareIR`, `WorkloadIR`, `ExactnessPolicy`, `Objective`, `canonical_payload()`, and `canonical_sha256()`.

- [ ] **Step 1: Write failing IR round-trip, ordering, and validation tests**

```python
from expertflow.compiler.schema import (
    ArtifactIdentity,
    ExactnessPolicy,
    ModelIR,
    MoELayerIR,
    Objective,
    canonical_sha256,
)


def test_model_ir_hash_is_stable_and_layer_order_is_canonical() -> None:
    identity = ArtifactIdentity(
        path="C:/models/gemma.gguf", size_bytes=10, sha256="a" * 64
    )
    model = ModelIR(
        schema_version="1.0.0",
        identity=identity,
        family="gemma4",
        architecture="gemma4-moe",
        quantization="Q6_K",
        expert_count=128,
        expert_top_k=8,
        moe_layers=(
            MoELayerIR(1, 128, 8, 30, 3_840),
            MoELayerIR(0, 128, 8, 20, 2_560),
        ),
        kv_kind="standard",
        mtp_kind="none",
    )
    assert [layer.layer_id for layer in model.moe_layers] == [0, 1]
    assert canonical_sha256(model) == canonical_sha256(model)


def test_exactness_and_objective_are_closed_enums() -> None:
    assert ExactnessPolicy.EXACT.value == "exact"
    assert Objective.DECODE_TPS.value == "decode_tps"
```

Also test rejection of invalid SHA-256, duplicate layers, nonpositive bytes, `top_k > expert_count`, concurrency other than one, and approximate settings inside an exact `WorkloadIR`.

- [ ] **Step 2: Run schema tests and verify RED**

Run: `uv run pytest -q tests/test_compiler_schema.py`

Expected: collection fails with `ModuleNotFoundError: expertflow.compiler`.

- [ ] **Step 3: Implement minimal immutable IRs and canonical serialization**

Use frozen, slotted dataclasses and closed enums:

```python
class ExactnessPolicy(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"


class Objective(str, Enum):
    DECODE_TPS = "decode_tps"


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class MoELayerIR:
    layer_id: int
    expert_count: int
    expert_top_k: int
    expert_bundle_bytes: int
    routed_expert_bank_bytes: int


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        canonical_payload(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

Sort `ModelIR.moe_layers` by `layer_id` in `__post_init__` using `object.__setattr__`; reject duplicate IDs and inconsistent expert/top-k values.

- [ ] **Step 4: Run focused and existing inventory tests**

Run: `uv run pytest -q tests/test_compiler_schema.py tests/test_q6_inventory.py tests/test_expert_layout.py`

Expected: PASS.

- [ ] **Step 5: Commit the typed compiler schema**

```powershell
git add src/expertflow/compiler/__init__.py src/expertflow/compiler/schema.py tests/test_compiler_schema.py
git commit -m "feat: add typed compiler intermediate representations"
```

---

### Task 3: Candidate and sealed execution-plan contracts

**Files:**
- Create: `src/expertflow/compiler/plan.py`
- Create: `tests/test_compiler_plan.py`

**Interfaces:**
- Consumes: Task 2 IR hashes and primitive runtime settings.
- Produces: `CandidateStatus`, `StaticPlacement`, `RuntimeSettings`, `CandidatePlan`, `ExecutionPlan`, `seal_candidate()`, `load_execution_plan()`, and `validate_execution_plan()`.

- [ ] **Step 1: Write failing plan-sealing tests**

```python
def test_only_measured_passing_candidate_can_be_sealed() -> None:
    candidate = candidate_fixture(
        status=CandidateStatus.MEASURED,
        measurement_ids=("m-1", "m-2", "m-3"),
        validation={"exact_tokens": True, "memory": True, "cleanup": True},
    )
    sealed = seal_candidate(candidate, fallback_id="stock-1")
    assert sealed.schema_version == "1.0.0"
    assert sealed.plan_sha256 == canonical_sha256(sealed.without_hash())


@pytest.mark.parametrize(
    "status", [CandidateStatus.ESTIMATED, CandidateStatus.REJECTED]
)
def test_unmeasured_or_rejected_candidate_cannot_be_sealed(status) -> None:
    with pytest.raises(ValueError, match="measured passing candidate"):
        seal_candidate(candidate_fixture(status=status), fallback_id="stock-1")
```

Also test identity mismatch, missing evidence, exact plan containing approximate settings, unsupported schema, tampered plan hash, and fallback incompatibility.

- [ ] **Step 2: Run plan tests and verify RED**

Run: `uv run pytest -q tests/test_compiler_plan.py`

Expected: import fails because `expertflow.compiler.plan` is absent.

- [ ] **Step 3: Implement minimal plan types and fail-closed validation**

`CandidatePlan` must carry model/hardware/workload/runtime hashes, status, settings, estimates, measurement IDs, rejection reasons, and validation results. `ExecutionPlan` adds compiler version, fallback ID, canonical hash, and sealed timestamp. Serialize enums as strings and tuples as arrays.

- [ ] **Step 4: Run focused plan/schema tests**

Run: `uv run pytest -q tests/test_compiler_schema.py tests/test_compiler_plan.py`

Expected: PASS.

- [ ] **Step 5: Commit plan contracts**

```powershell
git add src/expertflow/compiler/plan.py tests/test_compiler_plan.py
git commit -m "feat: add sealed execution plan contracts"
```

---

### Task 4: Generic model-adapter registry and Gemma 4 normalization

**Files:**
- Create: `src/expertflow/compiler/adapters/__init__.py`
- Create: `src/expertflow/compiler/adapters/base.py`
- Create: `src/expertflow/compiler/adapters/gemma4.py`
- Create: `tests/test_compiler_gemma4_adapter.py`

**Interfaces:**
- Consumes: metadata-only inventory emitted by `scripts/inventory_q6_gguf.py` and a `ModelDescriptor` containing family, architecture, quant, expert count, top-k, KV kind, and MTP kind.
- Produces: `ModelAdapter.normalize(descriptor, inventory, identity) -> ModelIR`, `AdapterRegistry.register()`, `AdapterRegistry.resolve()`, and built-in key `gemma4`.

- [ ] **Step 1: Write failing registry and Gemma normalization tests**

```python
def test_gemma_adapter_normalizes_inventory_without_raw_names_in_model_ir() -> None:
    adapter = AdapterRegistry.with_builtins().resolve("gemma4")
    model = adapter.normalize(descriptor_fixture(), inventory_fixture(), identity_fixture())
    assert model.family == "gemma4"
    assert len(model.moe_layers) == 2
    assert model.moe_layers[0].expert_bundle_bytes == 30
    assert model.expert_count == 128
    assert model.expert_top_k == 8


def test_adapter_rejects_inconsistent_component_sets() -> None:
    inventory = inventory_fixture(component_sets_consistent=False)
    with pytest.raises(ValueError, match="component sets"):
        Gemma4Adapter().normalize(descriptor_fixture(), inventory, identity_fixture())
```

Also test wrong family, noncontiguous/duplicate layer IDs, expert-count mismatch, absent routed layers, inconsistent per-layer bundle bytes, and duplicate adapter registration.

- [ ] **Step 2: Run adapter tests and verify RED**

Run: `uv run pytest -q tests/test_compiler_gemma4_adapter.py`

Expected: import fails because adapter modules are absent.

- [ ] **Step 3: Implement the protocol, registry, and Gemma-only parser**

```python
class ModelAdapter(Protocol):
    family: str

    def normalize(
        self,
        descriptor: ModelDescriptor,
        inventory: Mapping[str, object],
        identity: ArtifactIdentity,
    ) -> ModelIR: ...
```

Only `gemma4.py` may inspect Gemma-specific family metadata. It consumes the inventory's normalized layer/component byte accounting and emits only `MoELayerIR` values.

- [ ] **Step 4: Run adapter, inventory, and placement tests**

Run: `uv run pytest -q tests/test_compiler_gemma4_adapter.py tests/test_q6_inventory.py tests/test_q6_placement.py`

Expected: PASS.

- [ ] **Step 5: Commit the first model adapter**

```powershell
git add src/expertflow/compiler/adapters tests/test_compiler_gemma4_adapter.py
git commit -m "feat: normalize Gemma MoE models for compilation"
```

---

### Task 5: Dependency-checked compiler pass manager

**Files:**
- Create: `src/expertflow/compiler/passes/__init__.py`
- Create: `src/expertflow/compiler/passes/base.py`
- Create: `tests/test_compiler_pass_manager.py`

**Interfaces:**
- Consumes: `CompilerState(model, hardware, workload, candidates, analyses, diagnostics)`.
- Produces: `CompilerPass`, `PassResult`, `PassManager.resolve()`, and `PassManager.run()`.

- [ ] **Step 1: Write failing pass-order and rejection tests**

```python
def test_pass_manager_topologically_orders_declared_capabilities() -> None:
    manager = PassManager([PlacementPass(), BaselinePass(), InspectPass()])
    assert [item.name for item in manager.resolve()] == [
        "inspect", "stock-baseline", "static-placement"
    ]


def test_missing_capability_and_cycle_fail_before_execution() -> None:
    with pytest.raises(ValueError, match="missing capability"):
        PassManager([PlacementPass()]).resolve()
    with pytest.raises(ValueError, match="cycle"):
        PassManager([CycleA(), CycleB()]).resolve()
```

Also test deterministic tie-breaking by pass name, explicit conflicts, duplicate pass names, a pass returning undeclared capabilities, and diagnostics preserved after a no-op/rejected pass.

- [ ] **Step 2: Run pass-manager tests and verify RED**

Run: `uv run pytest -q tests/test_compiler_pass_manager.py`

Expected: import failure.

- [ ] **Step 3: Implement the minimal pass protocol and manager**

```python
class CompilerPass(Protocol):
    name: str
    requires: frozenset[str]
    provides: frozenset[str]
    conflicts: frozenset[str]

    def run(self, state: CompilerState) -> PassResult: ...
```

Resolve dependencies without third-party graph packages. Do not mutate an input state in place; return a replaced frozen state.

- [ ] **Step 4: Run pass and plan suites**

Run: `uv run pytest -q tests/test_compiler_pass_manager.py tests/test_compiler_plan.py`

Expected: PASS.

- [ ] **Step 5: Commit the pass manager**

```powershell
git add src/expertflow/compiler/passes tests/test_compiler_pass_manager.py
git commit -m "feat: add dependency checked compiler passes"
```

---

### Task 6: Append-only measurement store

**Files:**
- Create: `src/expertflow/compiler/evidence.py`
- Create: `tests/test_compiler_evidence.py`

**Interfaces:**
- Consumes: `MeasurementKey`, candidate ID, raw artifact paths/hashes, metrics, exit status, and validation results.
- Produces: `EvidenceStore(path)`, `append_measurement() -> str`, `measurements_for(key)`, `measurement(id)`, and `verify_artifacts(id)`.

- [ ] **Step 1: Write failing append-only and identity tests**

```python
def test_measurements_are_append_only_and_key_scoped(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "compiler.sqlite3")
    first = store.append_measurement(record_fixture(candidate_id="stock-a"))
    second = store.append_measurement(record_fixture(candidate_id="stock-b"))
    assert first != second
    assert [row.candidate_id for row in store.measurements_for(key_fixture())] == [
        "stock-a", "stock-b"
    ]
    with pytest.raises(ValueError, match="append-only"):
        store.replace_measurement(first, record_fixture())
```

Also test canonical key hashing, transaction rollback, duplicate explicit ID rejection, artifact hash mismatch, different workload/model key isolation, and stable chronological ordering.

- [ ] **Step 2: Run evidence tests and verify RED**

Run: `uv run pytest -q tests/test_compiler_evidence.py`

Expected: import failure.

- [ ] **Step 3: Implement SQLite schema and artifact verification**

Use only `sqlite3`. Create tables `measurement`, `artifact`, and `validation` with foreign keys enabled. Store canonical JSON payloads and SHA-256 values; expose no update/delete API.

- [ ] **Step 4: Run evidence and artifact tests**

Run: `uv run pytest -q tests/test_compiler_evidence.py tests/test_artifacts.py`

Expected: PASS.

- [ ] **Step 5: Commit the evidence store**

```powershell
git add src/expertflow/compiler/evidence.py tests/test_compiler_evidence.py
git commit -m "feat: add append only compiler evidence store"
```

---

### Task 7: Calibrated cost model and exploration quota

**Files:**
- Create: `src/expertflow/compiler/cost_model.py`
- Create: `tests/test_compiler_cost_model.py`

**Interfaces:**
- Consumes: `CostEstimate(candidate_id, metrics, uncertainty)`, measured residuals, hard constraints, and measurement budget.
- Produces: `CalibrationState`, `update_calibration()`, `select_measurement_batch()`, and `CalibrationDecision` with winner/boundary/uncertain/sentinel reasons.

- [ ] **Step 1: Write failing exploration and pruning-suspension tests**

```python
def test_batch_contains_winner_boundary_uncertain_and_sentinel() -> None:
    decision = select_measurement_batch(
        estimates_fixture(), budget=4, sentinel_period=1
    )
    assert {item.reason for item in decision.selected} == {
        "predicted_winner", "constraint_boundary", "max_uncertainty", "sentinel"
    }


def test_large_sentinel_residual_suspends_pruning() -> None:
    state = update_calibration(
        CalibrationState.empty(error_threshold_pct=10.0),
        predicted={"decode_tps": 30.0},
        measured={"decode_tps": 20.0},
        candidate_role="sentinel",
    )
    assert state.pruning_suspended is True
```

Also test exhaustive selection when candidates fit the budget, deterministic selection, no cross-key calibration, nonfinite metric rejection, and `inconclusive` when uncertainty remains excessive.

- [ ] **Step 2: Run cost-model tests and verify RED**

Run: `uv run pytest -q tests/test_compiler_cost_model.py`

Expected: import failure.

- [ ] **Step 3: Implement conservative residual calibration**

Use analytical estimates unchanged as the initial mean, observed absolute percentage residual as empirical error, and the maximum of declared/empirical uncertainty for pruning. Do not introduce NumPy, scikit-learn, or Bayesian optimization in this phase.

- [ ] **Step 4: Run cost/evidence tests**

Run: `uv run pytest -q tests/test_compiler_cost_model.py tests/test_compiler_evidence.py`

Expected: PASS.

- [ ] **Step 5: Commit calibrated candidate selection**

```powershell
git add src/expertflow/compiler/cost_model.py tests/test_compiler_cost_model.py
git commit -m "feat: calibrate compiler candidate selection"
```

---

### Task 8: Strongest-stock candidate generation and selection

**Files:**
- Create: `src/expertflow/compiler/stock.py`
- Create: `tests/test_compiler_stock.py`

**Interfaces:**
- Consumes: `HardwareIR`, `WorkloadIR`, runtime/model identities, `run_measured_baseline`, and performance-probe JSON parsed by `expertflow.benchmark.performance.parse_probe_result`.
- Produces: `StockCandidate`, `stock_candidate_matrix()`, `import_stock_measurement()`, and `select_strongest_stock()`.

- [ ] **Step 1: Write failing matrix and strongest-floor tests**

```python
def test_stock_matrix_is_bounded_and_deterministic() -> None:
    candidates = stock_candidate_matrix(hardware_fixture(), workload_fixture())
    assert [item.gpu_layers for item in candidates] == ["auto", "all", "99"]
    assert all(item.context_size == 4096 for item in candidates)
    assert len({item.candidate_id for item in candidates}) == 3


def test_strongest_stock_requires_measured_exact_memory_safe_runs() -> None:
    winner = select_strongest_stock(
        [
            stock_result("a", 22.9, exact=True, peak_mib=3000),
            stock_result("b", 99.0, exact=False, peak_mib=14000),
            stock_result("c", 23.1, exact=True, peak_mib=17000),
        ],
        total_vram_mib=16311,
        reserve_mib=256,
    )
    assert winner.candidate_id == "a"
```

Also test failed exit, missing repetitions, variance beyond configured tolerance, mismatched measurement key, missing cleanup, and deterministic tie-breaking by candidate ID.

- [ ] **Step 2: Run stock tests and verify RED**

Run: `uv run pytest -q tests/test_compiler_stock.py`

Expected: import failure.

- [ ] **Step 3: Implement bounded candidates and measured selection**

Build `BaselineRunConfig` values through the existing runtime module. Keep process execution injectable:

```python
class StockRunner(Protocol):
    def measure(
        self, candidate: StockCandidate, *, output_dir: Path
    ) -> StockMeasurement: ...
```

The production runner executes sequential warmup/measured processes, stores raw manifests through `EvidenceStore`, and imports probe JSON. Unit tests use a fake runner; they never launch a model.

- [ ] **Step 4: Run stock and existing baseline/benchmark suites**

Run: `uv run pytest -q tests/test_compiler_stock.py tests/test_baseline_command.py tests/test_baseline_cli.py tests/test_performance_benchmark.py`

Expected: PASS.

- [ ] **Step 5: Commit strongest-stock compilation**

```powershell
git add src/expertflow/compiler/stock.py tests/test_compiler_stock.py
git commit -m "feat: compile strongest stock runtime floor"
```

---

### Task 9: Generic static-MoE placement pass

**Files:**
- Create: `src/expertflow/compiler/static_placement.py`
- Create: `tests/test_compiler_static_placement.py`

**Interfaces:**
- Consumes: `ModelIR`, strongest stock plan, repeated per-layer profile records, exact bytes per layer, VRAM budget, and a measurement runner.
- Produces: `LayerBenefit`, `StaticPlacementCandidate`, `rank_layer_benefits()`, `generate_static_candidates()`, `StaticPlacementPass`, and lowered `StaticPlacement(layers, arena_bytes)`.

- [ ] **Step 1: Write failing generic ranking and budget tests**

```python
def test_ranking_uses_measured_time_per_vram_byte_without_family_names() -> None:
    benefits = rank_layer_benefits(
        model_ir_fixture(),
        profile_rows=[
            {"layer_id": 0, "total_us": 1000},
            {"layer_id": 1, "total_us": 500},
        ],
    )
    assert [item.layer_id for item in benefits] == [0, 1]
    assert benefits[0].score_us_per_mib > benefits[1].score_us_per_mib


def test_candidate_generation_never_crosses_vram_reserve() -> None:
    candidates = generate_static_candidates(
        benefits_fixture(), available_bytes=3_000, reserve_bytes=1_000
    )
    assert candidates
    assert all(item.arena_bytes <= 2_000 for item in candidates)
```

Also test duplicate/missing profile layers, profile/backend mismatch, nonfinite timings, exact arena accounting, deterministic greedy prefixes, inclusion of the recorded twelve-layer regression candidate, and rejection when no layer fits.

- [ ] **Step 2: Run static-placement tests and verify RED**

Run: `uv run pytest -q tests/test_compiler_static_placement.py`

Expected: import failure.

- [ ] **Step 3: Implement generic ranking and bounded prefixes**

Port the useful calculation from `scripts/analyze_q6_layer_profile.py` into the package, replacing `shadow_bytes` constants with `MoELayerIR.routed_expert_bank_bytes`. Generate an empty/static-off control, the recorded `[0,1,2,3,4,5,6,7,8,9,15,20]` regression candidate when compatible, and score-ordered prefixes that fit the budget. Do not implement arbitrary subsets or exhaustive combinatorics in Phase 3.

- [ ] **Step 4: Run compiler and historical static-analysis tests**

Run: `uv run pytest -q tests/test_compiler_static_placement.py tests/test_q6_layer_profile.py tests/test_q6_selected_static_analysis.py tests/test_q6_placement.py`

Expected: PASS.

- [ ] **Step 5: Commit static placement compilation**

```powershell
git add src/expertflow/compiler/static_placement.py tests/test_compiler_static_placement.py
git commit -m "feat: compile bounded static MoE placement"
```

---

### Task 10: Phase 0–3 pipeline and compiler CLI

**Files:**
- Create: `src/expertflow/compiler/pipeline.py`
- Create: `src/expertflow/compiler/commands.py`
- Create: `tests/test_compiler_pipeline.py`
- Create: `tests/test_compiler_cli.py`
- Modify: `src/expertflow/cli/main.py`

**Interfaces:**
- Consumes: Tasks 2–9 services and existing JSON inventory/profile/evidence artifacts.
- Produces: `CompilationRequest`, `compile_phase3(request, runner, store) -> CompilationResult` plus CLI commands `inspect`, `compile`, `validate`, and `explain`.

- [ ] **Step 1: Write failing pipeline and CLI tests**

```python
def test_phase3_pipeline_emits_stock_floor_static_winner_and_rejections(tmp_path) -> None:
    result = compile_phase3(
        request_fixture(tmp_path), FakeRunner(), EvidenceStore(tmp_path / "db.sqlite3")
    )
    assert result.execution_plan.settings.static_placement.layers
    assert result.execution_plan.fallback_id == result.stock_floor.candidate_id
    assert result.report["objective"] == "decode_tps"
    assert result.report["rejected_candidates"]


def test_compile_cli_writes_plan_and_explanation(tmp_path, capsys) -> None:
    exit_code = main(compile_argv_fixture(tmp_path))
    assert exit_code == 0
    assert (tmp_path / "execution-plan.json").is_file()
    assert (tmp_path / "explanation.json").is_file()
```

Also test unsupported adapter exit 2, identity mismatch exit 2, no valid stock candidate exit 3, no static winner returning the stock plan with an explicit no-go, plan tampering rejected by `validate`, and `explain` never relabeling estimates as measurements.

- [ ] **Step 2: Run pipeline/CLI tests and verify RED**

Run: `uv run pytest -q tests/test_compiler_pipeline.py tests/test_compiler_cli.py`

Expected: import/argument failures because pipeline and commands are absent.

- [ ] **Step 3: Implement orchestration and keep CLI wiring thin**

Define the request boundary exactly:

```python
@dataclass(frozen=True, slots=True)
class CompilationRequest:
    descriptor_path: Path
    inventory_path: Path
    hardware_path: Path
    workload_path: Path
    runtime_identity_path: Path
    layer_profile_paths: tuple[Path, ...]
    evidence_db_path: Path
    output_dir: Path
    recorded_evidence_path: Path | None = None
```

Add parser groups:

```python
inspect_cmd = commands.add_parser("inspect", help="Normalize a supported GGUF inventory into ModelIR.")
compile_cmd = commands.add_parser("compile", help="Compile a measured maximum-TPS execution plan.")
validate_cmd = commands.add_parser("validate", help="Validate a sealed execution plan and identities.")
explain_cmd = commands.add_parser("explain", help="Render compiler decisions and rejected candidates.")
```

Use these exact command contracts:

```text
expertflow inspect --descriptor FILE --inventory FILE --output FILE

expertflow compile --descriptor FILE --inventory FILE --hardware FILE
  --workload FILE --runtime-identity FILE --layer-profile FILE
  [--layer-profile FILE ...] --evidence-db FILE --output-dir DIR
  [--recorded-evidence FILE]

expertflow validate --plan FILE --descriptor FILE --inventory FILE
  --hardware FILE --runtime-identity FILE

expertflow explain --plan FILE --report FILE --output FILE
```

`--recorded-evidence` imports only hash-verified committed evidence and marks the
result `recorded_replay`; without it, `compile` uses the production sequential
measurement runner.

`main.py` imports only handler functions from `expertflow.compiler.commands`. Command handlers catch `ValueError`/identity errors, emit structured JSON failures, and return nonzero without tracebacks. All output files use UTF-8, sorted keys, trailing newline, and atomic temporary-file replacement.

- [ ] **Step 4: Run all compiler and existing CLI tests**

Run: `uv run pytest -q tests/test_compiler_*.py tests/test_product_cli.py tests/test_profile_cli.py tests/test_baseline_cli.py`

Expected: PASS and no existing command changes behavior.

- [ ] **Step 5: Commit the compiler product surface**

```powershell
git add src/expertflow/compiler/pipeline.py src/expertflow/compiler/commands.py src/expertflow/cli/main.py tests/test_compiler_pipeline.py tests/test_compiler_cli.py
git commit -m "feat: expose Phase 3 inference compiler"
```

---

### Task 11: Clean-checkout reproduction and Phase 3 exit gate

**Files:**
- Create: `docs/evidence/compiler-phase3/README.md`
- Create after execution: `docs/evidence/compiler-phase3/verification.json`
- Create after execution: `docs/evidence/compiler-phase3/stock-floor.json`
- Create after execution: `docs/evidence/compiler-phase3/static-plan.json`
- Create after execution: `docs/evidence/compiler-phase3/explanation.json`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: the implemented compiler, verified external Gemma Q6 model/inventory/runtime/profile paths, authoritative stock `22.966667` TPS reference, and authoritative ExpertFlow `28.13` TPS regression target.
- Produces: one clean-checkout Phase 3 verdict: `PASS`, `STATIC-REGRESSION-STOP`, or `ENVIRONMENT-BLOCKED`.

- [ ] **Step 1: Run the complete CPU-only verification before GPU work**

Run: `uv sync --frozen --extra quality`

Run: `uv run pytest -q`

Run: `uv run python -m compileall -q src/expertflow`

Run: `git diff --check`

Expected: all tests pass; the earlier pandas/NumPy ABI failure is gone. Source-contract skips remain explicitly reported when their external llama source is not supplied.

- [ ] **Step 2: Verify identities and compile a dry-run plan from recorded evidence**

Run `expertflow inspect` with the verified inventory and Gemma descriptor. Run `expertflow compile --recorded-evidence` against committed authoritative evidence. Verify that the stock floor is `22.966667` TPS, the static regression candidate is the twelve-layer set, all evidence remains labeled recorded/measured, and the plan validates.

- [ ] **Step 3: Run the live strongest-stock candidate matrix sequentially**

Preflight with `nvidia-smi`; stop with `ENVIRONMENT-BLOCKED` if another compute workload is active or reserve cannot be measured. Run one warmup plus three measured processes per stock candidate. Preserve raw commands, logs, probe JSON, process-owned VRAM, output hashes, and cleanup. Select the winner only through `select_strongest_stock()`.

- [ ] **Step 4: Run the recorded twelve-layer static regression candidate**

Use the pinned ExpertFlow llama.cpp fork and identical workload. Run one warmup plus ten alternating stock/static measured pairs, matching the authoritative protocol. Require exact prompt/generated token identity, stable process-owned memory, complete cleanup, and paired decode TPS at least `28.13 * 0.98 = 27.5674` to allow 2% reproduction tolerance. Do not claim a new improvement from the reproduction run.

- [ ] **Step 5: Enforce the Phase 3 decision**

Declare:

```text
PASS                    compiler selects a valid measured static plan and reproduces >=27.5674 TPS
STATIC-REGRESSION-STOP  exactness/memory/cleanup fails or static reproduction is below 27.5674 TPS
ENVIRONMENT-BLOCKED     required external identity, idle GPU, or supported build is unavailable
```

On `STATIC-REGRESSION-STOP`, do not begin CUDA autotuning, KV, adapters, MTP, or dynamic caching. On `ENVIRONMENT-BLOCKED`, preserve the dry-run plan but do not call it live-validated.

- [ ] **Step 6: Write verification evidence and append the project log**

`verification.json` records every command, exit code, artifact SHA-256, test count, source-contract skip, model/runtime/compiler identity, measurement ID, decision, and limitation. `README.md` separates recorded reproduction, live measurement, estimates, and rejected candidates. Append one factual entry to `PROJECT_LOG.md`; never rewrite prior entries.

- [ ] **Step 7: Final verification and scoped commit**

Run: `uv run pytest -q`

Run: `uv run python -m compileall -q src/expertflow`

Run: `git diff --check`

Run: `git status --short`

Stage only `docs/evidence/compiler-phase3/` and `PROJECT_LOG.md`, inspect the cached diff, then commit:

```powershell
git add docs/evidence/compiler-phase3 PROJECT_LOG.md
git diff --cached --check
git commit -m "docs: record Phase 3 compiler verification"
```

---

## Phase 3 completion boundary

This plan is complete only when Tasks 1–10 are implemented and Task 11 emits a declared verdict. A `PASS` authorizes separate specifications and plans for independent Phase 4–8 tracks. It does not authorize implementing those tracks inside this plan.

Recommended next planning order after `PASS`:

1. Phase 4 CUDA execution autotuning.
2. Phase 5 KV and shared memory-budget compilation.
3. Phase 6 second Gemma quant and Qwen MoE adapter.
4. Phase 7 upstream MTP/speculative compilation.
5. Phase 8 two-table dynamic residency.
