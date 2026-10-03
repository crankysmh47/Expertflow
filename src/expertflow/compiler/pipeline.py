"""Phase 3 orchestration: measured stock, explicit static no-go, sealed replay."""

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import uuid

from .adapters import AdapterRegistry, ModelDescriptor
from .evidence import EvidenceStore
from .passes.base import CompilerState, PassManager
from .plan import PlanIdentities, load_execution_plan, seal_candidate, validate_execution_plan
from .preflight import audit_historical_evidence, file_sha256
from .reference import load_reference_workload, read_json
from .runner import RuntimeBinding, ServerMeasurementRunner, WindowsGpuMemorySampler
from .schema import ArtifactIdentity, HardwareIR, WorkloadIR, canonical_payload, canonical_sha256
from .static_placement import StaticPlacementPass
from .stock import import_stock_measurement, select_strongest_stock, stock_candidate_matrix


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


@dataclass(frozen=True, slots=True)
class CompilerInputs:
    model: object
    hardware: HardwareIR
    workload: WorkloadIR
    stock: RuntimeBinding
    fork: RuntimeBinding
    profile_rows: tuple[dict, ...]
    provenance: dict

    def identities(self, binding):
        return PlanIdentities(canonical_sha256(self.model), canonical_sha256(self.hardware),
                              canonical_sha256(self.workload), binding.sha256, self.workload)


@dataclass(frozen=True, slots=True)
class CompilationResult:
    status: str
    execution_plan: object | None
    stock_floor: object | None
    report: dict


class EnvironmentBlocked(RuntimeError):
    pass


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(canonical_payload(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def inspect_model(descriptor_path, inventory_path):
    descriptor = ModelDescriptor(**read_json(Path(descriptor_path)))
    adapter = AdapterRegistry.with_builtins().resolve(descriptor.family)
    inventory = read_json(Path(inventory_path))
    provenance = inventory.get('model', {})
    identity = ArtifactIdentity(Path(provenance['path']).resolve().as_posix(), provenance['bytes'], provenance['sha256'])
    return adapter.normalize(descriptor, inventory, identity)


def _nvidia_query(query, gpu_uuid=None):
    command = ['nvidia-smi']
    if gpu_uuid:
        command += ['-i', gpu_uuid]
    command += ['--query-gpu=' + query, '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, timeout=10,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise EnvironmentBlocked('GPU telemetry unavailable: ' + result.stderr.strip())
    rows = [line.split(',') for line in result.stdout.strip().splitlines()]
    if len(rows) != 1:
        raise EnvironmentBlocked('Phase 3 requires one resolved GPU')
    return [value.strip() for value in rows[0]]


def validate_live_hardware(hardware):
    fields = _nvidia_query('uuid,name,compute_cap,driver_version,memory.total,memory.free,utilization.gpu', hardware.gpu_uuid)
    expected = (hardware.gpu_uuid, hardware.gpu_name, hardware.compute_capability, hardware.driver_version)
    if tuple(fields[:4]) != expected or int(fields[4]) << 20 != hardware.total_vram_bytes:
        raise ValueError('live hardware identity mismatch')
    if int(fields[5]) << 20 < hardware.minimum_reserve_bytes or int(fields[6]) > 10:
        raise EnvironmentBlocked('GPU is busy or device-free reserve unavailable')
    return {'gpu_fields': fields}


def capture_hardware(cuda_runtime, *, reserve_mib=256):
    fields = _nvidia_query('uuid,name,compute_cap,driver_version,memory.total,memory.free,utilization.gpu')
    return HardwareIR(fields[0], fields[1], fields[2], int(fields[4]) << 20,
                      (int(fields[5]) - reserve_mib) << 20, fields[3], '12.8',
                      file_sha256(Path(cuda_runtime)), reserve_mib << 20)


def load_compiler_inputs(request, *, live):
    descriptor = ModelDescriptor(**read_json(request.descriptor_path))
    adapter = AdapterRegistry.with_builtins().resolve(descriptor.family)
    inventory = read_json(request.inventory_path)
    runtime = read_json(request.runtime_identity_path)
    if runtime.get('schema_version') != '1.0.0':
        raise ValueError('unsupported runtime identity schema')
    identity = ArtifactIdentity(**runtime['model'])
    model_path = Path(identity.path)
    if not model_path.is_file():
        raise EnvironmentBlocked('model artifact unavailable')
    if model_path.stat().st_size != identity.size_bytes or file_sha256(model_path) != identity.sha256:
        raise ValueError('model artifact identity mismatch')
    model = adapter.normalize(descriptor, inventory, identity)
    hardware = HardwareIR(**read_json(request.hardware_path))
    workload = WorkloadIR.from_reference(load_reference_workload(Path.cwd(), request.workload_path))
    if workload.policy.value != 'exact':
        raise ValueError('Phase 3 requires exact policy')
    bindings = []
    for name in ('stock', 'fork'):
        entry = runtime[name]
        bindings.append(RuntimeBinding.from_manifest(Path.cwd(), entry['manifest_path'], entry['binary_dir'], entry['cuda_runtime']))
    stock, fork = bindings
    stock_manifest, fork_manifest = json.loads(stock.manifest_json), json.loads(fork.manifest_json)
    if stock_manifest['patches'] or not fork_manifest['patches'] or stock.sha256 == fork.sha256:
        raise ValueError('pristine/fork identities must remain distinct')
    if stock_manifest['upstream_commit'] != fork_manifest['upstream_commit']:
        raise ValueError('runtime comparison upstream mismatch')
    if stock.cuda_runtime.sha256 != hardware.cuda_runtime_sha256 or fork.cuda_runtime.sha256 != hardware.cuda_runtime_sha256:
        raise ValueError('CUDA runtime/hardware identity mismatch')
    profile_rows = []
    for path in request.layer_profile_paths:
        profile_rows.extend(adapter.normalize_profile(read_json(path), profile_id=file_sha256(path)))
    provenance = {'input_hashes': {str(Path(path).resolve()): file_sha256(path)
        for path in (request.descriptor_path, request.inventory_path, request.hardware_path,
                     request.workload_path, request.runtime_identity_path, *request.layer_profile_paths)}}
    if live:
        provenance['hardware_preflight'] = validate_live_hardware(hardware)
        source = runtime.get('source_path')
        if source:
            revision = subprocess.run(['git', '-C', source, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
            if revision.returncode or revision.stdout.strip() != fork_manifest['expertflow_commit']:
                raise ValueError('native source identity mismatch')
            environment = {**os.environ, 'EXPERTFLOW_LLAMA_SOURCE': source}
            contracts = subprocess.run([sys.executable, '-m', 'pytest', '-q',
                'tests/test_q6_quality_static_source_contract.py', 'tests/test_q6_selected_profile_source_contract.py'],
                env=environment, capture_output=True, text=True, timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            provenance['source_contracts'] = {'exit_code': contracts.returncode, 'output': contracts.stdout + contracts.stderr}
            if contracts.returncode:
                raise ValueError('pinned native source contracts failed')
    return CompilerInputs(model, hardware, workload, stock, fork, tuple(profile_rows), provenance)


def _recorded_report(request, inputs, store):
    path = request.recorded_evidence_path.resolve()
    root = Path.cwd().resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError('recorded evidence must be committed in this checkout') from error
    committed = subprocess.run(['git', 'show', 'HEAD:' + relative], capture_output=True, timeout=10)
    import hashlib
    if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != file_sha256(path):
        raise ValueError('recorded evidence is missing, dirty or not committed')
    recorded = read_json(path)
    historical = audit_historical_evidence(root)
    report = {'mode': 'recorded_replay', 'status': 'RECORDED-DIAGNOSTIC',
              'objective': inputs.workload.objective.value, 'source_sha256': file_sha256(path),
              'historical_cli': historical, 'earlier_strongest_stock_decode_tps': 22.966667,
              'rejected_candidates': [{'reason': 'historical_summary_without_native_token_evidence'}],
              'sealed': False}
    # A complete replay must point to existing append-only records with matching
    # keys and hash-verified, committed raw artifacts, never imported booleans.
    if recorded.get('measurement_ids'):
        candidates = stock_candidate_matrix(inputs.identities(inputs.stock))
        matches = []
        for candidate in candidates:
            ids = tuple(mid for mid in recorded['measurement_ids']
                        if store.measurement(mid).candidate_id == candidate.candidate_id)
            if not ids:
                continue
            for mid in ids:
                for artifact in store.measurement(mid).artifacts:
                    artifact_path = Path(artifact.identity.path).resolve()
                    try:
                        artifact_relative = artifact_path.relative_to(root).as_posix()
                    except ValueError as error:
                        raise ValueError('recorded raw artifacts must be committed') from error
                    raw = subprocess.run(['git', 'show', 'HEAD:' + artifact_relative], capture_output=True, timeout=10)
                    if raw.returncode or hashlib.sha256(raw.stdout).hexdigest() != artifact.identity.sha256:
                        raise ValueError('unverified committed raw artifact')
            matches.append(import_stock_measurement(candidate, ids, store))
        winner = select_strongest_stock(matches, total_vram_mib=inputs.hardware.total_vram_bytes >> 20,
                                        reserve_mib=inputs.workload.minimum_vram_reserve_mib)
        plan = seal_candidate(winner.candidate, store, winner.candidate.identities, None)
        report['sealed'] = True
        report['live_validated'] = False
        return CompilationResult('RECORDED-REPLAY', plan, winner, report)
    return CompilationResult('RECORDED-DIAGNOSTIC', None, None, report)


def _same_tokens(store, mid, baseline_id):
    actual, baseline = store.verify_measurement(mid), store.verify_measurement(baseline_id)
    if any(actual[name] != baseline[name] for name in ('prompt_tokens_sha256', 'generated_tokens_sha256')):
        raise ValueError('selected candidate token replay mismatch')


def replay_sealed_plan(plan_path, inputs, runner, store, output_dir):
    plan = load_execution_plan(plan_path, identities=inputs.identities(inputs.stock), store=store)
    if len(plan.candidate.measurement_ids) < inputs.workload.confirmation_pairs or any(
        store.measurement(mid).stage != 'confirmation' for mid in plan.candidate.measurement_ids
    ):
        raise ValueError('selected plan lacks complete confirmation evidence')
    validate_live = runner is None
    sampler = None
    if validate_live:
        validate_live_hardware(inputs.hardware)
        sampler = WindowsGpuMemorySampler(inputs.hardware.gpu_uuid)
        runner = ServerMeasurementRunner(store, memory_sampler=sampler)
    try:
        outcome = runner.run_once(plan.candidate, inputs.model.identity, inputs.stock,
            output_dir=output_dir, measured=True, stage='sealed_replay')
        if outcome.status != 'measured':
            return outcome.status.upper().replace('_', '-'), {'reason': outcome.reason, 'outcome': canonical_payload(outcome)}
        _same_tokens(store, outcome.measurement_id, plan.candidate.measurement_ids[0])
        confirmation = [store.verify_measurement(mid)['decode_tps'] for mid in plan.candidate.measurement_ids]
        reference = statistics.mean(confirmation)
        measured = store.verify_measurement(outcome.measurement_id)['decode_tps']
        difference = abs(measured - reference) * 100 / reference
        report = {'measurement_id': outcome.measurement_id, 'plan_sha256': plan.plan_sha256,
                  'confirmation_mean_tps': reference, 'replay_tps': measured, 'difference_pct': difference,
                  'tolerance_pct': inputs.workload.replay_tolerance_pct}
        return ('PASS-STOCK-FALLBACK' if difference <= inputs.workload.replay_tolerance_pct else 'VALIDATION-STOP'), report
    finally:
        if sampler:
            sampler.close()


def compile_phase3(request, runner=None, store=None):
    store = store or EvidenceStore(request.evidence_db_path)
    output = Path(request.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'execution-plan.json').exists():
        raise ValueError('output already contains a sealed plan; use a fresh output directory')
    report = {'objective': 'decode_tps', 'mode': 'live', 'rejected_candidates': [
                  {'scope': 'CPU-to-CUDA static MoE', 'reason': 'numerical_path_change'}],
              'duplicate_placements': [], 'measurements': [], 'sealed': False}
    stock_floor = None
    sampler = None
    try:
        inputs = load_compiler_inputs(request, live=request.recorded_evidence_path is None)
        report['identities'] = canonical_payload(inputs.identities(inputs.stock))
        report['provenance'] = inputs.provenance
        if request.recorded_evidence_path is not None:
            result = _recorded_report(request, inputs, store)
            atomic_json(output / 'explanation.json', result.report)
            if result.execution_plan:
                atomic_json(output / 'execution-plan.json', result.execution_plan)
            return result
        if runner is None:
            try:
                sampler = WindowsGpuMemorySampler(inputs.hardware.gpu_uuid)
            except RuntimeError as error:
                raise EnvironmentBlocked(str(error)) from error
            runner = ServerMeasurementRunner(store, memory_sampler=sampler)
        run_root = output / 'raw' / str(uuid.uuid4())
        matrix = stock_candidate_matrix(inputs.identities(inputs.stock))
        measurements, placements = [], {}
        for index, candidate in enumerate(matrix):
            warmup = runner.run_once(candidate, inputs.model.identity, inputs.stock,
                output_dir=run_root / f'stock-{index}-warmup', measured=False, stage='warmup')
            report['measurements'].append(canonical_payload(warmup))
            if warmup.status != 'measured':
                log = Path(warmup.output_dir) / 'stderr.log'
                text = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
                if re.search(r'out of memory|failed to allocate|unable to allocate|not enough memory', text, re.I):
                    report['rejected_candidates'].append({'candidate_id':candidate.candidate_id, 'reason':'allocation_failure', 'evidence':warmup.output_dir})
                    continue
                raise EnvironmentBlocked(warmup.reason or 'stock warmup failed')
            if warmup.resolved_placement is not None and warmup.resolved_placement in placements:
                report['duplicate_placements'].append({'candidate_id':candidate.candidate_id,
                    'representative':placements[warmup.resolved_placement], 'evidence':warmup.output_dir})
                continue
            if warmup.resolved_placement is not None:
                placements[warmup.resolved_placement] = candidate.candidate_id
            ids = []
            for repetition in range(inputs.workload.measured_runs):
                outcome = runner.run_once(candidate, inputs.model.identity, inputs.stock,
                    output_dir=run_root / f'stock-{index}-measured-{repetition}', measured=True)
                report['measurements'].append(canonical_payload(outcome))
                if outcome.status != 'measured':
                    if outcome.status == 'environment_blocked':
                        raise EnvironmentBlocked(outcome.reason or 'stock measurement failed')
                    raise ValueError(outcome.reason or 'stock validation failed')
                _same_tokens(store, outcome.measurement_id, warmup.measurement_id)
                ids.append(outcome.measurement_id)
            measurements.append(import_stock_measurement(candidate, ids, store))
        try:
            stock_floor = select_strongest_stock(measurements,
                total_vram_mib=inputs.hardware.total_vram_bytes >> 20,
                reserve_mib=inputs.workload.minimum_vram_reserve_mib)
        except ValueError as error:
            report.update(status='INCONCLUSIVE', reason=str(error))
            atomic_json(output / 'explanation.json', report)
            return CompilationResult('INCONCLUSIVE', None, None, report)
        # Explicit pristine/fork off relation, with identical CPU-MoE settings.
        cpu_floor = next((m for m in measurements if m.candidate.settings.cpu_moe), None)
        if cpu_floor is None:
            raise EnvironmentBlocked('no measured pristine CPU-MoE equivalence reference')
        fork_candidate = replace(cpu_floor.candidate, identities=inputs.identities(inputs.fork), measurement_ids=())
        equivalence = runner.run_once(fork_candidate, inputs.model.identity, inputs.fork,
            output_dir=run_root / 'fork-feature-off', measured=False, stage='startup_equivalence',
            numerical_path='fork_off_vs_pristine', comparison_ids=(cpu_floor.measurement_ids[0],))
        report['measurements'].append(canonical_payload(equivalence))
        if equivalence.status != 'measured':
            if equivalence.status == 'environment_blocked':
                raise EnvironmentBlocked(equivalence.reason or 'fork-off equivalence failed')
            raise ValueError(equivalence.reason or 'fork-off equivalence failed')
        store.verify_measurement(equivalence.measurement_id)
        state = CompilerState(inputs.model, inputs.hardware, inputs.workload,
                              (stock_floor.candidate,), capabilities=frozenset({'stock-baseline'}))
        state = PassManager([StaticPlacementPass(inputs.profile_rows, baseline_peak_bytes=stock_floor.peak_bytes)]).run(state)
        report['rejected_candidates'].extend(canonical_payload(c) for c in state.candidates if c.rejection_reasons)
        report['static_diagnostics'] = state.analysis('static_candidates') if state.analyses else []
        report['diagnostics'] = state.diagnostics
        report['stock_floor'] = canonical_payload(stock_floor)
        confirmed = []
        for index in range(inputs.workload.confirmation_pairs):
            outcome = runner.run_once(stock_floor.candidate, inputs.model.identity, inputs.stock,
                output_dir=run_root / f'confirmation-{index}', measured=True, stage='confirmation')
            report['measurements'].append(canonical_payload(outcome))
            if outcome.status != 'measured':
                if outcome.status == 'environment_blocked':
                    raise EnvironmentBlocked(outcome.reason or 'confirmation environment failure')
                raise ValueError(outcome.reason or 'confirmation validation failure')
            _same_tokens(store, outcome.measurement_id, stock_floor.measurement_ids[0])
            confirmed.append(outcome.measurement_id)
        selected = import_stock_measurement(stock_floor.candidate, confirmed, store)
        if selected.cv_pct > inputs.workload.maximum_cv_pct:
            report.update(status='INCONCLUSIVE', reason='confirmation variance', confirmation=canonical_payload(selected))
            atomic_json(output / 'explanation.json', report)
            return CompilationResult('INCONCLUSIVE', None, stock_floor, report)
        plan = seal_candidate(selected.candidate, store, selected.candidate.identities, None)
        candidate_path = output / 'execution-plan.pending.json'
        atomic_json(candidate_path, plan)
        verdict, replay = replay_sealed_plan(candidate_path, inputs, runner, store, run_root / 'sealed-replay')
        report.update(status=verdict, replay=replay, confirmation=canonical_payload(selected),
                      sealed=verdict == 'PASS-STOCK-FALLBACK')
        atomic_json(output / 'explanation.json', report)
        if verdict != 'PASS-STOCK-FALLBACK':
            return CompilationResult(verdict, None, stock_floor, report)
        atomic_json(output / 'execution-plan.json', plan)
        atomic_json(output / 'stock-floor.json', stock_floor)
        atomic_json(output / 'static-plan.json', {'eligible':False, 'candidates':report['static_diagnostics'],
                                                  'rejected_candidates':report['rejected_candidates']})
        return CompilationResult(verdict, plan, stock_floor, report)
    except EnvironmentBlocked as error:
        report.update(status='ENVIRONMENT-BLOCKED', reason=str(error))
    except (ValueError, KeyError, TypeError, OSError) as error:
        report.update(status='VALIDATION-STOP', reason=str(error))
    finally:
        if sampler:
            sampler.close()
    atomic_json(output / 'explanation.json', report)
    return CompilationResult(report['status'], None, stock_floor, report)
