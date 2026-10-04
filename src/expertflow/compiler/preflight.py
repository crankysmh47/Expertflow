"""Artifact feasibility and honest historical evidence before compiler build-out."""

import csv
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import json

from expertflow.artifacts import ArtifactSpec, verify_artifact
from expertflow.compiler.reference import load_reference_workload, load_runtime_manifest, read_json


DEFAULT_CUDA_RUNTIME = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/cudart64_12.dll')
_REQUIRED_DLLS = (
    'ggml-base.dll', 'ggml-cpu.dll', 'ggml-cuda.dll', 'ggml.dll',
    'llama-common.dll', 'llama-cli-impl.dll', 'llama-server-impl.dll', 'llama.dll',
)


def capture_power_policy(scheme):
    """Bind all visible and hidden AC/DC settings within the active scheme."""
    result = subprocess.run(['powercfg', '/qh', scheme], capture_output=True,
        text=True, timeout=10, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=True)
    settings = '\n'.join(line.strip() for line in result.stdout.splitlines() if line.strip())
    if not settings:
        raise RuntimeError('active power policy settings unavailable')
    return {'settings_sha256': hashlib.sha256(settings.encode('utf-8')).hexdigest(),
            'settings': settings}


def capture_host_environment():
    """Stable host controls omitted by the legacy GPU-only HardwareIR."""
    if os.name != 'nt':
        raise RuntimeError('stock product host capture requires a supported Windows provider')
    script = """$ErrorActionPreference='Stop';
$stockCpus=@(Get-CimInstance Win32_Processor | ForEach-Object {
    [ordered]@{name=$_.Name.Trim();cores=$_.NumberOfCores;logical_processors=$_.NumberOfLogicalProcessors;processor_id=$_.ProcessorId}});
$stockRam=@(Get-CimInstance Win32_PhysicalMemory | Sort-Object DeviceLocator | ForEach-Object {
    [ordered]@{slot=$_.DeviceLocator;capacity_bytes=$_.Capacity;configured_mhz=$_.ConfiguredClockSpeed}});
$stockSystem=Get-CimInstance Win32_ComputerSystem;
[ordered]@{cpu=$stockCpus;ram=$stockRam;ram_bytes=$stockSystem.TotalPhysicalMemory;os=[Environment]::OSVersion.Version.ToString()} | ConvertTo-Json -Depth 6 -Compress"""
    flags = subprocess.CREATE_NO_WINDOW
    result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
        capture_output=True, text=True, timeout=30, creationflags=flags, check=True)
    host = json.loads(result.stdout)
    power = subprocess.run(['powercfg', '/getactivescheme'], capture_output=True,
        text=True, timeout=10, creationflags=flags, check=True)
    match = re.search(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', power.stdout)
    if not match or not host.get('cpu') or not host.get('ram'):
        raise RuntimeError('host topology/power identity unavailable')
    import ctypes
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)):
        raise RuntimeError('process affinity identity unavailable')
    host.update(power_scheme=match.group().lower(), power_policy=capture_power_policy(match.group().lower()),
        architecture=platform.machine(),
        process_affinity_mask=process_mask.value, system_affinity_mask=system_mask.value,
        threading_environment={name:value for name,value in sorted(os.environ.items())
            if name.upper().startswith(('OMP_', 'KMP_', 'GOMP_', 'MKL_', 'OPENBLAS_'))})
    return host


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit_historical_evidence(root: Path) -> dict:
    directory = root / 'docs/evidence/q6-placement-final'
    paths = {name: directory / name for name in ('results.json', 'run-pairs.csv', 'quality-results.json')}
    summary = read_json(paths['results.json'])
    quality = read_json(paths['quality-results.json'])
    try:
        with paths['run-pairs.csv'].open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 20:
            raise ValueError('historical CSV must retain all 20 ordinary runs')
        samples = {'off': [], 'on': []}
        hashes = {'off': set(), 'on': set()}
        pair_orders = {}
        for row in rows:
            mode = row['mode']
            if mode not in samples or row['valid'] != 'True':
                raise ValueError('invalid historical mode/run')
            pair, order = int(row['pair']), int(row['order'])
            if not 1 <= pair <= 10 or order not in (1, 2) or (pair, order) in pair_orders:
                raise ValueError('invalid or duplicate historical pair/order')
            pair_orders[pair, order] = mode
            tps = float(row['decode_tps'])
            if not math.isfinite(tps) or tps <= 0:
                raise ValueError('nonfinite/nonpositive historical TPS')
            response_hash = row['response_sha256']
            if not re.fullmatch('[0-9a-f]{64}', response_hash):
                raise ValueError('invalid historical response hash')
            samples[mode].append(tps)
            hashes[mode].add(response_hash)
        for pair in range(1, 11):
            expected = ('off', 'on') if pair % 2 else ('on', 'off')
            if (pair_orders[pair, 1], pair_orders[pair, 2]) != expected:
                raise ValueError('historical pair ordering mismatch')
        for mode in samples:
            if len(samples[mode]) != 10 or len(hashes[mode]) != 1:
                raise ValueError('historical mode missing repetitions or unstable output')
        means = {mode: round(statistics.fmean(values), 8) for mode, values in samples.items()}
        performance = summary['performance']
        if performance['matched_pairs'] != 10 or performance['generated_tokens_per_run'] != 512:
            raise ValueError('historical summary repetition/token mismatch')
        for mode, field in [('off', 'stock_mean_decode_tps'), ('on', 'expertflow_mean_decode_tps')]:
            if not math.isclose(means[mode], float(performance[field]), abs_tol=1e-6):
                raise ValueError('historical summary/CSV TPS mismatch')
        threshold = float(summary['quality']['required_upper_pct'])
        upper = float(quality['perplexity']['paired_bootstrap_95_pct'][1])
        if not all(math.isfinite(value) for value in (threshold, upper)):
            raise ValueError('invalid historical quality bounds')
        quality_pass = upper <= threshold
        if quality_pass is not summary['quality']['ppl_gate_pass'] or quality_pass is not quality['perplexity']['upper_bound_gate_pass']:
            raise ValueError('historical quality summary mismatch')
    except (OSError, KeyError, TypeError, IndexError) as error:
        raise ValueError(f'invalid historical evidence: {error}') from error
    cross_identity = hashes['off'] == hashes['on']
    reasons = ['missing_native_token_ids', 'numerical_path_change']
    if not cross_identity:
        reasons.append('cross_mode_output_divergence')
    if not quality_pass:
        reasons.append('strict_quality_gate_failed')
    return {
        'measurement_kind': 'historical_cli_summary',
        'mean_decode_tps': means,
        'repetitions_per_mode': 10,
        'response_hashes': {mode: sorted(values) for mode, values in hashes.items()},
        'cross_mode_response_identity': cross_identity,
        'native_token_evidence_available': False,
        'strict_quality_gate_pass': quality_pass,
        'ppl_95_upper_pct': upper,
        'required_ppl_upper_pct': threshold,
        'exact_plan_eligible': False,
        'rejection_reasons': reasons,
        'input_hashes': {str(path.relative_to(root)).replace('\\', '/'): file_sha256(path)
                         for path in paths.values()},
    }


def verify_external_artifacts(
    root: Path, model: Path, stock_dir: Path, fork_dir: Path,
    *, cuda_runtime: Path = DEFAULT_CUDA_RUNTIME,
) -> dict:
    missing, errors, runtimes = {}, [], {}
    manifest = read_json(root / 'docs/evidence/q6-download/model-manifest.json')
    spec = ArtifactSpec(
        repository=manifest['repository'], revision=manifest['revision'],
        filename=manifest['filename'], size_bytes=manifest['exact_bytes'],
        sha256=manifest['sha256'],
    )
    if not model.is_file():
        missing['model'] = str(model)
    elif Path(str(model) + '.aria2').exists():
        missing['model_download_complete'] = str(model)
    if not cuda_runtime.is_file():
        missing['cuda_runtime'] = str(cuda_runtime)
    cuda_actual_hash = file_sha256(cuda_runtime) if cuda_runtime.is_file() else None
    # Small, independent runtime checks remain useful when model recovery is pending.
    for label, directory in [('stock', stock_dir), ('fork', fork_dir)]:
        runtime = load_runtime_manifest(root / f'configs/compiler/runtime-{label}.json')
        binary_hashes, patch_hashes = {}, {}
        for name, wanted in runtime['binaries'].items():
            path = directory / name
            if not path.is_file():
                missing[f'{label}:{name}'] = str(path)
                continue
            actual = file_sha256(path)
            binary_hashes[name] = actual
            if actual != wanted:
                errors.append(f'{label}:{name}: binary SHA-256 mismatch')
        for patch in runtime['patches']:
            path = root / patch['path']
            if not path.is_file():
                missing[f'patch:{patch["path"]}'] = str(path)
                continue
            actual = file_sha256(path)
            patch_hashes[patch['path']] = actual
            if actual != patch['sha256']:
                errors.append(f'{label}:{patch["path"]}: patch SHA-256 mismatch')
        for name in set(_REQUIRED_DLLS) | runtime['dependencies'].keys():
            if not (directory / name).is_file():
                missing[f'{label}:{name}'] = str(directory / name)
        dependencies = {path.name: file_sha256(path) for path in sorted(directory.glob('*.dll'))}
        for name, actual in dependencies.items():
            wanted = runtime['dependencies'].get(name)
            if wanted is None:
                errors.append(f'{label}:{name}: unpinned runtime dependency')
            elif actual != wanted:
                errors.append(f'{label}:{name}: dependency SHA-256 mismatch')
        if cuda_actual_hash is not None and cuda_actual_hash != runtime['cuda_runtime_sha256']:
            errors.append(f'{label}:CUDA runtime: dependency SHA-256 mismatch')
        runtimes[label] = {
            'directory': str(directory.resolve()),
            'manifest_sha256': file_sha256(root / f'configs/compiler/runtime-{label}.json'),
            'binary_hashes': binary_hashes,
            'patch_hashes': patch_hashes,
            'dependency_hashes': dependencies,
            'dependency_identity_kind': 'verified_against_frozen_local_build_manifest',
            'build': runtime['build'],
            'capabilities': runtime['capabilities'],
        }
    model_verified = False
    if 'model' not in missing and 'model_download_complete' not in missing:
        try:
            verify_artifact(model, spec)
            model_verified = True
        except ValueError as error:
            errors.append(f'model: {error}')
    return {
        'missing_artifacts': missing,
        'identity_errors': errors,
        'model': {'path': str(model.resolve()), 'expected_bytes': spec.size_bytes,
                  'expected_sha256': spec.sha256, 'verified': model_verified},
        'runtimes': runtimes,
        'cuda_runtime': {'path': str(cuda_runtime),
                         'sha256': cuda_actual_hash},
    }


def run_preflight(
    root: Path, model: Path, stock_dir: Path, fork_dir: Path,
    *, cuda_runtime: Path = DEFAULT_CUDA_RUNTIME,
) -> dict:
    report = {
        'schema_version': '1.0.0', 'phase': 'artifact_preflight',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'live_measurements_performed': False,
    }
    try:
        report['historical_static'] = audit_historical_evidence(root)
        product = load_reference_workload(root, root / 'configs/compiler/gemma4-q6-single-request.json')
        history = load_reference_workload(root, root / 'configs/compiler/gemma4-q6-historical-cli.json')
        report['product_workload'] = {'sha256': product.sha256, **product.payload}
        report['historical_workload'] = {'sha256': history.sha256, **history.payload}
        external = verify_external_artifacts(root, model, stock_dir, fork_dir, cuda_runtime=cuda_runtime)
        report['external_artifacts'] = external
        report['status'] = ('IDENTITY-STOP' if external['identity_errors'] else
                            'ENVIRONMENT-BLOCKED' if external['missing_artifacts'] else 'READY')
    except (OSError, ValueError, KeyError, TypeError) as error:
        report['status'] = 'IDENTITY-STOP'
        report['error'] = str(error)
    return report
