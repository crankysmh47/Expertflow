import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from expertflow.compiler.preflight import audit_historical_evidence, run_preflight


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence_root(tmp_path):
    root = tmp_path / 'repo'
    for relative in ['docs/research/evidence/q6-placement-final', 'docs/research/evidence/q6-download',
                     'configs/compiler', 'docs/research/release/expertflow-build-week/patches/llama.cpp']:
        shutil.copytree(ROOT / relative, root / relative)
    (root / 'configs/baseline-prompt.txt').write_bytes(b'A fixed product prompt.\n')
    return root


@pytest.fixture
def tiny_external(evidence_root):
    root = evidence_root
    model = root / 'tiny.gguf'
    model.write_bytes(b'GGUF tiny fixture')
    manifest_path = root / 'docs/research/evidence/q6-download/model-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['exact_bytes'] = model.stat().st_size
    manifest['sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    dirs = []
    for name in ('stock', 'fork'):
        runtime_dir = root / name
        runtime_dir.mkdir()
        runtime_manifest_path = root / f'configs/compiler/runtime-{name}.json'
        runtime_manifest = json.loads(runtime_manifest_path.read_text())
        for executable in runtime_manifest['binaries']:
            data = f'{name}:{executable}'.encode()
            (runtime_dir / executable).write_bytes(data)
            runtime_manifest['binaries'][executable] = hashlib.sha256(data).hexdigest()
        for dll in ('ggml-base.dll', 'ggml-cpu.dll', 'ggml-cuda.dll', 'ggml.dll',
                    'llama-common.dll', 'llama-cli-impl.dll', 'llama-server-impl.dll', 'llama.dll'):
            (runtime_dir / dll).write_bytes(f'{name}:{dll}'.encode())
        runtime_manifest['dependencies'] = {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in runtime_dir.glob('*.dll')
        }
        runtime_manifest['cuda_runtime_sha256'] = hashlib.sha256(b'CUDA dependency fixture').hexdigest()
        runtime_manifest_path.write_text(json.dumps(runtime_manifest))
        dirs.append(runtime_dir)
    cuda = root / 'cudart64_12.dll'
    cuda.write_bytes(b'CUDA dependency fixture')
    return root, model, *dirs, cuda


def run_fixture(paths):
    root, model, stock, fork, cuda = paths
    return run_preflight(root, model, stock, fork, cuda_runtime=cuda)


def test_historical_static_is_report_only():
    result = audit_historical_evidence(ROOT)
    assert result['exact_plan_eligible'] is False
    assert result['cross_mode_response_identity'] is False
    assert result['strict_quality_gate_pass'] is False
    assert result['mean_decode_tps'] == {'off': 22.28, 'on': 28.13}
    assert result['native_token_evidence_available'] is False
    assert 'missing_native_token_ids' in result['rejection_reasons']
    assert all(len(digest) == 64 for digest in result['input_hashes'].values())


@pytest.mark.parametrize('change', ['empty', 'invalid_tps', 'duplicate_pair', 'bad_hash', 'invalid_run'])
def test_malformed_historical_runs_cannot_be_imported(evidence_root, change):
    path = evidence_root / 'docs/research/evidence/q6-placement-final/run-pairs.csv'
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    fields = list(rows[0])
    if change == 'empty':
        rows = []
    elif change == 'invalid_tps':
        rows[0]['decode_tps'] = 'nan'
    elif change == 'duplicate_pair':
        rows[0]['pair'] = rows[2]['pair']
    elif change == 'bad_hash':
        rows[0]['response_sha256'] = 'no digest'
    else:
        rows[0]['valid'] = 'False'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        audit_historical_evidence(evidence_root)


def test_changed_summary_cannot_override_actual_measurements(evidence_root):
    path = evidence_root / 'docs/research/evidence/q6-placement-final/results.json'
    value = json.loads(path.read_text())
    value['performance']['expertflow_mean_decode_tps'] = 35
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='summary'):
        audit_historical_evidence(evidence_root)


def test_matching_text_hashes_still_do_not_supply_native_token_evidence(evidence_root):
    path = evidence_root / 'docs/research/evidence/q6-placement-final/run-pairs.csv'
    text = path.read_text().replace(
        'd486092ed666a0bbb5b8eaf0517996b6a4ab1b129b00d5f81bd65017a0887e5a',
        '7a5b8b8b055b3a6a884349ce7971b42827c736e2391b3e204178ccd23e21903d',
    )
    path.write_text(text)
    result = audit_historical_evidence(evidence_root)
    assert result['cross_mode_response_identity'] is True
    assert result['exact_plan_eligible'] is False


def test_missing_model_reports_blocker_and_verifies_independent_runtime_inputs(tiny_external):
    _, model, *_ = tiny_external
    model.unlink()
    result = run_fixture(tiny_external)
    assert result['status'] == 'ENVIRONMENT-BLOCKED'
    assert 'model' in result['external_artifacts']['missing_artifacts']
    assert result['external_artifacts']['runtimes']['stock']['binary_hashes']
    assert result['historical_static']['exact_plan_eligible'] is False


def test_pending_resumable_download_is_not_an_identity_failure(tiny_external):
    _, model, *_ = tiny_external
    model.write_bytes(b'partial')
    Path(str(model) + '.aria2').write_bytes(b'resumable download')
    result = run_fixture(tiny_external)
    assert result['status'] == 'ENVIRONMENT-BLOCKED'
    assert 'model_download_complete' in result['external_artifacts']['missing_artifacts']


@pytest.mark.parametrize('artifact', ['model_size', 'model_hash', 'stock_binary', 'fork_binary', 'patch', 'stock_dll', 'cuda_dll'])
def test_identity_mismatch_never_reports_ready(tiny_external, artifact):
    root, model, stock, fork, cuda = tiny_external
    if artifact == 'model_size':
        model.write_bytes(b'changed size')
    elif artifact == 'model_hash':
        model.write_bytes(b'X' * model.stat().st_size)
    elif artifact == 'stock_binary':
        (stock / 'llama-server.exe').write_bytes(b'changed server')
    elif artifact == 'fork_binary':
        (fork / 'llama-cli.exe').write_bytes(b'changed cli')
    elif artifact == 'stock_dll':
        (stock / 'llama.dll').write_bytes(b'changed inference implementation')
    elif artifact == 'cuda_dll':
        cuda.write_bytes(b'changed CUDA dependency')
    else:
        path = next((root / 'docs/research/release/expertflow-build-week/patches/llama.cpp').glob('0001-*'))
        path.write_bytes(b'changed patch')
    result = run_fixture(tiny_external)
    assert result['status'] == 'IDENTITY-STOP'
    assert result['external_artifacts']['identity_errors']


def test_ready_means_artifact_gate_only_and_snapshots_companion_dependencies(tiny_external):
    result = run_fixture(tiny_external)
    assert result['status'] == 'READY'
    assert result['phase'] == 'artifact_preflight'
    assert result['live_measurements_performed'] is False
    dependencies = result['external_artifacts']['runtimes']['fork']['dependency_hashes']
    assert dependencies['ggml-cuda.dll'] == hashlib.sha256(b'fork:ggml-cuda.dll').hexdigest()
    assert result['product_workload']['runtime_interface'] == 'server_completion'


def test_missing_runtime_dependency_blocks_execution(tiny_external):
    _, _, _, fork, _ = tiny_external
    (fork / 'ggml-cuda.dll').unlink()
    assert run_fixture(tiny_external)['status'] == 'ENVIRONMENT-BLOCKED'


def test_cli_writes_complete_report_and_nonzero_missing_artifact_exit(tiny_external):
    root, model, stock, fork, cuda = tiny_external
    model.unlink()
    output = root / 'report.json'
    output.write_text('old output')
    completed = subprocess.run([
        sys.executable, str(ROOT / 'scripts/compiler_preflight.py'),
        '--root', str(root), '--model', str(model), '--stock-dir', str(stock),
        '--fork-dir', str(fork), '--cuda-runtime', str(cuda), '--output', str(output),
    ], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 3, completed.stderr
    assert json.loads(output.read_text())['status'] == 'ENVIRONMENT-BLOCKED'
    assert not list(root.glob('report.json.*.tmp'))


def test_cli_replaces_old_report_on_unrepresentable_numeric_setting(tiny_external):
    root, model, stock, fork, cuda = tiny_external
    workload = root / 'configs/compiler/gemma4-q6-single-request.json'
    payload = json.loads(workload.read_text())
    payload['maximum_cv_pct'] = 10 ** 400
    workload.write_text(json.dumps(payload))
    output = root / 'report.json'
    output.write_text('{"status":"READY"}')
    completed = subprocess.run([
        sys.executable, str(ROOT / 'scripts/compiler_preflight.py'),
        '--root', str(root), '--model', str(model), '--stock-dir', str(stock),
        '--fork-dir', str(fork), '--cuda-runtime', str(cuda), '--output', str(output),
    ], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 2, completed.stderr
    assert json.loads(output.read_text())['status'] == 'IDENTITY-STOP'
    assert 'Traceback' not in completed.stderr
