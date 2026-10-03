import json
from pathlib import Path

import pytest

from expertflow.compiler.reference import load_reference_workload, load_runtime_manifest


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs/compiler'


def write_workload(tmp_path, **overrides):
    value = json.loads((CONFIGS / 'gemma4-q6-single-request.json').read_text())
    value['prompt_file'] = 'prompt.txt'
    value.update(overrides)
    (tmp_path / 'prompt.txt').write_bytes(b'A fixed prompt.\n')
    path = tmp_path / 'workload.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def test_reference_workload_keeps_prompt_bytes_and_frozen_completion_settings():
    workload = load_reference_workload(ROOT, CONFIGS / 'gemma4-q6-single-request.json')
    value = workload.payload
    assert value['prompt'] == (ROOT / 'configs/baseline-prompt.txt').read_bytes().decode('utf-8')
    assert value['runtime_interface'] == 'server_completion'
    assert (value['context_size'], value['predict_tokens'], value['concurrency']) == (4096, 512, 1)
    assert value['ignore_eos'] is True
    assert value['cache_prompt'] is False
    assert (value['kv_type_k'], value['kv_type_v']) == ('f16', 'f16')


def test_identity_changes_with_prompt_bytes(tmp_path):
    path = write_workload(tmp_path)
    before = load_reference_workload(tmp_path, path)
    (tmp_path / 'prompt.txt').write_bytes(b'A fixed prompt.\r\n')
    after = load_reference_workload(tmp_path, path)
    assert before.sha256 != after.sha256
    assert after.payload['prompt'].endswith('\r\n')


def test_returned_payload_cannot_change_frozen_identity(tmp_path):
    path = write_workload(tmp_path)
    workload = load_reference_workload(tmp_path, path)
    changed = workload.payload
    changed['context_size'] = 10
    assert workload.payload['context_size'] == 4096
    assert load_reference_workload(tmp_path, path).sha256 == workload.sha256


@pytest.mark.parametrize('overrides', [
    {'temperature': float('nan')}, {'temperature': 0.5}, {'seed': True},
    {'kv_type_v': 'q8_0'}, {'concurrency': 2}, {'policy': 'approximate'},
    {'runtime_interface': 'unknown'}, {'measured_runs': 0},
    {'minimum_vram_reserve_mib': -1}, {'batch_size': 128, 'microbatch_size': 512},
    {'cache_prompt': True}, {'health_timeout_seconds': 0},
    {'unexpected_setting': True},
    {'maximum_cv_pct': 10 ** 400},
])
def test_invalid_exact_reference_cannot_reach_measurement(tmp_path, overrides):
    path = write_workload(tmp_path, **overrides)
    with pytest.raises(ValueError):
        load_reference_workload(tmp_path, path)


def test_missing_or_empty_prompt_cannot_reach_measurement(tmp_path):
    path = write_workload(tmp_path)
    (tmp_path / 'prompt.txt').unlink()
    with pytest.raises(ValueError, match='prompt'):
        load_reference_workload(tmp_path, path)
    (tmp_path / 'prompt.txt').write_text('  ', encoding='utf-8')
    with pytest.raises(ValueError, match='prompt'):
        load_reference_workload(tmp_path, path)


def test_historical_workload_is_separate_and_cannot_be_exact_sealed():
    history = load_reference_workload(ROOT, CONFIGS / 'gemma4-q6-historical-cli.json')
    product = load_reference_workload(ROOT, CONFIGS / 'gemma4-q6-single-request.json')
    assert history.payload['prompt'] == 'Caching.'
    assert history.payload['context_size'] == 2048
    assert history.payload['exact_plan_eligible'] is False
    assert history.sha256 != product.sha256


def test_runtime_manifests_distinguish_stock_from_bounded_fork():
    stock = load_runtime_manifest(CONFIGS / 'runtime-stock.json')
    fork = load_runtime_manifest(CONFIGS / 'runtime-fork.json')
    assert stock['patches'] == []
    assert stock['capabilities']['max_static_layers'] == 0
    assert fork['capabilities']['max_static_layers'] == 12
    assert fork['capabilities']['max_static_shadows'] == 48
    assert len(fork['patches']) == 6
    assert stock['binaries']['llama-server.exe'] != fork['binaries']['llama-server.exe']


@pytest.mark.parametrize('change', ['digest', 'commit', 'abi', 'patch_order', 'capacity', 'missing_binary', 'missing_dependencies'])
def test_invalid_runtime_identity_fails_closed(tmp_path, change):
    value = json.loads((CONFIGS / 'runtime-fork.json').read_text())
    if change == 'digest':
        value['binaries']['llama-cli.exe'] = 'invalid'
    elif change == 'commit':
        value['upstream_commit'] = 'latest'
    elif change == 'abi':
        value['launcher_abi'] = '9.0.0'
    elif change == 'patch_order':
        value['patches'].reverse()
    elif change == 'capacity':
        value['capabilities']['max_static_layers'] = -1
    elif change == 'missing_binary':
        del value['binaries']['llama-server.exe']
    else:
        del value['dependencies']
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError):
        load_runtime_manifest(path)
