"""Reference CLI preflight and zero-native generation contracts."""

import importlib.util
import json
from pathlib import Path

from test_compiler_pipeline import inputs
from test_compiler_stock_discovery import FixtureEligibility, HOST


def driver():
    path = Path('scripts/benchmark_compiler_stock_reference.py')
    spec = importlib.util.spec_from_file_location('reference_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(tmp_path):
    return ['--descriptor', 'descriptor.json', '--inventory', 'inventory.json',
        '--hardware', 'hardware.json', '--workload', 'workload.json',
        '--runtime-identity', 'runtime.json', '--source-repository', str(tmp_path)]


def test_reference_cli_requires_fresh_database_before_input_load(tmp_path, monkeypatch, capsys):
    cli = driver()
    database = tmp_path / 'existing.sqlite3'
    database.write_bytes(b'user data')
    def unexpected(*args, **kwargs):
        raise AssertionError('must reject target before loading inputs')
    monkeypatch.setattr(cli, 'load_compiler_inputs', unexpected)
    code = cli.main([*arguments(tmp_path), '--action', 'run', '--output-dir', str(tmp_path / 'run'),
        '--evidence-db', str(database)])
    assert code == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'IDENTITY-STOP'
    assert database.read_bytes() == b'user data'


def test_reference_default_generation_does_not_launch_or_create_native_output(tmp_path, monkeypatch, capsys):
    cli = driver()
    inp = inputs(tmp_path)
    monkeypatch.setattr(cli, 'load_compiler_inputs', lambda *args, **kwargs: inp)
    monkeypatch.setattr(cli, 'capture_host_environment', lambda: HOST)
    monkeypatch.setattr('expertflow.compiler.stock_reference.EligibilityRegistry.with_builtins', FixtureEligibility)
    def unexpected(*args, **kwargs):
        raise AssertionError('generate must not create GPU sampler or native runner')
    monkeypatch.setattr(cli, 'WindowsGpuMemorySampler', unexpected)
    monkeypatch.setattr(cli, 'ServerMeasurementRunner', unexpected)
    output = tmp_path / 'native'
    preview = tmp_path / 'preview.json'
    code = cli.main([*arguments(tmp_path), '--output-dir', str(output), '--manifest-output', str(preview)])
    result = json.loads(capsys.readouterr().out)
    assert code == 0 and result['native_samples'] == 0
    assert result['status'] == 'GENERATED-REFERENCE-MANIFEST'
    assert json.loads(preview.read_text())['maximum_native_processes'] == 10
    assert not output.exists()


def test_reference_cli_validation_requires_existing_database(tmp_path, monkeypatch, capsys):
    cli = driver()
    def unexpected(*args, **kwargs):
        raise AssertionError('must reject missing database before loading inputs')
    monkeypatch.setattr(cli, 'load_compiler_inputs', unexpected)
    code = cli.main([*arguments(tmp_path), '--action', 'validate',
        '--evidence-db', str(tmp_path / 'missing.sqlite3'), '--reference-dir', str(tmp_path / 'reference')])
    assert code == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'IDENTITY-STOP'
