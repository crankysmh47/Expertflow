"""CPU controls for the single implementation review; no native evidence."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path('tests').resolve()))
from test_stock_wider import fixture_context, validate


@pytest.mark.parametrize('fail_at', [1, 10, 20])
def test_product_failed_prefix_retains_environment_reason(tmp_path, monkeypatch, fail_at):
    context = fixture_context(tmp_path, monkeypatch)
    wider, inputs, case, sequence, clock, factory, runners, host = context
    def failure_factory(store):
        runner = factory(store)
        original = runner.run_once
        def run(*args, **kwargs):
            runner.fail = store.path.name == 'product.sqlite3' and len(runner.calls) == fail_at - 1
            return original(*args, **kwargs)
        runner.run_once = run
        return runner
    report = wider.execute_case(inputs, case, sequence, runner_factory=failure_factory,
                                capture=lambda: deepcopy(host))
    assert report['status'] == 'ENVIRONMENT-BLOCKED'
    assert report['reason'] == 'counter unavailable'
    result = validate(context, report, monkeypatch)
    assert result['attempts'] == 86 + fail_at and not result['utility_gain_established']
    from scripts.benchmark_compiler_stock_coverage import can_continue
    assert not can_continue(result)


def test_product_prelaunch_resource_prefix_reconstructs(tmp_path, monkeypatch):
    context = fixture_context(tmp_path, monkeypatch)
    wider, inputs, case, sequence, clock, factory, runners, host = context
    def stop_before_product(store):
        runner = factory(store)
        if store.path.name == 'product.sqlite3':
            original_sleep = clock.sleep
            def exhausted(seconds):
                original_sleep(seconds)
                clock.value += 14400_000_000_000
            monkeypatch.setattr(wider.time, 'sleep', exhausted)
        return runner
    report = wider.execute_case(inputs, case, sequence, runner_factory=stop_before_product,
                                capture=lambda: deepcopy(host))
    assert report['status'] == 'RESOURCE-BUDGET-STOP'
    result = validate(context, report, monkeypatch)
    assert result['attempts'] == 87 and result['native_processes'] == 86
    assert not result['utility_gain_established']


def test_collection_finish_cannot_precede_any_attempt(tmp_path, monkeypatch):
    context = fixture_context(tmp_path, monkeypatch, fail=True)
    wider, inputs, case, sequence, clock, factory, runners, host = context
    report = wider.execute_case(inputs, case, sequence, runner_factory=factory,
                                capture=lambda: deepcopy(host))
    report['collection_finished_monotonic_ns'] = report['manifest']['case_started_monotonic_ns']
    report['collection_wall_seconds'] = 0
    with pytest.raises(ValueError, match='collection'):
        validate(context, report, monkeypatch)


def runtime_binding(root):
    from expertflow.compiler.runner import RuntimeBinding
    from expertflow.compiler.preflight import file_sha256
    from expertflow.compiler.schema import ArtifactIdentity
    root=root/'drift-runtime'
    root.mkdir()
    binaries = root/'binaries'
    binaries.mkdir()
    server, cli, cuda = binaries/'llama-server.exe', binaries/'llama-cli.exe', root/'cuda.dll'
    for path in (server, cli, cuda): path.write_bytes(path.name.encode())
    identity = lambda p: ArtifactIdentity(str(p.resolve()), p.stat().st_size, file_sha256(p))
    manifest = {'binaries': {p.name: file_sha256(p) for p in (server, cli)},
                'dependencies': {}, 'cuda_runtime_sha256': file_sha256(cuda), 'patches': []}
    return RuntimeBinding(identity(server), json.dumps(manifest), (), identity(cuda)), cli


@pytest.mark.parametrize('drift', ['companion', 'dll', 'resource'])
def test_actual_runner_blocks_drift_and_retains_resource_type(tmp_path, monkeypatch, drift):
    from expertflow.compiler import preflight
    from expertflow.compiler.evidence import EvidenceStore
    from expertflow.compiler.plan import CandidatePlan, RuntimeSettings
    from expertflow.compiler.runner import ServerMeasurementRunner
    context = fixture_context(tmp_path, monkeypatch)
    wider, inputs, case, sequence, clock, factory, runners, host = context
    binding, cli = runtime_binding(tmp_path)
    inputs = replace(inputs, stock=binding)
    candidate = CandidatePlan(inputs.identities(binding), RuntimeSettings(99, True))
    root = Path(case['planned_root']); root.mkdir()
    report = {'manifest': {'experiment_root': str(root), 'source_files': wider.sources(),
        'host_environment': host, 'case_started_monotonic_ns': clock.now(),
        'sequence_started_monotonic_ns': clock.now(), 'input_load_seconds': 0,
        'manifest_sha256': 'a'*64}, 'attempts': []}
    monkeypatch.setattr(preflight, 'capture_host_environment', lambda: deepcopy(host))
    def wait(seconds):
        clock.sleep(seconds)
        if drift == 'companion': cli.write_bytes(b'changed')
        if drift == 'dll': (cli.parent/'unpinned.dll').write_bytes(b'changed')
    monkeypatch.setattr(wider.time, 'sleep', wait)
    class Preparation:
        def check_idle(self):
            if drift == 'resource': clock.value += 14400_000_000_000
            return {}
    spawns = []
    def spawn(*args, **kwargs):
        spawns.append(True)
        raise RuntimeError('sentinel; no native process')
    inner = ServerMeasurementRunner(EvidenceStore(root/'utility.sqlite3'), process_factory=spawn,
                                    memory_sampler=Preparation())
    wrapper = wider.PacedRunner(inner, report, lambda: deepcopy(host))
    with pytest.raises(wider.ResourceStop if drift == 'resource' else ValueError):
        wrapper.run_once(candidate, inputs.model, binding, output_dir=root/'raw/one',
                         stage='fixture', measured=True)
    assert spawns == [] and inner.process_factory is spawn


def test_executing_artifact_verifier_in_source_freeze():
    from expertflow.stock.wider import sources
    assert str(Path('src/expertflow/artifacts.py').resolve()) in sources()


@pytest.mark.parametrize('change', ['lf','crlf','content'])
def test_committed_source_attestation_preserves_only_equivalent_line_endings(tmp_path,monkeypatch,change):
    from expertflow.compiler.preflight import file_sha256
    from scripts.benchmark_compiler_stock_coverage import require_committed_sources
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    marker=tmp_path/'docs/evidence/stock-coverage-20261005/implementation-review.md'
    marker.parent.mkdir(parents=True)
    marker.write_text('reviewed\n')
    (tmp_path/'.gitattributes').write_text('*.json text eol=lf\n')
    source=tmp_path/'input.json'
    source.write_bytes(b'{"value":1}\n')
    subprocess.run(['git','-C',str(tmp_path),'add','.'],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.email=fixture@example.invalid',
                    '-c','user.name=Fixture','commit','-qm','fixture'],check=True)
    if change=='crlf':source.write_bytes(b'{"value":1}\r\n')
    if change=='content':source.write_bytes(b'{"value":2}\r\n')
    before=source.read_bytes()
    monkeypatch.chdir(tmp_path)
    if change=='content':
        with pytest.raises(ValueError,match='committed'):require_committed_sources({str(source):file_sha256(source)})
    else:
        require_committed_sources({str(source):file_sha256(source)})
    assert source.read_bytes()==before


@pytest.mark.parametrize('module_name, relative', [
    ('expertflow.artifacts', 'artifacts.py'), ('expertflow.cli.main', 'cli/main.py')])
def test_installed_executing_sources_must_match_checkout(tmp_path, monkeypatch, module_name, relative):
    import importlib
    from scripts.benchmark_compiler_stock_coverage import require_matching_package
    module = importlib.import_module(module_name)
    counterfeit = tmp_path/Path(relative).name
    counterfeit.write_text('different installed source')
    monkeypatch.setattr(module, '__file__', str(counterfeit))
    with pytest.raises(ValueError, match='installed'):
        require_matching_package()


@pytest.mark.parametrize('mutation', ['launch', 'owner', 'teardown'])
@pytest.mark.parametrize('backend', ['public','raw'])
def test_failed_native_requires_bound_context_and_owned_cleanup(tmp_path, mutation, backend):
    from expertflow.stock.wider_audit import audit_attempts
    root = tmp_path.resolve(); output = root/'raw/one'; output.mkdir(parents=True)
    identity = {'pid': 1, 'creation_time_100ns': 2, 'creation_source': 'GetProcessTimes',
                'run_id': 'failed-owned-run', 'started_monotonic_ns': 31_000_000_000}
    launch = {'host_environment': {}, 'experiment_context': {'manifest_sha256': 'a'*64}}
    process = {'pid': 1, 'run_id': identity['run_id'], 'creation_time_100ns': 2,
               'exited': True, 'cleanup': True, 'memory_settled': True, 'forced_kill': False}
    memory = {'teardown_reading': {'pid': 1, 'state': 'absent', 'counter_available': True,
                                  'dedicated_bytes': 0}}
    for name, payload in [('run-start', identity), ('launch', launch), ('process', process), ('memory', memory)]:
        (output/f'{name}.json').write_text(json.dumps(payload))
    report = {'manifest': {'experiment_root': str(root), 'frozen_monotonic_ns': 1,
        'case_started_monotonic_ns': 1, 'sequence_started_monotonic_ns': 1,
        'input_load_seconds': 0, 'host_environment': {}, 'manifest_sha256': 'a'*64},
        'attempts': [{'output_dir': str(output), 'database': str(root/'utility.sqlite3'),
        'wait_started_monotonic_ns': 1_000_000_000, 'wait_finished_monotonic_ns': 31_000_000_000,
        'wait_elapsed_ns': 30_000_000_000, 'wait_seconds': 30,
        'finished_monotonic_ns': 32_000_000_000, 'native_started': True,
        'process_identity': identity, 'status': 'environment_blocked'}]}
    if backend=='public':
        audit=lambda: audit_attempts(report)
    else:
        import runpy
        from expertflow.compiler.schema import canonical_sha256
        raw_audit=runpy.run_path('docs/evidence/stock-coverage-20261005/independent_audit.py')['audit']
        case={'case_id':'failure','planned_root':str(root)}
        sequence={'registration':{'cases':[case]},'source_files':{},'host_environment':{}}
        sequence['manifest_sha256']=canonical_sha256(sequence)
        report['manifest'].update(case=case,sequence_manifest_sha256=sequence['manifest_sha256'])
        report['manifest'].pop('manifest_sha256')
        report['manifest']['manifest_sha256']=canonical_sha256(report['manifest'])
        launch['experiment_context']['manifest_sha256']=report['manifest']['manifest_sha256']
        (output/'launch.json').write_text(json.dumps(launch))
        report.update(status='ENVIRONMENT-BLOCKED',rows=[])
        (root/'report.json').write_text(json.dumps(report))
        outer=tmp_path/'outer.json'
        outer.write_text(json.dumps({'manifest':sequence,'native_processes':1}))
        audit=lambda:raw_audit(outer)
    audit()
    if mutation == 'launch': (output/'launch.json').unlink()
    elif mutation == 'owner':
        process['pid'] = 999; (output/'process.json').write_text(json.dumps(process))
    else:
        memory['teardown_reading']['pid'] = 999; (output/'memory.json').write_text(json.dumps(memory))
    with pytest.raises(ValueError): audit()
