"""Bounded fresh incumbent reference; diagnostic stability is not product acceptance."""

from dataclasses import replace
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import uuid

from .pipeline import atomic_json
from .plan import CandidatePlan, CandidateStatus, RuntimeSettings, _decode_plan, seal_candidate, validate_execution_plan
from .preflight import capture_host_environment, file_sha256
from .schema import canonical_payload, canonical_sha256
from .stock_discovery import _candidate, _checksum, _snapshot_inputs, _source_files
from .stock_eligibility import EligibilityRegistry
from .stock_search import topology_anchors

PROTOCOL = 'bounded-stock-reference-v1'
SPEC = Path('docs/research/protocols/specs/2026-10-04-stock-method-reuse.md')


def _sources():
    files = {**_source_files(), str(SPEC): file_sha256(SPEC)}
    driver = Path('scripts/benchmark_compiler_stock_reference.py')
    if driver.exists():
        files[str(driver)] = file_sha256(driver)
    return files


def _incumbent(inputs, proof=None):
    w = inputs.workload
    if w.policy.value != 'exact' or (w.kv_type_k, w.kv_type_v) != ('f16', 'f16'):
        raise ValueError('reference requires exact policy and audited F16 KV')
    cpu_moe = (proof or {}).get('baseline_cpu_moe', True)
    return CandidatePlan(inputs.identities(inputs.stock), RuntimeSettings(99, cpu_moe,
        w.cuda_graphs, w.kv_type_k, w.kv_type_v, w.batch_size, w.microbatch_size))


def prepare_reference(inputs, output_dir, *, host_environment, source_repository, registry=None):
    topology_anchors(host_environment, inputs.workload.threads)
    proof = (registry or EligibilityRegistry.with_builtins()).attest(inputs, host_environment, source_repository)
    candidate = _incumbent(inputs, proof)
    manifest = canonical_payload({'schema_version': '1.0.0', 'protocol_version': PROTOCOL,
        'experiment_id': uuid.uuid4().hex, 'frozen_monotonic_ns': time.monotonic_ns(),
        'experiment_root': str(Path(output_dir).resolve()),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_files': _sources(), 'protocol_sha256': file_sha256(SPEC),
        'source_repository': str(Path(source_repository).resolve()),
        'inputs': {'model': inputs.model, 'hardware': inputs.hardware, 'stock': inputs.stock},
        'candidate': candidate, 'host_environment': host_environment, 'eligibility': proof,
        'maximum_native_processes': 10, 'maximum_cv_pct': 10})
    manifest['manifest_sha256'] = canonical_sha256(manifest)
    return manifest


def _validate(manifest, host, registry):
    _checksum(manifest, 'manifest_sha256')
    if manifest.get('schema_version') != '1.0.0' or manifest.get('protocol_version') != PROTOCOL:
        raise ValueError('unsupported reference protocol')
    if manifest['protocol_sha256'] != file_sha256(SPEC) or manifest['source_files'] != _sources():
        raise ValueError('reference source/protocol changed')
    if manifest['host_environment'] != canonical_payload(host):
        raise ValueError('reference host mismatch')
    if not re.fullmatch('[0-9a-f]{32}', manifest['experiment_id']) or type(manifest['frozen_monotonic_ns']) is not int or manifest['frozen_monotonic_ns'] <= 0:
        raise ValueError('invalid fresh reference boundary')
    if manifest.get('maximum_native_processes') != 10 or manifest.get('maximum_cv_pct') != 10:
        raise ValueError('reference frozen budget/gate mismatch')
    candidate = _candidate(manifest['candidate'])
    inputs = _snapshot_inputs(manifest['inputs'], candidate.identities.workload)
    topology_anchors(host, inputs.workload.threads)
    proof = (registry or EligibilityRegistry.with_builtins()).attest(inputs, host, manifest['source_repository'])
    if canonical_payload(_incumbent(inputs, proof)) != manifest['candidate']:
        raise ValueError('reference candidate/input identity mismatch')
    if canonical_payload(proof) != manifest['eligibility']:
        raise ValueError('reference differs from trusted complete eligibility')
    return candidate


def _row(row, store, manifest, candidate, index, reference, owners):
    mid = row['measurement_id']
    native = store.verify_measurement(mid)
    record = store.measurement(mid)
    if native['measured'] is not True or native['exit_code'] != 0 or any(
            native['validations'].get(name) is not True for name in ('exact_tokens', 'memory', 'cleanup')):
        raise ValueError('reference requires measured passing native records')
    if row.get('reference_index') != index or record.stage != 'confirmation' or record.numerical_path != 'stock_same_runtime':
        raise ValueError('reference row order/stage/path mismatch')
    if native['candidate_id'] != candidate.candidate_id or native['identities'] != canonical_payload(candidate.identities) or native['settings_sha256'] != canonical_sha256(candidate.settings):
        raise ValueError('reference launch identity mismatch')
    if any(canonical_payload(row.get(key)) != canonical_payload(value) for key, value in native.items()):
        raise ValueError('reference row differs from verified native evidence')
    if native['owned_run_sha256'] in owners:
        raise ValueError('reference reused owned process')
    owners.add(native['owned_run_sha256'])
    if reference is not None and any(native[key] != reference[key] for key in ('prompt_tokens_sha256', 'generated_tokens_sha256')):
        raise ValueError('reference native tokens are unstable')
    paths = {a.role: Path(a.identity.path) for a in record.artifacts}
    root = Path(manifest['experiment_root']) / 'raw' / f'reference-{index:02}'
    if any(path.resolve().parent != root.resolve() for path in paths.values()):
        raise ValueError('reference artifacts outside frozen root')
    launch = json.loads(paths['launch'].read_text(encoding='utf-8'))
    if launch.get('host_environment') != manifest['host_environment'] or launch.get('experiment_context') != {'manifest_sha256': manifest['manifest_sha256']}:
        raise ValueError('native reference launch does not bind host/manifest')
    start = json.loads(paths['run-start'].read_text(encoding='utf-8'))
    if start['started_monotonic_ns'] < manifest['frozen_monotonic_ns']:
        raise ValueError('reference reused pre-freeze process')
    return native


def reconstruct_reference(report, store, *, host_environment, registry=None):
    candidate = _validate(report['manifest'], host_environment, registry)
    rows = report['rows']
    if len(rows) != 10 or len(report['outcomes']) != 10:
        raise ValueError('reference requires exactly ten complete runs')
    owners = set()
    reference = None
    for index, row in enumerate(rows):
        native = _row(row, store, report['manifest'], candidate, index, reference, owners)
        reference = reference or native
        outcome = report['outcomes'][index]
        if outcome.get('status') != 'measured' or outcome.get('measurement_id') != row['measurement_id']:
            raise ValueError('reference outcomes inconsistent')
    rates = [row['decode_tps'] for row in rows]
    mean = statistics.mean(rates)
    cv = 100 * statistics.stdev(rates) / mean
    if cv > 10 or report.get('status') != 'REFERENCE-STABLE' or report.get('product_accepted') is not False:
        raise ValueError('reference stability gate did not pass')
    if report.get('mean_tps') != mean or report.get('cv_pct') != cv:
        raise ValueError('reference statistics differ from native evidence')
    return candidate


def _publish(report, store, output, host, registry):
    base = reconstruct_reference(report, store, host_environment=host, registry=registry)
    candidate = replace(base, status=CandidateStatus.MEASURED,
        measurement_ids=tuple(row['measurement_id'] for row in report['rows']),
        validation=(('stock_reference_stable', True),))
    plan = seal_candidate(candidate, store, candidate.identities, None)
    receipt = canonical_payload({'schema_version': '1.0.0', 'protocol_version': PROTOCOL,
        'published_plan_sha256': plan.plan_sha256, 'experiment': report})
    receipt['receipt_sha256'] = canonical_sha256(receipt)
    pending = Path(tempfile.mkdtemp(prefix='.pending-reference-', dir=output))
    try:
        if (output / 'diagnostic').exists():
            raise ValueError('reference publication already exists')
        atomic_json(pending / 'execution-plan.json', plan)
        atomic_json(pending / 'reference-receipt.json', receipt)
        os.rename(pending, output / 'diagnostic')
    finally:
        if pending.exists():
            if pending.resolve().parent != output.resolve() or not pending.name.startswith('.pending-reference-'):
                raise ValueError('unsafe reference temporary cleanup')
            shutil.rmtree(pending)


def load_reference_plan(directory, store, *, identities, host_environment, registry=None):
    root = Path(directory)
    receipt = json.loads((root / 'reference-receipt.json').read_text(encoding='utf-8'))
    _checksum(receipt, 'receipt_sha256')
    if receipt.get('schema_version') != '1.0.0' or receipt.get('protocol_version') != PROTOCOL:
        raise ValueError('reference receipt protocol mismatch')
    candidate = reconstruct_reference(receipt['experiment'], store, host_environment=host_environment, registry=registry)
    plan = _decode_plan(json.loads((root / 'execution-plan.json').read_text(encoding='utf-8')))
    validate_execution_plan(plan, store=store)
    expected = tuple(row['measurement_id'] for row in receipt['experiment']['rows'])
    if plan.candidate.identities != identities or plan.candidate.candidate_id != candidate.candidate_id or plan.candidate.measurement_ids != expected or plan.plan_sha256 != receipt['published_plan_sha256']:
        raise ValueError('reference plan differs from verified stability evidence')
    return plan


def execute_stock_reference(inputs, store, runner, output_dir, *, host_capture=None, registry=None, source_repository):
    output = Path(output_dir)
    if output.exists():
        raise ValueError('fresh reference output required; no retries/resume')
    with store._connection() as connection:
        if connection.execute('SELECT COUNT(*) FROM measurement').fetchone()[0]:
            raise ValueError('fresh empty reference database required')
    capture = host_capture or capture_host_environment
    host = capture()
    store.prime_model(inputs.model)
    manifest = prepare_reference(inputs, output, host_environment=host,
        source_repository=source_repository, registry=registry)
    candidate = _candidate(manifest['candidate'])
    output.mkdir(parents=True)
    atomic_json(output / 'frozen-manifest.json', manifest)
    report = {'status': 'RUNNING', 'manifest': manifest, 'rows': [], 'outcomes': [], 'product_accepted': False}
    owners = set()
    reference = None
    try:
        for index in range(10):
            if canonical_payload(capture()) != manifest['host_environment'] or _sources() != manifest['source_files']:
                raise ValueError('reference host/source changed before launch')
            outcome = runner.run_once(candidate, inputs.model, inputs.stock,
                output_dir=output / 'raw' / f'reference-{index:02}', measured=True, stage='confirmation',
                host_environment=host, experiment_context={'manifest_sha256': manifest['manifest_sha256']})
            report['outcomes'].append(canonical_payload(outcome))
            if outcome.status != 'measured':
                report.update(status=outcome.status.upper().replace('_', '-'), reason=outcome.reason)
                atomic_json(output / 'report.json', report)
                return report
            row = {**store.verify_measurement(outcome.measurement_id),
                'measurement_id': outcome.measurement_id, 'reference_index': index}
            native = _row(row, store, manifest, candidate, index, reference, owners)
            reference = reference or native
            report['rows'].append(row)
            atomic_json(output / 'report.json', report)
        rates = [row['decode_tps'] for row in report['rows']]
        report.update(mean_tps=statistics.mean(rates), cv_pct=100 * statistics.stdev(rates) / statistics.mean(rates))
        report['status'] = 'REFERENCE-STABLE' if report['cv_pct'] <= 10 else 'INCONCLUSIVE'
        if report['status'] == 'REFERENCE-STABLE':
            if canonical_payload(capture()) != manifest['host_environment']:
                raise ValueError('reference host changed before publication')
            _publish(report, store, output, host, registry)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        report.update(status='VALIDATION-STOP', reason=str(error))
    atomic_json(output / 'report.json', report)
    return report
