"""Fresh paired stock-product evidence, distinct from historical diagnostics."""

from dataclasses import replace
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from .plan import CandidateStatus, _decode_plan, load_execution_plan, seal_candidate, validate_execution_plan
from .pipeline import atomic_json
from .refinement import PAIRS, RESAMPLES, SEED, balanced_schedule, evaluate_pairs, execute_pairs
from .schema import canonical_payload, canonical_sha256

PROTOCOL = 'paired-stock-product-v1'


def reconstruct_product(report, store, *, host_environment):
    freeze = report.get('frozen', {})
    if freeze.get('protocol_version') != PROTOCOL:
        raise ValueError('not fresh stock-product protocol evidence')
    from .preflight import file_sha256
    protocol_path = Path('docs/superpowers/specs/2026-10-04-stock-configuration-discovery.md')
    if freeze.get('protocol_sha256') != file_sha256(protocol_path):
        raise ValueError('product frozen protocol identity mismatch')
    experiment_id = freeze.get('experiment_id')
    if not isinstance(experiment_id, str) or not re.fullmatch('[0-9a-f]{32}', experiment_id):
        raise ValueError('invalid product experiment identity')
    if not host_environment or canonical_payload(host_environment) != freeze.get('host_environment'):
        raise ValueError('product host environment mismatch')
    if canonical_payload(freeze.get('schedule')) != canonical_payload(balanced_schedule()):
        raise ValueError('product pair schedule mismatch')
    if any(freeze.get(k) != v for k,v in {'pairs':PAIRS, 'seed':SEED,
            'bootstrap_samples':RESAMPLES, 'equivalence_margin_pct':2}.items()):
        raise ValueError('product protocol statistical controls mismatch')
    frozen_ns = freeze.get('frozen_monotonic_ns')
    if type(frozen_ns) is not int or frozen_ns <= 0:
        raise ValueError('product freeze time missing')
    root = Path(freeze['experiment_root']).resolve()
    source = _decode_plan(freeze['source_plan'])
    validate_execution_plan(source, store=store)
    if source.plan_sha256 != freeze.get('source_plan_sha256'):
        raise ValueError('product source plan hash mismatch')
    if source.candidate.settings.static is not None:
        raise ValueError('requires pristine stock configuration')
    if canonical_payload(source.candidate.identities) != freeze.get('identities') or (
            canonical_payload(source.candidate.settings) != freeze.get('settings')):
        raise ValueError('source plan differs from tested product configuration')
    rows = report.get('rows', [])
    if len(rows) != 20 or len(report.get('outcomes', [])) != 20:
        raise ValueError('requires twenty complete fresh product runs')
    source_owners = {store.verify_measurement(mid)['owned_run_sha256'] for mid in source.candidate.measurement_ids}
    source_reference = store.verify_measurement(source.candidate.measurement_ids[0])
    reconstructed = []
    for index, row in enumerate(rows):
        pair, order = divmod(index, 2)
        arm = balanced_schedule()[pair][order]
        mid = row['measurement_id']
        record = store.measurement(mid)
        verified = store.verify_measurement(mid)
        artifacts = {a.role:Path(a.identity.path) for a in record.artifacts}
        expected_root = root/'raw'/f'pair-{pair:02}-{arm}'
        if any(p.resolve().parent != expected_root for p in artifacts.values()):
            raise ValueError('product artifacts outside frozen fresh experiment')
        started = json.loads(artifacts['run-start'].read_text(encoding='utf-8'))
        if started['started_monotonic_ns'] < frozen_ns:
            raise ValueError('product reused a process started before protocol freeze')
        if row.get('pair') != pair or row.get('arm') != arm or record.stage != f'product-{experiment_id}-{pair:02}-{arm}':
            raise ValueError('product pair/stage mismatch')
        if record.numerical_path != 'stock_same_runtime' or verified['owned_run_sha256'] in source_owners:
            raise ValueError('product reused historical or non-pristine evidence')
        if any(canonical_payload(row.get(k)) != canonical_payload(v) for k,v in verified.items()):
            raise ValueError('product report does not bind to verified native artifacts')
        if verified['candidate_id'] != source.candidate.candidate_id or verified['identities'] != freeze['identities']:
            raise ValueError('product candidate identity mismatch')
        if verified['settings_sha256'] != canonical_sha256(source.candidate.settings):
            raise ValueError('product settings mismatch')
        if any(verified[k] != source_reference[k] for k in ('generated_tokens_sha256', 'prompt_tokens_sha256')):
            raise ValueError('product native tokens differ from source reference')
        outcome = report['outcomes'][index]
        if outcome.get('status') != 'measured' or outcome.get('measurement_id') != mid:
            raise ValueError('product outcome missing or inconsistent')
        reconstructed.append({**verified, 'pair': pair, 'arm': arm, 'measurement_id': mid})
    result = evaluate_pairs(reconstructed)
    if result['status'] != 'PASS-MEASUREMENT':
        raise ValueError('stock product paired acceptance did not pass')
    for key,value in result.items():
        if canonical_payload(report.get(key)) != canonical_payload(value):
            raise ValueError('product claimed statistics differ from reconstructed evidence')
    return source, reconstructed, result


def publish_stock_product(report, store, output_dir, *, host_environment):
    source, rows, result = reconstruct_product(report, store, host_environment=host_environment)
    candidate = replace(source.candidate, status=CandidateStatus.MEASURED,
        measurement_ids=tuple(r['measurement_id'] for r in rows), validation=(('paired_stock_product', True),))
    plan = seal_candidate(candidate, store, candidate.identities, None)
    receipt = {'schema_version': '1.0.0', 'protocol_version': PROTOCOL,
        'published_plan_sha256': plan.plan_sha256, 'source_plan_sha256': source.plan_sha256,
        'host_environment': canonical_payload(host_environment), 'statistics': result,
        'experiment': canonical_payload(report)}
    receipt['receipt_sha256'] = canonical_sha256(receipt)
    output = Path(output_dir)
    accepted = output/'accepted'
    if accepted.exists():
        raise ValueError('stock product already published; no overwrite')
    pending = Path(tempfile.mkdtemp(prefix='.pending-acceptance-', dir=output))
    if pending.resolve().parent != output.resolve() or accepted.resolve().parent != output.resolve():
        raise ValueError('product publication path outside intended output')
    try:
        atomic_json(pending/'execution-plan.json', plan)
        atomic_json(pending/'acceptance-receipt.json', receipt)
        # Consumers see both artifacts together, never a half-published plan.
        os.rename(pending, accepted)
    finally:
        if pending.exists():
            if pending.resolve().parent != output.resolve() or not pending.name.startswith('.pending-acceptance-'):
                raise ValueError('unsafe product temporary cleanup path')
            shutil.rmtree(pending)
    return plan


def load_validated_stock_plan(plan_path, receipt_path, store, *, identities, host_environment):
    plan = load_execution_plan(plan_path, identities=identities, store=store)
    receipt = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    digest = receipt.pop('receipt_sha256', None)
    if digest != canonical_sha256(receipt) or receipt.get('schema_version') != '1.0.0':
        raise ValueError('stock product receipt hash/schema mismatch')
    if receipt.get('protocol_version') != PROTOCOL or receipt.get('published_plan_sha256') != plan.plan_sha256:
        raise ValueError('stock product receipt plan/protocol mismatch')
    if canonical_payload(host_environment) != receipt.get('host_environment'):
        raise ValueError('stock product receipt host mismatch')
    source, rows, result = reconstruct_product(receipt['experiment'], store, host_environment=host_environment)
    if receipt.get('statistics') != canonical_payload(result) or receipt.get('source_plan_sha256') != source.plan_sha256:
        raise ValueError('stock product receipt evidence/statistics mismatch')
    if plan.candidate.candidate_id != source.candidate.candidate_id or plan.candidate.measurement_ids != tuple(r['measurement_id'] for r in rows):
        raise ValueError('published plan differs from accepted native configuration/evidence')
    return plan


def execute_stock_product(inputs, source_plan_path, source_store, target_store, runner, output_dir, *, host_capture=None):
    from .preflight import capture_host_environment
    capture = host_capture or capture_host_environment
    report = execute_pairs(inputs, source_plan_path, source_store, target_store, runner, output_dir,
        validation_protocol=PROTOCOL, host_capture=capture)
    if report['status'] == 'PASS-MEASUREMENT':
        try:
            publish_stock_product(report, target_store, output_dir, host_environment=capture())
            report.update(status='PASS-STOCK-FALLBACK', live_validated_product=True)
        except (ValueError, OSError, KeyError, TypeError) as error:
            report.update(status='VALIDATION-STOP', reason=str(error), live_validated_product=False)
    atomic_json(Path(output_dir)/'report.json', report)
    return report


def run_accepted_stock_plan(plan_path, receipt_path, inputs, store, output_dir, *, runner=None, host_capture=None):
    from .preflight import capture_host_environment
    from .pipeline import validate_live_hardware
    from .runner import ServerMeasurementRunner, WindowsGpuMemorySampler
    capture = host_capture or capture_host_environment
    host = capture()
    plan = load_validated_stock_plan(plan_path, receipt_path, store,
        identities=inputs.identities(inputs.stock), host_environment=host)
    sampler = None
    if runner is None:
        validate_live_hardware(inputs.hardware)
        sampler = WindowsGpuMemorySampler(inputs.hardware.gpu_uuid)
        runner = ServerMeasurementRunner(store, memory_sampler=sampler)
    try:
        outcome = runner.run_once(plan.candidate, inputs.model, inputs.stock,
            output_dir=output_dir, measured=True, stage='accepted-stock-run')
        if outcome.status != 'measured':
            return outcome.status.upper().replace('_','-'), {'reason':outcome.reason, 'outcome':canonical_payload(outcome)}
        measured = store.verify_measurement(outcome.measurement_id)
        reference = store.verify_measurement(plan.candidate.measurement_ids[0])
        if measured['candidate_id'] != plan.candidate.candidate_id or any(
                measured[k] != reference[k] for k in ('generated_tokens_sha256','prompt_tokens_sha256')):
            raise ValueError('accepted stock execution identity/token mismatch')
        owners = {store.verify_measurement(mid)['owned_run_sha256'] for mid in plan.candidate.measurement_ids}
        if measured['owned_run_sha256'] in owners:
            raise ValueError('accepted execution reused validation process')
        if canonical_payload(capture()) != canonical_payload(host):
            raise ValueError('host environment changed during accepted stock execution')
        return 'MEASURED-ACCEPTED-STOCK', {'plan_sha256':plan.plan_sha256,
            'measurement_id':outcome.measurement_id,'decode_tps':measured['decode_tps'],
            'validation_scope':'fresh exact execution of paired-validated stock plan'}
    finally:
        if sampler:
            sampler.close()
