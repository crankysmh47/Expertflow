"""Fixed-budget paired measurement; it never rewrites the Phase 3 verdict."""

import math
import random
import statistics
import json
from pathlib import Path
import re
import subprocess
import time

from .schema import require_number, canonical_payload

PAIRS = 10
SEED = 20261003
RESAMPLES = 10000


def balanced_schedule():
    schedule = [('direct', 'sealed')] * 5 + [('sealed', 'direct')] * 5
    random.Random(SEED).shuffle(schedule)
    return tuple(schedule)


def _percentile(values, fraction):
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def paired_statistics(direct, sealed):
    direct, sealed = tuple(direct), tuple(sealed)
    if len(direct) != PAIRS or len(sealed) != PAIRS:
        raise ValueError('requires exactly ten complete pairs')
    for value in (*direct, *sealed):
        require_number(value, 'paired native TPS', 1e-12)
    changes = [math.log(b) - math.log(a) for a, b in zip(direct, sealed)]
    rng = random.Random(SEED)
    try:
        draws = sorted(100 * math.expm1(statistics.mean(rng.choices(changes, k=PAIRS)))
                       for _ in range(RESAMPLES))
        point = 100 * math.expm1(statistics.mean(changes))
    except OverflowError as error:
        raise ValueError('paired percentage overflow') from error
    return {'geometric_change_pct': point,
            'ci95_pct': [_percentile(draws, .025), _percentile(draws, .975)],
            'ci90_pct': [_percentile(draws, .05), _percentile(draws, .95)],
            'one_sided95_lower_pct': _percentile(draws, .05),
            'direct_mean_tps': statistics.mean(direct), 'sealed_mean_tps': statistics.mean(sealed),
            'direct_cv_pct': statistics.stdev(direct) * 100 / statistics.mean(direct),
            'sealed_cv_pct': statistics.stdev(sealed) * 100 / statistics.mean(sealed),
            'direct_tps': direct, 'sealed_tps': sealed, 'paired_log_ratios': changes,
            'bootstrap_seed': SEED, 'bootstrap_samples': RESAMPLES,
            'method': 'percentile bootstrap of ten paired mean log ratios'}


def evaluate_pairs(rows):
    """Consume EvidenceStore-verified rows; callers must bind them to their schedule."""
    rows = tuple(rows)
    if len(rows) != PAIRS * 2:
        raise ValueError('requires twenty independently verified measurements')
    for name in ('measurement_id', 'owned_run_sha256'):
        if any(not r.get(name) for r in rows) or len({r[name] for r in rows}) != len(rows):
            raise ValueError('missing or duplicate ' + name)
    for name in ('candidate_id', 'identities', 'settings_sha256', 'generated_tokens_sha256', 'prompt_tokens_sha256'):
        if any(not r.get(name) or r[name] != rows[0].get(name) for r in rows):
            raise ValueError('paired evidence mismatch: ' + name)
    for row in rows:
        if row.get('measured') is not True or row.get('exit_code') != 0 or any(
                row.get('validations', {}).get(name) is not True for name in ('exact_tokens', 'memory', 'cleanup')):
            raise ValueError('paired evidence correctness failure')
    pairs = {}
    for row in rows:
        if type(row.get('pair')) is not int or row['pair'] not in range(PAIRS) or row.get('arm') not in ('direct', 'sealed'):
            raise ValueError('invalid pair or arm')
        key = (row['pair'], row['arm'])
        if key in pairs:
            raise ValueError('duplicate pair/arm')
        pairs[key] = row['decode_tps']
    if len(pairs) != PAIRS * 2:
        raise ValueError('missing pair arm')
    result = paired_statistics([pairs[i, 'direct'] for i in range(PAIRS)],
                               [pairs[i, 'sealed'] for i in range(PAIRS)])
    result['noninferior'] = result['one_sided95_lower_pct'] > -2
    result['equivalent'] = result['ci90_pct'][0] > -2 and result['ci90_pct'][1] < 2
    result['variance_pass'] = max(result['direct_cv_pct'], result['sealed_cv_pct']) <= 10
    result['status'] = ('PASS-MEASUREMENT' if result['noninferior'] and result['equivalent'] and result['variance_pass']
                        else 'VALIDATION-STOP' if result['ci90_pct'][1] < -2 else 'INCONCLUSIVE')
    result['optimization_gain_established'] = False
    return result


def diagnostic_summary(record):
    artifacts = {a.role: Path(a.identity.path) for a in record.artifacts}
    memory = json.loads(artifacts['memory'].read_text())
    samples = [s for s in memory['samples'] if s.get('phase') == 'measurement']
    summary = {'measurement_samples': len(samples),
               'peak_owned_bytes': max(s['dedicated_bytes'] for s in memory['samples']),
               'minimum_device_free_bytes': min(s['device_free_bytes'] for s in memory['samples']),
               'cpu_temperature_c': None, 'cpu_temperature_reason': 'no trusted CPU sensor'}
    for group, names in {'gpu': ('sm_clock_mhz', 'memory_clock_mhz', 'temperature_c', 'power_mw', 'gpu_utilization_pct', 'memory_utilization_pct'),
                         'cpu': ('performance_pct', 'reported_frequency_mhz', 'system_utilization_pct')}.items():
        for name in names:
            values = [s.get('diagnostics', {}).get(group, {}).get(name) for s in samples]
            values = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
            summary[group + '_' + name] = {'count': len(values), 'min': min(values), 'mean': statistics.mean(values),
                                          'max': max(values)} if values else {'count': 0, 'available': False}
    cpu = [(s.get('time_monotonic'), s.get('diagnostics', {}).get('cpu', {}).get('owned_cpu_time_100ns')) for s in samples]
    cpu = [(t, v) for t, v in cpu if type(v) is int and type(t) in (int, float)]
    summary['owned_cpu_busy_cores_mean'] = ((cpu[-1][1] - cpu[0][1]) / 1e7 / (cpu[-1][0] - cpu[0][0])
                                           if len(cpu) > 1 and cpu[-1][0] > cpu[0][0] else None)
    logs = ''.join((artifacts['launch'].parent / name).read_text(encoding='utf-8', errors='replace')
                   for name in ('stdout.log', 'stderr.log') if (artifacts['launch'].parent / name).exists())
    placement = re.findall(r'offloaded (\d+)/(\d+) layers', logs)
    summary['actual_offloaded_layers'] = list(map(int, placement[-1])) if placement else None
    summary['placement_note'] = 'native logged count' if placement else 'unknown; owned bytes are diagnostic only'
    summary['phase_wall_ms'] = json.loads(artifacts['phase-timing'].read_text()) if 'phase-timing' in artifacts else None
    return summary


def execute_pairs(inputs, source_plan_path, source_store, target_store, runner, output_dir):
    from .plan import CandidatePlan, RuntimeSettings, load_execution_plan
    from .preflight import file_sha256
    from .pipeline import atomic_json
    from .evidence import MeasurementKey
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source_plan_path = Path(source_plan_path)
    identities = inputs.identities(inputs.stock)
    plan = load_execution_plan(source_plan_path, identities=identities, store=source_store)
    w = inputs.workload
    direct = CandidatePlan(identities, RuntimeSettings(99, True, w.cuda_graphs, w.kv_type_k, w.kv_type_v,
                                                       w.batch_size, w.microbatch_size))
    if plan.candidate.candidate_id != direct.candidate_id:
        raise ValueError('source plan is not the frozen ngl99 CPU-MoE control')
    if len(plan.candidate.measurement_ids) != 10 or any(source_store.measurement(mid).stage != 'confirmation'
                                                       for mid in plan.candidate.measurement_ids):
        raise ValueError('source plan lacks ten confirmation records')
    baseline = source_store.verify_measurement(plan.candidate.measurement_ids[0])
    target_store.prime_model(inputs.model)
    protocol_path = Path('docs/superpowers/specs/2026-10-03-compiler-measurement-refinement.md')
    freeze = {'protocol_sha256': file_sha256(protocol_path),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_plan_file_sha256': file_sha256(source_plan_path), 'source_plan_sha256': plan.plan_sha256,
              'source_evidence_db': str(source_store.path.resolve()),
              'source_evidence_db_sha256': file_sha256(source_store.path),
              'source_plan_diagnostic_only': True, 'source_candidate_id': direct.candidate_id,
              'identities': canonical_payload(identities), 'settings': canonical_payload(direct.settings),
              'source_provenance': inputs.provenance, 'pairs': PAIRS, 'schedule': balanced_schedule(),
              'seed': SEED, 'bootstrap_samples': RESAMPLES, 'equivalence_margin_pct': 2}
    sources = [*sorted(Path('src/expertflow/compiler').rglob('*.py')), Path('scripts/benchmark_compiler_refinement.py')]
    freeze['source_files'] = {str(path): file_sha256(path) for path in sources}
    atomic_json(output / 'frozen-protocol.json', freeze)
    report = {'status': 'RUNNING', 'frozen': freeze, 'rows': [], 'outcomes': [],
              'live_validated_product': False, 'optimization_gain_established': False}
    try:
        for pair, order in enumerate(balanced_schedule()):
            for arm in order:
                started = time.perf_counter()
                if file_sha256(source_plan_path) != freeze['source_plan_file_sha256']:
                    raise ValueError('source plan changed during experiment')
                if any(file_sha256(Path(path)) != digest for path, digest in freeze['source_files'].items()):
                    raise ValueError('measured source changed during experiment')
                preparation_started = time.perf_counter()
                candidate = direct if arm == 'direct' else load_execution_plan(source_plan_path, identities=identities).candidate
                preparation_ms = (time.perf_counter() - preparation_started) * 1000
                outcome = runner.run_once(candidate, inputs.model, inputs.stock,
                    output_dir=output / 'raw' / f'pair-{pair:02}-{arm}', measured=True, stage=f'aa-{pair:02}-{arm}')
                report['outcomes'].append(canonical_payload(outcome))
                if outcome.status != 'measured':
                    report.update(status=outcome.status.upper().replace('_', '-'), reason=outcome.reason)
                    atomic_json(output / 'report.json', report)
                    return report
                record = target_store.measurement(outcome.measurement_id)
                if record.key != MeasurementKey.from_candidate(direct) or record.stage != f'aa-{pair:02}-{arm}':
                    raise ValueError('run does not bind to frozen pair/settings')
                verified = target_store.verify_measurement(outcome.measurement_id)
                for name in ('generated_tokens_sha256', 'prompt_tokens_sha256'):
                    if verified[name] != baseline[name]:
                        raise ValueError('native tokens differ from the frozen selected stock reference')
                report['rows'].append({**verified, 'pair': pair, 'arm': arm,
                    'measurement_id': outcome.measurement_id, 'run_wall_seconds': time.perf_counter() - started,
                    'candidate_preparation_wall_ms': preparation_ms,
                    'diagnostics': diagnostic_summary(record), 'artifacts': canonical_payload(record.artifacts)})
                atomic_json(output / 'report.json', report)
        report.update(evaluate_pairs(report['rows']))
        if file_sha256(source_store.path) != freeze['source_evidence_db_sha256']:
            raise ValueError('original source database changed during read-only experiment')
    except (ValueError, OSError, KeyError, TypeError) as error:
        report.update(status='VALIDATION-STOP', reason=str(error))
    atomic_json(output / 'report.json', report)
    return report
