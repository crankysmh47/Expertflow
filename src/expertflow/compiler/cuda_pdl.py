"""One qualified scheduling diagnostic; never publishes a product plan."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time

from .pipeline import atomic_json
from .plan import CandidatePlan
from .preflight import capture_host_environment, file_sha256
from .refinement import balanced_schedule, paired_statistics
from .schema import canonical_payload, canonical_sha256
from .stock_discovery import _candidate, _checksum, _snapshot_inputs
from .stock_eligibility import Gemma4Q6SchedulingProvider
from .stock_validation import load_validated_stock_plan

SPEC = Path('docs/superpowers/specs/2026-10-04-cuda-pdl-and-generalization.md')
PROBE = Path('docs/evidence/compiler-cuda-pdl-20261004/feasibility.json')


def source_files():
    paths = [*sorted(Path('src/expertflow/compiler').rglob('*.py')), SPEC, PROBE,
             Path('scripts/benchmark_compiler_cuda_pdl.py')]
    return {str(p): file_sha256(p) for p in paths}


def qualify_control(inputs, host, repository):
    scope = Gemma4Q6SchedulingProvider().attest(inputs, host, repository)
    proof = json.loads(PROBE.read_text())
    _checksum(proof, 'proof_sha256')
    expected = {'shapes': 18, 'computes': 54, 'cuda_launch_errors': 0}
    for arm, pdl, classic in [('on', 84, 72), ('off', 0, 156)]:
        if proof[arm]['counts'] != {**expected, 'pdl_launches': pdl, 'classic_launches': classic}:
            raise ValueError('native PDL launch proof mismatch')
        for path, digest in proof[arm]['artifacts'].items():
            if file_sha256(Path(path)) != digest:
                raise ValueError('native PDL proof artifact changed')
    if proof['on']['output_sha256'] != proof['off']['output_sha256']:
        raise ValueError('PDL output bytes differ')
    if proof['probe_source_sha256'] != file_sha256(PROBE.with_name('probe.cpp')):
        raise ValueError('PDL probe source changed')
    executable = proof['probe_executable']
    if file_sha256(Path(executable['path'])) != executable['sha256']:
        raise ValueError('PDL probe executable changed')
    for arm in ('on', 'off'):
        artifacts = proof[arm]['artifacts']
        binary = next(Path(p) for p in artifacts if p.endswith('.bin'))
        stdout = next(Path(p) for p in artifacts if p.endswith('-stdout.log'))
        stderr = next(Path(p) for p in artifacts if p.endswith('-stderr.log'))
        if (file_sha256(binary) != proof[arm]['output_sha256'] or
                json.loads(stdout.read_text()) != proof[arm]['counts'] or
                'CUPTI unsubscribe: 0 (CUPTI_SUCCESS)' not in stderr.read_text()):
            raise ValueError('PDL proof claims differ from native artifacts')
        launch = json.loads(Path(proof[arm]['launch_path']).read_text())
        if (launch['exit_code'] != 0 or launch['command'] != [executable['path'], str(binary)] or
                launch['environment']['GGML_CUDA_PDL'] != ('1' if arm == 'on' else '0')):
            raise ValueError('PDL probe launch/exit mismatch')
        libraries = launch['loaded_libraries']
        expected = {Path(dep.path).name.lower(): dep for dep in inputs.stock.dependencies
                    if Path(dep.path).name.lower() in ('ggml-base.dll', 'ggml-cuda.dll')}
        expected[Path(inputs.stock.cuda_runtime.path).name.lower()] = inputs.stock.cuda_runtime
        loaded = {name.lower(): item for name, item in libraries.items()}
        for name, identity in expected.items():
            if (name not in loaded or Path(loaded[name]['path']).resolve() != Path(identity.path).resolve() or
                    loaded[name]['sha256'] != identity.sha256):
                raise ValueError('PDL probe loaded inference DLL mismatch')
        if not any(name.startswith('cupti') for name in loaded):
            raise ValueError('PDL probe missing loaded CUPTI')
        for name, item in libraries.items():
            if file_sha256(Path(item['path'])) != item['sha256']:
                raise ValueError('PDL probe loaded library changed')
            if f'MODULE\t{name}\t{item["path"]}' not in stderr.read_text():
                raise ValueError('PDL probe module claims differ from native capture')
    inputs.stock.verify_manifest_bindings()
    if proof['runtime_sha256'] != inputs.stock.sha256:
        raise ValueError('PDL proof runtime mismatch')
    return {'base_scope': scope, 'probe_sha256': proof['proof_sha256'],
            'allowed_control': 'cuda_pdl', 'arithmetic': 'same compiled kernel and arguments'}


def audit(report, store, *, host_environment, _collecting=False):
    frozen = report['frozen']
    _checksum(frozen, 'manifest_sha256')
    if frozen['protocol'] != 'cuda-pdl-paired-v1' or frozen['source_files'] != source_files():
        raise ValueError('PDL frozen source/protocol mismatch')
    if frozen['host_environment'] != canonical_payload(host_environment):
        raise ValueError('PDL frozen host mismatch')
    if frozen['schedule'] != canonical_payload(balanced_schedule()):
        raise ValueError('PDL schedule mismatch')
    if file_sha256(Path(frozen['source_database'])) != frozen['source_database_sha256']:
        raise ValueError('PDL prerequisite database changed')
    candidates = {arm: _candidate(payload) for arm, payload in frozen['candidates'].items()}
    baseline = candidates['direct']
    from .evidence import EvidenceStore
    source = EvidenceStore(Path(frozen['source_database']))
    inputs = _snapshot_inputs(frozen['inputs'], baseline.identities.workload)
    accepted = Path(frozen['accepted_directory'])
    plan = load_validated_stock_plan(accepted / 'execution-plan.json', accepted / 'acceptance-receipt.json',
        source, identities=inputs.identities(inputs.stock), host_environment=host_environment)
    if (plan.plan_sha256 != frozen['accepted_plan_sha256'] or
            baseline != CandidatePlan(plan.candidate.identities, plan.candidate.settings) or
            frozen['control_proof'] != qualify_control(inputs, host_environment, frozen['source_repository'])):
        raise ValueError('PDL prerequisite/control proof mismatch')
    reference = source.verify_measurement(plan.candidate.measurement_ids[0])
    expected_tokens = {k: reference[k] for k in ('generated_tokens_sha256', 'prompt_tokens_sha256')}
    expected_owners = [source.verify_measurement(mid)['owned_run_sha256'] for mid in plan.candidate.measurement_ids]
    if frozen['reference_tokens'] != expected_tokens or frozen['source_owners'] != expected_owners:
        raise ValueError('PDL prerequisite token/owner mismatch')
    if (baseline.settings.cuda_pdl is not None or baseline.settings.static is not None or
            baseline.settings.gpu_layers != 99 or not baseline.settings.cpu_moe or
            candidates['sealed'] != replace(baseline, settings=replace(baseline.settings, cuda_pdl='off'))):
        raise ValueError('PDL requires only the qualified launch switch')
    rows, outcomes = report['rows'], report['outcomes']
    if len(rows) != 20 or len(outcomes) != 20:
        raise ValueError('PDL requires twenty complete fresh runs')
    owners, mids, rates = set(frozen['source_owners']), set(), {'direct': [], 'sealed': []}
    for index, row in enumerate(rows):
        pair, order = divmod(index, 2)
        arm = balanced_schedule()[pair][order]
        mid = row['measurement_id']
        verified = store.verify_measurement(mid)
        record = store.measurement(mid)
        paths = {a.role: Path(a.identity.path) for a in record.artifacts}
        root = Path(frozen['experiment_root']) / 'raw' / f'pair-{pair:02}-{arm}'
        if (row.get('pair') != pair or row.get('arm') != arm or mid in mids or
                verified['owned_run_sha256'] in owners or
                record.stage != f'pdl-{pair:02}-{arm}' or record.numerical_path != 'stock_same_runtime'):
            raise ValueError('PDL run ownership/order/path mismatch')
        if any(p.resolve().parent != root.resolve() for p in paths.values()):
            raise ValueError('PDL artifacts outside frozen root')
        launch = json.loads(paths['launch'].read_text())
        started = json.loads(paths['run-start'].read_text())
        if (launch.get('host_environment') != frozen['host_environment'] or
                launch.get('experiment_context') != {'manifest_sha256': frozen['manifest_sha256']} or
                started['started_monotonic_ns'] < frozen['frozen_monotonic_ns']):
            raise ValueError('PDL native freeze binding mismatch')
        candidate = candidates[arm]
        if (verified['candidate_id'] != candidate.candidate_id or
                verified['settings_sha256'] != canonical_sha256(candidate.settings) or
                verified['identities'] != canonical_payload(candidate.identities)):
            raise ValueError('PDL native candidate mismatch')
        if (verified['measured'] is not True or verified['exit_code'] != 0 or any(
                verified['validations'].get(k) is not True for k in ('exact_tokens', 'memory', 'cleanup'))):
            raise ValueError('PDL correctness gate failed')
        if any(verified[k] != frozen['reference_tokens'][k] for k in frozen['reference_tokens']):
            raise ValueError('PDL tokens differ from own model reference')
        if any(canonical_payload(row.get(k)) != canonical_payload(v) for k, v in verified.items()):
            raise ValueError('PDL claimed row differs from native evidence')
        if outcomes[index].get('status') != 'measured' or outcomes[index].get('measurement_id') != mid:
            raise ValueError('PDL outcome mismatch')
        owners.add(verified['owned_run_sha256']); mids.add(mid)
        rates[arm].append(verified['decode_tps'])
    stats = paired_statistics(rates['direct'], rates['sealed'])
    # Arms arrive in balanced order; each list still has pair0..9 order.
    stats['gain_pass'] = (stats['geometric_change_pct'] >= 2 and stats['ci95_pct'][0] > 0 and
                         max(stats['direct_cv_pct'], stats['sealed_cv_pct']) <= 10)
    if report.get('statistics') is not None and report['statistics'] != canonical_payload(stats):
        raise ValueError('PDL claimed statistics mismatch')
    expected_status = 'PASS-DIAGNOSTIC' if stats['gain_pass'] else 'NO-GO'
    allowed = ('RUNNING', expected_status) if _collecting else (expected_status,)
    if (report['status'] not in allowed or report['product_accepted'] is not False or
            (not _collecting and report.get('statistics') is None)):
        raise ValueError('PDL diagnostic verdict mismatch')
    return stats


def execute(inputs, accepted, source, target, runner, output, *, source_repository,
            host_capture=capture_host_environment):
    output, accepted = Path(output), Path(accepted)
    host = host_capture()
    plan = load_validated_stock_plan(accepted / 'execution-plan.json', accepted / 'acceptance-receipt.json',
        source, identities=inputs.identities(inputs.stock), host_environment=host)
    proof = qualify_control(inputs, host, source_repository)
    base = CandidatePlan(plan.candidate.identities, plan.candidate.settings)
    candidates = {'direct': base, 'sealed': replace(base, settings=replace(base.settings, cuda_pdl='off'))}
    reference = source.verify_measurement(plan.candidate.measurement_ids[0])
    frozen = canonical_payload({'protocol': 'cuda-pdl-paired-v1', 'source_files': source_files(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'accepted_plan_sha256': plan.plan_sha256, 'control_proof': proof, 'source_database': str(source.path.resolve()),
        'accepted_directory': str(accepted.resolve()), 'source_repository': str(Path(source_repository).resolve()),
        'inputs': {'model': inputs.model, 'hardware': inputs.hardware, 'stock': inputs.stock},
        'source_database_sha256': file_sha256(source.path), 'host_environment': host,
        'candidates': candidates, 'reference_tokens': {k: reference[k] for k in
            ('generated_tokens_sha256', 'prompt_tokens_sha256')},
        'source_owners': [source.verify_measurement(mid)['owned_run_sha256'] for mid in plan.candidate.measurement_ids],
        'frozen_monotonic_ns': time.monotonic_ns(), 'experiment_root': str(output.resolve()),
        'schedule': balanced_schedule()})
    frozen['manifest_sha256'] = canonical_sha256(frozen)
    output.mkdir(parents=True, exist_ok=False)
    target.prime_model(inputs.model)
    atomic_json(output / 'frozen-protocol.json', frozen)
    report = {'status': 'RUNNING', 'frozen': frozen, 'rows': [], 'outcomes': [], 'product_accepted': False}
    atomic_json(output / 'report.json', report)
    try:
        for pair, order in enumerate(balanced_schedule()):
            for arm in order:
                if source_files() != frozen['source_files']:
                    raise ValueError('PDL source changed during collection')
                outcome = runner.run_once(candidates[arm], inputs.model, inputs.stock,
                    output_dir=output / 'raw' / f'pair-{pair:02}-{arm}', measured=True,
                    stage=f'pdl-{pair:02}-{arm}', host_environment=host,
                    experiment_context={'manifest_sha256': frozen['manifest_sha256']})
                report['outcomes'].append(canonical_payload(outcome))
                if outcome.status != 'measured':
                    report.update(status='ENVIRONMENT-BLOCKED' if outcome.status == 'environment_blocked'
                                  else 'VALIDATION-STOP', reason=outcome.reason)
                    atomic_json(output / 'report.json', report)
                    return report
                row = target.verify_measurement(outcome.measurement_id)
                if any(row[k] != frozen['reference_tokens'][k] for k in frozen['reference_tokens']):
                    raise ValueError('PDL native token mismatch')
                report['rows'].append({**row, 'pair': pair, 'arm': arm, 'measurement_id': outcome.measurement_id})
                atomic_json(output / 'report.json', report)
        stats = audit(report, target, host_environment=host_capture(), _collecting=True)
        report.update(statistics=canonical_payload(stats), status='PASS-DIAGNOSTIC' if stats['gain_pass'] else 'NO-GO')
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        report.update(status='VALIDATION-STOP', reason=str(error))
    atomic_json(output / 'report.json', report)
    return report
