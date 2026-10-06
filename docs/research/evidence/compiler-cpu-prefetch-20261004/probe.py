"""One frozen kernel-only experiment; no retries or compiler-plan publication."""
import json
from pathlib import Path
import subprocess
import time

from expertflow.compiler.diagnostics import DiagnosticSampler
from expertflow.compiler.evidence import EvidenceStore, MeasurementKey
from expertflow.compiler.pipeline import CompilationRequest, atomic_json, load_compiler_inputs
from expertflow.compiler.plan import CandidatePlan, RuntimeSettings
from expertflow.compiler.preflight import file_sha256
from expertflow.compiler.refinement import balanced_schedule, diagnostic_summary, evaluate_pairs, paired_statistics
from expertflow.compiler.runner import ServerMeasurementRunner, WindowsGpuMemorySampler
from expertflow.compiler.schema import canonical_payload

ROOT = Path('C:/models/expertflow/runs/compiler-cpu-prefetch-20261004')
SOURCE = Path('C:/models/expertflow/worktrees/llama-cpu-prefetch-20261004')
AA_REPORT = Path('C:/models/expertflow/runs/compiler-refinement-20261003/aa/report.json')
AA_DB = Path('C:/models/expertflow/runs/compiler-refinement-20261003/compiler.sqlite3')


def source_identity():
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True)
    if dirty:
        raise ValueError('candidate native source is dirty')
    return commit


def main():
    if (ROOT/'compiler.sqlite3').exists() or (ROOT/'report.json').exists():
        raise ValueError('fresh database/report required; no restarting retained experiment')
    request = CompilationRequest(Path('configs/compiler/gemma4-q6-model.json'),
        Path('docs/evidence/q6-download/tensor-inventory.json'),
        Path('docs/evidence/compiler-phase3/inputs/hardware.json'),
        Path('configs/compiler/gemma4-q6-single-request.json'), ROOT/'runtime-identity.json',
        (), ROOT/'compiler.sqlite3', ROOT/'native')
    inputs = load_compiler_inputs(request, live=True)
    aa = json.loads(AA_REPORT.read_text())
    original_hashes = {'aa_report': file_sha256(AA_REPORT), 'aa_database': file_sha256(AA_DB)}
    prior = EvidenceStore(AA_DB)
    verified_aa = []
    for index, row in enumerate(aa['rows']):
        pair, order = divmod(index, 2)
        expected_arm = balanced_schedule()[pair][order]
        verified = prior.verify_measurement(row['measurement_id'])
        if row['pair'] != pair or row['arm'] != expected_arm or any(
                canonical_payload(row.get(k)) != canonical_payload(v) for k,v in verified.items()):
            raise ValueError('A/A report does not bind to verified schedule/artifacts')
        verified_aa.append({**verified, 'pair': pair, 'arm': expected_arm,
                            'measurement_id': row['measurement_id']})
    if aa['status'] != 'PASS-MEASUREMENT' or evaluate_pairs(verified_aa)['status'] != 'PASS-MEASUREMENT':
        raise ValueError('measurement prerequisite failed')
    w = inputs.workload
    if w.threads != 12:
        raise ValueError('requires frozen twelve-thread workload')
    settings = RuntimeSettings(99, True, w.cuda_graphs, w.kv_type_k, w.kv_type_v, w.batch_size, w.microbatch_size)
    candidates = {arm: CandidatePlan(inputs.identities(binding), settings)
                  for arm,binding in [('stock',inputs.stock),('prefetch',inputs.fork)]}
    baseline = verified_aa[0]
    if canonical_payload(candidates['stock'].identities) != baseline['identities']:
        raise ValueError('stock candidate differs from accepted A/A reference')
    numerical_path = Path('.superpowers/sdd/cpu-expert-prefetch-20261004/numerical-gate.json')
    numerical = json.loads(numerical_path.read_text())
    if numerical['status'] != 'PASS-NUMERICAL-GATE' or numerical['bitwise_outputs_equal'] is not True:
        raise ValueError('numerical prerequisite failed')
    native_commit = source_identity()
    if native_commit != json.loads(inputs.fork.manifest_json)['expertflow_commit']:
        raise ValueError('candidate source does not match runtime manifest')
    store = EvidenceStore(ROOT/'compiler.sqlite3')
    store.prime_model(inputs.model)
    reference_id = baseline['measurement_id']
    store.append_measurement(prior.measurement(reference_id), measurement_id=reference_id)
    sources = [*sorted(Path('src/expertflow/compiler').rglob('*.py')), Path(__file__), numerical_path,
        Path('tests/native/cpu_expert_prefetch_parity.cpp'),
        Path('docs/superpowers/specs/2026-10-04-cpu-expert-prefetch-experiment.md'),
        Path('docs/superpowers/plans/2026-10-04-cpu-expert-prefetch-experiment.md'),
        Path('configs/compiler/runtime-cpu-prefetch.json'), Path('configs/compiler/runtime-stock.json'),
        Path('docs/evidence/compiler-cpu-prefetch-20261004/0001-cpu-expert-row-prefetch.patch'), ROOT/'runtime-identity.json']
    freeze = {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'native_source_commit': native_commit, 'source_hashes': {str(p): file_sha256(p) for p in sources},
        'inputs': inputs.provenance, 'candidates': canonical_payload(candidates),
        'runtime_bindings': canonical_payload({'stock':inputs.stock,'prefetch':inputs.fork}),
        'reference_measurement_id': reference_id, 'imported_reference_records': 1,
        'prior_hashes': original_hashes, 'native_budget': 20, 'pairs': 10, 'seed': 20261003,
        'schedule': [['stock' if a == 'direct' else 'prefetch' for a in order] for order in balanced_schedule()],
        'bootstrap_samples': 10000, 'minimum_point_gain_pct': 5,
        'minimum_two_sided_ci95_lower_pct': 0, 'maximum_cv_pct': 10}
    atomic_json(ROOT/'frozen-protocol.json', freeze)
    report = {'status': 'RUNNING', 'frozen':freeze, 'rows':[], 'outcomes':[],
        'product_plan_published': False, 'optimization_gain_established': False, 'retry_count':0}
    atomic_json(ROOT/'report.json', report)
    sampler = None
    try:
        sampler = DiagnosticSampler(WindowsGpuMemorySampler(inputs.hardware.gpu_uuid))
        runner = ServerMeasurementRunner(store, memory_sampler=sampler)
        owners = {r['owned_run_sha256'] for r in verified_aa}
        for pair,order in enumerate(freeze['schedule']):
            for arm in order:
                if source_identity() != native_commit or any(file_sha256(Path(p)) != digest
                        for p,digest in freeze['source_hashes'].items()):
                    raise ValueError('frozen source/protocol changed during experiment')
                binding = inputs.stock if arm == 'stock' else inputs.fork
                candidate = candidates[arm]
                started = time.perf_counter()
                outcome = runner.run_once(candidate, inputs.model, binding,
                    output_dir=ROOT/'native'/f'pair-{pair:02}-{arm}', measured=True,
                    stage=f'prefetch-{pair:02}-{arm}',
                    numerical_path='stock_same_runtime' if arm == 'stock' else 'fork_off_vs_pristine',
                    comparison_ids=() if arm == 'stock' else (reference_id,))
                report['outcomes'].append(canonical_payload(outcome))
                if outcome.status != 'measured':
                    report.update(status=outcome.status.upper().replace('_','-'), reason=outcome.reason)
                    atomic_json(ROOT/'report.json', report)
                    return 2
                record = store.measurement(outcome.measurement_id)
                if record.key != MeasurementKey.from_candidate(candidate) or record.stage != f'prefetch-{pair:02}-{arm}':
                    raise ValueError('run does not bind to frozen candidate/arm')
                verified = store.verify_measurement(outcome.measurement_id)
                for name in ('prompt_tokens_sha256','generated_tokens_sha256'):
                    if verified[name] != baseline[name]:
                        raise ValueError('candidate altered reference native tokens')
                if verified['owned_run_sha256'] in owners:
                    raise ValueError('duplicate/prior native process reused')
                owners.add(verified['owned_run_sha256'])
                report['rows'].append({**verified, 'pair':pair, 'arm':arm,
                    'measurement_id':outcome.measurement_id, 'run_wall_seconds':time.perf_counter()-started,
                    'diagnostics':diagnostic_summary(record), 'artifacts':canonical_payload(record.artifacts)})
                atomic_json(ROOT/'report.json', report)
                print(json.dumps({'completed':len(report['rows']), 'pair':pair, 'arm':arm,
                                  'decode_tps':verified['decode_tps']}), flush=True)
        rates = {(r['pair'],r['arm']):r['decode_tps'] for r in report['rows']}
        result = paired_statistics([rates[i,'stock'] for i in range(10)],
                                   [rates[i,'prefetch'] for i in range(10)])
        passed = result['geometric_change_pct'] >= 5 and result['ci95_pct'][0] > 0 and max(
            result['direct_cv_pct'],result['sealed_cv_pct']) <= 10
        if file_sha256(AA_REPORT) != original_hashes['aa_report'] or file_sha256(AA_DB) != original_hashes['aa_database']:
            raise ValueError('read-only prerequisite changed')
        report.update(result, status='PASS-OPTIMIZATION' if passed else
            'VALIDATION-STOP' if result['ci95_pct'][1] < 0 else 'INCONCLUSIVE',
            statistics_arm_mapping={'direct':'stock','sealed':'prefetch'}, optimization_gain_established=passed)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report.update(status='VALIDATION-STOP', reason=str(error), optimization_gain_established=False)
    finally:
        if sampler:
            sampler.close()
        atomic_json(ROOT/'report.json', report)
    print(json.dumps({'status':report['status'], 'runs':len(report['rows']), 'reason':report.get('reason')}), flush=True)
    return 0 if report['status'] == 'PASS-OPTIMIZATION' else 2


if __name__ == '__main__':
    raise SystemExit(main())
