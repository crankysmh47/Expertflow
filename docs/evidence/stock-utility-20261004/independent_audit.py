"""Independent native-timing/statistics audit, outside the collector source map."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.plan import load_execution_plan
from expertflow.compiler.refinement import balanced_schedule, paired_source_files
from expertflow.compiler.schema import canonical_payload, canonical_sha256


def paired(control, selected):
    logs = [math.log(b)-math.log(a) for a,b in zip(control,selected)]
    generator = random.Random(20261003)
    draws = []
    for _ in range(10000):
        draw = generator.choices(logs,k=10)
        draws.append(100*math.expm1(sum(draw)/10))
    draws.sort()
    def quantile(probability):
        # Registered empirical percentile: inverse ECDF (nearest rank).
        return draws[max(0,math.ceil(len(draws)*probability)-1)]
    return {
        'geometric_change_pct':100*math.expm1(sum(logs)/10),
        'ci95_pct':[quantile(.025),quantile(.975)],
        'ci90_pct':[quantile(.05),quantile(.95)],
        'control_mean_tps':statistics.mean(control),'selected_mean_tps':statistics.mean(selected),
        'control_cv_pct':statistics.stdev(control)*100/statistics.mean(control),
        'selected_cv_pct':statistics.stdev(selected)*100/statistics.mean(selected)}


def audit(report_path, database):
    report = json.loads(report_path.read_text())
    store = EvidenceStore(database)
    rows = report['rows']
    owners = set()
    reference = None
    rates = {}
    phase_seconds = {}
    peaks = []
    minimum_free = []
    for row in rows:
        record = store.measurement(row['measurement_id'])
        verified = store.verify_measurement(row['measurement_id'])
        artifacts = {a.role:Path(a.identity.path) for a in record.artifacts}
        raw = {role:json.loads(path.read_text()) for role,path in artifacts.items()}
        tokens = (raw['tokenize']['tokens'],raw['completion']['tokens'])
        assert reference is None or reference == tokens, 'native token divergence'
        reference = reference or tokens
        process = raw['run-start']
        owner = (process['pid'],process['creation_time_100ns'],process['run_id'])
        assert owner not in owners, 'reused native owner'
        owners.add(owner)
        timings = raw['completion']['timings']
        rate = timings['predicted_n']*1000/timings['predicted_ms']
        assert math.isclose(rate,row['decode_tps'],abs_tol=1e-12), 'rate mismatch'
        assert all(verified['validations'][name] for name in ('exact_tokens','memory','cleanup'))
        phases = raw['phase-timing']
        wall = raw['completion-wall']['elapsed_ms']
        native_seconds = (wall+sum(phases.get(name) or 0 for name in ('load_health_ms','tokenize_ms','teardown_ms')))/1000
        assert math.isclose(native_seconds,row['run_wall_seconds'],abs_tol=1e-12), 'phase cost mismatch'
        label = row['label']
        rates[label] = rate
        phase_seconds[label] = native_seconds
        peaks.extend(sample['dedicated_bytes'] for sample in raw['memory']['samples'])
        minimum_free.extend(sample['device_free_bytes'] for sample in raw['memory']['samples'])
    comparisons = {}
    if len(rows) == 86:
        for purpose in ('defaults','manual'):
            control = [rates[f'{purpose}-pair-{n:02}-direct'] for n in range(10)]
            selected = [rates[f'{purpose}-pair-{n:02}-sealed'] for n in range(10)]
            comparisons[purpose] = paired(control,selected)
        claimed = report['statistics']
        for purpose,key in (('defaults','gain'),('manual','manual_equivalence')):
            for metric in ('geometric_change_pct','ci95_pct','ci90_pct'):
                actual,expected = comparisons[purpose][metric],claimed[key][metric]
                if isinstance(actual,list):
                    assert all(math.isclose(a,b,abs_tol=1e-10) for a,b in zip(actual,expected))
                else: assert math.isclose(actual,expected,abs_tol=1e-10)
        gain,equivalence = comparisons['defaults'],comparisons['manual']
        stable = max(gain['control_cv_pct'],gain['selected_cv_pct'],equivalence['control_cv_pct'],equivalence['selected_cv_pct']) <= 10
        gain_pass = gain['geometric_change_pct'] >= 5 and gain['ci95_pct'][0] > 0
        manual_pass = equivalence['ci90_pct'][0] > -2 and equivalence['ci90_pct'][1] < 2
        verdict = 'VARIANCE-STOP' if not stable else 'NO-UTILITY-GAIN' if not gain_pass else 'MANUAL-BASELINE-STOP' if not manual_pass else 'PASS-STOCK-UTILITY'
        assert verdict == claimed['status']
    else: verdict = 'INCOMPLETE-NATIVE-PROOF'
    tuning = {}
    for method in ('automatic','manual'):
        labels = [label for label in rates if label.startswith(method+'-screen-')]
        tuning[method] = {'native_evaluations':len(labels),
            'native_phase_seconds':sum(phase_seconds[label] for label in labels)}
    root = Path(report['manifest']['experiment_root'])
    product_result = None
    product_path = root/'product/report.json'
    if product_path.is_file():
        product = json.loads(product_path.read_text())
        freeze = product['frozen']
        product_store = EvidenceStore(root/'product.sqlite3')
        selected = load_execution_plan(root/'selected-plan.json',store=store)
        assert freeze['source_plan_sha256'] == selected.plan_sha256
        assert freeze['source_plan'] == canonical_payload(selected)
        assert freeze['source_files'] == paired_source_files(product=True)
        assert freeze['host_environment'] == report['manifest']['host_environment']
        assert freeze['frozen_monotonic_ns'] >= report['manifest']['frozen_monotonic_ns']
        assert len(product['rows']) == len(product['outcomes']) == 20
        assert canonical_payload(balanced_schedule()) == freeze['schedule']
        product_rates = {}
        for index,row in enumerate(product['rows']):
            pair,position = divmod(index,2)
            arm = balanced_schedule()[pair][position]
            mid = row['measurement_id']
            record = product_store.measurement(mid)
            verified = product_store.verify_measurement(mid)
            artifacts = {a.role:Path(a.identity.path) for a in record.artifacts}
            raw = {role:json.loads(path.read_text()) for role,path in artifacts.items()}
            assert (raw['tokenize']['tokens'],raw['completion']['tokens']) == reference
            start = raw['run-start']
            owner = (start['pid'],start['creation_time_100ns'],start['run_id'])
            assert owner not in owners, 'reused utility/product process'
            owners.add(owner)
            expected = root/'product/raw'/f'pair-{pair:02}-{arm}'
            assert all(path.parent.resolve() == expected.resolve() for path in artifacts.values())
            assert start['started_monotonic_ns'] >= freeze['frozen_monotonic_ns']
            assert raw['launch']['experiment_context'] == {'manifest_sha256':report['manifest']['manifest_sha256']}
            assert raw['launch']['host_environment'] == freeze['host_environment']
            assert record.stage == f"product-{freeze['experiment_id']}-{pair:02}-{arm}"
            assert verified['candidate_id'] == selected.candidate.candidate_id
            assert verified['identities'] == canonical_payload(selected.candidate.identities)
            assert verified['settings_sha256'] == canonical_sha256(selected.candidate.settings)
            assert verified['measured'] is True and verified['exit_code'] == 0
            assert all(verified['validations'][name] for name in ('exact_tokens','memory','cleanup'))
            assert row['pair'] == pair and row['arm'] == arm
            assert all(canonical_payload(row.get(key)) == canonical_payload(value) for key,value in verified.items())
            assert product['outcomes'][index]['measurement_id'] == mid
            timing = raw['completion']['timings']
            product_rates[pair,arm] = timing['predicted_n']*1000/timing['predicted_ms']
            peaks.extend(sample['dedicated_bytes'] for sample in raw['memory']['samples'])
            minimum_free.extend(sample['device_free_bytes'] for sample in raw['memory']['samples'])
        product_result = paired([product_rates[n,'direct'] for n in range(10)],
                               [product_rates[n,'sealed'] for n in range(10)])
        for metric in ('geometric_change_pct','ci95_pct','ci90_pct'):
            actual,claimed = product_result[metric],product[metric]
            if isinstance(actual,list):assert all(math.isclose(a,b,abs_tol=1e-10) for a,b in zip(actual,claimed))
            else:assert math.isclose(actual,claimed,abs_tol=1e-10)
        product_result['equivalent'] = product_result['ci90_pct'][0] > -2 and product_result['ci90_pct'][1] < 2
        product_result['variance_pass'] = max(product_result['control_cv_pct'],product_result['selected_cv_pct']) <= 10
        product_result['reported_status'] = product['status']
        if product['status'] == 'INCONCLUSIVE':
            assert product_result['equivalent'] is False
            assert report['status'] == 'PRODUCT-VALIDATION-STOP'
            assert not (root/'product/accepted').exists() and not (root/'consumer').exists()
    native_starts = sorted(root.rglob('run-start.json'))
    assert len(native_starts) <= report['manifest']['maximum_native_processes']
    assert len(native_starts) == len(owners) == len(report['attempts'])
    assert all(attempt['native_started'] is True and attempt['status']=='measured' for attempt in report['attempts'])
    retained = {(a['process_identity']['pid'],a['process_identity']['creation_time_100ns'],a['process_identity']['run_id']) for a in report['attempts']}
    assert retained == owners
    output = {'status':'AUDITED','utility_verdict':verdict,'terminal_status':report['status'],
        'complete_utility_records':len(rows),'actual_native_processes':len(native_starts),
        'comparisons':comparisons,'product_comparison':product_result,'tuning_cost':tuning,
        'max_owned_memory_mib':max(peaks,default=0)/2**20,
        'min_device_free_mib':min(minimum_free,default=0)/2**20,
        'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'database_sha256':hashlib.sha256(database.read_bytes()).hexdigest(),
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'audit_scope':'all complete utility/product native records, independent raw timing/statistics/cost, original source/host/context, retained owners and terminal product gate; no consumer after product stop'}
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.report,args.database)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'utility_verdict':result['utility_verdict'],
        'native_processes':result['actual_native_processes']}))
