"""Independent native-timing/statistics audit, outside the collector source map."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from expertflow.compiler.evidence import EvidenceStore


def paired(control, selected):
    logs = [math.log(b)-math.log(a) for a,b in zip(control,selected)]
    generator = random.Random(20261003)
    draws = []
    for _ in range(10000):
        draw = generator.choices(logs,k=10)
        draws.append(100*math.expm1(sum(draw)/10))
    draws.sort()
    def quantile(probability):
        index = (len(draws)-1)*probability
        lower = int(index)
        return draws[lower]+(draws[min(lower+1,len(draws)-1)]-draws[lower])*(index-lower)
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
    native_starts = sorted(root.rglob('run-start.json'))
    assert len(native_starts) <= report['manifest']['maximum_native_processes']
    output = {'status':'AUDITED','utility_verdict':verdict,'terminal_status':report['status'],
        'complete_utility_records':len(rows),'actual_native_processes':len(native_starts),
        'comparisons':comparisons,'tuning_cost':tuning,
        'max_owned_memory_mib':max(peaks,default=0)/2**20,
        'min_device_free_mib':min(minimum_free,default=0)/2**20,
        'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'database_sha256':hashlib.sha256(database.read_bytes()).hexdigest(),
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'audit_scope':'all complete utility records and independently reconstructed statistics/cost; product/consumer checked separately by validate action'}
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
