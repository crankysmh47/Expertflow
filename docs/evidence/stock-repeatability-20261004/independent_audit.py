"""Read-only raw timing, ownership, pacing, diagnostic and cost reconstruction."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from expertflow.compiler.evidence import EvidenceStore


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(direct, sealed):
    logs = [math.log(b)-math.log(a) for a,b in zip(direct,sealed)]
    assert len(logs)==10
    generator = random.Random(20261003)
    draws = sorted(100*math.expm1(sum(generator.choices(logs,k=10))/10) for _ in range(10000))
    def percentile(probability):
        return draws[max(0,math.ceil(probability*len(draws))-1)]
    result = {'geometric_change_pct':100*math.expm1(sum(logs)/10),
        'ci90_pct':[percentile(.05),percentile(.95)],'ci95_pct':[percentile(.025),percentile(.975)],
        'one_sided95_lower_pct':percentile(.05),
        'direct_mean_tps':statistics.mean(direct),'sealed_mean_tps':statistics.mean(sealed),
        'direct_cv_pct':100*statistics.stdev(direct)/statistics.mean(direct),
        'sealed_cv_pct':100*statistics.stdev(sealed)/statistics.mean(sealed)}
    result.update(noninferior=result['one_sided95_lower_pct']>-2,
        equivalent=result['ci90_pct'][0]>-2 and result['ci90_pct'][1]<2,
        variance_pass=max(result['direct_cv_pct'],result['sealed_cv_pct'])<=10)
    result['status'] = ('PASS-MEASUREMENT' if all(result[key] for key in ('noninferior','equivalent','variance_pass'))
        else 'VALIDATION-STOP' if result['ci90_pct'][1]<-2 else 'INCONCLUSIVE')
    return result


def audit(report_path):
    report = json.loads(report_path.read_text())
    root = Path(report['manifest']['experiment_root'])
    owners, tokens, rates, costs, sensors = set(), {}, {}, {}, {}
    peaks, free, waits = [], [], []
    previous = report['manifest']['frozen_monotonic_ns']
    measured = 0
    native_seconds = 0
    for attempt in report['attempts']:
        output = Path(attempt['output_dir'])
        assert output.is_relative_to(root)
        assert attempt['wait_started_monotonic_ns']>=previous
        previous = attempt['finished_monotonic_ns']
        if not attempt['native_started']:
            assert not (output/'run-start.json').exists()
            continue
        waits.append(attempt['wait_elapsed_ns']/1e9)
        assert waits[-1]>=30 and attempt['wait_elapsed_ns']==attempt['wait_finished_monotonic_ns']-attempt['wait_started_monotonic_ns']
        start = json.loads((output/'run-start.json').read_text())
        assert start==attempt['process_identity']
        assert attempt['wait_finished_monotonic_ns']<=start['started_monotonic_ns']<=attempt['finished_monotonic_ns']
        owner = (start['pid'],start['creation_time_100ns'],start['creation_source'])
        assert owner not in owners
        owners.add(owner)
        launch = json.loads((output/'launch.json').read_text())
        scope = 'transfer' if output.is_relative_to(root/'transfer') else 'main'
        manifest = (json.loads((root/'transfer/utility/frozen-manifest.json').read_text())
            if scope=='transfer' else report['manifest'])
        expected_context = {'manifest_sha256':manifest['manifest_sha256']} if scope=='transfer' else {'repeatability_manifest_sha256':manifest['manifest_sha256']}
        assert launch['experiment_context']==expected_context==attempt['experiment_context']
        assert launch['host_environment']==report['manifest']['host_environment']
        if attempt['status']!='measured':
            continue
        store = EvidenceStore(Path(attempt['database']))
        record = store.measurement(attempt['measurement_id'])
        verified = store.verify_measurement(attempt['measurement_id'])
        assert record.stage==attempt['stage'] and verified['measured'] is True and verified['exit_code']==0
        assert all(verified['validations'][key] for key in ('exact_tokens','memory','cleanup'))
        raw = {a.role:json.loads(Path(a.identity.path).read_text()) for a in record.artifacts}
        assert all(Path(a.identity.path).parent==output for a in record.artifacts)
        actual_tokens = (raw['tokenize']['tokens'],raw['completion']['tokens'])
        assert scope not in tokens or tokens[scope]==actual_tokens
        tokens[scope] = actual_tokens
        timing = raw['completion']['timings']
        rate = timing['predicted_n']*1000/timing['predicted_ms']
        assert math.isclose(rate,verified['decode_tps'],abs_tol=1e-12)
        rates[attempt['measurement_id']] = rate
        phases = raw.get('phase-timing',{})
        seconds = (raw['completion-wall']['elapsed_ms']+sum(phases.get(key) or 0 for key in ('load_health_ms','tokenize_ms','teardown_ms')))/1000
        native_seconds += seconds
        phase = output.relative_to(root).parts[0]
        costs[phase] = costs.get(phase,0)+seconds
        for sample in raw['memory']['samples']:
            peaks.append(sample['dedicated_bytes'])
            free.append(sample['device_free_bytes'])
            if sample.get('phase')!='measurement':
                continue
            for group, payload in sample.get('diagnostics',{}).items():
                if not isinstance(payload,dict):
                    continue
                for key,value in payload.items():
                    if type(value) in (int,float) and math.isfinite(value):
                        sensors.setdefault(f'{group}.{key}',[]).append(value)
        measured += 1
    assert len(list(root.rglob('run-start.json')))==len(owners)
    blocks = {}
    for name in ('block-a','block-b'):
        path = root/name/'report.json'
        if not path.is_file():
            continue
        block = json.loads(path.read_text())
        if len(block['rows'])!=20:
            blocks[name] = {'status':block['status'],'complete_records':len(block['rows'])}
            continue
        values = {(row['pair'],row['arm']):rates[row['measurement_id']] for row in block['rows']}
        comparison = paired([values[pair,'direct'] for pair in range(10)], [values[pair,'sealed'] for pair in range(10)])
        for key,value in comparison.items():
            if isinstance(value,list):
                assert all(math.isclose(a,b,abs_tol=1e-10) for a,b in zip(value,block[key]))
            elif type(value) is float:
                assert math.isclose(value,block[key],abs_tol=1e-10)
            else:
                assert value==block[key]
        blocks[name] = comparison
    if report['status']=='REPEATABILITY-STOP':
        assert any(block['status']!='PASS-MEASUREMENT' for block in blocks.values())
        assert not (root/'block-b/accepted').exists() and not (root/'consumer').exists() and not (root/'transfer').exists()
    if report['status']=='PASS-STOCK-REPEATABILITY-TRANSFER':
        assert len(owners)==measured==len(report['attempts'])==148
        assert all(block['status']=='PASS-MEASUREMENT' for block in blocks.values()) and len(blocks)==2
    collector = report['collector_elapsed_ns']/1e9
    assert collector==(report['collector_finished_monotonic_ns']-report['collector_started_monotonic_ns'])/1e9
    assert report['collector_finished_monotonic_ns']>=previous
    assert collector>=sum(waits)
    return {'status':'AUDITED','terminal_status':report['status'],'actual_native_processes':len(owners),
        'complete_measured_records':measured,'blocks':blocks,'cost':{'collector_seconds':collector,
            'prelaunch_wait_seconds':sum(waits),'minimum_wait_seconds':min(waits,default=None),
            'native_phase_seconds':native_seconds,'native_phase_seconds_by_phase':costs},
        'diagnostics':{name:{'count':len(values),'min':min(values),'mean':statistics.mean(values),'max':max(values)} for name,values in sorted(sensors.items())},
        'cpu_temperature':'unavailable; no trusted sensor','max_owned_memory_mib':max(peaks,default=0)/2**20,
        'min_device_free_mib':min(free,default=0)/2**20,'report_sha256':sha(report_path),
        'auditor_sha256':sha(__file__),'scope':'independent raw native TPS, paired bootstrap gates, kernel owners, tokens, source context, waits, diagnostics and separate costs; production validator separately reconstructs complete phase/state/source/receipt bindings'}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.report)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':result['status'],'native_processes':result['actual_native_processes'],'terminal_status':result['terminal_status']}))
