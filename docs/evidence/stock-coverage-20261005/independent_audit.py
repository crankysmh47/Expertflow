"""Independent raw arithmetic/order audit; does not call the wider validator."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics

from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.preflight import file_sha256
from expertflow.compiler.schema import canonical_payload,canonical_sha256


def paired(direct,selected):
    if len(direct)!=10 or len(selected)!=10 or any(x<=0 or not math.isfinite(x) for x in (*direct,*selected)):
        raise ValueError('ten positive independent paired rates required')
    logs=[math.log(b)-math.log(a) for a,b in zip(direct,selected)]
    rng=random.Random(20261003)
    draws=sorted(100*math.expm1(statistics.mean(rng.choices(logs,k=10))) for _ in range(10000))
    percentile=lambda q:draws[max(0,math.ceil(len(draws)*q)-1)]
    return {'geometric_change_pct':100*math.expm1(statistics.mean(logs)),
        'ci95_pct':[percentile(.025),percentile(.975)],'ci90_pct':[percentile(.05),percentile(.95)],
        'one_sided95_lower_pct':percentile(.05),'direct_cv_pct':statistics.stdev(direct)*100/statistics.mean(direct),
        'sealed_cv_pct':statistics.stdev(selected)*100/statistics.mean(selected)}


def audit(path):
    outer=json.loads(Path(path).read_text())
    sequence=outer['manifest']
    payload=dict(sequence)
    claimed=payload.pop('manifest_sha256')
    if canonical_sha256(payload)!=claimed:
        raise ValueError('outer freeze checksum mismatch')
    for filename,digest in sequence['source_files'].items():
        if file_sha256(Path(filename))!=digest:
            raise ValueError('audited source changed: '+filename)
    stores={}
    owners=set()
    results=[]
    total=0
    for case in sequence['registration']['cases']:
        root=Path(case['planned_root'])
        if not (root/'report.json').is_file():
            results.append({'case_id':case['case_id'],'status':'NOT-RUN'})
            continue
        report=json.loads((root/'report.json').read_text())
        manifest=report['manifest']
        payload=dict(manifest)
        claimed=payload.pop('manifest_sha256')
        if canonical_sha256(payload)!=claimed or manifest['case']!=case or manifest['sequence_manifest_sha256']!=sequence['manifest_sha256']:
            raise ValueError('case freeze differs from registered sequence')
        native=[]
        for entry in report['attempts']:
            output=Path(entry['output_dir'])
            start=output/'run-start.json'
            if entry['native_started']!=start.is_file():
                raise ValueError('raw/native journal start mismatch')
            if not start.is_file():
                if entry.get('measurement_id'):
                    raise ValueError('measurement without native start')
                continue
            identity=json.loads(start.read_text())
            owner=(identity['pid'],identity['creation_time_100ns'],identity['creation_source'])
            if owner in owners or identity!=entry['process_identity'] or entry['wait_elapsed_ns']<30_000_000_000:
                raise ValueError('reused owner or incomplete pacing')
            owners.add(owner)
            if not entry['wait_finished_monotonic_ns']<=identity['started_monotonic_ns']<=entry['finished_monotonic_ns']:
                raise ValueError('raw native/wait order mismatch')
            total+=1
            if not entry.get('measurement_id'):
                process=output/'process.json'
                if not process.is_file() or json.loads(process.read_text()).get('cleanup') is not True:
                    raise ValueError('failed start lacks raw owned cleanup')
                continue
            db=Path(entry['database']).resolve()
            if db not in stores:stores[db]=EvidenceStore(db)
            store=stores[db]
            verified=store.verify_measurement(entry['measurement_id'])
            record=store.measurement(entry['measurement_id'])
            artifacts={a.role:Path(a.identity.path) for a in record.artifacts}
            if any(p.parent.resolve()!=output.resolve() for p in artifacts.values()):
                raise ValueError('raw artifact escaped recorded owner directory')
            launch=json.loads(artifacts['launch'].read_text())
            if launch.get('experiment_context')!={'manifest_sha256':manifest['manifest_sha256']} or launch.get('host_environment')!=sequence['host_environment']:
                raise ValueError('raw launch lost outer freeze')
            completion=json.loads(artifacts['completion'].read_text())
            rate=completion['timings']['predicted_n']*1000/completion['timings']['predicted_ms']
            if (rate!=verified['decode_tps'] or record.stage!=entry['stage'] or verified['measured'] is not True or
                    verified['exit_code']!=0 or any(verified['validations'].get(k) is not True for k in ('exact_tokens','memory','cleanup'))):
                raise ValueError('raw completion differs from database')
            if native and any(verified[k]!=native[0][1][k] for k in ('prompt_tokens_sha256','generated_tokens_sha256')):
                raise ValueError('raw tokens differ from own case reference')
            native.append((entry,verified))
        if len(list(root.rglob('run-start.json')))!=sum(e['native_started'] for e in report['attempts']):
            raise ValueError('hidden native start')
        utility=[(e,n) for e,n in native if Path(e['database']).name=='utility.sqlite3']
        result={'case_id':case['case_id'],'status':report['status'],'attempts':len(report['attempts']),
            'native_processes':sum(e['native_started'] for e in report['attempts']),'complete_utility':len(utility)==86}
        if len(utility)==86:
            schedule=case['screening_schedule']
            ids=case['candidate_ids']
            default=case['default_id']
            labels=[(f'reference-{n:02}',default) for n in range(10)]
            labels.extend((f'{method}-screen-{block:02}-{cid}',cid)
                for method in ('automatic','manual') for block,order in enumerate(schedule) for cid in order)
            automatic={(i//6,n['candidate_id']):n['decode_tps'] for i,(e,n) in enumerate(utility[10:28])}
            auto=min(ids,key=lambda cid:(-math.exp(statistics.mean(math.log(automatic[b,cid])-math.log(automatic[b,default]) for b in range(3))),cid!=default,cid))
            manual=min(ids,key=lambda cid:(-statistics.mean(n['decode_tps'] for e,n in utility[28:46] if n['candidate_id']==cid),cid!=default,cid))
            pair_order=sequence['registration']['paired_schedule']
            labels.extend((f'{purpose}-pair-{pair:02}-{arm}',control if arm=='direct' else auto)
                for purpose,control in (('defaults',default),('manual',manual))
                for pair,order in enumerate(pair_order) for arm in order)
            for (entry,native_row),(label,cid) in zip(utility,labels):
                if Path(entry['output_dir']).name!=label or native_row['candidate_id']!=cid:
                    raise ValueError('raw utility order/control mismatch')
            comparisons={}
            for purpose,start in (('gain',46),('manual_equivalence',66)):
                values={(i//2,pair_order[i//2][i%2]):n['decode_tps'] for i,(e,n) in enumerate(utility[start:start+20])}
                comparisons[purpose]=paired([values[i,'direct'] for i in range(10)],[values[i,'sealed'] for i in range(10)])
            for name,rebuilt in comparisons.items():
                if any(canonical_payload(report['statistics'][name][k])!=canonical_payload(v) for k,v in rebuilt.items()):
                    raise ValueError('raw reconstructed interval differs from report')
            gain,equiv=comparisons['gain'],comparisons['manual_equivalence']
            stable=max(gain['direct_cv_pct'],gain['sealed_cv_pct'],equiv['direct_cv_pct'],equiv['sealed_cv_pct'])<=10
            status=('VARIANCE-STOP' if not stable else 'NO-UTILITY-GAIN' if not (gain['geometric_change_pct']>=5 and gain['ci95_pct'][0]>0)
                else 'MANUAL-BASELINE-STOP' if not (equiv['ci90_pct'][0]>-2 and equiv['ci90_pct'][1]<2) else 'PASS-STOCK-UTILITY')
            if status!=report['statistics']['status'] or report['automatic_id']!=auto or report['manual_id']!=manual:
                raise ValueError('raw utility selection/verdict differs')
            result.update(utility_verdict=status,automatic_id=auto,manual_id=manual,selected_default=auto==default,
                automatic_evaluations=18,manual_evaluations=18,**comparisons)
            product=[n for e,n in native if Path(e['output_dir']).is_relative_to(root/'product')]
            if product:
                if status!='PASS-STOCK-UTILITY' or any(n['candidate_id']!=auto for n in product):
                    raise ValueError('product followed failed utility or wrong selection')
                if len(product)==20:
                    values={(i//2,pair_order[i//2][i%2]):n['decode_tps'] for i,n in enumerate(product)}
                    stats=paired([values[i,'direct'] for i in range(10)],[values[i,'sealed'] for i in range(10)])
                    accepted=(stats['ci90_pct'][0]>-2 and stats['ci90_pct'][1]<2 and stats['one_sided95_lower_pct']>-2 and
                        max(stats['direct_cv_pct'],stats['sealed_cv_pct'])<=10)
                    result.update(product_statistics=stats,product_gate_pass=accepted)
                    if report['status']=='PASS-STOCK-UTILITY-PRODUCT' and not accepted:
                        raise ValueError('failed raw product gate promoted')
        results.append(result)
    if len(owners)!=outer['native_processes'] or total>428:
        raise ValueError('sequence raw owner/count budget mismatch')
    return {'status':'RAW-AUDIT-PASS','native_processes':total,'cases':results,
        'additional_native_calls':0,'scope':'independent raw rates/order/owners/bootstrap arithmetic; public validator separately checks acceptance/source receipts'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.report),indent=2,sort_keys=True))
