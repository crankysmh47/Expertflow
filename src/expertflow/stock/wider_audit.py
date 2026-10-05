"""Read-only reconstruction of wider native records, including stopped prefixes."""
from dataclasses import replace
import json
from pathlib import Path
import statistics
import math

from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.plan import load_execution_plan
from expertflow.compiler.refinement import balanced_schedule, evaluate_pairs
from expertflow.compiler.schema import canonical_payload, canonical_sha256
from expertflow.compiler.stock_search import rank_screening
from expertflow.compiler.stock_validation import load_validated_stock_plan
from scripts import benchmark_compiler_stock_utility as utility
from scripts import benchmark_compiler_stock_repeatability as repeatability


def audit_attempts(report):
    m=report['manifest']
    root=Path(m['experiment_root']).resolve()
    attempts=report['attempts']
    if len(attempts)>107:
        raise ValueError('wider attempt budget exceeded')
    outputs,owners=set(),set()
    previous=m['frozen_monotonic_ns']
    for entry in attempts:
        output,database=Path(entry['output_dir']).resolve(),Path(entry['database']).resolve()
        output.relative_to(root)
        if database not in (root/'utility.sqlite3',root/'product.sqlite3'):
            raise ValueError('attempt database outside frozen case')
        if output in outputs or entry['wait_started_monotonic_ns']<previous:
            raise ValueError('duplicate/out-of-order wider attempt')
        outputs.add(output)
        finished=entry['finished_monotonic_ns']
        if finished<entry['wait_started_monotonic_ns']:
            raise ValueError('wider attempt clock order mismatch')
        previous=finished
        started=output/'run-start.json'
        if entry['native_started'] is not started.is_file():
            raise ValueError('native start journal mismatch')
        if started.is_file():
            observed=json.loads(started.read_text())
            owner=(observed['pid'],observed['creation_time_100ns'],observed['creation_source'])
            if (entry['wait_seconds']!=30 or entry['wait_elapsed_ns']<30_000_000_000 or
                entry['wait_elapsed_ns']!=entry['wait_finished_monotonic_ns']-entry['wait_started_monotonic_ns'] or
                not entry['wait_finished_monotonic_ns']<=observed['started_monotonic_ns']<=finished or
                owner in owners or entry['process_identity']!=observed):
                raise ValueError('wider native owner/wait evidence mismatch')
            if ((observed['started_monotonic_ns']-m['case_started_monotonic_ns'])/1e9+m['input_load_seconds']>=14400 or
                    (observed['started_monotonic_ns']-m['sequence_started_monotonic_ns'])/1e9>=57600):
                raise ValueError('native launch after wall budget')
            owners.add(owner)
        launch_path=output/'launch.json'
        if launch_path.is_file():
            launch=json.loads(launch_path.read_text())
            if (launch.get('host_environment')!=m['host_environment'] or
                    launch.get('experiment_context')!={'manifest_sha256':m['manifest_sha256']}):
                raise ValueError('wider native outer freeze/context mismatch')
        if entry.get('measurement_id'):
            store=EvidenceStore(database)
            record=store.measurement(entry['measurement_id'])
            native=store.verify_measurement(entry['measurement_id'])
            artifacts={a.role:Path(a.identity.path) for a in record.artifacts}
            if (entry['status']!='measured' or not started.is_file() or record.stage!=entry['stage'] or
                    any(p.parent.resolve()!=output for p in artifacts.values()) or
                    native['measured'] is not True or native['exit_code']!=0 or
                    any(native['validations'].get(k) is not True for k in ('exact_tokens','memory','cleanup'))):
                raise ValueError('wider attempt native correctness mismatch')
        elif entry['status']=='measured':
            raise ValueError('measured attempt has no record')
        elif started.is_file():
            # A failed native call is retained but cannot be accepted as valid evidence.
            process_path=output/'process.json'
            if not process_path.is_file() or json.loads(process_path.read_text()).get('cleanup') is not True:
                raise ValueError('failed native call lacks owned cleanup evidence')
    if len(list(root.rglob('run-start.json')))!=len(owners):
        raise ValueError('unaccounted native starts in case')
    return owners


def reconstruct_case(report,inputs,sequence,*,host_environment):
    from . import wider
    m=report['manifest']
    payload=dict(m)
    claimed=payload.pop('manifest_sha256')
    root=Path(m['experiment_root']).resolve()
    if (claimed!=canonical_sha256(payload) or m['protocol_version']!=wider.PROTOCOL or
            m!=json.loads((root/'frozen-manifest.json').read_text()) or
            m['source_files']!=wider.sources() or m['source_files']!=sequence['source_files'] or
            m['sequence_manifest_sha256']!=sequence['manifest_sha256'] or
            m['source_commit']!=sequence['source_commit'] or
            m['host_environment']!=canonical_payload(host_environment) or
            m['host_environment']!=sequence['host_environment'] or
            m['sequence_started_monotonic_ns']!=sequence['sequence_started_monotonic_ns'] or
            m['maximum_native_processes']!=107 or m['reference_processes']!=10):
        raise ValueError('wider frozen manifest/source/host/budget mismatch')
    if (type(m['input_load_seconds']) not in (int,float) or not math.isfinite(m['input_load_seconds']) or
            m['input_load_seconds']<0 or not sequence['frozen_monotonic_ns']<=m['case_started_monotonic_ns']<=m['frozen_monotonic_ns']):
        raise ValueError('wider freeze/loading cost order mismatch')
    if (report['collection_finished_monotonic_ns']<m['case_started_monotonic_ns'] or
            report['collection_wall_seconds']!=(report['collection_finished_monotonic_ns']-m['case_started_monotonic_ns'])/1e9):
        raise ValueError('wider collection wall cost mismatch')
    matches=[c for c in sequence['registration']['cases'] if c['case_id']==m['case']['case_id']]
    if len(matches)!=1 or matches[0]!=m['case'] or root!=Path(m['case']['planned_root']).resolve():
        raise ValueError('case outside registered sequence roots')
    proof,default,candidates=wider.scope_case(inputs,m['case'],sequence,host_environment)
    if (m['eligibility']!=canonical_payload(proof) or
            m['inputs']!=canonical_payload({'model':inputs.model,'hardware':inputs.hardware,'stock':inputs.stock}) or
            m['source_repository']!=sequence['source_repository'] or
            m['default_source_proof']!=sequence['default_source_proof'] or
            m['default_id']!=default.candidate_id or m['default_controls']!=[8,'on'] or
            m['candidates']!={cid:canonical_payload(c) for cid,c in candidates.items()} or
            m['screening_schedule']!=m['case']['screening_schedule'] or
            m['confirmation_schedule']!=canonical_payload(balanced_schedule())):
        raise ValueError('wider input/eligibility/controls mismatch')
    journal_owners=audit_attempts(report)
    store=EvidenceStore(root/'utility.sqlite3')
    rows,outcomes,entries=report['rows'],report['outcomes'],report['attempts']
    if not 0<=len(rows)<=len(outcomes)<=min(86,len(entries)) or len(outcomes)>len(rows)+1:
        raise ValueError('wider utility prefix budget mismatch')
    if len(rows)>86 or len(entries)>len(rows)+1 and len(rows)<86:
        raise ValueError('wider partial grid contains extra attempts')
    owners=set()
    reference=None
    rebuilt=[]
    expected=[(f'reference-{n:02}',default.candidate_id) for n in range(10)]
    schedule=m['screening_schedule']
    expected.extend((f'{method}-screen-{block:02}-{cid}',cid)
        for method in ('automatic','manual') for block,order in enumerate(schedule) for cid in order)
    auto=manual=None
    # Native rows are checked before they are used to choose winners.
    def verify(index,label,cid):
        nonlocal reference
        entry=entries[index]
        if (Path(entry['output_dir']).resolve()!=root/'raw'/label or
                Path(entry['database']).resolve()!=store.path.resolve() or
                entry['stage']!=utility.measurement_stage(label,m)):
            raise ValueError('wider utility attempt order/path/stage mismatch')
        if index>=len(rows):
            if index<len(outcomes) and (outcomes[index]['status']!=entry['status'] or
                    outcomes[index].get('measurement_id')!=entry.get('measurement_id')):
                raise ValueError('wider stopped utility outcome mismatch')
            return
        row,outcome=rows[index],outcomes[index]
        if (row['label']!=label or row['candidate_id']!=cid or outcome['status']!='measured' or
                outcome['measurement_id']!=row['measurement_id'] or entry['status']!='measured' or
                entry.get('measurement_id')!=row['measurement_id']):
            raise ValueError('wider utility row/outcome/order mismatch')
        native=utility.verify_row(row,candidates[cid],store,m,owners,reference)
        reference=reference or native
        rebuilt.append(row)
    for index,(label,cid) in enumerate(expected):
        if index>=min(46,len(entries)):break
        verify(index,label,cid)
    reference_cv=None
    if len(rows)>=10:
        reference_cv=statistics.stdev(r['decode_tps'] for r in rebuilt[:10])*100/statistics.mean(r['decode_tps'] for r in rebuilt[:10])
    if reference_cv is not None and reference_cv>10 and len(entries)>10:
        raise ValueError('unstable reference was followed by extra calls')
    if len(rows)>=46:
        screens={}
        for method,start in (('automatic',10),('manual',28)):
            screens[method]=[dict(rebuilt[start+i],block=i//6) for i in range(18)]
        auto=rank_screening(screens['automatic'],default.candidate_id,schedule)[0]['candidate_id']
        manual=utility.manual_winner(screens['manual'],candidates,default.candidate_id)
        if (report.get('automatic_id'),report.get('manual_id'))!=(auto,manual):
            raise ValueError('wider selection is not reconstructed from own grid')
        pairs=[(f'{purpose}-pair-{pair:02}-{arm}',control if arm=='direct' else auto)
            for purpose,control in (('defaults',default.candidate_id),('manual',manual))
            for pair,order in enumerate(balanced_schedule()) for arm in order]
        for index,(label,cid) in enumerate(pairs,46):
            if index>=min(86,len(entries)):break
            verify(index,label,cid)
    elif report.get('automatic_id') is not None or report.get('manual_id') is not None:
        raise ValueError('partial wider grid claims a winner')
    stats=None
    complete=len(rows)==86
    if complete:
        values={}
        for purpose,start in (('defaults',46),('manual',66)):
            samples={}
            for index,row in enumerate(rebuilt[start:start+20]):
                pair,pos=divmod(index,2)
                samples[pair,balanced_schedule()[pair][pos]]=row['decode_tps']
            values[purpose]=([samples[i,'direct'] for i in range(10)],[samples[i,'sealed'] for i in range(10)])
        stats=utility.evaluate_utility(*values['defaults'],*values['manual'],automatic_evaluations=18,manual_evaluations=18)
        if report.get('statistics')!=canonical_payload(stats):
            raise ValueError('wider claimed statistics/costs mismatch')
    elif report.get('statistics') is not None:
        raise ValueError('partial wider case cannot claim intervals')
    status=report['status']
    if complete and stats['status']!='PASS-STOCK-UTILITY':
        if status!=stats['status'] or len(entries)!=86:
            raise ValueError('failed wider utility promoted/continued')
    elif not complete:
        permitted={'VALIDATION-STOP','ENVIRONMENT-BLOCKED','RESOURCE-BUDGET-STOP'}
        if reference_cv is not None and reference_cv>10:permitted.add('VARIANCE-STOP')
        if status not in permitted:
            raise ValueError('incomplete wider proof claims statistical success')
    else:
        verify_product(report,store,candidates[auto],sequence,host_environment,entries[86:])
    if status=='RESOURCE-BUDGET-STOP' and not report.get('reason'):
        raise ValueError('resource stop lacks reason')
    native_times=[utility.native_phase_seconds(EvidenceStore(e['database']),e['measurement_id'])
                  for e in entries if e.get('measurement_id')]
    phase_costs={}
    for entry in entries:
        output=Path(entry['output_dir'])
        label=output.name
        phase=('product' if output.is_relative_to(root/'product') else 'consumer' if output==root/'consumer' else
            next((name for name in ('reference','automatic-screen','manual-screen','defaults-pair','manual-pair')
                  if label.startswith(name)),'unknown'))
        costs=phase_costs.setdefault(phase,{'attempts':0,'native_calls':0,'wait_seconds':0,'load_health_seconds':0,
            'tokenize_seconds':0,'completion_seconds':0,'teardown_seconds':0})
        costs['attempts']+=1
        costs['native_calls']+=int(entry['native_started'])
        costs['wait_seconds']+=entry.get('wait_elapsed_ns',0)/1e9
        timing=output/'phase-timing.json'
        if timing.is_file():
            data=json.loads(timing.read_text())
            for name in ('load_health','tokenize','teardown'):
                costs[name+'_seconds']+=(data.get(name+'_ms') or 0)/1000
        completion=output/'completion-wall.json'
        if completion.is_file():
            costs['completion_seconds']+=json.loads(completion.read_text())['elapsed_ms']/1000
    peak,reserve=0,None
    for entry in entries:
        memory=Path(entry['output_dir'])/'memory.json'
        if memory.is_file():
            for sample in json.loads(memory.read_text()).get('samples',[]):
                peak=max(peak,sample.get('dedicated_bytes') or 0)
                value=sample.get('device_free_bytes')
                if type(value) is int:reserve=value if reserve is None else min(value,reserve)
    return {'status':status,'complete_utility':complete,'utility_gain_established':status=='PASS-STOCK-UTILITY-PRODUCT',
        'selected_default':None if auto is None else auto==default.candidate_id,
        'automatic_id':auto,'manual_id':manual,'statistics':stats,'native_processes':len(journal_owners),
        'attempts':len(entries),'all_raw_records_valid':all(e['status']=='measured' for e in entries),
        'utility_gate_pass':complete and stats['status']=='PASS-STOCK-UTILITY',
        'phase_costs':phase_costs,
        'wait_seconds':sum(e.get('wait_elapsed_ns',0) for e in entries)/1e9,
        'native_phase_seconds':sum(native_times),'input_load_seconds':m['input_load_seconds'],
        'collection_wall_seconds':report['collection_wall_seconds'],
        'peak_owned_bytes':peak,'minimum_device_free_bytes':reserve,
        'scope':{'case_id':m['case']['case_id'],'model':m['case']['model_artifact'],
                 'workload_sha256':m['case']['workload_sha256'],'runtime_sha256':m['case']['runtime_sha256'],
                 'host_environment_sha256':canonical_sha256(host_environment)}}


def verify_product(report,source_store,automatic,sequence,host,entries):
    root=Path(report['manifest']['experiment_root']).resolve()
    if not (root/'selected-plan.json').is_file():
        if report['status'] not in ('RESOURCE-BUDGET-STOP','VALIDATION-STOP') or entries:
            raise ValueError('passing utility has no sealed source plan')
        return
    source=load_execution_plan(root/'selected-plan.json',store=source_store)
    expected=tuple(r['measurement_id'] for r in report['rows']
        if r['label'].startswith('manual-pair-') and r['label'].endswith('-sealed'))
    if source.candidate.candidate_id!=automatic.candidate_id or source.candidate.measurement_ids!=expected:
        raise ValueError('wider product source differs from selected confirmation')
    product_path=root/'product/report.json'
    if not product_path.is_file():
        if report['status'] not in ('RESOURCE-BUDGET-STOP','VALIDATION-STOP') or entries:
            raise ValueError('wider product report missing')
        return
    product=json.loads(product_path.read_text())
    product_store=EvidenceStore(root/'product.sqlite3')
    passed=repeatability.verify_block(product,root/'product',product_store,root/'selected-plan.json',
        source_store,entries[:20],host,report['manifest'])
    if len(entries)<=20 and report['status']=='PASS-STOCK-UTILITY-PRODUCT':
        raise ValueError('wider product pass lacks fresh consumer')
    if report['status']=='PRODUCT-VALIDATION-STOP':
        if passed or len(entries)!=20 or evaluate_pairs(product['rows'])['status']=='PASS-MEASUREMENT':
            raise ValueError('claimed product statistical stop differs from native gates')
        return
    if not passed:
        if report['status'] not in ('VALIDATION-STOP','RESOURCE-BUDGET-STOP'):
            raise ValueError('incomplete/invalid product falsely accepted')
        return
    plan=load_validated_stock_plan(root/'product/accepted/execution-plan.json',
        root/'product/accepted/acceptance-receipt.json',product_store,
        identities=automatic.identities,host_environment=host)
    if len(entries)==21:
        repeatability.verify_consumer(report.get('consumer'),entries[-1],plan,product_store,root/'consumer')
        if report['status']=='PASS-STOCK-UTILITY-PRODUCT' and report.get('consumer',{}).get('status')!='MEASURED-ACCEPTED-STOCK':
            raise ValueError('wider success lacks accepted consumer')
    elif report['status'] not in ('VALIDATION-STOP','RESOURCE-BUDGET-STOP'):
        raise ValueError('wider consumer proof incomplete')
