"""Bounded stock tuning utility proof; no historical receipt or gate is rewritten."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
import uuid

from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.pipeline import CompilationRequest, atomic_json, load_compiler_inputs
from expertflow.compiler.plan import CandidatePlan, CandidateStatus, seal_candidate
from expertflow.compiler.preflight import capture_host_environment, file_sha256
from expertflow.compiler.refinement import balanced_schedule, paired_statistics
from expertflow.compiler.runner import ServerMeasurementRunner, WindowsGpuMemorySampler
from expertflow.compiler.schema import WorkloadIR, canonical_payload, canonical_sha256
from expertflow.compiler.stock_discovery import _candidate, _snapshot_inputs
from expertflow.compiler.stock_eligibility import EligibilityRegistry, UPSTREAM
from expertflow.compiler.stock_search import rank_screening, scheduling_space, screening_schedule, topology_anchors


SPEC = Path('docs/research/protocols/specs/2026-10-04-stock-utility-proof.md')
PROTOCOL = 'stock-utility-proof-v1'
Q6_SHA256 = '089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba'


def audit_protocol_scope(inputs):
    from expertflow.compiler.reference import load_reference_workload
    if inputs.model.identity.sha256 != Q6_SHA256:
        raise ValueError('utility protocol requires the registered Gemma Q6 weights')
    actual = replace(inputs.workload,threads=8,cuda_graphs='on')
    for name in ('gemma4-q6-single-request.json','gemma4-q6-utility-transfer.json'):
        path = Path('configs/compiler')/name
        registered = WorkloadIR.from_reference(load_reference_workload(Path.cwd(),path))
        if actual == replace(registered,threads=8,cuda_graphs='on'):
            return {'registered_workload':str(path),'workload_sha256':canonical_sha256(actual),
                    'model_weights_sha256':Q6_SHA256}
    raise ValueError('workload is outside the two registered utility inputs')


class FrozenRunner:
    """Carry the original utility freeze and durable attempts across every stage."""
    def __init__(self,runner,manifest,report,capture):
        self.runner,self.manifest,self.report,self.capture=runner,manifest,report,capture

    def run_once(self,*args,**kwargs):
        manifest,report=self.manifest,self.report
        if canonical_payload(self.capture())!=manifest['host_environment'] or sources()!=manifest['source_files']:
            raise ValueError('utility source/host changed before launch')
        if len(report['attempts'])>=manifest['maximum_native_processes']:
            raise ValueError('utility native attempt budget exhausted')
        root=Path(manifest['experiment_root'])
        output=Path(kwargs['output_dir']).resolve()
        relative=output.relative_to(root)
        utility=output.parent==(root/'raw')
        attempt={'output_dir':str(output),'stage':kwargs['stage'],'status':'attempting',
                 'native_started':False,'process_identity':None}
        report['attempts'].append(attempt)
        index=len(report['outcomes'])
        if utility:
            report['outcomes'].append({'status':'attempting','measurement_id':None,'label':relative.name})
        atomic_json(root/'report.json',report)
        kwargs['host_environment']=manifest['host_environment']
        kwargs['experiment_context']={'manifest_sha256':manifest['manifest_sha256']}
        try:
            outcome=self.runner.run_once(*args,**kwargs)
            attempt.update(status=outcome.status,measurement_id=outcome.measurement_id)
            if utility:report['outcomes'][index]=canonical_payload(outcome)
            return outcome
        except Exception as error:
            attempt.update(status='exception',reason=str(error),exception_type=type(error).__name__)
            if utility:report['outcomes'][index]={'status':'exception','measurement_id':None,
                'label':relative.name,'reason':str(error),'exception_type':type(error).__name__}
            raise
        finally:
            started=output/'run-start.json'
            if started.is_file():
                try:
                    attempt.update(native_started=True,process_identity=json.loads(started.read_text()))
                except (ValueError,OSError) as error:
                    attempt.update(native_started=None,process_identity_error=str(error))
            atomic_json(root/'report.json',report)


def resolved_defaults(host):
    anchors = topology_anchors(host, host['cpu'][0]['cores'])
    if host['cpu'][0]['cores'] != 8 or host['cpu'][0]['logical_processors'] != 16 or anchors != (8,12,16):
        raise ValueError('default proof is scoped to the pinned single-socket 8/16 host')
    return 8, 'on'


def audit_defaults(repository, host):
    resolved_defaults(host)
    paths = ['common/common.cpp', 'common/common.h', 'ggml/src/ggml-cuda/ggml-cuda.cu',
             'ggml/src/ggml-cuda/common.cuh']
    objects = {}
    for name in paths:
        blob = subprocess.check_output(['git','-C',str(repository),'show',f'{UPSTREAM}:{name}'])
        objects[name] = hashlib.sha256(blob).hexdigest()
        if name == 'common/common.cpp' and b'return common_cpu_get_num_physical_cores();' not in blob:
            raise ValueError('unknown native default thread resolution')
        if name == 'common/common.h' and b'n_threads                   = -1;' not in blob:
            raise ValueError('unknown native default thread parameter')
        if name == 'ggml/src/ggml-cuda/common.cuh' and (
                b'(getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr)' not in blob or
                b'return !(disable_due_to_gpu_arch || disable_cuda_graphs_due_to_env);' not in blob):
            raise ValueError('unknown native default graph resolution')
    return {'upstream_commit':UPSTREAM,'source_objects':objects,'resolved_threads':8,
        'resolved_graphs':'on','scope':'resolved thread/graph deployment defaults; explicit CPU-MoE fit, workload and F16 controls fixed; not all out-of-box flags'}


def evaluate_utility(default, automatic, manual, automatic_again, *, automatic_evaluations, manual_evaluations):
    if type(automatic_evaluations) is not int or type(manual_evaluations) is not int or min(automatic_evaluations,manual_evaluations)<1:
        raise ValueError('positive measured native evaluation counts required')
    gain = paired_statistics(default, automatic)
    equivalence = paired_statistics(manual, automatic_again)
    cost = automatic_evaluations <= manual_evaluations
    stable = max(gain['direct_cv_pct'],gain['sealed_cv_pct'],equivalence['direct_cv_pct'],equivalence['sealed_cv_pct']) <= 10
    gain_pass = gain['geometric_change_pct'] >= 5 and gain['ci95_pct'][0] > 0
    manual_pass = equivalence['ci90_pct'][0] > -2 and equivalence['ci90_pct'][1] < 2
    status = ('VARIANCE-STOP' if not stable else 'NO-UTILITY-GAIN' if not gain_pass else
        'MANUAL-BASELINE-STOP' if not manual_pass else 'TUNING-COST-STOP' if not cost else 'PASS-STOCK-UTILITY')
    return {'status':status,'gain':gain,'manual_equivalence':equivalence,
        'native_evaluation_cost_pass':cost,'automatic_evaluations':automatic_evaluations,
        'manual_evaluations':manual_evaluations,'cost_unit':'independent native candidate evaluations',
        'wall_time_scope':'native load/tokenize/completion/teardown phases reported separately; no lower wall-time or human-effort claim'}


def sources():
    paths = [*sorted(Path('src/expertflow/compiler').rglob('*.py')),Path(__file__),SPEC,
        Path('configs/compiler/gemma4-q6-single-request.json'),Path('configs/baseline-prompt.txt'),
        Path('configs/compiler/gemma4-q6-utility-transfer.json'),Path('configs/compiler/stock-utility-heldout-prompt.txt')]
    return {str(p.resolve()):file_sha256(p) for p in paths}


def native_phase_seconds(store, mid):
    artifacts = {a.role:Path(a.identity.path) for a in store.measurement(mid).artifacts}
    completion = json.loads(artifacts['completion-wall'].read_text())['elapsed_ms']
    phase = json.loads(artifacts['phase-timing'].read_text()) if 'phase-timing' in artifacts else {}
    return (completion+sum(phase.get(k) or 0 for k in ('load_health_ms','tokenize_ms','teardown_ms')))/1000


def measurement_stage(label, manifest):
    # Ten held-out selected-plan arms supply the unchanged product validator.
    # Root/context/order binding identifies their utility experiment and phase.
    if label.startswith('manual-pair-') and label.endswith('-sealed'):
        return 'confirmation'
    return f"utility-{manifest['experiment_id']}-{label}"


def verify_row(row, candidate, store, manifest, owners, reference=None):
    mid = row['measurement_id']
    native = store.verify_measurement(mid)
    record = store.measurement(mid)
    if native['measured'] is not True or native['exit_code'] != 0 or any(native['validations'].get(k) is not True for k in ('exact_tokens','memory','cleanup')):
        raise ValueError('utility requires complete passing native evidence')
    if record.stage != measurement_stage(row['label'],manifest) or record.numerical_path != 'stock_same_runtime':
        raise ValueError('utility stage/numerical path mismatch')
    if native['candidate_id'] != candidate.candidate_id or native['identities'] != canonical_payload(candidate.identities) or native['settings_sha256'] != canonical_sha256(candidate.settings):
        raise ValueError('utility candidate binding mismatch')
    if any(canonical_payload(row.get(k)) != canonical_payload(v) for k,v in native.items()):
        raise ValueError('utility row differs from immutable native evidence')
    if row['run_wall_seconds'] != native_phase_seconds(store,mid):
        raise ValueError('native phase wall metric mismatch')
    if native['owned_run_sha256'] in owners:
        raise ValueError('utility reused native owner')
    owners.add(native['owned_run_sha256'])
    artifacts = {a.role:Path(a.identity.path) for a in record.artifacts}
    expected_root = Path(manifest['experiment_root'])/'raw'/row['label']
    if any(p.resolve().parent != expected_root.resolve() for p in artifacts.values()):
        raise ValueError('utility artifacts outside frozen root')
    launch = json.loads(artifacts['launch'].read_text())
    if launch.get('host_environment') != manifest['host_environment'] or launch.get('experiment_context') != {'manifest_sha256':manifest['manifest_sha256']}:
        raise ValueError('utility native launch does not bind host/protocol')
    start = json.loads(artifacts['run-start'].read_text())
    if start['started_monotonic_ns'] < manifest['frozen_monotonic_ns']:
        raise ValueError('utility reused pre-freeze native process')
    if reference and any(native[k] != reference[k] for k in ('generated_tokens_sha256','prompt_tokens_sha256')):
        raise ValueError('utility native tokens differ from own reference')
    return native


def manual_winner(rows, ids, default_id):
    return min(ids,key=lambda cid:(-statistics.mean(r['decode_tps'] for r in rows if r['candidate_id']==cid),cid!=default_id,cid))


def check_live_inputs(inputs, manifest, source_repository):
    supplied = canonical_payload({'model':inputs.model,'hardware':inputs.hardware,'stock':inputs.stock})
    workload = replace(inputs.workload,threads=8,cuda_graphs='on')
    default = _candidate(manifest['candidates'][manifest['default_id']])
    if (supplied != manifest['inputs'] or workload != default.identities.workload or
            Path(source_repository).resolve() != Path(manifest['source_repository']).resolve()):
        raise ValueError('supplied live inputs/source differ from frozen utility experiment')


def validate_result(report, store, inputs, *, source_repository, host_environment):
    check_live_inputs(inputs,report['manifest'],source_repository)
    result = reconstruct_utility(report,store,host_environment=host_environment)
    status = report['status']
    if result['status'] != 'PASS-STOCK-UTILITY':
        if status != result['status']:
            raise ValueError('utility terminal status differs from reconstructed gate')
        return result
    if status not in ('PASS-STOCK-UTILITY','PASS-STOCK-UTILITY-PRODUCT',
                      'PRODUCT-VALIDATION-STOP','CONSUMER-VALIDATION-STOP'):
        raise ValueError('unknown utility terminal status')
    if status != 'PASS-STOCK-UTILITY-PRODUCT':
        return {**result,'status':status}
    from expertflow.compiler.stock_validation import load_validated_stock_plan
    root = Path(report['manifest']['experiment_root'])
    product_store = EvidenceStore(root/'product.sqlite3')
    automatic = _candidate(report['manifest']['candidates'][report['automatic_id']])
    plan = load_validated_stock_plan(root/'product/accepted/execution-plan.json',
        root/'product/accepted/acceptance-receipt.json',product_store,
        identities=automatic.identities,host_environment=host_environment)
    receipt = json.loads((root/'product/accepted/acceptance-receipt.json').read_text())
    expected_ids = [r['measurement_id'] for r in report['rows']
        if r['label'].startswith('manual-pair-') and r['label'].endswith('-sealed')]
    source = receipt['experiment']['frozen']['source_plan']['candidate']
    if source['measurement_ids'] != expected_ids or _candidate(source).candidate_id != report['automatic_id']:
        raise ValueError('product source differs from utility selection/confirmation')
    if report.get('product_status') != 'PASS-STOCK-FALLBACK' or report.get('product_native_processes') != 20:
        raise ValueError('utility product process/status mismatch')
    if len(report['attempts'])!=107:
        raise ValueError('complete utility/product proof requires 107 retained attempts')
    for mid in plan.candidate.measurement_ids:
        artifacts={a.role:Path(a.identity.path) for a in product_store.measurement(mid).artifacts}
        launch=json.loads(artifacts['launch'].read_text())
        if launch.get('experiment_context')!={'manifest_sha256':report['manifest']['manifest_sha256']}:
            raise ValueError('product launch lost original utility protocol binding')
    consumer = report['consumer']
    record = product_store.measurement(consumer['measurement_id'])
    native = product_store.verify_measurement(consumer['measurement_id'])
    reference = store.verify_measurement(expected_ids[0])
    owners = {r['owned_run_sha256'] for r in report['rows']}
    owners.update(product_store.verify_measurement(mid)['owned_run_sha256'] for mid in plan.candidate.measurement_ids)
    if (consumer['status'] != 'MEASURED-ACCEPTED-STOCK' or consumer['plan_sha256'] != plan.plan_sha256 or
            consumer['decode_tps'] != native['decode_tps'] or record.stage != 'accepted-stock-run' or
            native['candidate_id'] != automatic.candidate_id or native['identities'] != canonical_payload(automatic.identities) or
            native['settings_sha256'] != canonical_sha256(automatic.settings) or native['owned_run_sha256'] in owners or
            native['measured'] is not True or native['exit_code'] != 0 or
            any(native['validations'].get(k) is not True for k in ('exact_tokens','memory','cleanup')) or
            any(native[k] != reference[k] for k in ('generated_tokens_sha256','prompt_tokens_sha256'))):
        raise ValueError('utility consumer differs from accepted fresh native execution')
    artifacts = {a.role:Path(a.identity.path) for a in record.artifacts}
    if any(p.resolve().parent != (root/'consumer').resolve() for p in artifacts.values()):
        raise ValueError('consumer artifacts outside utility experiment')
    launch = json.loads(artifacts['launch'].read_text())
    start = json.loads(artifacts['run-start'].read_text())
    if (launch.get('host_environment') != canonical_payload(host_environment) or
            launch.get('experiment_context')!={'manifest_sha256':report['manifest']['manifest_sha256']} or
            start['started_monotonic_ns'] < report['manifest']['frozen_monotonic_ns']):
        raise ValueError('consumer host/freshness binding mismatch')
    return {**result,'status':status,'native_processes':107,'accepted_plan_sha256':plan.plan_sha256}


def reconstruct_utility(report, store, *, host_environment):
    m = report['manifest']
    payload = dict(m)
    claimed = payload.pop('manifest_sha256')
    if claimed != canonical_sha256(payload) or m['protocol_version'] != PROTOCOL:
        raise ValueError('utility manifest hash/protocol mismatch')
    if m['source_files'] != sources() or canonical_payload(host_environment) != m['host_environment']:
        raise ValueError('utility source/host changed')
    if m['maximum_native_processes'] != 107 or m['reference_processes'] != 10:
        raise ValueError('utility budget changed')
    candidates = {cid:_candidate(p) for cid,p in m['candidates'].items()}
    default = candidates[m['default_id']]
    snapshot_inputs = _snapshot_inputs(m['inputs'], default.identities.workload)
    if m.get('protocol_scope') != audit_protocol_scope(snapshot_inputs):
        raise ValueError('utility registered protocol scope mismatch')
    proof = EligibilityRegistry.with_builtins().attest(snapshot_inputs,host_environment,m['source_repository'])
    if m['eligibility'] != proof or default.identities != snapshot_inputs.identities(snapshot_inputs.stock):
        raise ValueError('utility eligibility/input bindings changed')
    if default.settings.cpu_moe is not proof.get('baseline_cpu_moe',True) or default.settings.gpu_layers!=99 or default.settings.static is not None:
        raise ValueError('utility default placement outside trusted provider scope')
    if m['default_source_proof'] != audit_defaults(m['source_repository'],host_environment):
        raise ValueError('utility native default source proof changed')
    space = scheduling_space(default,host_environment)
    if set(candidates) != {c.candidate_id for c in space.candidates} or any(c.candidate_id!=cid for cid,c in candidates.items()):
        raise ValueError('utility candidate space changed')
    if tuple(m['default_controls']) != resolved_defaults(host_environment):
        raise ValueError('default controls changed')
    schedule = screening_schedule(candidates)
    if canonical_payload(schedule) != m['screening_schedule'] or canonical_payload(balanced_schedule()) != m['confirmation_schedule']:
        raise ValueError('utility schedule changed')
    rows = report['rows']
    if len(rows)!=86 or len(report['outcomes'])!=86:
        raise ValueError('utility proof requires exactly 86 retained runs')
    if not 86 <= len(report['attempts']) <= 107:
        raise ValueError('utility retained attempt budget mismatch')
    owners=set()
    reference=None
    index=0
    def take(label,cid):
        nonlocal index,reference
        row=rows[index]
        outcome=report['outcomes'][index]
        if row['label']!=label or row['candidate_id']!=cid or outcome['status']!='measured' or outcome['measurement_id']!=row['measurement_id']:
            raise ValueError('utility outcome/order mismatch')
        verified=verify_row(row,candidates[cid],store,m,owners,reference)
        attempt=report['attempts'][index]
        artifacts={a.role:Path(a.identity.path) for a in store.measurement(row['measurement_id']).artifacts}
        if (attempt.get('status')!='measured' or attempt.get('measurement_id')!=row['measurement_id'] or
                attempt.get('native_started') is not True or
                attempt.get('process_identity')!=json.loads(artifacts['run-start'].read_text()) or
                Path(attempt['output_dir']).resolve()!=(Path(m['experiment_root'])/'raw'/label).resolve()):
            raise ValueError('utility retained attempt does not match native record')
        reference=reference or verified
        index+=1
        return row
    refs=[take(f'reference-{n:02}',m['default_id']) for n in range(10)]
    if statistics.stdev(r['decode_tps'] for r in refs)*100/statistics.mean(r['decode_tps'] for r in refs)>10:
        raise ValueError('utility reference unstable')
    screens={}
    for method in ('automatic','manual'):
        screens[method]=[dict(take(f'{method}-screen-{block:02}-{cid}',cid),block=block)
            for block,order in enumerate(schedule) for cid in order]
    auto=rank_screening(screens['automatic'],m['default_id'],schedule)[0]['candidate_id']
    manual=manual_winner(screens['manual'],candidates,m['default_id'])
    if (report['automatic_id'],report['manual_id'])!=(auto,manual):
        raise ValueError('utility selected candidate not derived from its own screen')
    paired={}
    for purpose,control in [('defaults',m['default_id']),('manual',manual)]:
        pairs={}
        for pair,order in enumerate(balanced_schedule()):
            for arm in order:
                cid=control if arm=='direct' else auto
                pairs[pair,arm]=take(f'{purpose}-pair-{pair:02}-{arm}',cid)['decode_tps']
        paired[purpose]=([pairs[i,'direct'] for i in range(10)],[pairs[i,'sealed'] for i in range(10)])
    result=evaluate_utility(*paired['defaults'],*paired['manual'],automatic_evaluations=18,manual_evaluations=18)
    if report.get('statistics') != canonical_payload(result):
        raise ValueError('utility claimed statistics differ from native reconstruction')
    return result


def execute_utility(inputs, store, runner, output, *, source_repository, host_capture=None,
                    registry=None, default_source_proof=None, include_product=True, product_runner_factory=None):
    capture=host_capture or capture_host_environment
    host=canonical_payload(capture())
    proof=(registry or EligibilityRegistry.with_builtins()).attest(inputs,host,source_repository)
    protocol_scope=audit_protocol_scope(inputs)
    if proof.get('allowed_controls') != ['threads','cuda_graphs']:
        raise ValueError('utility requires reviewed stock scheduling scope')
    threads,graphs=resolved_defaults(host)
    workload=replace(inputs.workload,threads=threads,cuda_graphs=graphs)
    identities=replace(inputs.identities(inputs.stock),workload=workload,workload_sha256=canonical_sha256(workload))
    from expertflow.compiler.plan import RuntimeSettings
    default=CandidatePlan(identities,RuntimeSettings(99,proof.get('baseline_cpu_moe',True)))
    space=scheduling_space(default,host)
    if len(space.candidates)!=6:
        raise ValueError('utility scope requires six complete configurations')
    candidates={c.candidate_id:c for c in space.candidates}
    schedule=screening_schedule(candidates)
    output=Path(output).resolve()
    if output.exists():
        raise ValueError('fresh utility output required; no retries or resume')
    m={'protocol_version':PROTOCOL,'experiment_id':uuid.uuid4().hex,
       'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'source_files':sources(),'host_environment':host,'eligibility':proof,'protocol_scope':protocol_scope,
       'inputs':canonical_payload({'model':inputs.model,'hardware':inputs.hardware,'stock':inputs.stock}),
       'source_repository':str(Path(source_repository).resolve()),
       'default_source_proof':default_source_proof or audit_defaults(source_repository,host),
       'default_controls':[threads,graphs],'default_id':default.candidate_id,
       'candidates':{cid:canonical_payload(c) for cid,c in candidates.items()},
       'screening_schedule':canonical_payload(schedule),'confirmation_schedule':canonical_payload(balanced_schedule()),
       'reference_processes':10,'maximum_native_processes':107,
       'experiment_root':str(output),'frozen_monotonic_ns':time.monotonic_ns()}
    m['manifest_sha256']=canonical_sha256(m)
    output.mkdir(parents=True)
    atomic_json(output/'frozen-manifest.json',m)
    store.prime_model(inputs.model)
    report={'status':'RUNNING','manifest':m,'rows':[],'outcomes':[],'attempts':[]}
    frozen_runner=FrozenRunner(runner,m,report,capture)
    owners=set()
    reference=None
    def collect(label,cid):
        nonlocal reference
        if canonical_payload(capture())!=host or sources()!=m['source_files']:
            raise ValueError('utility source/host changed before launch')
        outcome=frozen_runner.run_once(candidates[cid],inputs.model,inputs.stock,
            output_dir=output/'raw'/label,measured=True,stage=measurement_stage(label,m),
            host_environment=host,experiment_context={'manifest_sha256':m['manifest_sha256']})
        if outcome.status!='measured':
            report.update(status=outcome.status.upper().replace('_','-'),reason=outcome.reason)
            atomic_json(output/'report.json',report)
            return None
        row={**store.verify_measurement(outcome.measurement_id),'measurement_id':outcome.measurement_id,
             'label':label,'run_wall_seconds':native_phase_seconds(store,outcome.measurement_id)}
        verified=verify_row(row,candidates[cid],store,m,owners,reference)
        reference=reference or verified
        report['rows'].append(row)
        atomic_json(output/'report.json',report)
        print(json.dumps({'process':len(report['outcomes']),'label':label,'tps':row['decode_tps']}),flush=True)
        return row
    try:
        refs=[]
        for n in range(10):
            row=collect(f'reference-{n:02}',default.candidate_id)
            if row is None: return report
            refs.append(row)
        if statistics.stdev(r['decode_tps'] for r in refs)*100/statistics.mean(r['decode_tps'] for r in refs)>10:
            raise ValueError('utility reference unstable')
        screens={}
        for method in ('automatic','manual'):
            screens[method]=[]
            for block,order in enumerate(schedule):
                for cid in order:
                    row=collect(f'{method}-screen-{block:02}-{cid}',cid)
                    if row is None: return report
                    screens[method].append({**row,'block':block})
        auto=rank_screening(screens['automatic'],default.candidate_id,schedule)[0]['candidate_id']
        manual=manual_winner(screens['manual'],candidates,default.candidate_id)
        report.update(automatic_id=auto,manual_id=manual)
        paired={}
        for purpose,control in [('defaults',default.candidate_id),('manual',manual)]:
            pairs={}
            for pair,order in enumerate(balanced_schedule()):
                for arm in order:
                    row=collect(f'{purpose}-pair-{pair:02}-{arm}',control if arm=='direct' else auto)
                    if row is None:return report
                    pairs[pair,arm]=row['decode_tps']
            paired[purpose]=([pairs[i,'direct'] for i in range(10)],[pairs[i,'sealed'] for i in range(10)])
        stats=evaluate_utility(*paired['defaults'],*paired['manual'],automatic_evaluations=18,manual_evaluations=18)
        report.update(statistics=canonical_payload(stats),status=stats['status'])
        reconstruct_utility(report,store,host_environment=capture())
        if stats['status']=='PASS-STOCK-UTILITY' and include_product:
            selected=replace(candidates[auto],status=CandidateStatus.MEASURED,
                measurement_ids=tuple(r['measurement_id'] for r in report['rows'] if r['label'].startswith('manual-pair-') and r['label'].endswith('-sealed')),
                validation=(('utility_confirmation',True),))
            plan=seal_candidate(selected,store,selected.identities,None)
            atomic_json(output/'selected-plan.json',plan)
            adjusted=replace(inputs,workload=selected.identities.workload)
            from expertflow.compiler.stock_validation import execute_stock_product,run_accepted_stock_plan
            product_store=EvidenceStore(output/'product.sqlite3')
            product_runner=(product_runner_factory(product_store) if product_runner_factory else
                ServerMeasurementRunner(product_store,memory_sampler=runner.memory_sampler))
            product_runner=FrozenRunner(product_runner,m,report,capture)
            product=execute_stock_product(adjusted,output/'selected-plan.json',store,product_store,product_runner,
                output/'product',host_capture=capture)
            report['product_status']=product['status']
            report['product_native_processes']=len(product['outcomes'])
            if product['status']!='PASS-STOCK-FALLBACK':
                report.update(status='PRODUCT-VALIDATION-STOP',reason=product.get('reason'))
            else:
                status,consumer=run_accepted_stock_plan(output/'product/accepted/execution-plan.json',
                    output/'product/accepted/acceptance-receipt.json',adjusted,product_store,
                    output/'consumer',runner=product_runner,host_capture=capture)
                report['consumer']={'status':status,**consumer}
                if status!='MEASURED-ACCEPTED-STOCK':report['status']='CONSUMER-VALIDATION-STOP'
                else:
                    report['status']='PASS-STOCK-UTILITY-PRODUCT'
                    validate_result(report,store,inputs,source_repository=source_repository,host_environment=capture())
    except Exception as error:
        report.update(status='VALIDATION-STOP',reason=str(error))
    atomic_json(output/'report.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('run','validate'),required=True)
    for name in ('descriptor','inventory','hardware','workload','runtime-identity','evidence-db','output-dir','source-repository'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    sampler=None
    try:
        request=CompilationRequest(args.descriptor,args.inventory,args.hardware,args.workload,args.runtime_identity,(),args.evidence_db,args.output_dir)
        inputs=load_compiler_inputs(request,live=True)
        audit_protocol_scope(inputs)
        if args.action=='validate':
            if not args.evidence_db.is_file():raise ValueError('existing utility database required')
            report=json.loads((args.output_dir/'report.json').read_text())
            result=validate_result(report,EvidenceStore(args.evidence_db),inputs,
                source_repository=args.source_repository,host_environment=capture_host_environment())
        else:
            if args.evidence_db.exists() or args.output_dir.exists():raise ValueError('fresh database/output required; no retries')
            host=capture_host_environment()
            default_proof=audit_defaults(args.source_repository,host)
            store=EvidenceStore(args.evidence_db)
            sampler=WindowsGpuMemorySampler(inputs.hardware.gpu_uuid)
            runner=ServerMeasurementRunner(store,memory_sampler=sampler)
            result=execute_utility(inputs,store,runner,args.output_dir,source_repository=args.source_repository,
                default_source_proof=default_proof)
        print(json.dumps({'status':result['status'],'reason':result.get('reason'),'report':str(args.output_dir/'report.json')}))
        return 0 if result['status'].startswith('PASS-') else 2
    finally:
        if sampler:sampler.close()


if __name__=='__main__':
    raise SystemExit(main())
