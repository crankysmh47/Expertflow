"""Independent terminal audit; read-only SQLite, no native calls/full weight reads."""
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib, importlib.util, json, math, random, sqlite3, statistics, subprocess, sys, time, zipfile
from pathlib import Path

REPO=Path('C:/sem4/expertflow');sys.path.insert(0,str(REPO))
ROOT=Path('C:/models/expertflow/runs/compiler-stock-coverage-20261005')
WS=REPO/'.superpowers/sdd/2026-10-05-wider-stock-collector'
COMMIT='78ad5f0063f2bc372a528e37ec1f45d8be1ceb11'
from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.schema import ArtifactIdentity, canonical_payload, canonical_sha256, WorkloadIR
from expertflow.compiler.runner import RuntimeBinding
from expertflow.compiler.preflight import file_sha256, capture_host_environment
from expertflow.compiler.pipeline import inspect_model
from expertflow.compiler.reference import load_reference_workload
from expertflow.compiler.stock_discovery import _snapshot_inputs
from expertflow.compiler.plan import CandidatePlan,RuntimeSettings
from expertflow.compiler.stock_search import scheduling_space, screening_schedule
from expertflow.compiler.stock_eligibility import EligibilityRegistry
from expertflow.stock.coverage import verify_registration
from expertflow.stock.wider import sources as executing_sources
from scripts.benchmark_compiler_stock_utility import audit_defaults

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda data:hashlib.sha256(data).hexdigest()
def check(condition,reason):
    if not condition:raise AssertionError(reason)
def eq(actual,expected,reason):
    if isinstance(actual,(set,Path)):check(actual==expected,reason)
    else:check(canonical_payload(actual)==canonical_payload(expected),reason)
def near(actual,expected,reason):check(math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-8),reason)
def frozen(payload):
    rest=dict(payload);digest=rest.pop('manifest_sha256');eq(canonical_sha256(rest),digest,'manifest checksum')
def statkey(path):
    s=Path(path).stat();return s.st_size,s.st_mtime_ns,s.st_ctime_ns

start=time.perf_counter();outer=read(ROOT/'report.json');seq=outer['manifest'];reg=seq['registration']
check(outer['status']=='COMPLETE-STOCK-COVERAGE','terminal sequence status')
eq(outer['attempts'],344,'attempt count');eq(outer['native_processes'],344,'native count')
frozen(seq);eq(seq,read(ROOT/'frozen-manifest.json'),'outer persisted freeze');eq(seq['source_commit'],COMMIT,'executed source commit')
check(seq['sequence_started_monotonic_ns']<=seq['frozen_monotonic_ns']<=outer['sequence_finished_monotonic_ns'],'outer clock order')
eq(seq['sequence_started_monotonic_ns'],outer['sequence_started_monotonic_ns'],'outer start agreement')
near((outer['sequence_finished_monotonic_ns']-seq['sequence_started_monotonic_ns'])/1e9,outer['sequence_wall_seconds'],'sequence wall')
check(outer['sequence_wall_seconds']<57600,'sequence cap')
eq([seq[k] for k in ('case_wall_cap_seconds','sequence_wall_cap_seconds','maximum_native_processes')],[14400,57600,428],'fixed budgets')
registration_path=REPO/'configs/compiler/stock-coverage-20261005.json'
eq(read(registration_path),reg,'registration copy');verify_registration(reg,REPO)
eq(reg['registration_sha256'],'0376ac4d7c5e146bc594f5c22a4b1be803b446ced4c3d9d04a4cb0737af55603','immutable registration')
check(registration_path.read_bytes()==subprocess.check_output(['git','show','5da6786:configs/compiler/stock-coverage-20261005.json']),'registration raw bytes changed')
for name,digest in reg['input_files'].items():eq(file_sha256(REPO/name),digest,'registered input '+name)
eq(seq['host_environment'],reg['host_environment'],'frozen host');eq(canonical_payload(capture_host_environment()),reg['host_environment'],'live host')
eq(seq['source_files'],executing_sources(),'complete current source inventory')
eq(seq['registration_sha256'],reg['registration_sha256'],'outer registration hash')
eq(seq['source_repository'],reg['source_repository'],'registered source repository')
eq(seq['default_source_proof'],reg['default_source_proof'],'registered default proof')
eq(audit_defaults(seq['source_repository'],seq['host_environment']),reg['default_source_proof'],'live upstream default proof')
snapshot=read(REPO/'docs/evidence/stock-coverage-20261005/corrected-source-snapshot.json')
archive=REPO/'docs/evidence/stock-coverage-20261005/corrected-source-snapshot.zip'
eq(snapshot['source_commit'],COMMIT,'snapshot commit');eq(file_sha256(archive),snapshot['archive_sha256'],'archive checksum');eq(snapshot['source_files'],seq['source_files'],'archive/freeze source maps')
line_endings=[]
with zipfile.ZipFile(archive) as z:
    eq(len(z.namelist()),len(set(z.namelist())),'archive duplicates');eq(set(z.namelist()),{e['member'] for e in snapshot['entries']},'archive entry inventory')
    eq({e['original_path']:e['sha256'] for e in snapshot['entries']},seq['source_files'],'archive entry hashes')
    for e in snapshot['entries']:
        data=z.read(e['member']);eq(sha(data),e['sha256'],'archive raw '+e['member']);eq(len(data),e['size_bytes'],'archive size')
        eq(file_sha256(Path(e['original_path'])),e['sha256'],'live pinned source '+e['member'])
        blob=subprocess.check_output(['git','show',COMMIT+':'+e['member']])
        if data!=blob:
            attrs=subprocess.check_output(['git','check-attr','--source='+COMMIT,'text','eol','--',e['member']],text=True)
            check(': text: set' in attrs and ': eol: lf' in attrs and data.replace(b'\r\n',b'\n')==blob,'undeclared source difference '+e['member'])
            line_endings.append(e['member'])
print('Source, registration, host and snapshot checks passed.',flush=True)

# Explicitly exclude fresh weight hashing. Every record still binds the exact
# registered identity, the physical file/size, and unchanged stat tuple.
allowed_models={ArtifactIdentity(**c['model_artifact']) for c in reg['cases']}
model_stats={identity:statkey(identity.path) for identity in allowed_models}
for identity,key in model_stats.items():eq(key[0],identity.size_bytes,'model size')
class AuditStore(EvidenceStore):
    def __init__(self,path):
        self.path=Path(path).resolve();check(self.path.is_file(),'missing database');self._model_verifications=set();self.verified={}
    @contextmanager
    def _connection(self):
        conn=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)
        conn.execute('PRAGMA query_only=ON')
        try:yield conn
        finally:conn.close()
    def prime_model(self,model):
        check(model.identity in allowed_models,'unregistered weight identity')
        eq(statkey(model.identity.path),model_stats[model.identity],'weight file stat drift')
    def verify_measurement(self,mid,_seen=frozenset()):
        value=super().verify_measurement(mid,_seen);self.verified[mid]=value;return value
stores={}
def store_for(path):
    path=Path(path).resolve()
    if path not in stores:stores[path]=AuditStore(path)
    return stores[path]
# Runtime digest verification once per identical stat-guarded binding. All raw
# launch/record controls continue through the unmodified native evidence checker.
original_runtime_verify=RuntimeBinding.verify
runtime_verified={}
def runtime_verify(binding):
    identities=(binding.server,*binding.dependencies,*((binding.cuda_runtime,) if binding.cuda_runtime else ()))
    current=tuple((a,statkey(a.path)) for a in identities)
    if binding.sha256 in runtime_verified:eq(current,runtime_verified[binding.sha256],'runtime stat drift')
    else:
        original_runtime_verify(binding);runtime_verified[binding.sha256]=current
RuntimeBinding.verify=runtime_verify
spec=importlib.util.spec_from_file_location('independent_frozen_auditor',REPO/'docs/evidence/stock-coverage-20261005/independent_audit.py')
auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor);auditor.EvidenceStore=store_for
protected={str(p):file_sha256(p) for p in [ROOT/'report.json',ROOT/'frozen-manifest.json',*[Path(c['planned_root'])/name for c in reg['cases'] for name in ('report.json','frozen-manifest.json','utility.sqlite3')]]}
raw_audit=auditor.audit(ROOT/'report.json')
check(raw_audit['status']=='RAW-AUDIT-PASS' and raw_audit['native_processes']==344,'source-bound raw audit')
print('Source-bound independent auditor checked all344 native records.',flush=True)

# Independent extended reconstruction: SQLite inventories, all outcomes/rows,
# scope/grid, chronology, reference CV, complete statistic fields, phase costs.
previous=seq['frozen_monotonic_ns'];owners=set();run_ids=set();mids=set();all_outputs=set();summaries=[]
for case,saved,ar in zip(reg['cases'],outer['cases'],raw_audit['cases']):
    case_id=case['case_id'];root=Path(case['planned_root']).resolve();report=read(root/'report.json');m=report['manifest'];frozen(m)
    eq(m,read(root/'frozen-manifest.json'),'case persisted freeze');eq(m['case'],case,'registered case');eq(saved['case_id'],case_id,'sequence case order')
    eq(m['source_commit'],COMMIT,'case commit');eq(m['source_files'],seq['source_files'],'case source map');eq(m['sequence_manifest_sha256'],seq['manifest_sha256'],'outer freeze binding')
    eq(m['sequence_started_monotonic_ns'],seq['sequence_started_monotonic_ns'],'case sequence start');eq(m['host_environment'],seq['host_environment'],'case host')
    eq(m['source_repository'],seq['source_repository'],'source repository');eq(m['default_source_proof'],seq['default_source_proof'],'default source proof')
    eq(m['input_load_seconds'],seq['case_input_load_seconds'][case_id],'input loading cost');check(m['input_load_seconds']>=0,'negative loading cost')
    eq(sorted(p.name for p in root.iterdir()),['frozen-manifest.json','raw','report.json','utility.sqlite3'],'unexpected product/consumer/other case artifacts')
    eq(len(report['rows']),86,'complete utility rows');eq(len(report['outcomes']),86,'complete utility outcomes');eq(len(report['attempts']),86,'complete utility attempts')
    eq(report['status'],'NO-UTILITY-GAIN','case stop');eq(saved['status'],report['status'],'outer case stop');eq(ar['utility_verdict'],report['status'],'independent verdict')
    check(previous<=m['case_started_monotonic_ns']<=m['frozen_monotonic_ns'],'overlapping case chronology')
    inp_snapshot=seq['case_inputs'][case_id]
    model=inspect_model(REPO/case['descriptor'],REPO/case['inventory'])
    workload=replace(WorkloadIR.from_reference(load_reference_workload(REPO,REPO/case['workload'])),threads=8,cuda_graphs='on')
    inputs=_snapshot_inputs(inp_snapshot,workload)
    eq(canonical_payload(inputs.model),canonical_payload(model),'metadata model versus frozen input')
    eq(m['inputs'],{name:inp_snapshot[name] for name in ('model','hardware','stock')},'case input snapshot')
    eq(canonical_sha256(workload),case['workload_sha256'],'registered workload')
    proof=EligibilityRegistry.with_builtins().attest(inputs,seq['host_environment'],seq['source_repository'])
    eq(canonical_payload(proof),m['eligibility'],'live eligibility/source objects')
    default=CandidatePlan(inputs.identities(inputs.stock),RuntimeSettings(99,proof.get('baseline_cpu_moe',True)))
    candidates=scheduling_space(default,seq['host_environment']).candidates
    eq([c.candidate_id for c in candidates],case['candidate_ids'],'candidate identities');eq(default.candidate_id,case['default_id'],'default identity')
    eq({c.candidate_id:canonical_payload(c) for c in candidates},m['candidates'],'candidate freeze')
    eq(canonical_payload(screening_schedule(c.candidate_id for c in candidates)),case['screening_schedule'],'seeded screens')
    eq(m['screening_schedule'],case['screening_schedule'],'case screen freeze');eq(m['confirmation_schedule'],reg['paired_schedule'],'pair schedule')
    eq(m['maximum_native_processes'],107,'case budget');eq(m['reference_processes'],10,'references')
    store=store_for(root/'utility.sqlite3')
    with store._connection() as db:
        eq(db.execute('PRAGMA integrity_check').fetchall(),[('ok',)],'SQLite integrity')
        eq(db.execute('PRAGMA foreign_key_check').fetchall(),[],'SQLite foreign keys')
        database_rows=db.execute('SELECT id,key_sha256,owned_run_sha256 FROM measurement ORDER BY sequence').fetchall()
        eq([r[0] for r in database_rows],[a['measurement_id'] for a in report['attempts']],'database exact order/count')
        eq(db.execute('SELECT COUNT(*) FROM artifact').fetchone()[0],86*10,'artifact table count')
        eq(db.execute('SELECT COUNT(*) FROM validation').fetchone()[0],86*3,'validation table count')
    prev=m['frozen_monotonic_ns'];phase_costs={};waits=[];phase_total=0;peak=0;reserve=None;observed_reserve=None;sample_count=0;max_gap=0;max_load=0;max_completion=0
    reference=None;rates=[];artifact_count=0;earliest_creation=None;latest_creation=None
    for row,outcome,entry,dbrow in zip(report['rows'],report['outcomes'],report['attempts'],database_rows):
        mid=entry['measurement_id'];check(mid not in mids,'measurement ID reuse');mids.add(mid)
        eq(entry['status'],'measured','failed attempt');eq(entry['reason'],None,'unexpected attempt reason');eq(outcome['status'],'measured','outcome status');eq(outcome['reason'],None,'outcome reason')
        eq(outcome['measurement_id'],mid,'outcome ID');eq(row['measurement_id'],mid,'row ID');eq(Path(outcome['output_dir']).resolve(),Path(entry['output_dir']).resolve(),'outcome root')
        eq(Path(entry['database']).resolve(),store.path,'exact database')
        output=Path(entry['output_dir']).resolve();eq(output,root/'raw'/row['label'],'raw output label/root');check(output not in all_outputs,'output reuse');all_outputs.add(output)
        check(entry['native_started'] is True,'missing native flag');eq(entry['wait_seconds'],30,'fixed pacing');check(prev<=entry['wait_started_monotonic_ns']<=entry['wait_finished_monotonic_ns'],'attempt order')
        eq(entry['wait_elapsed_ns'],entry['wait_finished_monotonic_ns']-entry['wait_started_monotonic_ns'],'wait arithmetic');check(entry['wait_elapsed_ns']>=30_000_000_000,'short wait');waits.append(entry['wait_elapsed_ns']/1e9)
        start_record=read(output/'run-start.json');eq(start_record,entry['process_identity'],'journal/raw owner');owner=tuple(start_record[k] for k in ('pid','creation_time_100ns','creation_source'))
        check(owner not in owners and start_record['run_id'] not in run_ids,'owner/run reuse');owners.add(owner);run_ids.add(start_record['run_id'])
        check(entry['wait_finished_monotonic_ns']<=start_record['started_monotonic_ns']<=entry['finished_monotonic_ns'],'native chronology');prev=entry['finished_monotonic_ns']
        check((start_record['started_monotonic_ns']-m['case_started_monotonic_ns'])/1e9+m['input_load_seconds']<14400,'case launch cap')
        check((start_record['started_monotonic_ns']-seq['sequence_started_monotonic_ns'])/1e9<57600,'sequence launch cap')
        native=store.verified[mid];record=store.measurement(mid)
        eq(record.key,record.key,'typed record');eq(dbrow[1],canonical_sha256(record.key),'database key hash');eq(dbrow[2],record.owned_run_sha256,'database owner hash')
        eq(record.numerical_path,'stock_same_runtime','numerical path');eq(record.comparison_ids,(),'unexpected comparisons')
        expected_stage='confirmation' if row['label'].startswith('manual-pair-') and row['label'].endswith('-sealed') else 'utility-'+m['experiment_id']+'-'+row['label']
        eq(entry['stage'],expected_stage,'stage/label');eq(record.stage,expected_stage,'database stage')
        for key,value in native.items():eq(row[key],value,'row/raw '+key)
        if reference is None:reference=native
        for key in ('prompt_tokens_sha256','generated_tokens_sha256'):eq(native[key],reference[key],'own-reference exact tokens')
        rates.append(native['decode_tps'])
        with store._connection() as db:
            artifacts={role:read_data for role,read_data in db.execute('SELECT role,payload FROM artifact WHERE measurement_id=?',(mid,))}
            for artifact in record.artifacts:eq(json.loads(artifacts[artifact.role]),canonical_payload(artifact),'artifact table binding')
            eq(dict(db.execute('SELECT name,passed FROM validation WHERE measurement_id=?',(mid,))),{'exact_tokens':1,'memory':1,'cleanup':1},'validation table binding')
        artifact_count+=len(record.artifacts)
        phase=read(output/'phase-timing.json');wall=read(output/'completion-wall.json');memory=read(output/'memory.json');process=read(output/'process.json');request=read(output/'request.json')
        eq(memory['sample_interval_seconds'],0.2,'mandatory sampler interval');eq(request['n_predict'],512,'decode length');eq(process['forced_kill'],False,'forced cleanup');eq(process['owned_termination'],True,'owned termination')
        check(start_record['started_monotonic_ns']<=wall['started_monotonic_ns']<=wall['finished_monotonic_ns']<=entry['finished_monotonic_ns'],'completion chronology')
        for name in ('load_health_ms','tokenize_ms','teardown_ms'):check(math.isfinite(phase[name]) and phase[name]>=0,'invalid phase cost')
        max_load=max(max_load,phase['load_health_ms']/1000);max_completion=max(max_completion,wall['elapsed_ms']/1000)
        check(phase['load_health_ms']/1000<180 and wall['elapsed_ms']/1000<300,'per-call health/completion cap')
        duration=(wall['elapsed_ms']+sum(phase[k] for k in ('load_health_ms','tokenize_ms','teardown_ms')))/1000
        near(row['run_wall_seconds'],duration,'native phase row cost');phase_total+=duration
        label=row['label'];which=next(p for p in ('reference','automatic-screen','manual-screen','defaults-pair','manual-pair') if label.startswith(p))
        cost=phase_costs.setdefault(which,{'attempts':0,'native_calls':0,'wait_seconds':0,'load_health_seconds':0,'tokenize_seconds':0,'completion_seconds':0,'teardown_seconds':0})
        cost['attempts']+=1;cost['native_calls']+=1;cost['wait_seconds']+=waits[-1];cost['completion_seconds']+=wall['elapsed_ms']/1000
        for name in ('load_health','tokenize','teardown'):cost[name+'_seconds']+=phase[name+'_ms']/1000
        for sample in memory['samples']:
            peak=max(peak,sample['dedicated_bytes']);reserve=sample['device_free_bytes'] if reserve is None else min(reserve,sample['device_free_bytes']);sample_count+=1
        obs=memory['observations']
        for observation in obs:
            if observation['state']=='allocated':observed_reserve=observation['device_free_bytes'] if observed_reserve is None else min(observed_reserve,observation['device_free_bytes'])
        max_gap=max([max_gap,*[b['time_monotonic']-a['time_monotonic'] for a,b in zip(obs,obs[1:])]])
    eq(set(root.glob('raw/*/run-start.json')),{p/'run-start.json' for p in all_outputs if p.is_relative_to(root)},'case raw start inventory')
    check(prev<=report['collection_finished_monotonic_ns']<=saved['reconstruction_started_monotonic_ns']<=saved['reconstruction_finished_monotonic_ns']<=outer['sequence_finished_monotonic_ns'],'collection/reconstruction chronology')
    near(report['collection_wall_seconds'],(report['collection_finished_monotonic_ns']-m['case_started_monotonic_ns'])/1e9,'case collection wall')
    near(saved['reconstruction_wall_seconds'],(saved['reconstruction_finished_monotonic_ns']-saved['reconstruction_started_monotonic_ns'])/1e9,'reconstruction wall')
    case_wall=m['input_load_seconds']+(saved['reconstruction_finished_monotonic_ns']-m['case_started_monotonic_ns'])/1e9
    near(saved['case_wall_seconds'],case_wall,'full case wall');check(case_wall<14400 and saved['resource_budget_pass'] is True,'case wall cap')
    previous=saved['reconstruction_finished_monotonic_ns']
    ref_cv=statistics.stdev(rates[:10])*100/statistics.mean(rates[:10]);check(ref_cv<=10,'reference stability')
    eq(saved['statistics'],report['statistics'],'outer statistical copy');eq(saved['automatic_id'],ar['automatic_id'],'outer automatic');eq(saved['manual_id'],ar['manual_id'],'outer manual')
    eq(saved['selected_default'],ar['selected_default'],'selected-default claim')
    eq(saved['scope'],{'case_id':case_id,'model':case['model_artifact'],'workload_sha256':case['workload_sha256'],
        'runtime_sha256':case['runtime_sha256'],'host_environment_sha256':canonical_sha256(seq['host_environment'])},'reported measured scope')
    for key in ('utility_gain_established','raw_utility_product_gate_pass','utility_gate_pass'):eq(saved[key],False,'invalid utility/product claim')
    for key in ('all_raw_records_valid','complete_utility'):eq(saved[key],True,'completeness claim')
    for key in ('automatic_evaluations','manual_evaluations'):eq(report['statistics'][key],18,'evaluation budget')
    eq(report['statistics']['native_evaluation_cost_pass'],True,'tuning count gate')
    for purpose,offset in (('gain',46),('manual_equivalence',66)):
        direct=[];selected=[]
        for pair,order in enumerate(reg['paired_schedule']):
            values=dict(zip(order,rates[offset+2*pair:offset+2*pair+2]));direct.append(values['direct']);selected.append(values['sealed'])
        stats=report['statistics'][purpose]
        eq(stats['direct_tps'],direct,'direct paired samples');eq(stats['sealed_tps'],selected,'selected paired samples')
        eq(stats['paired_log_ratios'],[math.log(b)-math.log(a) for a,b in zip(direct,selected)],'paired logs')
        eq(stats['direct_mean_tps'],statistics.mean(direct),'direct mean');eq(stats['sealed_mean_tps'],statistics.mean(selected),'selected mean')
        eq(stats['bootstrap_seed'],20261003,'bootstrap seed');eq(stats['bootstrap_samples'],10000,'bootstrap count')
    for phase,cost in phase_costs.items():
        for key,value in cost.items():near(saved['phase_costs'][phase][key],value,'phase cost '+phase+'/'+key)
    near(saved['native_phase_seconds'],phase_total,'total native phase cost');near(saved['wait_seconds'],sum(waits),'total waits')
    near(saved['collection_wall_seconds'],report['collection_wall_seconds'],'outer collection cost');near(saved['input_load_seconds'],m['input_load_seconds'],'outer load cost')
    eq(saved['peak_owned_bytes'],peak,'peak memory');eq(saved['minimum_device_free_bytes'],reserve,'minimum reserve')
    candidate_lookup={c.candidate_id:c for c in candidates}
    controls=lambda cid:{'threads':candidate_lookup[cid].identities.workload.threads,'cuda_graphs':candidate_lookup[cid].settings.cuda_graphs}
    summaries.append({'case_id':case_id,'status':report['status'],'attempts':86,'native_starts':86,'database_records':86,'hashed_native_artifacts':artifact_count,
        'automatic_controls':controls(ar['automatic_id']),'manual_controls':controls(ar['manual_id']),'selected_default':ar['selected_default'],'automatic_evaluations':18,'manual_evaluations':18,
        'reference_cv_pct':ref_cv,'own_reference_prompt_sha256':reference['prompt_tokens_sha256'],'own_reference_output_sha256':reference['generated_tokens_sha256'],
        'defaults':ar['gain'],'manual_equivalence':ar['manual_equivalence'],'terminal_reason':'defaults geometric gain below registered5percent minimum' if ar['gain']['geometric_change_pct']<5 else 'defaults CI95 lower not strictly positive',
        'input_load_seconds':m['input_load_seconds'],'collection_wall_seconds':report['collection_wall_seconds'],'reconstruction_wall_seconds':saved['reconstruction_wall_seconds'],'case_wall_seconds':case_wall,
        'wait_seconds':sum(waits),'minimum_wait_seconds':min(waits),'native_phase_seconds':phase_total,'phase_costs':phase_costs,'peak_owned_bytes':peak,'minimum_sampled_device_free_bytes':reserve,
        'minimum_observed_device_free_bytes':observed_reserve,'memory_sample_count':sample_count,'maximum_observation_gap_seconds':max_gap,'maximum_load_health_seconds':max_load,'maximum_completion_seconds':max_completion,
        'resource_caps_pass':True,'utility_gain_established':False,'product_or_consumer_present':False})
    print('Extended raw/database/timing checks passed: '+case_id,flush=True)

check(len(owners)==len(run_ids)==len(mids)==len(all_outputs)==344,'whole-sequence unique inventory')
eq(set(ROOT.rglob('run-start.json')),{p/'run-start.json' for p in all_outputs},'unaccounted whole-sequence starts')
eq(sorted(p.name for p in ROOT.iterdir()),sorted(['report.json','frozen-manifest.json',*[c['case_id'] for c in reg['cases']]]),'unexpected sequence outputs')
supervisor=read(WS/'native-job.json');public_log=read(WS/'native-collection.log')
eq(supervisor['status'],'COMPLETED','supervisor terminal status');eq(supervisor['exit_code'],0,'supervisor exit')
supervisor_start=datetime.fromisoformat(supervisor['started_at_utc']);supervisor_end=datetime.fromisoformat(supervisor['finished_at_utc'])
supervisor_wall=(supervisor_end-supervisor_start).total_seconds()
check(supervisor_wall>=outer['sequence_wall_seconds'],'supervisor cost below collection cost')
check(datetime.fromisoformat(snapshot['created_at_utc'])<supervisor_start,'snapshot creation after launch')
eq(public_log['status'],outer['status'],'public terminal status');eq(public_log['cases'],outer['cases'],'public case results')
eq(public_log['attempts'],344,'public attempts');eq(public_log['native_processes'],344,'public native count')
for key in ('utility_gain_established','family_wide_gain_established','global_optimum_established','serving_throughput_established'):
    eq(public_log['decision'][key],False,'public unsupported claim '+key)
# Closed-study integrity: source/history pin bytes and process identities.
historical=read(REPO/'docs/evidence/stock-repeatability-20261004/source-integrity.json')
for name,digest in {**historical['frozen_files'],**historical['historical_files']}.items():eq(file_sha256(Path(name)),digest,'original source/history '+name)
oldroot=Path('C:/models/expertflow/runs/compiler-stock-repeatability-20261004');oldreport=read(oldroot/'report.json');oldstarts=set(oldroot.rglob('run-start.json'))
eq(len(oldstarts),148,'original starts');eq(len(oldreport['attempts']),148,'original attempts');eq(file_sha256(oldroot/'report.json'),'464defd327d43c4f510f3768266d1dacaefc77d25c239afefef5cc4226c1d0c5','original report')
for entry in oldreport['attempts']:
    path=Path(entry['output_dir'])/'run-start.json';check(path in oldstarts,'original start');identity=read(path);eq(identity,entry['process_identity'],'original identity');eq(entry['status'],'measured','original status')
    check(tuple(identity[k] for k in ('pid','creation_time_100ns','creation_source')) not in owners,'historical process reuse')
for name,digest in protected.items():eq(file_sha256(Path(name)),digest,'audit modified record/database '+name)
for identity,key in model_stats.items():eq(statkey(identity.path),key,'model changed during audit')
for name,digest in seq['source_files'].items():eq(file_sha256(Path(name)),digest,'source changed during audit')
RuntimeBinding.verify=original_runtime_verify
result={'status':'PASS-FINAL-INDEPENDENT-RAW-EVIDENCE-AUDIT','audited_at_utc':datetime.now(timezone.utc).isoformat(),'audit_wall_seconds':time.perf_counter()-start,'material_discrepancies':[],
    'executed_commit':COMMIT,'source_snapshot_files':len(snapshot['entries']),'source_archive_sha256':snapshot['archive_sha256'],'declared_line_ending_equivalences':line_endings,
    'registration_sha256':reg['registration_sha256'],'registration_raw_sha256':file_sha256(registration_path),'registered_input_files':len(reg['input_files']),
    'sequence_report_sha256':file_sha256(ROOT/'report.json'),'sequence_status':outer['status'],'attempts':344,'native_starts':344,'unique_measurements':344,'hashed_native_artifacts':sum(s['hashed_native_artifacts'] for s in summaries),
    'sequence_wall_seconds':outer['sequence_wall_seconds'],'supervisor_wall_seconds':supervisor_wall,'input_load_seconds_total':sum(seq['case_input_load_seconds'].values()),'actual_wait_seconds_total':sum(s['wait_seconds'] for s in summaries),'resource_caps_pass':True,
    'cases':summaries,'all_raw_records_valid':True,'all_reference_and_paired_cv_pass':True,'product_or_consumer_artifacts':0,'unaccounted_native_starts':0,
    'original_frozen_files_verified':len(historical['frozen_files']),'original_history_pins_verified':len(historical['historical_files']),'original_native_starts':148,'original_report_sha256':file_sha256(oldroot/'report.json'),
    'native_calls_from_audit':0,'fresh_full_weight_hash_reads':0,'readonly_sqlite':True,'protected_report_manifest_database_hashes_unchanged':True,'full_weight_hash_verification_owner':'separate root public read-only validation',
    'limits':['Model identities and unchanged physical file sizes/stat tuples were verified, but this audit intentionally did not reread complete weight files.','Runtime digests were verified once per binding with unchanged file-stat guards on subsequent checks.','Recorded phase/wall/sampling evidence was reconstructed; no independent external profiler or simultaneous family-wide utility claim.']}
(WS/'final-independent-audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
(WS/'final-independent-source-bound-auditor.json').write_text(json.dumps(raw_audit,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2),flush=True)

