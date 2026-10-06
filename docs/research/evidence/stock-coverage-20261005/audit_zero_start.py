import sys,json,hashlib,subprocess,zipfile
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import replace
sys.path.insert(0,str(Path.cwd()))
from expertflow.compiler.schema import canonical_payload,canonical_sha256,WorkloadIR
from expertflow.compiler.pipeline import inspect_model
from expertflow.compiler.reference import load_reference_workload
from expertflow.compiler.stock_discovery import _snapshot_inputs
from expertflow.compiler.plan import CandidatePlan,RuntimeSettings
from expertflow.compiler.stock_search import scheduling_space,screening_schedule
repo=Path.cwd();ws=repo/'.superpowers/sdd/2026-10-05-wider-stock-collector'
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
commit='337615a226b8c0746c2cb268ac90ac180599371c'
root=Path('C:/models/expertflow/runs/compiler-stock-coverage-20261005')
report=read(root/'report.json');supervisor=read(ws/'native-job.json');log=read(ws/'native-collection.log')
assert sorted(str(p.relative_to(root)) for p in root.rglob('*'))==['report.json']
assert report['status']==log['status']=='IDENTITY-STOP'
assert report['attempts']==report['native_processes']==log['attempts']==log['native_processes']==0
assert (report['sequence_finished_monotonic_ns']-report['sequence_started_monotonic_ns'])/1e9==report['sequence_wall_seconds']==23.391
assert all(c['status']=='NOT-RUN' for c in report['cases']) and report['cases']==log['cases']
assert supervisor['status']=='COMPLETED' and supervisor['exit_code']==2
start=datetime.fromisoformat(supervisor['started_at_utc']);end=datetime.fromisoformat(supervisor['finished_at_utc'])
assert (end-start).total_seconds()>=report['sequence_wall_seconds']
snapshot=read('docs/evidence/stock-coverage-20261005/source-snapshot.json')
archive=Path('docs/evidence/stock-coverage-20261005/measured-source-snapshot.zip')
assert snapshot['source_commit']==commit and sha(archive.read_bytes())==snapshot['archive_sha256']
assert datetime.fromisoformat(snapshot['created_at_utc'])<start
line_endings=[]
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))==len(snapshot['entries'])
    assert set(z.namelist())=={e['member'] for e in snapshot['entries']}
    assert {e['original_path']:e['sha256'] for e in snapshot['entries']}==snapshot['source_files']
    for entry in snapshot['entries']:
        data=z.read(entry['member'])
        assert sha(data)==entry['sha256'] and len(data)==entry['size_bytes'],entry['member']
        blob=subprocess.check_output(['git','show',commit+':'+entry['member']])
        if blob!=data:
            attrs=subprocess.check_output(['git','check-attr','--source='+commit,'text','eol','--',entry['member']],text=True)
            assert ': text: set' in attrs and ': eol: lf' in attrs and data.replace(b'\r\n',b'\n')==blob,entry['member']
            line_endings.append(entry['member'])
    archived_registration=z.read('configs/compiler/stock-coverage-20261005.json')
registration_path=Path('configs/compiler/stock-coverage-20261005.json');registration=read(registration_path)
assert registration_path.read_bytes()==archived_registration
assert registration_path.read_bytes()==subprocess.check_output(['git','show','5da6786:configs/compiler/stock-coverage-20261005.json'])
payload=dict(registration);digest=payload.pop('registration_sha256')
assert canonical_sha256(payload)==digest=='0376ac4d7c5e146bc594f5c22a4b1be803b446ced4c3d9d04a4cb0737af55603'
for name,digest in registration['input_files'].items():assert sha(Path(name).read_bytes())==digest,name
assert all(not Path(c['planned_root']).exists() for c in registration['cases'])
historical=read('docs/evidence/stock-repeatability-20261004/source-integrity.json')
for name,digest in {**historical['frozen_files'],**historical['historical_files']}.items():assert sha(Path(name).read_bytes())==digest,name
oldroot=Path('C:/models/expertflow/runs/compiler-stock-repeatability-20261004')
oldreport=read(oldroot/'report.json');oldstarts=list(oldroot.rglob('run-start.json'))
assert len(oldstarts)==len(oldreport['attempts'])==148 and all(a['status']=='measured' for a in oldreport['attempts'])
assert sha((oldroot/'report.json').read_bytes())=='464defd327d43c4f510f3768266d1dacaefc77d25c239afefef5cc4226c1d0c5'
for a in oldreport['attempts']:
    p=Path(a['output_dir'])/'run-start.json'
    assert p in oldstarts and read(p)==a['process_identity']
reference=read('docs/evidence/stock-repeatability-20261004/main-frozen-manifest.json')
diagnosis=read(ws/'preflight-scope-diagnosis.json');normalization=read(ws/'path-normalization-proof.json')
path_checks=[]
for case in registration['cases'][2:]:
    model=inspect_model(Path(case['descriptor']),Path(case['inventory']))
    raw_model=replace(model,identity=replace(model.identity,path=str(Path(model.identity.path).resolve())))
    normalized=replace(raw_model,identity=replace(raw_model.identity,path=Path(raw_model.identity.path).resolve().as_posix()))
    assert Path(raw_model.identity.path).samefile(normalized.identity.path)
    assert Path(model.identity.path).stat().st_size==case['model_artifact']['size_bytes']
    assert canonical_payload(normalized)==canonical_payload(model)
    assert canonical_sha256(normalized)==case['model_ir_sha256']
    workload=replace(WorkloadIR.from_reference(load_reference_workload(repo,Path(case['workload']))),threads=8,cuda_graphs='on')
    inp=_snapshot_inputs({**reference['main_inputs'],'model':canonical_payload(normalized)},workload)
    default=CandidatePlan(inp.identities(inp.stock),RuntimeSettings(99,False))
    candidates=scheduling_space(default,registration['host_environment']).candidates
    assert default.candidate_id==case['default_id']
    assert [c.candidate_id for c in candidates]==case['candidate_ids']
    assert canonical_payload(screening_schedule(c.candidate_id for c in candidates))==case['screening_schedule']
    if case['case_id']==diagnosis['case_id']:
        assert canonical_payload(raw_model.identity)==diagnosis['differences']['model_artifact']['live']
        assert canonical_sha256(raw_model)==diagnosis['differences']['model_ir_sha256']['live']
        assert canonical_sha256(normalized)==normalization['normalized_model_ir_sha256']
    path_checks.append({'case_id':case['case_id'],'same_physical_file':True,'file_size':Path(model.identity.path).stat().st_size,'registered_grid_recomputed':True})
result={'status':'PASS-ZERO-START-PREFLIGHT-EVIDENCE-AUDIT','audited_at_utc':datetime.now(timezone.utc).isoformat(),'material_discrepancies':[],
 'report_sha256':sha((root/'report.json').read_bytes()),'report_inventory':['report.json'],'status_observed':report['status'],'attempts':0,'native_starts':0,
 'sequence_wall_seconds':report['sequence_wall_seconds'],'supervisor_wall_seconds':(end-start).total_seconds(),'supervisor_exit_code':supervisor['exit_code'],
 'source_commit':commit,'source_archive_sha256':snapshot['archive_sha256'],'snapshot_files_verified':len(snapshot['entries']),'declared_line_ending_equivalences':line_endings,
 'registration_canonical_sha256':registration['registration_sha256'],'registration_raw_sha256':sha(registration_path.read_bytes()),'registration_input_files_verified':len(registration['input_files']),
 'original_frozen_files_verified':len(historical['frozen_files']),'original_history_pins_verified':len(historical['historical_files']),'original_native_starts':len(oldstarts),'original_report_sha256':sha((oldroot/'report.json').read_bytes()),
 'granite_path_checks':path_checks,'fresh_full_weight_hash_read':False,'audit_native_calls':0,
 'limitations':['No sequence freeze or case manifests were emitted; source provenance is checked through the prelaunch source archive and supervisor artifacts.','Weight digest equality is verified against saved metadata/diagnostic records; physical file and size were checked without reading full weights.','Post-stop working-tree edits are intentionally not compared with the earlier archived snapshot.']}
(ws/'zero-start-preflight-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
