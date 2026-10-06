"""Execute/reconstruct only the four registered wider stock utility cases."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from expertflow.compiler.diagnostics import DiagnosticSampler
from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.pipeline import CompilationRequest, atomic_json, load_compiler_inputs
from expertflow.compiler.preflight import capture_host_environment, file_sha256
from expertflow.compiler.runner import ServerMeasurementRunner, WindowsGpuMemorySampler
from expertflow.compiler.schema import canonical_payload, canonical_sha256
from expertflow.stock.coverage import verify_registration
from expertflow.stock import wider
from scripts import benchmark_compiler_stock_utility as utility

REGISTRATION=Path('configs/compiler/stock-coverage-20261005.json')
REGISTERED_SHA='0376ac4d7c5e146bc594f5c22a4b1be803b446ced4c3d9d04a4cb0737af55603'
PROTOCOL='wider-stock-utility-sequence-v1'


def load_case_inputs(case,registration):
    root=Path(case['planned_root'])
    request=CompilationRequest(Path(case['descriptor']),Path(case['inventory']),Path(registration['hardware']),
        Path(case['workload']),Path(case['runtime_identity']),(),root/'utility.sqlite3',root)
    return load_compiler_inputs(request,live=True)


def require_committed_sources(source_files):
    root=Path.cwd().resolve()
    review=root/'docs/evidence/stock-coverage-20261005/implementation-review.md'
    if not review.is_file():
        raise ValueError('independent reviewed implementation record required before collection')
    for name,digest in source_files.items():
        relative=Path(name).resolve().relative_to(root).as_posix()
        blob=subprocess.check_output(['git','show','HEAD:'+relative])
        import hashlib
        if hashlib.sha256(blob).hexdigest()!=digest:
            raise ValueError('reviewed source must be committed before collection: '+relative)


def require_matching_package():
    import expertflow.stock
    actual=Path(expertflow.stock.__file__).resolve().parent
    declared=Path('src/expertflow/stock').resolve()
    files={p.relative_to(actual) for p in actual.rglob('*.py')}
    if files!={p.relative_to(declared) for p in declared.rglob('*.py')} or any(
            file_sha256(actual/p)!=file_sha256(declared/p) for p in files):
        raise ValueError('installed stock source differs from reviewed project source')
    import expertflow.artifacts
    import expertflow.cli.main
    import expertflow.compiler
    compiler=Path(expertflow.compiler.__file__).resolve().parent
    declared_compiler=Path('src/expertflow/compiler').resolve()
    compiler_files={p.relative_to(compiler) for p in compiler.rglob('*.py')}
    if compiler_files!={p.relative_to(declared_compiler) for p in declared_compiler.rglob('*.py')} or any(
            file_sha256(compiler/p)!=file_sha256(declared_compiler/p) for p in compiler_files):
        raise ValueError('installed compiler source differs from reviewed project source')
    for module,relative in ((expertflow.artifacts,'artifacts.py'),(expertflow.cli.main,'cli/main.py')):
        if file_sha256(Path(module.__file__))!=file_sha256(Path('src/expertflow')/relative):
            raise ValueError('installed executing source differs from reviewed project source: '+relative)


def sequence_root(registration):
    parents={Path(c['planned_root']).resolve().parent for c in registration['cases']}
    if len(parents)!=1:
        raise ValueError('all registered cases require one sequence root')
    return parents.pop()


def can_continue(result):
    return result['all_raw_records_valid'] is True and result['status'] in (
        wider.STATISTICAL_STOPS|{'PASS-STOCK-UTILITY-PRODUCT'})


def case_wall_seconds(report,*,finished_ns):
    return report['manifest']['input_load_seconds']+(finished_ns-report['manifest']['case_started_monotonic_ns'])/1e9


def apply_resource_gate(report,result,*,start,finish,sequence_started):
    wall=case_wall_seconds(report,finished_ns=finish)
    passed=wall<14400 and (finish-sequence_started)/1e9<57600
    raw_pass=result.get('utility_gain_established',False)
    return {**result,'reconstruction_started_monotonic_ns':start,'reconstruction_finished_monotonic_ns':finish,
        'reconstruction_wall_seconds':(finish-start)/1e9,'case_wall_seconds':wall,
        'resource_budget_pass':passed,'raw_utility_product_gate_pass':raw_pass,
        'utility_gain_established':raw_pass and passed}


def run_sequence(registration,*,loader=None,runner_factory=None,capture=None):
    loader,capture=loader or load_case_inputs,capture or capture_host_environment
    started=time.monotonic_ns()
    root=sequence_root(registration)
    if root.exists():
        raise ValueError('fresh sequence root required; no retries/resume')
    root.mkdir(parents=True)
    report={'status':'PREFLIGHT','cases':[{'case_id':c['case_id'],'status':'NOT-RUN'} for c in registration['cases']],
        'attempts':0,'native_processes':0,'sequence_started_monotonic_ns':started}
    atomic_json(root/'report.json',report)
    sampler=None
    try:
        host=canonical_payload(capture())
        if host!=registration['host_environment']:
            raise ValueError('registered host environment changed')
        source_files=wider.sources()
        require_committed_sources(source_files)
        default_proof=utility.audit_defaults(registration['source_repository'],host)
        if default_proof!=registration['default_source_proof']:
            raise ValueError('registered defaults proof changed')
        inputs,load_seconds={},{}
        for case in registration['cases']:
            before=time.monotonic_ns()
            inputs[case['case_id']]=loader(case,registration)
            wider.scope_case(inputs[case['case_id']],case,{'source_repository':registration['source_repository'],
                'default_source_proof':default_proof},host)
            load_seconds[case['case_id']]=(time.monotonic_ns()-before)/1e9
            if load_seconds[case['case_id']]>=wider.CASE_SECONDS or (time.monotonic_ns()-started)/1e9>=wider.SEQUENCE_SECONDS:
                raise wider.ResourceStop('registered preflight/load wall budget exhausted')
            report['last_loaded_case']=case['case_id']
            atomic_json(root/'report.json',report)
        m={'protocol_version':PROTOCOL,'registration':registration,'registration_sha256':registration['registration_sha256'],
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_files':source_files,'host_environment':host,'source_repository':registration['source_repository'],
            'default_source_proof':default_proof,'sequence_started_monotonic_ns':started,
            'frozen_monotonic_ns':time.monotonic_ns(),'experiment_root':str(root),
            'maximum_native_processes':428,'case_wall_cap_seconds':14400,'sequence_wall_cap_seconds':57600,
            'case_inputs':{key:canonical_payload(value) for key,value in inputs.items()},
            'case_input_load_seconds':load_seconds}
        if wider.sources()!=source_files or canonical_payload(capture())!=host:
            raise ValueError('source/host changed during preflight freeze')
        m['manifest_sha256']=canonical_sha256(m)
        atomic_json(root/'frozen-manifest.json',m)
        report.update(status='RUNNING',manifest=m)
        atomic_json(root/'report.json',report)
        if runner_factory is None:
            first=inputs[registration['cases'][0]['case_id']]
            sampler=WindowsGpuMemorySampler(first.hardware.gpu_uuid)
            diagnostic=DiagnosticSampler(sampler)
            runner_factory=lambda store:ServerMeasurementRunner(store,memory_sampler=diagnostic,sample_interval=0.2)
        for index,case in enumerate(registration['cases']):
            if (time.monotonic_ns()-started)/1e9>=57600:
                raise wider.ResourceStop('registered sequence wall budget exhausted')
            case_report=wider.execute_case(inputs[case['case_id']],case,m,runner_factory=runner_factory,
                capture=capture,input_load_seconds=load_seconds[case['case_id']])
            before=time.monotonic_ns()
            result=wider.validate_case(case_report,inputs[case['case_id']],m,host_environment=canonical_payload(capture()))
            result=apply_resource_gate(case_report,result,start=before,finish=time.monotonic_ns(),sequence_started=started)
            report['cases'][index]={'case_id':case['case_id'],**result}
            report['attempts']=sum(c.get('attempts',0) for c in report['cases'])
            report['native_processes']=sum(c.get('native_processes',0) for c in report['cases'])
            if report['attempts']>428:
                raise wider.ResourceStop('registered sequence attempt budget exceeded')
            if not can_continue(result) or not result['resource_budget_pass']:
                report.update(status='SEQUENCE-STOP',stopped_case=case['case_id'])
                break
            atomic_json(root/'report.json',report)
        else:
            report['status']='COMPLETE-STOCK-COVERAGE'
    except wider.ResourceStop as error:
        report.update(status='RESOURCE-BUDGET-STOP',reason=str(error))
    except Exception as error:
        report.update(status='IDENTITY-STOP',reason=str(error),exception_type=type(error).__name__)
    finally:
        if sampler:sampler.close()
        # Retain honest attempt/start counts even when raw reconstruction invalidates a case.
        for index,case in enumerate(registration['cases']):
            path=Path(case['planned_root'])/'report.json'
            if path.is_file() and report['cases'][index]['status']=='NOT-RUN':
                case_report=json.loads(path.read_text())
                report['cases'][index]={'case_id':case['case_id'],'status':case_report['status'],
                    'evidence_verified':False,'invalidation_reason':report.get('reason'),
                    'attempts':len(case_report.get('attempts',[])),
                    'native_processes':len(list(path.parent.rglob('run-start.json')))}
        report['attempts']=sum(c.get('attempts',0) for c in report['cases'])
        report['native_processes']=sum(c.get('native_processes',0) for c in report['cases'])
        report['sequence_finished_monotonic_ns']=time.monotonic_ns()
        report['sequence_wall_seconds']=(report['sequence_finished_monotonic_ns']-started)/1e9
        atomic_json(root/'report.json',report)
    return report


def validate_sequence(report,*,loader=None,capture=None):
    loader,capture=loader or load_case_inputs,capture or capture_host_environment
    m=report['manifest']
    registration=m['registration']
    verify_registration(registration,Path.cwd())
    payload=dict(m)
    claimed=payload.pop('manifest_sha256')
    root=sequence_root(registration)
    host=canonical_payload(capture())
    if (claimed!=canonical_sha256(payload) or m!=json.loads((root/'frozen-manifest.json').read_text()) or
            m['protocol_version']!=PROTOCOL or m['registration_sha256']!=REGISTERED_SHA or
            m['source_files']!=wider.sources() or m['host_environment']!=host or
            m['maximum_native_processes']!=428 or m['case_wall_cap_seconds']!=14400 or
            m['sequence_wall_cap_seconds']!=57600 or m['experiment_root']!=str(root)):
        raise ValueError('sequence freeze/source/host/protocol mismatch')
    if (report['sequence_started_monotonic_ns']!=m['sequence_started_monotonic_ns'] or
            not m['sequence_started_monotonic_ns']<=m['frozen_monotonic_ns']<=report['sequence_finished_monotonic_ns'] or
            report['sequence_wall_seconds']!=(report['sequence_finished_monotonic_ns']-m['sequence_started_monotonic_ns'])/1e9):
        raise ValueError('sequence wall cost/clock order mismatch')
    results=[]
    seen_stop=False
    owners=set()
    previous=m['frozen_monotonic_ns']
    for index,case in enumerate(registration['cases']):
        saved=report['cases'][index]
        if saved['case_id']!=case['case_id']:
            raise ValueError('sequence case order changed')
        path=Path(case['planned_root'])/'report.json'
        if not path.is_file():
            if saved['status']!='NOT-RUN' or Path(case['planned_root']).exists():
                raise ValueError('case journal/result missing')
            if not seen_stop and report['status']=='COMPLETE-STOCK-COVERAGE':
                raise ValueError('complete sequence missing case')
            results.append(saved)
            continue
        if seen_stop:
            raise ValueError('sequence continued after mandatory stop')
        inputs=loader(case,registration)
        if canonical_payload(inputs)!=m['case_inputs'][case['case_id']]:
            # Dynamic hardware preflight telemetry belongs to provenance, not identities.
            frozen=dict(m['case_inputs'][case['case_id']])
            current=dict(canonical_payload(inputs))
            frozen.pop('provenance',None)
            current.pop('provenance',None)
            if frozen!=current:
                raise ValueError('live case inputs changed from sequence freeze')
        case_report=json.loads(path.read_text())
        if case_report['manifest']['case_started_monotonic_ns']<previous:
            raise ValueError('case collection overlaps previous reconstruction/freeze')
        for start_path in path.parent.rglob('run-start.json'):
            identity=json.loads(start_path.read_text())
            owner=(identity['pid'],identity['creation_time_100ns'],identity['creation_source'])
            if owner in owners:
                raise ValueError('native owner reused across cases')
            owners.add(owner)
        if case_report['manifest']['input_load_seconds']!=m['case_input_load_seconds'][case['case_id']]:
            raise ValueError('case loading cost changed')
        result=wider.validate_case(case_report,inputs,m,host_environment=host)
        start,finish=saved['reconstruction_started_monotonic_ns'],saved['reconstruction_finished_monotonic_ns']
        result=apply_resource_gate(case_report,result,start=start,finish=finish,sequence_started=m['sequence_started_monotonic_ns'])
        for key,value in result.items():
            if canonical_payload(saved.get(key))!=canonical_payload(value):
                raise ValueError('sequence claimed case result/statistics/cost mismatch: '+key)
        if (not case_report['collection_finished_monotonic_ns']<=start<=finish<=report['sequence_finished_monotonic_ns'] or
                saved['reconstruction_wall_seconds']!=(finish-start)/1e9 or
                saved['case_wall_seconds']!=case_wall_seconds(case_report,finished_ns=finish) or
                saved['resource_budget_pass'] is not (saved['case_wall_seconds']<14400 and
                    (finish-m['sequence_started_monotonic_ns'])/1e9<57600)):
            raise ValueError('case reconstruction/resource wall cost mismatch')
        result.update(case_id=case['case_id'],reconstruction_wall_seconds=saved['reconstruction_wall_seconds'],
            case_wall_seconds=saved['case_wall_seconds'],resource_budget_pass=saved['resource_budget_pass'],
            reconstruction_started_monotonic_ns=start,reconstruction_finished_monotonic_ns=finish)
        seen_stop=not can_continue(result) or not result['resource_budget_pass']
        previous=finish
        results.append(result)
    attempts=sum(c.get('attempts',0) for c in results)
    native=sum(c.get('native_processes',0) for c in results)
    if attempts!=report['attempts'] or native!=report['native_processes'] or attempts>428:
        raise ValueError('sequence process budget/count mismatch')
    if len(list(root.rglob('run-start.json')))!=native:
        raise ValueError('unaccounted sequence native starts')
    expected='SEQUENCE-STOP' if seen_stop else 'COMPLETE-STOCK-COVERAGE'
    if report['status']!=expected:
        raise ValueError('sequence terminal verdict differs from reconstruction')
    return {'status':expected,'cases':results,'attempts':attempts,'native_processes':native,
        'additional_native_calls':0,'family_wide_gain_established':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--action',choices=('run','validate'),required=True)
    parser.add_argument('--registration',type=Path,default=REGISTRATION)
    args=parser.parse_args(argv)
    require_matching_package()
    registration=verify_registration(json.loads(args.registration.read_text()),Path.cwd())
    if registration['registration_sha256']!=REGISTERED_SHA:
        raise ValueError('requires the exact registered four-case study')
    root=sequence_root(registration)
    if args.action=='run':
        result=run_sequence(registration)
    else:
        result=validate_sequence(json.loads((root/'report.json').read_text()))
    summary={k:result[k] for k in ('status','cases','attempts','native_processes') if k in result}
    summary.update(report=str(root/'report.json'),reason=result.get('reason'),
        family_wide_gain_established=False)
    print(json.dumps(summary))
    return 0 if result['status']=='COMPLETE-STOCK-COVERAGE' else 2


if __name__=='__main__':
    raise SystemExit(main())
