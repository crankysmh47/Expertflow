"""Registered wider utility collection, separate from the closed Q6 scope."""
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time
import uuid

from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.pipeline import atomic_json
from expertflow.compiler.plan import CandidatePlan, CandidateStatus, RuntimeSettings, seal_candidate
from expertflow.compiler.preflight import file_sha256
from expertflow.compiler.refinement import balanced_schedule
from expertflow.compiler.refinement import evaluate_pairs
from expertflow.compiler.schema import canonical_payload, canonical_sha256
from expertflow.compiler.stock_eligibility import EligibilityRegistry
from expertflow.compiler.stock_search import rank_screening, scheduling_space, screening_schedule
from expertflow.compiler.stock_validation import execute_stock_product, run_accepted_stock_plan
from scripts import benchmark_compiler_stock_utility as utility

PROTOCOL = 'wider-stock-utility-case-v1'
CASE_SECONDS = 14400
SEQUENCE_SECONDS = 57600
STATISTICAL_STOPS = {'NO-UTILITY-GAIN','VARIANCE-STOP','MANUAL-BASELINE-STOP',
                    'TUNING-COST-STOP','PRODUCT-VALIDATION-STOP'}


class ResourceStop(RuntimeError):
    """No further native call may start after a registered resource cap."""


def sources():
    paths = [*Path('src/expertflow/compiler').rglob('*.py'),
        *Path('src/expertflow/stock').rglob('*.py'), Path('src/expertflow/cli/main.py'),
        Path('src/expertflow/artifacts.py'),
        Path('scripts/benchmark_compiler_stock_utility.py'),
        Path('scripts/benchmark_compiler_stock_repeatability.py'),
        Path('scripts/benchmark_compiler_stock_coverage.py'),
        Path('docs/superpowers/specs/2026-10-05-stock-coverage.md'),
        Path('docs/superpowers/specs/2026-10-04-stock-configuration-discovery.md'),
        *Path('tests').glob('test_stock_wider*.py')]
    registration = Path('configs/compiler/stock-coverage-20261005.json')
    paths.append(registration)
    data = json.loads(registration.read_text())
    paths.extend(Path(p) for p in data['input_files'])
    review = Path('docs/evidence/stock-coverage-20261005/implementation-review.md')
    if review.is_file():
        paths.append(review)
    auditor=Path('docs/evidence/stock-coverage-20261005/independent_audit.py')
    if auditor.is_file():
        paths.append(auditor)
    return {str(p.resolve()):file_sha256(p) for p in sorted(set(paths)) if p.is_file()}


def scope_case(inputs, case, sequence, host):
    """Attest only the normalized identities and grid in this registration."""
    proof = EligibilityRegistry.with_builtins().attest(inputs,host,sequence['source_repository'])
    workload = replace(inputs.workload,threads=8,cuda_graphs='on')
    identities = replace(inputs.identities(inputs.stock),workload=workload,workload_sha256=canonical_sha256(workload))
    default = CandidatePlan(identities,RuntimeSettings(99,proof.get('baseline_cpu_moe',True)))
    candidates = scheduling_space(default,host).candidates
    expected = {'model_artifact':canonical_payload(inputs.model.identity),
        'model_ir_sha256':canonical_sha256(inputs.model),'workload_sha256':canonical_sha256(workload),
        'runtime_sha256':inputs.stock.sha256,'provider_id':proof['provider_id'],
        'fixed_settings':canonical_payload(default.settings),'default_id':default.candidate_id,
        'candidate_ids':[c.candidate_id for c in candidates],
        'screening_schedule':canonical_payload(screening_schedule(c.candidate_id for c in candidates))}
    if any(canonical_payload(case.get(k))!=v for k,v in expected.items()) or len(candidates)!=6:
        raise ValueError('live inputs/grid outside registered wider case')
    if utility.audit_defaults(sequence['source_repository'],host)!=sequence['default_source_proof']:
        raise ValueError('registered default source proof changed')
    return proof, default, {c.candidate_id:c for c in candidates}


class PacedRunner:
    """Journal before waiting; one guard spans utility, product and consumer."""
    def __init__(self, runner, report, capture):
        self.runner,self.report,self.capture=runner,report,capture

    def check_scope_budget(self):
        m=self.report['manifest']
        if sources()!=m['source_files'] or canonical_payload(self.capture())!=m['host_environment']:
            raise ValueError('wider source/host changed')
        now=time.monotonic_ns()
        if ((now-m['case_started_monotonic_ns'])/1e9+m['input_load_seconds']>=CASE_SECONDS or
                (now-m['sequence_started_monotonic_ns'])/1e9>=SEQUENCE_SECONDS):
            raise ResourceStop('registered case/sequence wall budget exhausted')
    def guard(self):
        self.check_scope_budget()
        if len(self.report['attempts'])>=107:
            raise ResourceStop('registered case native attempt budget exhausted')

    def run_once(self,*args,**kwargs):
        self.guard()
        report,m=self.report,self.report['manifest']
        root=Path(m['experiment_root']).resolve()
        output=Path(kwargs['output_dir']).resolve()
        output.relative_to(root)
        database=self.runner.store.path.resolve()
        if database not in (root/'utility.sqlite3',root/'product.sqlite3'):
            raise ValueError('wider database outside registered roots')
        if output.exists():
            raise ValueError('fresh native output required; no retries')
        attempt={'output_dir':str(output),'database':str(database),'stage':kwargs['stage'],
            'status':'waiting','native_started':False,'process_identity':None,
            'wait_seconds':30,'wait_started_monotonic_ns':time.monotonic_ns()}
        report['attempts'].append(attempt)
        atomic_json(root/'report.json',report)
        original_factory=getattr(self.runner,'process_factory',None)
        spawn_error=None
        try:
            time.sleep(30)
            attempt['wait_finished_monotonic_ns']=time.monotonic_ns()
            attempt['wait_elapsed_ns']=attempt['wait_finished_monotonic_ns']-attempt['wait_started_monotonic_ns']
            if attempt['wait_elapsed_ns']<30_000_000_000:
                raise ResourceStop('fixed prelaunch wait incomplete')
            # The reserved attempt itself is allowed to use slot107.
            if len(report['attempts'])>107:
                raise ResourceStop('registered native budget exhausted')
            m_now=time.monotonic_ns()
            if ((m_now-m['case_started_monotonic_ns'])/1e9+m['input_load_seconds']>=CASE_SECONDS or
                    (m_now-m['sequence_started_monotonic_ns'])/1e9>=SEQUENCE_SECONDS):
                raise ResourceStop('registered case/sequence wall budget exhausted')
            if sources()!=m['source_files'] or canonical_payload(self.capture())!=m['host_environment']:
                raise ValueError('wider source/host changed during fixed wait')
            self.runner.store.prime_model(args[1])
            args[2].verify_manifest_bindings()
            self.check_scope_budget()
            if original_factory is not None:
                def guarded_spawn(*argv,**options):
                    nonlocal spawn_error
                    try:
                        args[2].verify_manifest_bindings()
                        self.runner.store.prime_model(args[1])
                        self.check_scope_budget()
                    except (ResourceStop,ValueError) as error:
                        spawn_error=error
                        raise
                    return original_factory(*argv,**options)
                self.runner.process_factory=guarded_spawn
            kwargs.update(host_environment=m['host_environment'],
                experiment_context={'manifest_sha256':m['manifest_sha256']})
            attempt['status']='attempting'
            atomic_json(root/'report.json',report)
            outcome=self.runner.run_once(*args,**kwargs)
            attempt.update(status=outcome.status,measurement_id=outcome.measurement_id,reason=outcome.reason)
            if spawn_error is not None:
                raise spawn_error
            return outcome
        except Exception as error:
            attempt.update(status='exception',reason=str(error),exception_type=type(error).__name__)
            raise
        finally:
            if original_factory is not None:
                self.runner.process_factory=original_factory
            started=output/'run-start.json'
            if started.is_file():
                attempt.update(native_started=True,process_identity=json.loads(started.read_text()))
            attempt['finished_monotonic_ns']=time.monotonic_ns()
            atomic_json(root/'report.json',report)


def execute_case(inputs,case,sequence,*,runner_factory,capture,input_load_seconds=0):
    case_started=time.monotonic_ns()
    proof,default,candidates=scope_case(inputs,case,sequence,canonical_payload(capture()))
    root=Path(case['planned_root']).resolve()
    if root.exists():
        raise ValueError('fresh registered case required; no retries/resume')
    root.mkdir(parents=True)
    m={'protocol_version':PROTOCOL,'experiment_id':uuid.uuid4().hex,
        'source_commit':sequence['source_commit'],'source_files':sequence['source_files'],
        'host_environment':sequence['host_environment'],'eligibility':proof,'case':case,
        'inputs':canonical_payload({'model':inputs.model,'hardware':inputs.hardware,'stock':inputs.stock}),
        'source_repository':sequence['source_repository'],'default_source_proof':sequence['default_source_proof'],
        'default_controls':[8,'on'],'default_id':default.candidate_id,
        'candidates':{cid:canonical_payload(c) for cid,c in candidates.items()},
        'screening_schedule':case['screening_schedule'],'confirmation_schedule':canonical_payload(balanced_schedule()),
        'reference_processes':10,'maximum_native_processes':107,'experiment_root':str(root),
        'frozen_monotonic_ns':time.monotonic_ns(),'case_started_monotonic_ns':case_started,
        'sequence_started_monotonic_ns':sequence['sequence_started_monotonic_ns'],
        'sequence_manifest_sha256':sequence['manifest_sha256'],'input_load_seconds':input_load_seconds}
    m['manifest_sha256']=canonical_sha256(m)
    atomic_json(root/'frozen-manifest.json',m)
    report={'status':'RUNNING','manifest':m,'rows':[],'outcomes':[],'attempts':[],'statistics':None}
    atomic_json(root/'report.json',report)
    store=EvidenceStore(root/'utility.sqlite3')
    runner=PacedRunner(runner_factory(store),report,capture)
    owners=set()
    reference=None
    def collect(label,cid):
        nonlocal reference
        outcome=runner.run_once(candidates[cid],inputs.model,inputs.stock,output_dir=root/'raw'/label,
            measured=True,stage=utility.measurement_stage(label,m))
        report['outcomes'].append(canonical_payload(outcome))
        if outcome.status!='measured':
            report.update(status=outcome.status.upper().replace('_','-'),reason=outcome.reason)
            atomic_json(root/'report.json',report)
            return None
        row={**store.verify_measurement(outcome.measurement_id),'measurement_id':outcome.measurement_id,
            'label':label,'run_wall_seconds':utility.native_phase_seconds(store,outcome.measurement_id)}
        verified=utility.verify_row(row,candidates[cid],store,m,owners,reference)
        reference=reference or verified
        report['rows'].append(row)
        atomic_json(root/'report.json',report)
        return row
    try:
        refs=[]
        for n in range(10):
            row=collect(f'reference-{n:02}',default.candidate_id)
            if row is None:return report
            refs.append(row)
        cv=statistics.stdev(r['decode_tps'] for r in refs)*100/statistics.mean(r['decode_tps'] for r in refs)
        if cv>10:
            report.update(status='VARIANCE-STOP',reason='own-reference CV exceeds10%',reference_cv_pct=cv)
            return report
        screens={}
        for method in ('automatic','manual'):
            screens[method]=[]
            for block,order in enumerate(m['screening_schedule']):
                for cid in order:
                    row=collect(f'{method}-screen-{block:02}-{cid}',cid)
                    if row is None:return report
                    screens[method].append({**row,'block':block})
        auto=rank_screening(screens['automatic'],default.candidate_id,m['screening_schedule'])[0]['candidate_id']
        manual=utility.manual_winner(screens['manual'],candidates,default.candidate_id)
        report.update(automatic_id=auto,manual_id=manual)
        pairs={}
        for purpose,control in (('defaults',default.candidate_id),('manual',manual)):
            values={}
            for pair,order in enumerate(balanced_schedule()):
                for arm in order:
                    row=collect(f'{purpose}-pair-{pair:02}-{arm}',control if arm=='direct' else auto)
                    if row is None:return report
                    values[pair,arm]=row['decode_tps']
            pairs[purpose]=([values[i,'direct'] for i in range(10)],[values[i,'sealed'] for i in range(10)])
        stats=utility.evaluate_utility(*pairs['defaults'],*pairs['manual'],automatic_evaluations=18,manual_evaluations=18)
        report.update(statistics=canonical_payload(stats),status=stats['status'])
        if stats['status']=='PASS-STOCK-UTILITY':
            selected=replace(candidates[auto],status=CandidateStatus.MEASURED,
                measurement_ids=tuple(r['measurement_id'] for r in report['rows']
                    if r['label'].startswith('manual-pair-') and r['label'].endswith('-sealed')),
                validation=(('utility_confirmation',True),))
            atomic_json(root/'selected-plan.json',seal_candidate(selected,store,selected.identities,None))
            adjusted=replace(inputs,workload=selected.identities.workload)
            product_store=EvidenceStore(root/'product.sqlite3')
            product_runner=PacedRunner(runner_factory(product_store),report,capture)
            product=execute_stock_product(adjusted,root/'selected-plan.json',store,product_store,
                product_runner,root/'product',host_capture=capture)
            report['product_status']=product['status']
            report['product_native_processes']=len(product['outcomes'])
            if product['status']!='PASS-STOCK-FALLBACK':
                statistical=len(product['rows'])==20 and evaluate_pairs(product['rows'])['status']!='PASS-MEASUREMENT'
                report.update(status='PRODUCT-VALIDATION-STOP' if statistical else product['status'],
                    reason=product.get('reason'))
            else:
                status,consumer=run_accepted_stock_plan(root/'product/accepted/execution-plan.json',
                    root/'product/accepted/acceptance-receipt.json',adjusted,product_store,
                    root/'consumer',runner=product_runner,host_capture=capture)
                report['consumer']={'status':status,**consumer}
                report['status']='PASS-STOCK-UTILITY-PRODUCT' if status=='MEASURED-ACCEPTED-STOCK' else status
                if status!='MEASURED-ACCEPTED-STOCK':report['reason']=consumer.get('reason')
    except ResourceStop as error:
        report.update(status='RESOURCE-BUDGET-STOP',reason=str(error))
    except Exception as error:
        report.update(status='VALIDATION-STOP',reason=str(error),exception_type=type(error).__name__)
    finally:
        # execute_pairs persists after an outcome; a typed guard may interrupt
        # its first call before that point. Retain the already-frozen prefix.
        product_root=root/'product'
        if (product_root/'frozen-protocol.json').is_file() and not (product_root/'report.json').is_file():
            atomic_json(product_root/'report.json',{'status':'RUNNING',
                'frozen':json.loads((product_root/'frozen-protocol.json').read_text()),
                'rows':[],'outcomes':[],'live_validated_product':False,'optimization_gain_established':False})
        report['collection_finished_monotonic_ns']=time.monotonic_ns()
        report['collection_wall_seconds']=(report['collection_finished_monotonic_ns']-m['case_started_monotonic_ns'])/1e9
        atomic_json(root/'report.json',report)
    return report


def validate_case(report,inputs,sequence,*,host_environment):
    import sys
    from . import wider_audit
    from .readers import EvidenceReaders,reuse_readers
    if isinstance(EvidenceStore,EvidenceReaders):
        return wider_audit.reconstruct_case(report,inputs,sequence,host_environment=host_environment)
    with reuse_readers(sys.modules[__name__],wider_audit,utility,wider_audit.repeatability):
        return wider_audit.reconstruct_case(report,inputs,sequence,host_environment=host_environment)
