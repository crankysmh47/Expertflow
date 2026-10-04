"""One bounded, paced repeatability study; historical sources/verdicts stay fixed."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import runpy
import subprocess
import time
import uuid

from expertflow.compiler.diagnostics import DiagnosticSampler
from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.pipeline import CompilationRequest, atomic_json, load_compiler_inputs
from expertflow.compiler.plan import load_execution_plan
from expertflow.compiler.preflight import capture_host_environment, file_sha256
from expertflow.compiler.refinement import evaluate_pairs, execute_pairs
from expertflow.compiler.runner import ServerMeasurementRunner, WindowsGpuMemorySampler
from expertflow.compiler.schema import canonical_payload, canonical_sha256
from expertflow.compiler.stock_validation import load_validated_stock_plan, publish_stock_product, reconstruct_product, run_accepted_stock_plan
if __package__:
    from scripts import benchmark_compiler_stock_utility as utility
else:
    import benchmark_compiler_stock_utility as utility

SPEC = Path('docs/superpowers/specs/2026-10-04-stock-repeatability.md')
PROTOCOL = 'paced-stock-repeatability-v1'
PRIOR_REPORT_SHA256 = '1475cb9830b4e572004b5a7b4bd60fe0c2e7e7b3877391a5f3a6105bd7b8c128'
WAIT_SECONDS = 30
MAXIMUM_NATIVE_PROCESSES = 148


def snapshot(inputs):
    return canonical_payload({name: getattr(inputs, name) for name in ('model', 'hardware', 'stock', 'workload')})


def sources():
    result = utility.sources()
    for path in (SPEC, Path(__file__), Path('docs/evidence/stock-utility-20261004/independent_audit.py')):
        result[str(path.resolve())] = file_sha256(path)
    return result


def verify_prerequisite(inputs, transfer_inputs, source_plan, source_store, prior_path, repository, host):
    if file_sha256(prior_path) != PRIOR_REPORT_SHA256:
        raise ValueError('requires the original closed utility report bytes')
    prior = json.loads(Path(prior_path).read_text())
    result = utility.validate_result(prior, source_store, inputs, source_repository=repository, host_environment=host)
    if result['status'] != 'PRODUCT-VALIDATION-STOP' or result['gain']['geometric_change_pct'] < 5:
        raise ValueError('original utility prerequisite did not qualify')
    scopes = [Path(utility.audit_protocol_scope(value)['registered_workload']).as_posix() for value in (inputs, transfer_inputs)]
    if scopes != ['configs/compiler/gemma4-q6-single-request.json', 'configs/compiler/gemma4-q6-utility-transfer.json']:
        raise ValueError('requires the registered main and untouched transfer workloads')
    plan = load_execution_plan(source_plan, identities=inputs.identities(inputs.stock), store=source_store)
    confirmation = tuple(row['measurement_id'] for row in prior['rows']
                         if row['label'].startswith('manual-pair-') and row['label'].endswith('-sealed'))
    if (plan.candidate.candidate_id != prior['automatic_id'] or plan.candidate.measurement_ids != confirmation
            or inputs.workload.threads != 12 or plan.candidate.settings.cuda_graphs != 'on'):
        raise ValueError('repeatability input differs from the fixed utility selection')
    auditor = runpy.run_path('docs/evidence/stock-utility-20261004/independent_audit.py')['audit']
    audit = auditor(Path(prior_path), source_store.path)
    if audit['actual_native_processes'] != 106 or audit['utility_verdict'] != 'PASS-STOCK-UTILITY':
        raise ValueError('original raw utility/product prerequisite is incomplete')
    return {'utility_verdict': audit['utility_verdict'], 'original_terminal_status': audit['terminal_status'],
            'source_plan_sha256': plan.plan_sha256, 'original_report_sha256': PRIOR_REPORT_SHA256}


class GuardedRunner:
    """Keep the outer freeze, pacing and attempts across every nested phase."""
    def __init__(self, runner, report, capture, source_capture, *, sleep_fn=None, clock=None):
        self.runner, self.report, self.capture, self.source_capture = runner, report, capture, source_capture
        self.sleep_fn = sleep_fn or time.sleep
        self.clock = clock or time.monotonic_ns

    def guard(self):
        manifest = self.report['manifest']
        if canonical_payload(self.capture()) != manifest['host_environment'] or self.source_capture() != manifest['source_files']:
            raise ValueError('repeatability source/host changed')

    def run_once(self, *args, **kwargs):
        report, manifest = self.report, self.report['manifest']
        self.guard()
        root, output = Path(manifest['experiment_root']).resolve(), Path(kwargs['output_dir']).resolve()
        output.relative_to(root)
        if len(report['attempts']) >= MAXIMUM_NATIVE_PROCESSES:
            raise ValueError('repeatability native attempt budget exhausted')
        database = self.runner.store.path.resolve()
        database.relative_to(root)
        attempt = {'output_dir': str(output), 'database': str(database),
                   'stage': kwargs['stage'], 'status': 'waiting', 'native_started': False, 'process_identity': None,
                   'wait_seconds': WAIT_SECONDS, 'wait_started_monotonic_ns': self.clock()}
        report['attempts'].append(attempt)
        atomic_json(root/'report.json', report)
        try:
            self.sleep_fn(WAIT_SECONDS)
            attempt['wait_finished_monotonic_ns'] = self.clock()
            attempt['wait_elapsed_ns'] = attempt['wait_finished_monotonic_ns'] - attempt['wait_started_monotonic_ns']
            self.guard()
            if attempt['wait_elapsed_ns'] < WAIT_SECONDS * 1e9:
                raise ValueError('repeatability fixed wait was not completed')
            kwargs['host_environment'] = manifest['host_environment']
            kwargs.setdefault('experiment_context', {'repeatability_manifest_sha256': manifest['manifest_sha256']})
            attempt.update(status='attempting', experiment_context=canonical_payload(kwargs['experiment_context']))
            atomic_json(root/'report.json', report)
            outcome = self.runner.run_once(*args, **kwargs)
            attempt.update(status=outcome.status, measurement_id=outcome.measurement_id)
            print(json.dumps({'process': len(report['attempts']), 'stage': kwargs['stage'], 'status': outcome.status}), flush=True)
            return outcome
        except Exception as error:
            attempt.update(status='exception', reason=str(error), exception_type=type(error).__name__)
            raise
        finally:
            started = output/'run-start.json'
            if started.is_file():
                attempt['native_started'] = True
                try:
                    attempt['process_identity'] = json.loads(started.read_text())
                except (ValueError, OSError) as error:
                    attempt['observation_error'] = str(error)
            atomic_json(root/'report.json', report)


def tracked_sources(source_plan, source_store, prior_path):
    return {**sources(), **{str(Path(path).resolve()): file_sha256(path)
                           for path in (source_plan, source_store.path, prior_path)}}


def run_followup(inputs, transfer_inputs, source_plan, source_store, prior_path, output_dir,
                 *, source_repository, runner_factory, host_capture=None):
    capture = host_capture or capture_host_environment
    host = canonical_payload(capture())
    prerequisite = verify_prerequisite(inputs, transfer_inputs, source_plan, source_store, prior_path, source_repository, host)
    source = load_execution_plan(source_plan, identities=inputs.identities(inputs.stock), store=source_store)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = {'protocol_version': PROTOCOL, 'protocol_sha256': file_sha256(SPEC), 'experiment_id': uuid.uuid4().hex,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_files': tracked_sources(source_plan, source_store, prior_path), 'host_environment': host,
        'source_repository': str(Path(source_repository).resolve()), 'source_plan_sha256': source.plan_sha256,
        'main_inputs': snapshot(inputs), 'transfer_inputs': snapshot(transfer_inputs), 'prerequisite': prerequisite,
        'experiment_root': str(root), 'frozen_monotonic_ns': time.monotonic_ns(),
        'maximum_native_processes': MAXIMUM_NATIVE_PROCESSES, 'wait_seconds': WAIT_SECONDS,
        'main_blocks': 2, 'native_processes_per_block': 20, 'consumer_processes': 1, 'transfer_maximum': 107,
        'diagnostics': 'DiagnosticSampler at native memory interval 0.2s; optional sensors'}
    manifest['manifest_sha256'] = canonical_sha256(manifest)
    atomic_json(root/'frozen-manifest.json', manifest)
    report = {'status': 'RUNNING', 'manifest': manifest, 'attempts': [], 'blocks': []}
    atomic_json(root/'report.json', report)
    source_capture = lambda: tracked_sources(source_plan, source_store, prior_path)
    def runner(store):
        return GuardedRunner(runner_factory(store), report, capture, source_capture)
    try:
        for name in ('block-a', 'block-b'):
            store = EvidenceStore(root/f'{name}.sqlite3')
            block = execute_pairs(inputs, source_plan, source_store, store, runner(store), root/name,
                validation_protocol='paired-stock-product-v1', host_capture=capture)
            path = root/name/'report.json'
            report['blocks'].append({'name': name, 'status': block['status'], 'report_sha256': file_sha256(path)})
            if block['status'] != 'PASS-MEASUREMENT':
                report.update(status='REPEATABILITY-STOP', reason=f'{name}: {block["status"]}')
                atomic_json(root/'report.json', report)
                return report
        publish_stock_product(block, store, root/'block-b', host_environment=capture())
        accepted = root/'block-b/accepted'
        status, consumer = run_accepted_stock_plan(accepted/'execution-plan.json', accepted/'acceptance-receipt.json',
            inputs, store, root/'consumer', runner=runner(store), host_capture=capture)
        report['consumer'] = {'status': status, **consumer}
        if status != 'MEASURED-ACCEPTED-STOCK':
            report.update(status='CONSUMER-VALIDATION-STOP', reason=consumer.get('reason'))
        else:
            transfer_store = EvidenceStore(root/'transfer/utility.sqlite3')
            transfer = utility.execute_utility(transfer_inputs, transfer_store, runner(transfer_store), root/'transfer/utility',
                source_repository=source_repository, host_capture=capture, product_runner_factory=runner)
            report['transfer'] = {'status': transfer['status'], 'report_sha256': file_sha256(root/'transfer/utility/report.json')}
            report.update(status='PASS-STOCK-REPEATABILITY-TRANSFER' if transfer['status'] == 'PASS-STOCK-UTILITY-PRODUCT'
                          else 'TRANSFER-VALIDATION-STOP', reason=transfer.get('reason'))
    except Exception as error:
        report.update(status='VALIDATION-STOP', reason=str(error), exception_type=type(error).__name__)
    atomic_json(root/'report.json', report)
    return report


def audit_attempts(report):
    manifest, owners = report['manifest'], set()
    root = Path(manifest['experiment_root']).resolve()
    if len(report['attempts']) > MAXIMUM_NATIVE_PROCESSES:
        raise ValueError('repeatability attempt budget exceeded')
    for attempt in report['attempts']:
        output, database = Path(attempt['output_dir']).resolve(), Path(attempt['database']).resolve()
        output.relative_to(root)
        database.relative_to(root)
        started = output/'run-start.json'
        if attempt['native_started'] != started.is_file():
            raise ValueError('retained native start mismatch')
        if started.is_file():
            observed = json.loads(started.read_text())
            owner = (observed['pid'], observed['creation_time_100ns'], observed['run_id'])
            if (owner in owners or attempt['process_identity'] != observed
                    or observed['started_monotonic_ns'] < attempt.get('wait_finished_monotonic_ns',manifest['frozen_monotonic_ns'])):
                raise ValueError('retained native owner/wait mismatch')
            owners.add(owner)
        if attempt['status'] != 'measured':
            continue
        if (attempt['wait_seconds'] != WAIT_SECONDS or attempt['wait_elapsed_ns'] < WAIT_SECONDS * 1e9
                or attempt['wait_elapsed_ns'] != attempt['wait_finished_monotonic_ns'] - attempt['wait_started_monotonic_ns']
                or attempt['wait_started_monotonic_ns'] < manifest['frozen_monotonic_ns']):
            raise ValueError('retained fixed wait evidence mismatch')
        store = EvidenceStore(database)
        record = store.measurement(attempt['measurement_id'])
        verified = store.verify_measurement(attempt['measurement_id'])
        artifacts = {a.role: Path(a.identity.path) for a in record.artifacts}
        if any(path.parent.resolve() != output for path in artifacts.values()) or record.stage != attempt['stage']:
            raise ValueError('retained native root/stage mismatch')
        start, launch = (json.loads(artifacts[name].read_text()) for name in ('run-start', 'launch'))
        if (attempt['process_identity'] != start or attempt['native_started'] is not True
                or start['started_monotonic_ns'] < attempt['wait_finished_monotonic_ns']):
            raise ValueError('retained native owner/wait mismatch')
        if output.is_relative_to(root/'transfer/utility'):
            nested = json.loads((root/'transfer/utility/frozen-manifest.json').read_text())
            expected_context = {'manifest_sha256': nested['manifest_sha256']}
        else:
            expected_context = {'repeatability_manifest_sha256': manifest['manifest_sha256']}
        if (launch.get('experiment_context') != attempt['experiment_context']
                or launch.get('experiment_context') != expected_context
                or launch.get('host_environment') != manifest['host_environment']
                or verified['measured'] is not True or verified['exit_code'] != 0
                or any(verified['validations'].get(name) is not True for name in ('exact_tokens','memory','cleanup'))):
            raise ValueError('retained native context/correctness mismatch')
    if len(list(root.rglob('run-start.json'))) != sum(attempt['native_started'] for attempt in report['attempts']):
        raise ValueError('unaccounted native process starts')
    return len(owners)


def validate_followup(report, inputs, transfer_inputs, source_plan, source_store, prior_path,
                      *, source_repository, host_environment):
    manifest = report['manifest']
    payload = dict(manifest)
    digest = payload.pop('manifest_sha256')
    if (digest != canonical_sha256(payload) or manifest['protocol_version'] != PROTOCOL
            or manifest['protocol_sha256'] != file_sha256(SPEC)
            or manifest['source_files'] != tracked_sources(source_plan, source_store, prior_path)
            or manifest['host_environment'] != canonical_payload(host_environment)
            or manifest['main_inputs'] != snapshot(inputs) or manifest['transfer_inputs'] != snapshot(transfer_inputs)
            or manifest['source_repository'] != str(Path(source_repository).resolve())
            or any(manifest[key] != value for key, value in {'maximum_native_processes':148,'wait_seconds':30,
                'main_blocks':2,'native_processes_per_block':20,'consumer_processes':1,'transfer_maximum':107}.items())):
        raise ValueError('repeatability frozen inputs/source/host/protocol mismatch')
    prerequisite = verify_prerequisite(inputs, transfer_inputs, source_plan, source_store, prior_path, source_repository, host_environment)
    if prerequisite != manifest['prerequisite']:
        raise ValueError('repeatability prerequisite mismatch')
    root = Path(manifest['experiment_root'])
    native_processes = audit_attempts(report)
    for index, summary in enumerate(report['blocks']):
        name = ('block-a','block-b')[index]
        path = root/name/'report.json'
        block = json.loads(path.read_text())
        if summary != {'name':name,'status':block['status'],'report_sha256':file_sha256(path)}:
            raise ValueError('repeatability block summary mismatch')
        if block['status'] == 'PASS-MEASUREMENT':
            source, _, _ = reconstruct_product(block, EvidenceStore(root/f'{name}.sqlite3'), host_environment=host_environment)
            if source.plan_sha256 != manifest['source_plan_sha256']:
                raise ValueError('repeatability block source plan mismatch')
        elif len(block['rows']) == 20:
            store = EvidenceStore(root/f'{name}.sqlite3')
            rows = [{**store.verify_measurement(row['measurement_id']), 'pair':row['pair'], 'arm':row['arm']}
                    for row in block['rows']]
            if evaluate_pairs(rows)['status'] != block['status']:
                raise ValueError('repeatability negative block statistics mismatch')
            if report['status'] != 'REPEATABILITY-STOP' or (root/'block-b/accepted').exists():
                raise ValueError('failed repeatability block was promoted')
    if report['status'] != 'PASS-STOCK-REPEATABILITY-TRANSFER':
        return {'status': report['status'], 'native_processes': native_processes}
    if len(report['blocks']) != 2 or any(block['status'] != 'PASS-MEASUREMENT' for block in report['blocks']):
        raise ValueError('complete repeatability requires two independently passing blocks')
    store = EvidenceStore(root/'block-b.sqlite3')
    plan = load_validated_stock_plan(root/'block-b/accepted/execution-plan.json', root/'block-b/accepted/acceptance-receipt.json',
        store, identities=inputs.identities(inputs.stock), host_environment=host_environment)
    consumer = report['consumer']
    measured = store.verify_measurement(consumer['measurement_id'])
    if (consumer['status'] != 'MEASURED-ACCEPTED-STOCK' or consumer['plan_sha256'] != plan.plan_sha256
            or consumer['decode_tps'] != measured['decode_tps'] or measured['candidate_id'] != plan.candidate.candidate_id
            or store.measurement(consumer['measurement_id']).stage != 'accepted-stock-run'):
        raise ValueError('repeatability fresh consumer mismatch')
    transfer_path = root/'transfer/utility/report.json'
    transfer = json.loads(transfer_path.read_text())
    reconstructed = utility.validate_result(transfer, EvidenceStore(root/'transfer/utility.sqlite3'), transfer_inputs,
        source_repository=source_repository, host_environment=host_environment)
    if (report['transfer'] != {'status':transfer['status'],'report_sha256':file_sha256(transfer_path)}
            or reconstructed['status'] != 'PASS-STOCK-UTILITY-PRODUCT' or native_processes != 148):
        raise ValueError('repeatability transfer/process budget mismatch')
    return {'status': report['status'], 'native_processes': native_processes, 'accepted_main_plan_sha256':plan.plan_sha256}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('run','validate'), required=True)
    defaults = {'descriptor':'configs/compiler/gemma4-q6-model.json','inventory':'docs/evidence/q6-download/tensor-inventory.json',
        'hardware':'docs/evidence/compiler-phase3/inputs/hardware.json','workload':'configs/compiler/gemma4-q6-single-request.json',
        'transfer-workload':'configs/compiler/gemma4-q6-utility-transfer.json','runtime-identity':'docs/evidence/compiler-phase3/inputs/runtime-identity.json',
        'source-plan':'C:/models/expertflow/runs/compiler-stock-utility-main-20261004/utility/selected-plan.json',
        'source-evidence-db':'C:/models/expertflow/runs/compiler-stock-utility-main-20261004/utility.sqlite3',
        'prior-report':'C:/models/expertflow/runs/compiler-stock-utility-main-20261004/utility/report.json',
        'source-repository':'C:/models/expertflow/worktrees/llama-q6-placement-final'}
    for name, default in defaults.items(): parser.add_argument('--'+name,type=Path,default=Path(default))
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args(argv)
    sampler = None
    try:
        if not args.source_evidence_db.is_file(): raise ValueError('original utility database unavailable')
        if args.action == 'run' and args.output_dir.exists(): raise ValueError('fresh study output required; no retries')
        request = CompilationRequest(args.descriptor,args.inventory,args.hardware,args.workload,args.runtime_identity,(),
                                     args.output_dir/'unused.sqlite3',args.output_dir)
        inputs = load_compiler_inputs(request,live=True)
        transfer_inputs = load_compiler_inputs(replace(request,workload_path=args.transfer_workload),live=True)
        source_store = EvidenceStore(args.source_evidence_db)
        if args.action == 'validate':
            report = json.loads((args.output_dir/'report.json').read_text())
            result = validate_followup(report,inputs,transfer_inputs,args.source_plan,source_store,args.prior_report,
                source_repository=args.source_repository,host_environment=capture_host_environment())
        else:
            sampler = DiagnosticSampler(WindowsGpuMemorySampler(inputs.hardware.gpu_uuid))
            result = run_followup(inputs,transfer_inputs,args.source_plan,source_store,args.prior_report,args.output_dir,
                source_repository=args.source_repository,runner_factory=lambda store:ServerMeasurementRunner(store,memory_sampler=sampler))
        print(json.dumps({'status':result['status'],'reason':result.get('reason'),'report':str(args.output_dir/'report.json')}))
        return 0 if result['status'].startswith('PASS-') else 2
    finally:
        if sampler: sampler.close()


if __name__ == '__main__':
    raise SystemExit(main())
