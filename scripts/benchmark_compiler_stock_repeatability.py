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
from expertflow.compiler.refinement import balanced_schedule, evaluate_pairs, execute_pairs, paired_source_files
from expertflow.compiler.runner import ServerMeasurementRunner, WindowsGpuMemorySampler
from expertflow.compiler.schema import canonical_payload, canonical_sha256
from expertflow.compiler.stock_validation import load_validated_stock_plan, publish_stock_product, reconstruct_product, run_accepted_stock_plan
if __package__:
    from scripts import benchmark_compiler_stock_utility as utility
else:
    import benchmark_compiler_stock_utility as utility

SPEC = Path('docs/research/protocols/specs/2026-10-04-stock-repeatability.md')
PROTOCOL = 'paced-stock-repeatability-v1'
PRIOR_REPORT_SHA256 = '1475cb9830b4e572004b5a7b4bd60fe0c2e7e7b3877391a5f3a6105bd7b8c128'
WAIT_SECONDS = 30
MAXIMUM_NATIVE_PROCESSES = 148


def snapshot(inputs):
    return canonical_payload({name: getattr(inputs, name) for name in ('model', 'hardware', 'stock', 'workload')})


def sources():
    result = utility.sources()
    result.update({str(Path(path).resolve()): digest for path, digest in paired_source_files(product=True).items()})
    for path in (SPEC, Path(__file__), Path('docs/research/evidence/stock-utility-20261004/independent_audit.py')):
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
    auditor = runpy.run_path('docs/research/evidence/stock-utility-20261004/independent_audit.py')['audit']
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
            self.guard()
            print(json.dumps({'process': len(report['attempts']), 'stage': kwargs['stage'], 'status': outcome.status}), flush=True)
            return outcome
        except Exception as error:
            attempt.update(status='exception', reason=str(error), exception_type=type(error).__name__)
            raise
        finally:
            attempt['finished_monotonic_ns'] = self.clock()
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
    collector_started = time.monotonic_ns()
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
    report = {'status': 'RUNNING', 'manifest': manifest, 'attempts': [], 'blocks': [],
              'collector_started_monotonic_ns': collector_started}
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
            report.update(status='PENDING-RECONSTRUCTION' if transfer['status'] == 'PASS-STOCK-UTILITY-PRODUCT'
                          else 'TRANSFER-VALIDATION-STOP', reason=transfer.get('reason'))
    except Exception as error:
        report.update(status='VALIDATION-STOP', reason=str(error), exception_type=type(error).__name__)
    finally:
        report['collector_finished_monotonic_ns'] = time.monotonic_ns()
        report['collector_elapsed_ns'] = report['collector_finished_monotonic_ns'] - collector_started
        atomic_json(root/'report.json', report)
    if report['status'] == 'PENDING-RECONSTRUCTION':
        try:
            reconstructed = validate_followup({**report,'status':'PASS-STOCK-REPEATABILITY-TRANSFER'},inputs,transfer_inputs,source_plan,source_store,prior_path,
                source_repository=source_repository,host_environment=capture())
            report.update(status='PASS-STOCK-REPEATABILITY-TRANSFER',reconstruction=reconstructed)
        except Exception as error:
            report.update(status='VALIDATION-STOP',reason=str(error),exception_type=type(error).__name__)
    atomic_json(root/'report.json', report)
    return report


def audit_attempts(report):
    manifest, owners, outputs = report['manifest'], set(), set()
    root = Path(manifest['experiment_root']).resolve()
    if len(report['attempts']) > MAXIMUM_NATIVE_PROCESSES:
        raise ValueError('repeatability attempt budget exceeded')
    previous = manifest['frozen_monotonic_ns']
    for attempt in report['attempts']:
        output, database = Path(attempt['output_dir']).resolve(), Path(attempt['database']).resolve()
        output.relative_to(root)
        database.relative_to(root)
        if output in outputs or attempt['wait_started_monotonic_ns'] < previous:
            raise ValueError('duplicate or out-of-order native attempt')
        outputs.add(output)
        finished = attempt['finished_monotonic_ns']
        if finished < attempt['wait_started_monotonic_ns']:
            raise ValueError('attempt elapsed order mismatch')
        previous = finished
        started = output/'run-start.json'
        if attempt['native_started'] != started.is_file():
            raise ValueError('retained native start mismatch')
        if started.is_file():
            if (attempt['wait_seconds'] != WAIT_SECONDS or attempt['wait_elapsed_ns'] < WAIT_SECONDS * 1e9
                    or attempt['wait_elapsed_ns'] != attempt['wait_finished_monotonic_ns'] - attempt['wait_started_monotonic_ns']):
                raise ValueError('retained fixed wait evidence mismatch')
            observed = json.loads(started.read_text())
            owner = (observed['pid'], observed['creation_time_100ns'], observed['creation_source'])
            if (owner in owners or attempt['process_identity'] != observed
                    or not attempt['wait_finished_monotonic_ns'] <= observed['started_monotonic_ns'] <= finished):
                raise ValueError('retained native owner/wait mismatch')
            owners.add(owner)
        if output.is_relative_to(root/'transfer/utility'):
            nested = json.loads((root/'transfer/utility/frozen-manifest.json').read_text())
            expected_context = {'manifest_sha256': nested['manifest_sha256']}
        else:
            expected_context = {'repeatability_manifest_sha256': manifest['manifest_sha256']}
        launch_path = output/'launch.json'
        if launch_path.is_file():
            launch = json.loads(launch_path.read_text())
            if (launch.get('experiment_context') != attempt.get('experiment_context')
                    or launch.get('experiment_context') != expected_context
                    or launch.get('host_environment') != manifest['host_environment']):
                raise ValueError('retained native context/host mismatch')
        if attempt.get('measurement_id'):
            store = EvidenceStore(database)
            record = store.measurement(attempt['measurement_id'])
            verified = store.verify_measurement(attempt['measurement_id'])
            artifacts = {a.role: Path(a.identity.path) for a in record.artifacts}
            if (any(path.parent.resolve() != output for path in artifacts.values()) or record.stage != attempt['stage']
                    or attempt['process_identity'] != json.loads(artifacts['run-start'].read_text())
                    or attempt['native_started'] is not True or verified['measured'] is not True or verified['exit_code'] != 0
                    or any(verified['validations'].get(name) is not True for name in ('exact_tokens','memory','cleanup'))):
                raise ValueError('retained native record/correctness mismatch')
        elif attempt['status'] == 'measured':
            raise ValueError('measured attempt lacks native record')
    if len(list(root.rglob('run-start.json'))) != len(owners):
        raise ValueError('unaccounted native process starts')
    return len(owners)


def match_entry(entry, output, database, stage, outcome=None):
    if (Path(entry['output_dir']).resolve() != Path(output).resolve()
            or Path(entry['database']).resolve() != Path(database).resolve() or entry['stage'] != stage):
        raise ValueError('native phase journal path/database/stage mismatch')
    if outcome is not None and (entry['status'] != outcome['status']
            or entry.get('measurement_id') != outcome.get('measurement_id')):
        raise ValueError('native phase journal outcome mismatch')


def verify_block(block, root, store, source_path, source_store, entries, host, outer):
    """Reconstruct complete gates and incomplete prefixes with the same bindings."""
    root = Path(root).resolve()
    freeze = block['frozen']
    source = load_execution_plan(source_path, store=source_store)
    expected = {'protocol_version':'paired-stock-product-v1',
        'protocol_sha256':file_sha256(Path('docs/research/protocols/specs/2026-10-04-stock-configuration-discovery.md')),
        'source_files':paired_source_files(product=True), 'source_commit':outer['source_commit'],
        'source_plan':canonical_payload(source), 'source_plan_sha256':source.plan_sha256,
        'source_plan_file_sha256':file_sha256(source_path), 'source_candidate_id':source.candidate.candidate_id,
        'source_evidence_db':str(source_store.path.resolve()), 'source_evidence_db_sha256':file_sha256(source_store.path),
        'identities':canonical_payload(source.candidate.identities), 'settings':canonical_payload(source.candidate.settings),
        'schedule':canonical_payload(balanced_schedule()), 'pairs':10, 'seed':20261003,
        'bootstrap_samples':10000, 'equivalence_margin_pct':2, 'host_environment':host,
        'experiment_root':str(root), 'source_plan_diagnostic_only':True}
    if (any(canonical_payload(freeze.get(k)) != canonical_payload(v) for k,v in expected.items())
            or freeze != json.loads((root/'frozen-protocol.json').read_text())
            or not outer['frozen_monotonic_ns'] <= freeze['frozen_monotonic_ns']
            or len(source.candidate.measurement_ids) != 10):
        raise ValueError('repeatability block frozen source/controls mismatch')
    rows, outcomes = block['rows'], block['outcomes']
    if not 0 <= len(rows) <= len(outcomes) <= len(entries) <= 20 or len(entries) > len(rows)+1:
        raise ValueError('repeatability block prefix budget mismatch')
    reference = source_store.verify_measurement(source.candidate.measurement_ids[0])
    source_owners = {source_store.verify_measurement(mid)['owned_run_sha256'] for mid in source.candidate.measurement_ids}
    rebuilt = []
    for index, entry in enumerate(entries):
        pair, position = divmod(index,2)
        arm = balanced_schedule()[pair][position]
        stage = f"product-{freeze['experiment_id']}-{pair:02}-{arm}"
        outcome = outcomes[index] if index < len(outcomes) else None
        match_entry(entry,root/'raw'/f'pair-{pair:02}-{arm}',store.path,stage,outcome)
        if index >= len(rows):
            if entry['status']=='measured' and block['status'] not in ('VALIDATION-STOP','RUNNING'):
                raise ValueError('missing verified product row')
            continue
        row = rows[index]
        native = store.verify_measurement(row['measurement_id'])
        record = store.measurement(row['measurement_id'])
        if (entry['status']!='measured' or entry.get('measurement_id')!=row['measurement_id']
                or row.get('pair')!=pair or row.get('arm')!=arm or record.numerical_path!='stock_same_runtime'
                or native['candidate_id']!=source.candidate.candidate_id
                or native['identities']!=freeze['identities'] or native['settings_sha256']!=canonical_sha256(source.candidate.settings)
                or native['owned_run_sha256'] in source_owners
                or any(native[k]!=reference[k] for k in ('generated_tokens_sha256','prompt_tokens_sha256'))
                or any(canonical_payload(row.get(k))!=canonical_payload(v) for k,v in native.items())):
            raise ValueError('repeatability block row/source/reference mismatch')
        if json.loads((Path(entry['output_dir'])/'run-start.json').read_text())['started_monotonic_ns'] < freeze['frozen_monotonic_ns']:
            raise ValueError('product native call precedes block freeze')
        rebuilt.append({**native,'pair':pair,'arm':arm,'measurement_id':row['measurement_id']})
    if len(rows)==20:
        result = evaluate_pairs(rebuilt)
        expected_status = 'PASS-STOCK-FALLBACK' if block['status']=='PASS-STOCK-FALLBACK' else result['status']
        if expected_status=='PASS-STOCK-FALLBACK' and result['status']!='PASS-MEASUREMENT':
            raise ValueError('failed product block was promoted')
        for key,value in result.items():
            claimed = expected_status if key=='status' else value
            if canonical_payload(block.get(key))!=canonical_payload(claimed):
                raise ValueError('repeatability block claimed statistics mismatch')
    elif block['status'] not in ('RUNNING','VALIDATION-STOP','ENVIRONMENT-BLOCKED'):
        raise ValueError('partial product cannot claim a statistical result')
    return block['status'] in ('PASS-MEASUREMENT','PASS-STOCK-FALLBACK')


def verify_consumer(consumer, entry, plan, store, output):
    outcome = None if consumer is None else consumer.get('outcome')
    match_entry(entry,output,store.path,'accepted-stock-run',outcome)
    if consumer is None or consumer['status']!='MEASURED-ACCEPTED-STOCK':
        if entry['status']=='measured':
            raise ValueError('consumer native record lacks reconstructed acceptance')
        return False
    native = store.verify_measurement(consumer['measurement_id'])
    reference = store.verify_measurement(plan.candidate.measurement_ids[0])
    if (entry['status']!='measured' or entry.get('measurement_id')!=consumer['measurement_id']
            or consumer['plan_sha256']!=plan.plan_sha256 or consumer['decode_tps']!=native['decode_tps']
            or native['candidate_id']!=plan.candidate.candidate_id
            or native['identities']!=canonical_payload(plan.candidate.identities)
            or native['settings_sha256']!=canonical_sha256(plan.candidate.settings)
            or any(native[k]!=reference[k] for k in ('generated_tokens_sha256','prompt_tokens_sha256'))):
        raise ValueError('consumer differs from accepted reference/plan/journal')
    return True


def validate_followup(report, inputs, transfer_inputs, source_plan, source_store, prior_path,
                      *, source_repository, host_environment):
    statuses = {'REPEATABILITY-STOP','CONSUMER-VALIDATION-STOP','TRANSFER-VALIDATION-STOP',
                'VALIDATION-STOP','PASS-STOCK-REPEATABILITY-TRANSFER'}
    if report['status'] not in statuses:
        raise ValueError('unknown repeatability terminal status')
    manifest = report['manifest']
    payload = dict(manifest)
    digest = payload.pop('manifest_sha256')
    root = Path(manifest['experiment_root']).resolve()
    if (digest != canonical_sha256(payload) or manifest['protocol_version'] != PROTOCOL
            or manifest != json.loads((root/'frozen-manifest.json').read_text())
            or manifest['protocol_sha256'] != file_sha256(SPEC)
            or manifest['source_files'] != tracked_sources(source_plan, source_store, prior_path)
            or manifest['host_environment'] != canonical_payload(host_environment)
            or manifest['main_inputs'] != snapshot(inputs) or manifest['transfer_inputs'] != snapshot(transfer_inputs)
            or manifest['source_repository'] != str(Path(source_repository).resolve())
            or any(manifest[key] != value for key, value in {'maximum_native_processes':148,'wait_seconds':30,
                'main_blocks':2,'native_processes_per_block':20,'consumer_processes':1,'transfer_maximum':107}.items())):
        raise ValueError('repeatability frozen inputs/source/host/protocol mismatch')
    if verify_prerequisite(inputs,transfer_inputs,source_plan,source_store,prior_path,source_repository,host_environment)!=manifest['prerequisite']:
        raise ValueError('repeatability prerequisite mismatch')
    if load_execution_plan(source_plan,store=source_store).plan_sha256!=manifest['source_plan_sha256']:
        raise ValueError('repeatability selected source plan mismatch')
    start,end,elapsed = (report[key] for key in ('collector_started_monotonic_ns','collector_finished_monotonic_ns','collector_elapsed_ns'))
    if (any(type(value) is not int for value in (start,end,elapsed)) or not 0 < start <= manifest['frozen_monotonic_ns'] <= end
            or elapsed!=end-start or elapsed < sum(a.get('wait_elapsed_ns',0) for a in report['attempts'])
            or any(a['finished_monotonic_ns']>end for a in report['attempts'])):
        raise ValueError('collector elapsed cost mismatch')
    native_processes = audit_attempts(report)
    cursor = 0
    def take_phase(path):
        nonlocal cursor
        entries = []
        while cursor < len(report['attempts']) and Path(report['attempts'][cursor]['output_dir']).resolve().is_relative_to(path.resolve()):
            entries.append(report['attempts'][cursor])
            cursor += 1
        return entries
    summaries = report['blocks']
    if len(summaries)>2 or (root/'block-a/accepted').exists():
        raise ValueError('unregistered main publication/block')
    passed = 0
    main_failed = False
    for index,name in enumerate(('block-a','block-b')):
        path = root/name/'report.json'
        frozen = root/name/'frozen-protocol.json'
        if passed!=index or not frozen.exists():
            if (root/name).exists() or len(summaries)>index:
                raise ValueError('main block activated before prerequisite')
            break
        block = json.loads(path.read_text()) if path.exists() else {'status':'RUNNING','frozen':json.loads(frozen.read_text()),'rows':[],'outcomes':[]}
        if index < len(summaries):
            if summaries[index] != {'name':name,'status':block['status'],'report_sha256':file_sha256(path)}:
                raise ValueError('repeatability block summary mismatch')
        elif report['status']!='VALIDATION-STOP':
            raise ValueError('main block summary missing')
        if verify_block(block,root/name,EvidenceStore(root/f'{name}.sqlite3'),source_plan,source_store,take_phase(root/name),host_environment,manifest):
            passed += 1
        else:
            main_failed = True
            if report['status'] not in ('REPEATABILITY-STOP','VALIDATION-STOP'):
                raise ValueError('failed main block terminal mismatch')
            break
    accepted = root/'block-b/accepted'
    consumer_pass = False
    transfer_status = None
    if passed!=2:
        if accepted.exists() or (root/'consumer').exists() or (root/'transfer').exists() or report.get('consumer') or report.get('transfer'):
            raise ValueError('conditional phase activated after failed main block')
        if (root/'block-b').exists() and passed==0:
            raise ValueError('block B activated after failed A')
    elif accepted.exists():
        store = EvidenceStore(root/'block-b.sqlite3')
        plan = load_validated_stock_plan(accepted/'execution-plan.json',accepted/'acceptance-receipt.json',store,
            identities=inputs.identities(inputs.stock),host_environment=host_environment)
        receipt = json.loads((accepted/'acceptance-receipt.json').read_text())
        if receipt['experiment']!=json.loads((root/'block-b/report.json').read_text()):
            raise ValueError('main publication differs from block B')
        entries = take_phase(root/'consumer')
        if len(entries)>1 or (report.get('consumer') and not entries):
            raise ValueError('consumer journal count mismatch')
        if entries:
            consumer_pass = verify_consumer(report.get('consumer'),entries[0],plan,store,root/'consumer')
        if consumer_pass:
            if (root/'transfer/utility/frozen-manifest.json').exists():
                transfer_status = verify_transfer(root,report,transfer_inputs,source_repository,host_environment,manifest,take_phase)
            elif report['status']!='VALIDATION-STOP':
                raise ValueError('qualified consumer requires transfer evidence')
        elif (root/'transfer').exists() or report.get('transfer'):
            raise ValueError('transfer activated before consumer acceptance')
    elif (root/'consumer').exists() or (root/'transfer').exists() or report.get('consumer') or report.get('transfer'):
        raise ValueError('consumer requires published acceptance')
    expected = ('REPEATABILITY-STOP' if passed!=2 else
                'CONSUMER-VALIDATION-STOP' if not consumer_pass else
                'PASS-STOCK-REPEATABILITY-TRANSFER' if transfer_status=='PASS-STOCK-UTILITY-PRODUCT' else 'TRANSFER-VALIDATION-STOP')
    if report['status'] not in (expected,'VALIDATION-STOP') or cursor!=len(report['attempts']):
        raise ValueError('terminal phase state/journal mismatch')
    if (report['status']=='REPEATABILITY-STOP' and not main_failed
            or report['status']=='CONSUMER-VALIDATION-STOP' and not report.get('consumer')):
        raise ValueError('terminal stop lacks its activated failed gate')
    if report['status']=='PASS-STOCK-REPEATABILITY-TRANSFER' and (native_processes!=148 or len(report['attempts'])!=148):
        raise ValueError('complete repeatability process count mismatch')
    cost = {'collector_seconds':elapsed/1e9,'prelaunch_wait_seconds':sum(a.get('wait_elapsed_ns',0) for a in report['attempts'])/1e9}
    return {'status':report['status'],'native_processes':native_processes,'cost':cost}


def verify_transfer(root, outer_report, inputs, repository, host, outer, take_phase):
    path = root/'transfer/utility/report.json'
    nested = root/'transfer/utility'
    transfer = json.loads(path.read_text())
    m = transfer['manifest']
    digest_payload = dict(m)
    digest = digest_payload.pop('manifest_sha256')
    utility.check_live_inputs(inputs,m,repository)
    if (digest!=canonical_sha256(digest_payload) or m!=json.loads((nested/'frozen-manifest.json').read_text())
            or m['protocol_version']!=utility.PROTOCOL or m['source_files']!=utility.sources()
            or m['source_commit']!=outer['source_commit'] or m['host_environment']!=host
            or Path(m['experiment_root']).resolve()!=nested or m['frozen_monotonic_ns']<outer['frozen_monotonic_ns']
            or m['maximum_native_processes']!=107 or m['reference_processes']!=10):
        raise ValueError('transfer frozen protocol/source/host mismatch')
    candidates = {cid:utility._candidate(value) for cid,value in m['candidates'].items()}
    default = candidates[m['default_id']]
    snapshot_inputs = utility._snapshot_inputs(m['inputs'],default.identities.workload)
    proof = utility.EligibilityRegistry.with_builtins().attest(snapshot_inputs,host,repository)
    schedule = utility.screening_schedule(candidates)
    if (m['protocol_scope']!=utility.audit_protocol_scope(snapshot_inputs) or m['eligibility']!=proof
            or m['default_source_proof']!=utility.audit_defaults(repository,host)
            or default.identities!=snapshot_inputs.identities(snapshot_inputs.stock)
            or set(candidates)!={c.candidate_id for c in utility.scheduling_space(default,host).candidates}
            or any(c.candidate_id!=cid for cid,c in candidates.items())
            or default.settings.cpu_moe is not proof.get('baseline_cpu_moe',True)
            or default.settings.gpu_layers!=99 or default.settings.static is not None
            or m['default_controls']!=list(utility.resolved_defaults(host))
            or m['screening_schedule']!=canonical_payload(schedule) or m['confirmation_schedule']!=canonical_payload(balanced_schedule())):
        raise ValueError('transfer registered candidate/default schedule mismatch')
    entries = take_phase(nested)
    if len(entries)!=len(transfer['attempts']) or len(entries)>107:
        raise ValueError('nested transfer attempt budget mismatch')
    for entry, inner in zip(entries,transfer['attempts']):
        if any(entry.get(key)!=inner.get(key) for key in ('output_dir','stage','status','measurement_id','native_started','process_identity')):
            raise ValueError('outer/nested transfer journal mismatch')
    rows = transfer['rows']
    expected = [(f'reference-{i:02}',m['default_id']) for i in range(10)]
    expected += [(f'{method}-screen-{block:02}-{cid}',cid) for method in ('automatic','manual') for block,order in enumerate(schedule) for cid in order]
    if len(rows)>=46 and ('automatic_id' in transfer or len(rows)>46):
        auto = utility.rank_screening([dict(r,block=i//len(candidates)) for i,r in enumerate(rows[10:28])],m['default_id'],schedule)[0]['candidate_id']
        manual = utility.manual_winner(rows[28:46],candidates,m['default_id'])
        if (auto,manual)!=(transfer['automatic_id'],transfer['manual_id']):
            raise ValueError('transfer selection differs from its own screens')
        expected += [(f'{purpose}-pair-{pair:02}-{arm}',control if arm=='direct' else auto)
            for purpose,control in (('defaults',m['default_id']),('manual',manual))
            for pair,order in enumerate(balanced_schedule()) for arm in order]
    raw_entries = [e for e in entries if Path(e['output_dir']).parent==nested/'raw']
    if not 0 <= len(rows) <= len(transfer['outcomes']) <= len(raw_entries) <= min(86,len(expected)) or len(raw_entries)>len(rows)+1:
        raise ValueError('transfer raw prefix mismatch')
    store = EvidenceStore(root/'transfer/utility.sqlite3')
    owners, reference = set(), None
    for index,entry in enumerate(raw_entries):
        label,cid = expected[index]
        outcome = transfer['outcomes'][index] if index<len(transfer['outcomes']) else None
        match_entry(entry,nested/'raw'/label,store.path,utility.measurement_stage(label,m),outcome)
        if index<len(rows):
            row = rows[index]
            if row['label']!=label or row['candidate_id']!=cid or entry['status']!='measured' or entry.get('measurement_id')!=row['measurement_id']:
                raise ValueError('transfer row/order/journal mismatch')
            native = utility.verify_row(row,candidates[cid],store,m,owners,reference)
            reference = reference or native
    if len(rows)==86:
        utility.validate_result(transfer,store,inputs,source_repository=repository,host_environment=host)
    elif transfer['status'] not in ('VALIDATION-STOP','ENVIRONMENT-BLOCKED'):
        raise ValueError('partial transfer cannot claim utility statistics')
    offset = len(raw_entries)
    product = nested/'product/report.json'
    if (nested/'product').exists():
        if len(rows)!=86 or transfer['statistics']['status']!='PASS-STOCK-UTILITY':
            raise ValueError('transfer product activated before utility acceptance')
        block = json.loads(product.read_text()) if product.exists() else {'status':'RUNNING','frozen':json.loads((nested/'product/frozen-protocol.json').read_text()),'rows':[],'outcomes':[]}
        block_entries = [entry for entry in entries[offset:] if Path(entry['output_dir']).is_relative_to(nested/'product')]
        product_store = EvidenceStore(nested/'product.sqlite3')
        passed = verify_block(block,nested/'product',product_store,nested/'selected-plan.json',store,block_entries,host,outer)
        offset += len(block_entries)
        if passed:
            selected = load_execution_plan(nested/'selected-plan.json',store=store)
            accepted = nested/'product/accepted'
            plan = load_validated_stock_plan(accepted/'execution-plan.json',accepted/'acceptance-receipt.json',product_store,
                identities=selected.candidate.identities,host_environment=host)
            if entries[offset:]:
                if len(entries[offset:])!=1 or not verify_consumer(transfer.get('consumer'),entries[offset],plan,product_store,nested/'consumer'):
                    if transfer['status'] not in ('CONSUMER-VALIDATION-STOP','VALIDATION-STOP'):
                        raise ValueError('transfer consumer gate mismatch')
                offset += 1
            elif transfer['status']=='PASS-STOCK-UTILITY-PRODUCT':
                raise ValueError('transfer consumer evidence missing')
        elif (nested/'product/accepted').exists() or (nested/'consumer').exists():
            raise ValueError('transfer failed product was promoted')
    elif (nested/'consumer').exists():
        raise ValueError('transfer consumer lacks accepted product')
    if offset!=len(entries) or transfer['status'] not in {'ENVIRONMENT-BLOCKED','VALIDATION-STOP','NO-UTILITY-GAIN',
            'VARIANCE-STOP','MANUAL-BASELINE-STOP','PRODUCT-VALIDATION-STOP','CONSUMER-VALIDATION-STOP','PASS-STOCK-UTILITY-PRODUCT'}:
        raise ValueError('transfer terminal journal/state mismatch')
    summary = outer_report.get('transfer')
    if summary is not None and summary!={'status':transfer['status'],'report_sha256':file_sha256(path)}:
        raise ValueError('transfer summary mismatch')
    if summary is None and outer_report['status']!='VALIDATION-STOP':
        raise ValueError('transfer summary missing')
    return transfer['status']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('run','validate'), required=True)
    defaults = {'descriptor':'configs/compiler/gemma4-q6-model.json','inventory':'docs/research/evidence/q6-download/tensor-inventory.json',
        'hardware':'docs/research/evidence/compiler-phase3/inputs/hardware.json','workload':'configs/compiler/gemma4-q6-single-request.json',
        'transfer-workload':'configs/compiler/gemma4-q6-utility-transfer.json','runtime-identity':'docs/research/evidence/compiler-phase3/inputs/runtime-identity.json',
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
            if Path(report['manifest']['experiment_root']).resolve()!=args.output_dir.resolve():
                raise ValueError('requested study root differs from frozen result')
            result = validate_followup(report,inputs,transfer_inputs,args.source_plan,source_store,args.prior_report,
                source_repository=args.source_repository,host_environment=capture_host_environment())
        else:
            sampler = DiagnosticSampler(WindowsGpuMemorySampler(inputs.hardware.gpu_uuid))
            result = run_followup(inputs,transfer_inputs,args.source_plan,source_store,args.prior_report,args.output_dir,
                source_repository=args.source_repository,runner_factory=lambda store:ServerMeasurementRunner(store,memory_sampler=sampler))
        print(json.dumps({'status':result['status'],'reason':result.get('reason'),'report':str(args.output_dir/'report.json')}))
        return 0 if result['status']=='PASS-STOCK-REPEATABILITY-TRANSFER' else 2
    finally:
        if sampler: sampler.close()


if __name__ == '__main__':
    raise SystemExit(main())
