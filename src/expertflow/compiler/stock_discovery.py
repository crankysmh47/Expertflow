"""Fresh bounded stock search and evidence-backed recommendation reconstruction."""

from dataclasses import replace
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import uuid

from .pipeline import atomic_json, CompilerInputs
from .plan import CandidatePlan, CandidateStatus, PlanIdentities, RuntimeSettings, _decode_plan, seal_candidate, validate_execution_plan
from .preflight import capture_host_environment, file_sha256
from .refinement import balanced_schedule, paired_statistics
from .schema import ArtifactIdentity, HardwareIR, ModelIR, MoELayerIR, WorkloadIR, canonical_payload, canonical_sha256, require_int
from .runner import RuntimeBinding
from .stock_eligibility import EligibilityRegistry
from .stock_search import scheduling_space, semantic_fingerprint, screening_schedule, rank_screening
from .stock_validation import load_validated_stock_plan

PROTOCOL = 'bounded-stock-search-v1'
SPEC = Path('docs/superpowers/specs/2026-10-04-bounded-stock-search.md')


def _candidate(payload):
    data = dict(payload)
    identities = dict(data['identities'])
    identities['workload'] = WorkloadIR(**identities['workload'])
    data['identities'] = PlanIdentities(**identities)
    data['settings'] = RuntimeSettings(**data['settings'])
    return CandidatePlan(**data)


def _checksum(payload, name):
    value = dict(payload)
    claimed = value.pop(name,None)
    if claimed != canonical_sha256(value):
        raise ValueError(f'{name} mismatch')


def _source_files():
    paths = [*sorted(Path('src/expertflow/compiler').rglob('*.py')),SPEC]
    driver = Path('scripts/benchmark_compiler_stock_search.py')
    if driver.exists():
        paths.append(driver)
    return {str(p):file_sha256(p) for p in paths}


def _snapshot_inputs(snapshot,workload):
    model = dict(snapshot['model'])
    model['identity'] = ArtifactIdentity(**model['identity'])
    model['moe_layers'] = tuple(MoELayerIR(**row) for row in model['moe_layers'])
    binding = dict(snapshot['stock'])
    binding['server'] = ArtifactIdentity(**binding['server'])
    binding['dependencies'] = tuple(ArtifactIdentity(**row) for row in binding['dependencies'])
    if binding['cuda_runtime'] is not None:
        binding['cuda_runtime'] = ArtifactIdentity(**binding['cuda_runtime'])
    stock = RuntimeBinding(**binding)
    return CompilerInputs(ModelIR(**model),HardwareIR(**snapshot['hardware']),workload,stock,stock,(),{})


def _space(base,host,exclusions=None,config=None):
    if config is None:
        cpus = host.get('cpu',[])
        if len(cpus) != 1 or (cpus[0].get('cores'),cpus[0].get('logical_processors'),base.identities.workload.threads) != (8,16,12):
            raise ValueError('this topology requires an explicit search space/budget config')
        exclusions = exclusions or {8:'prior rejected eight-thread hypothesis; no compatible cached TPS'}
        if set(exclusions) != {8}:
            raise ValueError('approved current search requires only the threads8 exclusion')
        config = {'policy':'approved-current','excluded_threads':canonical_payload(sorted(exclusions.items())),
                  'maximum_native_processes':32}
    else:
        if exclusions or set(config) != {'policy','excluded_threads','maximum_native_processes'} or config['policy'] != 'explicit':
            raise ValueError('explicit space config must declare exclusions and coherent budget')
        exclusions = dict(config['excluded_threads'])
        require_int(config['maximum_native_processes'],'explicit search budget')
    space = scheduling_space(base,host,excluded_threads=exclusions)
    if config['maximum_native_processes'] != 3*len(space.candidates)+20:
        raise ValueError('search candidate coverage exceeds or differs from declared budget')
    return space,canonical_payload(config)


def prepare_search(inputs, plan_path, receipt_path, source_store, output_dir, *, host_environment,
                   source_repository, registry=None, excluded_threads=None, space_config=None):
    source_store.prime_model(inputs.model)
    plan = load_validated_stock_plan(plan_path,receipt_path,source_store,
        identities=inputs.identities(inputs.stock),host_environment=host_environment)
    eligibility = (registry or EligibilityRegistry.with_builtins()).attest(inputs,host_environment,source_repository)
    expected = {'model_ir_sha256':plan.candidate.identities.model_sha256,
        'runtime_sha256':plan.candidate.identities.runtime_sha256,
        'hardware_sha256':plan.candidate.identities.hardware_sha256,
        'host_environment_sha256':canonical_sha256(host_environment)}
    if any(eligibility.get(k) != v for k,v in expected.items()) or eligibility.get('allowed_controls') != ['threads','cuda_graphs']:
        raise ValueError('search eligibility identity/scope mismatch')
    space,config = _space(plan.candidate,host_environment,excluded_threads,space_config)
    manifest = {'schema_version':'1.0.0','protocol_version':PROTOCOL,'experiment_id':uuid.uuid4().hex,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_files':_source_files(),'protocol_sha256':file_sha256(SPEC),
        'source_repository':str(Path(source_repository).resolve()),'space_config':config,
        'inputs':canonical_payload({'model':inputs.model,'hardware':inputs.hardware,'stock':inputs.stock}),
        'experiment_root':str(Path(output_dir).resolve()),'frozen_monotonic_ns':time.monotonic_ns(),
        'incumbent_id':plan.candidate.candidate_id,'semantic_sha256':semantic_fingerprint(plan.candidate),
        'host_environment':canonical_payload(host_environment),'eligibility':eligibility,
        'candidates':{c.candidate_id:canonical_payload(c) for c in space.candidates},
        'excluded_threads':canonical_payload(space.excluded_threads),'untested_threads':space.untested_threads,
        'screening_schedule':screening_schedule([c.candidate_id for c in space.candidates]),
        'confirmation_schedule':balanced_schedule(),'screening_seed':20261004,'confirmation_seed':20261003,
        'bootstrap_samples':10000,'minimum_gain_pct':2,'minimum_ci95_lower_pct':0,'maximum_cv_pct':10,
        'screening_processes':3*len(space.candidates),'maximum_native_processes':3*len(space.candidates)+20,
        'prerequisite_plan':canonical_payload(plan),
        'prerequisite_receipt':json.loads(Path(receipt_path).read_text(encoding='utf-8')),
        'prerequisite_files':{str(Path(p).resolve()):file_sha256(Path(p))
            for p in (plan_path,receipt_path,source_store.path)}}
    manifest = canonical_payload(manifest)
    manifest['manifest_sha256'] = canonical_sha256(manifest)
    return manifest


def _validate_manifest(manifest, host, *, registry=None):
    _checksum(manifest,'manifest_sha256')
    if manifest.get('schema_version') != '1.0.0' or manifest.get('protocol_version') != PROTOCOL:
        raise ValueError('unsupported search protocol')
    if manifest['protocol_sha256'] != file_sha256(SPEC) or manifest['host_environment'] != canonical_payload(host):
        raise ValueError('search protocol/host mismatch')
    if not re.fullmatch('[0-9a-f]{32}',manifest['experiment_id']) or type(manifest['frozen_monotonic_ns']) is not int or manifest['frozen_monotonic_ns'] <= 0:
        raise ValueError('invalid fresh search boundary')
    plan = _decode_plan(manifest['prerequisite_plan'])
    candidates = {cid:_candidate(c) for cid,c in manifest['candidates'].items()}
    config = manifest['space_config']
    if config.get('policy') == 'approved-current':
        space,expected_config = _space(plan.candidate,host,dict(config['excluded_threads']))
    else:
        space,expected_config = _space(plan.candidate,host,config=config)
    if config != expected_config or manifest['excluded_threads'] != canonical_payload(space.excluded_threads):
        raise ValueError('search approved space/exclusion mismatch')
    if canonical_payload({c.candidate_id:c for c in space.candidates}) != manifest['candidates']:
        raise ValueError('search candidate coverage/identity mismatch')
    if manifest['incumbent_id'] != plan.candidate.candidate_id or manifest['semantic_sha256'] != semantic_fingerprint(plan.candidate):
        raise ValueError('search incumbent/semantic identity mismatch')
    if manifest['untested_threads'] != canonical_payload(space.untested_threads):
        raise ValueError('search coverage exclusions mismatch')
    proof = manifest['eligibility']
    expected = {'model_ir_sha256':plan.candidate.identities.model_sha256,
        'runtime_sha256':plan.candidate.identities.runtime_sha256,
        'hardware_sha256':plan.candidate.identities.hardware_sha256,'host_environment_sha256':canonical_sha256(host)}
    if not proof.get('provider_id') or any(proof.get(k) != v for k,v in expected.items()) or proof.get('allowed_controls') != ['threads','cuda_graphs']:
        raise ValueError('search eligibility binding mismatch')
    inputs = _snapshot_inputs(manifest['inputs'],plan.candidate.identities.workload)
    if inputs.identities(inputs.stock) != plan.candidate.identities:
        raise ValueError('search input snapshot identity mismatch')
    trusted = (registry or EligibilityRegistry.with_builtins()).attest(inputs,host,manifest['source_repository'])
    if canonical_payload(trusted) != proof:
        raise ValueError('search eligibility differs from trusted complete attestation')
    controls = {'screening_seed':20261004,'confirmation_seed':20261003,'bootstrap_samples':10000,
        'minimum_gain_pct':2,'minimum_ci95_lower_pct':0,'maximum_cv_pct':10,
        'screening_processes':3*len(candidates),'maximum_native_processes':3*len(candidates)+20}
    if any(manifest.get(k) != v for k,v in controls.items()) or (
            manifest['screening_schedule'] != canonical_payload(screening_schedule(candidates))) or (
            manifest['confirmation_schedule'] != canonical_payload(balanced_schedule())):
        raise ValueError('search frozen budget/schedule/gate mismatch')
    if manifest['source_files'] != _source_files():
        raise ValueError('search source changed')
    return plan,candidates


def _verify_run(row, store, manifest, candidate, label, reference, owners):
    mid = row['measurement_id']
    verified = store.verify_measurement(mid)
    record = store.measurement(mid)
    if verified['measured'] is not True or verified['exit_code'] != 0 or any(
            verified['validations'].get(name) is not True for name in ('exact_tokens','memory','cleanup')):
        raise ValueError('search requires measured passing native evidence')
    if record.stage != f"search-{manifest['experiment_id']}-{label}" or record.numerical_path != 'stock_same_runtime':
        raise ValueError('search stage/numerical path mismatch')
    if verified['candidate_id'] != candidate.candidate_id or verified['identities'] != canonical_payload(candidate.identities) or verified['settings_sha256'] != canonical_sha256(candidate.settings):
        raise ValueError('search candidate/launch identity mismatch')
    if verified['owned_run_sha256'] in owners:
        raise ValueError('search reused owned process')
    owners.add(verified['owned_run_sha256'])
    if any(verified[k] != reference[k] for k in ('generated_tokens_sha256','prompt_tokens_sha256')):
        raise ValueError('search changed exact native tokens')
    if any(canonical_payload(row.get(k)) != canonical_payload(v) for k,v in verified.items()):
        raise ValueError('search row differs from native artifacts')
    paths = {a.role:Path(a.identity.path) for a in record.artifacts}
    root = Path(manifest['experiment_root'])/'raw'/label
    if any(p.resolve().parent != root.resolve() for p in paths.values()):
        raise ValueError('search artifacts outside frozen root')
    launch = json.loads(paths['launch'].read_text(encoding='utf-8'))
    if launch.get('host_environment') != manifest['host_environment'] or launch.get('experiment_context') != {'manifest_sha256':manifest['manifest_sha256']}:
        raise ValueError('native launch does not bind search host/manifest')
    start = json.loads(paths['run-start'].read_text(encoding='utf-8'))
    if start['started_monotonic_ns'] < manifest['frozen_monotonic_ns']:
        raise ValueError('search reused pre-freeze process')
    return verified


def _decision(ranking, incumbent_id, confirmation):
    finalist = ranking[0]['candidate_id']
    statistics = None
    accepted = False
    if finalist != incumbent_id:
        if len(confirmation) != 20:
            raise ValueError('search requires independent twenty-run confirmation')
        rates = {(r['pair'],r['arm']):r['decode_tps'] for r in confirmation}
        statistics = paired_statistics([rates[i,'direct'] for i in range(10)],[rates[i,'sealed'] for i in range(10)])
        accepted = statistics['geometric_change_pct'] >= 2 and statistics['ci95_pct'][0] > 0 and max(statistics['direct_cv_pct'],statistics['sealed_cv_pct']) <= 10
    elif confirmation:
        raise ValueError('incumbent screening winner must not consume self-confirmation')
    return {'ranking':canonical_payload(ranking),'finalist_id':finalist,
        'statistics':canonical_payload(statistics),'confirmation_accepted':accepted,
        'recommended_id':finalist if accepted else incumbent_id,
        'status':'RECOMMENDED-CHALLENGER' if accepted else 'RECOMMENDED-INCUMBENT'}


def reconstruct_search(report, store, *, host_environment, registry=None):
    manifest = report['manifest']
    plan,candidates = _validate_manifest(manifest,host_environment,registry=registry)
    validate_execution_plan(plan,store=store)
    receipt = manifest['prerequisite_receipt']
    _checksum(receipt,'receipt_sha256')
    # Reconstruct the existing Stage A prerequisite using the imported evidence.
    from .stock_validation import reconstruct_product
    source,_,result = reconstruct_product(receipt['experiment'],store,host_environment=host_environment)
    if receipt['published_plan_sha256'] != plan.plan_sha256 or receipt['statistics'] != canonical_payload(result) or receipt['source_plan_sha256'] != source.plan_sha256:
        raise ValueError('search prerequisite receipt mismatch')
    if plan.candidate.measurement_ids != tuple(r['measurement_id'] for r in receipt['experiment']['rows']):
        raise ValueError('search prerequisite plan does not bind product evidence')
    owners = {store.verify_measurement(mid)['owned_run_sha256'] for mid in (*plan.candidate.measurement_ids,*source.candidate.measurement_ids)}
    reference = store.verify_measurement(plan.candidate.measurement_ids[0])
    screening = report['screening']
    expected = [(block,cid) for block,order in enumerate(manifest['screening_schedule']) for cid in order]
    if len(screening) != len(expected):
        raise ValueError('partial screening cannot recommend')
    for row,(block,cid) in zip(screening,expected):
        if row.get('block') != block or row.get('candidate_id') != cid:
            raise ValueError('screening order mismatch')
        _verify_run(row,store,manifest,candidates[cid],f'screen-{block:02}-{cid}',reference,owners)
    ranking = rank_screening(screening,manifest['incumbent_id'],manifest['screening_schedule'])
    finalist = ranking[0]['candidate_id']
    confirmation = report['confirmation']
    if finalist != manifest['incumbent_id']:
        if len(confirmation) != 20:
            raise ValueError('partial confirmation cannot recommend')
        for index,row in enumerate(confirmation):
            pair,order = divmod(index,2)
            arm = manifest['confirmation_schedule'][pair][order]
            cid = manifest['incumbent_id'] if arm == 'direct' else finalist
            if row.get('pair') != pair or row.get('arm') != arm:
                raise ValueError('confirmation pair/order mismatch')
            _verify_run(row,store,manifest,candidates[cid],f'confirm-{pair:02}-{arm}',reference,owners)
    decision = _decision(ranking,manifest['incumbent_id'],confirmation)
    if any(canonical_payload(report.get(k)) != canonical_payload(v) for k,v in decision.items()):
        raise ValueError('search claimed selection/statistics mismatch')
    rows = [*screening,*confirmation]
    if len(report['outcomes']) != len(rows) or any(o.get('status') != 'measured' or o.get('measurement_id') != r['measurement_id'] for o,r in zip(report['outcomes'],rows)):
        raise ValueError('search outcomes incomplete or inconsistent')
    return plan,candidates,decision


def _publish(report,store,output,host,registry=None):
    incumbent,candidates,decision = reconstruct_search(report,store,host_environment=host,registry=registry)
    if decision['confirmation_accepted']:
        candidate = candidates[decision['recommended_id']]
        mids = tuple(r['measurement_id'] for r in report['confirmation'] if r['arm'] == 'sealed')
        plan = seal_candidate(replace(candidate,status=CandidateStatus.MEASURED,measurement_ids=mids,
            validation=(('stock_search_confirmation',True),)),store,candidate.identities,None)
    else:
        plan = incumbent
    receipt = {'schema_version':'1.0.0','protocol_version':PROTOCOL,
        'published_plan_sha256':plan.plan_sha256,'experiment':canonical_payload(report)}
    receipt['receipt_sha256'] = canonical_sha256(receipt)
    pending = Path(tempfile.mkdtemp(prefix='.pending-search-',dir=output))
    destination = output/'recommended'
    try:
        if destination.exists():
            raise ValueError('search recommendation already exists')
        atomic_json(pending/'execution-plan.json',plan)
        atomic_json(pending/'search-receipt.json',receipt)
        os.rename(pending,destination)
    finally:
        if pending.exists():
            if pending.resolve().parent != output.resolve() or not pending.name.startswith('.pending-search-'):
                raise ValueError('unsafe search temporary cleanup')
            shutil.rmtree(pending)
    return plan


def load_search_recommendation(directory,store,*,host_environment,registry=None):
    root = Path(directory)
    receipt = json.loads((root/'search-receipt.json').read_text(encoding='utf-8'))
    _checksum(receipt,'receipt_sha256')
    if receipt.get('schema_version') != '1.0.0' or receipt.get('protocol_version') != PROTOCOL:
        raise ValueError('search receipt protocol mismatch')
    incumbent,candidates,decision = reconstruct_search(receipt['experiment'],store,host_environment=host_environment,registry=registry)
    plan = _decode_plan(json.loads((root/'execution-plan.json').read_text(encoding='utf-8')))
    validate_execution_plan(plan,store=store)
    if plan.plan_sha256 != receipt['published_plan_sha256'] or plan.candidate.candidate_id != decision['recommended_id']:
        raise ValueError('recommended plan differs from verified search')
    expected = tuple(r['measurement_id'] for r in receipt['experiment']['confirmation'] if r['arm'] == 'sealed') if decision['confirmation_accepted'] else incumbent.candidate.measurement_ids
    if plan.candidate.measurement_ids != expected:
        raise ValueError('recommendation evidence differs from independent validation')
    return plan


def execute_stock_search(inputs,plan_path,receipt_path,source_store,target_store,runner,output_dir,*,
                         host_capture=None,registry=None,source_repository,excluded_threads=None,space_config=None):
    capture = host_capture or capture_host_environment
    host = capture()
    output = Path(output_dir)
    if output.exists():
        raise ValueError('fresh search output required; no retry/resume')
    with target_store._connection() as connection:
        if connection.execute('SELECT COUNT(*) FROM measurement').fetchone()[0]:
            raise ValueError('fresh empty search database required')
    manifest = prepare_search(inputs,plan_path,receipt_path,source_store,output,
        host_environment=host,source_repository=source_repository,registry=registry,excluded_threads=excluded_threads,space_config=space_config)
    output.mkdir(parents=True)
    target_store.prime_model(inputs.model)
    source_plan = _decode_plan(manifest['prerequisite_receipt']['experiment']['frozen']['source_plan'])
    incumbent = _decode_plan(manifest['prerequisite_plan'])
    for mid in (*source_plan.candidate.measurement_ids,*incumbent.candidate.measurement_ids):
        target_store.append_measurement(source_store.measurement(mid),measurement_id=mid)
    atomic_json(output/'frozen-manifest.json',manifest)
    report = {'status':'RUNNING','manifest':manifest,'screening':[],'confirmation':[],'outcomes':[]}
    candidates = {cid:_candidate(payload) for cid,payload in manifest['candidates'].items()}
    reference = source_store.verify_measurement(incumbent.candidate.measurement_ids[0])
    owners = {source_store.verify_measurement(mid)['owned_run_sha256'] for mid in (*source_plan.candidate.measurement_ids,*incumbent.candidate.measurement_ids)}

    def run(cid,label,fields,destination):
        if canonical_payload(capture()) != manifest['host_environment']:
            raise ValueError('search host changed')
        if any(file_sha256(Path(p)) != sha for p,sha in {**manifest['source_files'],**manifest['prerequisite_files']}.items()):
            raise ValueError('search source or prerequisite changed')
        outcome = runner.run_once(candidates[cid],inputs.model,inputs.stock,
            output_dir=output/'raw'/label,measured=True,stage=f"search-{manifest['experiment_id']}-{label}",
            host_environment=host,experiment_context={'manifest_sha256':manifest['manifest_sha256']})
        report['outcomes'].append(canonical_payload(outcome))
        if outcome.status != 'measured':
            report.update(status=outcome.status.upper().replace('_','-'),reason=outcome.reason)
            atomic_json(output/'report.json',report)
            return False
        row = {**target_store.verify_measurement(outcome.measurement_id),**fields,'measurement_id':outcome.measurement_id}
        _verify_run(row,target_store,manifest,candidates[cid],label,reference,owners)
        destination.append(row)
        atomic_json(output/'report.json',report)
        return True

    try:
        for block,order in enumerate(manifest['screening_schedule']):
            for cid in order:
                if not run(cid,f'screen-{block:02}-{cid}',{'block':block},report['screening']):
                    return report
        ranking = rank_screening(report['screening'],manifest['incumbent_id'],manifest['screening_schedule'])
        finalist = ranking[0]['candidate_id']
        if finalist != manifest['incumbent_id']:
            for pair,order in enumerate(manifest['confirmation_schedule']):
                for arm in order:
                    cid = manifest['incumbent_id'] if arm == 'direct' else finalist
                    if not run(cid,f'confirm-{pair:02}-{arm}',{'pair':pair,'arm':arm},report['confirmation']):
                        return report
        report.update(_decision(ranking,manifest['incumbent_id'],report['confirmation']))
        if canonical_payload(capture()) != manifest['host_environment'] or any(file_sha256(Path(p)) != sha for p,sha in manifest['prerequisite_files'].items()):
            raise ValueError('search host/prerequisite changed before publication')
        _publish(report,target_store,output,host,registry)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as error:
        report.update(status='VALIDATION-STOP',reason=str(error))
    atomic_json(output/'report.json',report)
    return report
