"""Inspect a preregistration without treating it as collected model evidence."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from expertflow.compiler.pipeline import EnvironmentBlocked, inspect_model
from expertflow.compiler.plan import CandidatePlan, RuntimeSettings
from expertflow.compiler.preflight import file_sha256
from expertflow.compiler.reference import load_reference_workload
from expertflow.compiler.refinement import balanced_schedule
from expertflow.compiler.schema import WorkloadIR, canonical_payload, canonical_sha256
from expertflow.compiler.stock_discovery import _snapshot_inputs
from expertflow.compiler.stock_search import scheduling_space, screening_schedule

GATES = {'minimum_defaults_gain_pct':5, 'defaults_ci95_lower_strictly_above_pct':0,
    'manual_ci90_strict_bounds_pct':[-2,2], 'product_ci90_strict_bounds_pct':[-2,2],
    'product_one_sided95_lower_strictly_above_pct':-2, 'maximum_arm_cv_pct':10,
    'automatic_evaluations':18, 'manual_evaluations':18, 'exact_tokens':True,
    'owned_memory_reserve_cleanup':True}
CASES = (('gemma4-q4-prose','gemma4','Q4_0'), ('gemma4-q4-code','gemma4','Q4_0'),
         ('granite-q6-prose','granitemoe','Q6_K'), ('granite-q6-code','granitemoe','Q6_K'))
HOST_REFERENCE = 'docs/evidence/stock-repeatability-20261004/main-frozen-manifest.json'
DEFAULT_REFERENCE = 'docs/evidence/stock-repeatability-20261004/transfer-frozen-manifest.json'


def archived_input(root, relative):
    """Resolve frozen logical names without changing their recorded identity."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('registration input path outside project')
    if path.is_file():
        return path
    for old, new in (('docs/evidence/', 'docs/research/evidence/'),
                     ('docs/superpowers/', 'docs/research/protocols/')):
        if relative.startswith(old):
            path = (root / new / relative[len(old):]).resolve()
            if not path.is_relative_to(root):
                raise ValueError('registration input path outside project')
            return path
    return path


def verify_registration(data, project):
    payload = dict(data)
    claimed = payload.pop('registration_sha256', None)
    if claimed != canonical_sha256(payload):
        raise ValueError('registration checksum mismatch')
    if (data['protocol_version'] != 'wider-stock-utility-coverage-v1'
            or data['status'] != 'REGISTERED-NOT-RUN' or canonical_sha256(data['gates']) != canonical_sha256(GATES)
            or data['maximum_native_processes'] != 428 or data['wait_seconds'] != 30
            or data['collection_wall_cap_seconds_per_case'] != 14400
            or data['utility_processes_per_case'] != 86
            or data['implementation_freeze_required'] is not True):
        raise ValueError('registered controls/gates/budget/state mismatch')
    controls = {'bootstrap_samples':10000,'bootstrap_seed':20261003,'screening_seed':20261004,
        'default_controls':[8,'on'],'reference_processes':10,'memory_sample_interval_seconds':0.2,
        'paired_schedule':canonical_payload(balanced_schedule())}
    if any(canonical_sha256(data.get(k)) != canonical_sha256(v) for k,v in controls.items()):
        raise ValueError('registered statistical/default/pacing controls mismatch')
    root = Path(project).resolve()
    for relative, expected in data['input_files'].items():
        path = archived_input(root, relative)
        if not path.is_relative_to(root):
            raise ValueError('registration input path outside project')
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError('registration input identity mismatch: ' + relative)
    if any(p not in data['input_files'] for p in (HOST_REFERENCE,DEFAULT_REFERENCE)):
        raise ValueError('registration host/default source references not pinned')
    host_reference = json.loads(archived_input(root, HOST_REFERENCE).read_text())
    default_reference = json.loads(archived_input(root, DEFAULT_REFERENCE).read_text())
    host = host_reference['host_environment']
    if (canonical_sha256(data['host_environment']) != canonical_sha256(host)
            or data['host_environment_sha256'] != canonical_sha256(host)
            or data['default_source_proof'] != default_reference['default_source_proof']
            or data['source_repository'] != default_reference['source_repository']):
        raise ValueError('registration host/default source scope mismatch')
    if len(data['cases']) != 4:
        raise ValueError('requires exactly four registered cases')
    for case, (case_id, family, quantization) in zip(data['cases'], CASES):
        if (case['case_id'],case['family'],case['quantization'],case['status'],case['maximum_native_processes']) != (
                case_id,family,quantization,'NOT-RUN',107):
            raise ValueError('registered case scope/budget/state mismatch')
        required = (case['descriptor'],case['inventory'],case['workload'],case['prompt'],
                    case['runtime_identity'],data['hardware'],data['specification'])
        if any(path not in data['input_files'] for path in required):
            raise ValueError('registered case input is not pinned')
        descriptor = json.loads(archived_input(root, case['descriptor']).read_text())
        if (descriptor['family'], descriptor['quantization']) != (family,quantization):
            raise ValueError('registered descriptor scope mismatch')
        ids = case['candidate_ids']
        if (len(ids) != len(set(ids)) or len(ids) != 6 or case['default_id'] not in ids
                or len(case['screening_schedule']) != 3
                or any(len(block) != 6 or set(block) != set(ids) for block in case['screening_schedule'])):
            raise ValueError('registered candidate grid/schedule mismatch')
        model = inspect_model(archived_input(root, case['descriptor']),archived_input(root, case['inventory']))
        workload = WorkloadIR.from_reference(load_reference_workload(root,root/case['workload']))
        workload = replace(workload,threads=8,cuda_graphs='on')
        inputs = _snapshot_inputs({**host_reference['main_inputs'],'model':canonical_payload(model)},workload)
        default = CandidatePlan(inputs.identities(inputs.stock),RuntimeSettings(99,family == 'gemma4'))
        candidates = scheduling_space(default,host).candidates
        expected = {'model_artifact':canonical_payload(model.identity),'model_ir_sha256':canonical_sha256(model),
            'workload_sha256':canonical_sha256(workload),'runtime_sha256':inputs.stock.sha256,
            'fixed_settings':canonical_payload(default.settings),'default_id':default.candidate_id,
            'candidate_ids':[c.candidate_id for c in candidates],
            'screening_schedule':canonical_payload(screening_schedule(c.candidate_id for c in candidates)),
            'candidate_controls':[{'candidate_id':c.candidate_id,'threads':c.identities.workload.threads,
                'cuda_graphs':c.settings.cuda_graphs} for c in candidates],
            'provider_id':('gemma4-q4-x86-scheduling-v1' if family == 'gemma4' else 'granitemoe-q6-gpu-scheduling-v1')}
        if any(canonical_sha256(case.get(k)) != canonical_sha256(v) for k,v in expected.items()):
            raise ValueError('registration normalized inputs/candidates/schedule mismatch')
    return data


def inspect_registration(action, argv, project):
    if action != 'inspect':
        raise ValueError('coverage registration supports inspect only; wider collection needs a reviewed frozen collector')
    parser = argparse.ArgumentParser(prog='expertflow stock coverage inspect', allow_abbrev=False)
    parser.add_argument('--registration',type=Path,default=Path('configs/compiler/stock-coverage-20261005.json'))
    args = parser.parse_args(argv)
    root = Path(project).resolve()
    path = args.registration if args.registration.is_absolute() else root/args.registration
    if not path.is_file():
        raise EnvironmentBlocked('coverage registration unavailable; supply --project or --registration')
    data = verify_registration(json.loads(path.read_text(encoding='utf-8')),root)
    result = {'status':data['status'], 'registration_sha256':data['registration_sha256'],
        'native_calls':0, 'execution_ready':False, 'utility_gain_established':False,
        'weight_hashes_verified':False, 'input_file_pins_verified':len(data['input_files']),
        'maximum_native_processes':data['maximum_native_processes'],
        'cases':[{k:c[k] for k in ('case_id','family','quantization','status','maximum_native_processes')}
                 for c in data['cases']],
        'next':'Implement/review the separate wider collector and freeze all source/live identities before collection.'}
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0
