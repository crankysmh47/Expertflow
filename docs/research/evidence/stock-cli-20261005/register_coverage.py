"""Prepare exact wider test metadata; no model loading or native collection."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess

from expertflow.compiler.pipeline import inspect_model, atomic_json
from expertflow.compiler.plan import CandidatePlan, RuntimeSettings
from expertflow.compiler.preflight import file_sha256
from expertflow.compiler.reference import load_reference_workload
from expertflow.compiler.refinement import balanced_schedule
from expertflow.compiler.schema import WorkloadIR, canonical_payload, canonical_sha256
from expertflow.compiler.stock_discovery import _snapshot_inputs
from expertflow.compiler.stock_search import scheduling_space, screening_schedule
from expertflow.stock.coverage import GATES, HOST_REFERENCE, DEFAULT_REFERENCE, verify_registration

ROOT = Path.cwd()
SPEC = 'docs/superpowers/specs/2026-10-05-stock-coverage.md'
OUTPUT = Path('configs/compiler/stock-coverage-20261005.json')


def main():
    if OUTPUT.exists():
        raise ValueError('registration exists; no overwrite')
    original = json.loads(Path('docs/evidence/stock-repeatability-20261004/main-frozen-manifest.json').read_text())
    prior = json.loads(Path('C:/models/expertflow/runs/compiler-stock-repeatability-20261004/transfer/utility/frozen-manifest.json').read_text())
    definitions = [
        ('gemma4-q4','gemma4','Q4_0','configs/compiler/gemma4-q4-model.json',
         'docs/evidence/stock-discovery-20261004/q4-tensor-inventory.json',
         'docs/evidence/stock-discovery-20261004/q4-runtime-identity.json',
         'gemma4-q4-x86-scheduling-v1','configs/compiler/gemma4-q6-single-request.json',True),
        ('granite-q6','granitemoe','Q6_K','configs/compiler/granite-q6-model.json',
         'docs/evidence/compiler-granite-20261004/tensor-inventory.json',
         'docs/evidence/compiler-granite-20261004/runtime-identity.json',
         'granitemoe-q6-gpu-scheduling-v1','configs/compiler/granite-q6-single-request.json',False)]
    files = {SPEC,HOST_REFERENCE,DEFAULT_REFERENCE,'configs/compiler/runtime-stock.json',
             'docs/evidence/compiler-phase3/inputs/hardware.json'}
    cases = []
    for prefix,family,quant,descriptor,inventory,runtime,provider,base,cpu_moe in definitions:
        model = inspect_model(descriptor,inventory)
        runtime_model = json.loads(Path(runtime).read_text())['model']
        if (runtime_model['sha256'],runtime_model['size_bytes']) != (model.identity.sha256,model.identity.size_bytes):
            raise ValueError('model metadata and runtime input differ')
        for purpose,prompt in [('prose','configs/baseline-prompt.txt'),('code','configs/compiler/stock-utility-heldout-prompt.txt')]:
            case_id = prefix+'-'+purpose
            workload_file = f'configs/compiler/{case_id}-coverage.json'
            raw = json.loads(Path(base).read_text())
            raw['prompt_file'] = prompt
            atomic_json(workload_file,raw)
            workload = WorkloadIR.from_reference(load_reference_workload(ROOT,Path(workload_file)))
            workload = replace(workload,threads=8,cuda_graphs='on')
            snapshot = {**original['main_inputs'],'model':canonical_payload(model)}
            inputs = _snapshot_inputs(snapshot,workload)
            default = CandidatePlan(inputs.identities(inputs.stock),RuntimeSettings(99,cpu_moe))
            space = scheduling_space(default,original['host_environment'])
            ids = [c.candidate_id for c in space.candidates]
            cases.append({'case_id':case_id,'family':family,'quantization':quant,'status':'NOT-RUN',
                'maximum_native_processes':107,'descriptor':descriptor,'inventory':inventory,
                'runtime_identity':runtime,'workload':workload_file,'prompt':prompt,
                'model_artifact':canonical_payload(model.identity),'model_ir_sha256':canonical_sha256(model),
                'workload_sha256':canonical_sha256(workload),'runtime_sha256':inputs.stock.sha256,
                'provider_id':provider,'fixed_settings':canonical_payload(default.settings),
                'default_id':default.candidate_id,'candidate_ids':ids,
                'screening_schedule':canonical_payload(screening_schedule(ids)),
                'candidate_controls':[{ 'candidate_id':c.candidate_id,'threads':c.identities.workload.threads,
                    'cuda_graphs':c.settings.cuda_graphs } for c in space.candidates],
                'planned_root':f'C:/models/expertflow/runs/compiler-stock-coverage-20261005/{case_id}'})
            files.update((descriptor,inventory,runtime,workload_file,prompt))
    data = {'protocol_version':'wider-stock-utility-coverage-v1','status':'REGISTERED-NOT-RUN',
        'registered_at':'2026-10-05','source_checkpoint':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'implementation_freeze_required':True,'specification':SPEC,
        'hardware':'docs/evidence/compiler-phase3/inputs/hardware.json',
        'host_environment':original['host_environment'],'host_environment_sha256':canonical_sha256(original['host_environment']),
        'source_repository':prior['source_repository'],'default_source_proof':prior['default_source_proof'],
        'default_controls':[8,'on'],'paired_schedule':canonical_payload(balanced_schedule()),
        'bootstrap_samples':10000,'bootstrap_seed':20261003,'screening_seed':20261004,
        'wait_seconds':30,'memory_sample_interval_seconds':0.2,'reference_processes':10,
        'utility_processes_per_case':86,'maximum_native_processes':428,
        'collection_wall_cap_seconds_per_case':14400,'gates':GATES,
        'input_files':{p:file_sha256(Path(p)) for p in sorted(files)},'cases':cases,
        'scope':'One pinned Windows/NVIDIA host; four new utility comparisons, no untouched-workload or serving claim.'}
    data['registration_sha256'] = canonical_sha256(data)
    verify_registration(data,ROOT)
    atomic_json(OUTPUT,data)
    print(json.dumps({'status':data['status'],'cases':len(cases),'maximum_native_processes':428,'native_calls':0}))


if __name__ == '__main__':
    main()
