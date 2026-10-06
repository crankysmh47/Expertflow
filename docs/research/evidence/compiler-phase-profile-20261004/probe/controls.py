"""Throwaway matched-workload profiling controls; no product-plan publication."""
import json
from pathlib import Path
import subprocess
from expertflow.compiler.pipeline import CompilationRequest, load_compiler_inputs, atomic_json
from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.runner import ServerMeasurementRunner, WindowsGpuMemorySampler
from expertflow.compiler.diagnostics import DiagnosticSampler
from expertflow.compiler.plan import CandidatePlan, RuntimeSettings
from expertflow.compiler.preflight import file_sha256
from expertflow.compiler.schema import canonical_payload

root = Path('C:/models/expertflow/runs/compiler-phase-profile-20261004')
assert not (root/'controls.sqlite3').exists()
request = CompilationRequest(Path('configs/compiler/gemma4-q6-model.json'),
    Path('docs/evidence/q6-download/tensor-inventory.json'), Path('docs/evidence/compiler-phase3/inputs/hardware.json'),
    Path('configs/compiler/gemma4-q6-single-request.json'), Path('C:/models/expertflow/runs/compiler-phase-profile-20261004/runtime-identity.json'),
    (),root/'controls.sqlite3',root/'controls')
inputs=load_compiler_inputs(request,live=True)
store=EvidenceStore(root/'controls.sqlite3');store.prime_model(inputs.model)
w=inputs.workload
settings=RuntimeSettings(99,True,w.cuda_graphs,w.kv_type_k,w.kv_type_v,w.batch_size,w.microbatch_size)
report={'status':'RUNNING','diagnostic_only':True,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'probe_sha256':file_sha256(Path(__file__)), 'inputs':inputs.provenance,'runs':[]}
atomic_json(root/'controls-frozen.json',report)
sampler=DiagnosticSampler(WindowsGpuMemorySampler(inputs.hardware.gpu_uuid))
runner=ServerMeasurementRunner(store,memory_sampler=sampler)
try:
    for arm,binding in [('stock',inputs.stock),('fork-off',inputs.fork)]:
        candidate=CandidatePlan(inputs.identities(binding),settings)
        outcome=runner.run_once(candidate,inputs.model,binding,output_dir=root/'controls'/arm,
            measured=False,stage='profile-control-'+arm,
            numerical_path='stock_same_runtime' if arm=='stock' else 'fork_off_vs_pristine',
            comparison_ids=() if arm=='stock' else (report['runs'][0]['outcome']['measurement_id'],))
        row={'arm':arm,'outcome':canonical_payload(outcome)}
        report['runs'].append(row)
        if outcome.status!='measured':
            report.update(status=outcome.status.upper().replace('_','-'),reason=outcome.reason);break
        row['verified']=store.verify_measurement(outcome.measurement_id)
        if arm=='stock':
            aa=json.loads(Path('docs/evidence/compiler-refinement/aa-report.json').read_text())['rows'][0]
            assert all(row['verified'][k]==aa[k] for k in ('identities','settings_sha256','prompt_tokens_sha256','generated_tokens_sha256'))
        atomic_json(root/'controls-report.json',report)
    else:
        report['status']='PASS-CONTROLS'
finally:
    sampler.close();atomic_json(root/'controls-report.json',report)
print(json.dumps({'status':report['status'],'runs':len(report['runs']),'reason':report.get('reason')}))
