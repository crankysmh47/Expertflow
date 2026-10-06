"""Throwaway three-run synchronized profiler; intentionally not a compiler product path."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
from ctypes import wintypes

sys.path.insert(0,str(Path(__file__).parent))
from console_stop import creation
from expertflow.compiler.pipeline import CompilationRequest,load_compiler_inputs,atomic_json
from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler import runner as native
from expertflow.compiler.diagnostics import DiagnosticSampler
from expertflow.compiler.plan import CandidatePlan,RuntimeSettings
from expertflow.compiler.preflight import file_sha256
from expertflow.compiler.schema import canonical_payload
from expertflow.compiler.refinement import diagnostic_summary

root=Path('C:/models/expertflow/runs/compiler-profile-20261004')
controls=json.loads((root/'controls-report.json').read_text())
assert controls['status']=='PASS-CONTROLS'
assert not (root/'profiles.sqlite3').exists()
request=CompilationRequest(Path('configs/compiler/gemma4-q6-model.json'),Path('docs/evidence/q6-download/tensor-inventory.json'),
    Path('docs/evidence/compiler-phase3/inputs/hardware.json'),Path('configs/compiler/gemma4-q6-single-request.json'),
    Path('docs/evidence/compiler-phase3/inputs/runtime-identity.json'),(),root/'profiles.sqlite3',root/'profiles')
inputs=load_compiler_inputs(request,live=True)
shutil.copyfile(root/'controls.sqlite3',root/'profiles.sqlite3')
store=EvidenceStore(root/'profiles.sqlite3');store.prime_model(inputs.model)
w=inputs.workload
candidate=CandidatePlan(inputs.identities(inputs.fork),RuntimeSettings(99,True,w.cuda_graphs,w.kv_type_k,w.kv_type_v,w.batch_size,w.microbatch_size))
base_lower=native.lower_launch
active_profile=None

def diagnostic_lower(*args,**kwargs):
    lowered=base_lower(*args,**kwargs)
    binding=args[2]
    if active_profile is not None and binding.sha256==inputs.fork.sha256:
        return replace(lowered,environment={**lowered.environment,'LLAMA_EXPERTFLOW_SPLIT_PROFILE':str(active_profile)})
    return lowered

children=[]
class GracefulChild(subprocess.Popen):
    def __init__(self,*args,**kwargs):
        kwargs['creationflags']=subprocess.CREATE_NEW_CONSOLE
        info=subprocess.STARTUPINFO();info.dwFlags=subprocess.STARTF_USESHOWWINDOW;info.wShowWindow=0
        kwargs['startupinfo']=info
        super().__init__(*args,**kwargs)
        self.interrupt_error=None;children.append(self)
    def terminate(self):
        if self.poll() is not None:return
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('console_stop.py')),
            str(self.pid),str(creation(wintypes.HANDLE(int(self._handle))))],
            creationflags=subprocess.CREATE_NO_WINDOW,capture_output=True,timeout=10)
        if result.returncode:
            self.interrupt_error=result.stderr.decode(errors='replace');super().terminate()

report={'status':'RUNNING','diagnostic_only':True,'fixed_profile_budget':3,'runs':[],
    'profile_policy':'isolated throwaway lower_launch override permits only the frozen fork profiling output path; measured=False; ordinary product verifier rejects profiling controls',
    'phase_limitation':'native counters aggregate initialization, warmup, prefill and decode; decode_calls is a tensor-shape heuristic, not a trustworthy phase label',
    'candidate':canonical_payload(candidate),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'source_files':{str(p):file_sha256(p) for p in [Path(__file__),Path(__file__).with_name('console_stop.py'),Path('src/expertflow/compiler/runner.py')]},
    'controls_database_sha256':file_sha256(root/'controls.sqlite3'),'inputs':inputs.provenance}
atomic_json(root/'profiles-frozen.json',report)
sampler=DiagnosticSampler(native.WindowsGpuMemorySampler(inputs.hardware.gpu_uuid))
runner=native.ServerMeasurementRunner(store,memory_sampler=sampler,process_factory=GracefulChild)
native.lower_launch=diagnostic_lower
try:
    for index in range(3):
        directory=(root/'profiles'/f'run-{index+1}').resolve()
        active_profile=directory/'split-profile.json'
        outcome=runner.run_once(candidate,inputs.model,inputs.fork,output_dir=directory,measured=False,
            stage=f'synchronized-profile-{index+1}',numerical_path='fork_off_vs_pristine',
            comparison_ids=(controls['runs'][0]['outcome']['measurement_id'],))
        row={'outcome':canonical_payload(outcome)};report['runs'].append(row)
        if outcome.status!='measured' or children[-1].interrupt_error:
            report.update(status='PROFILE-STOP',reason=outcome.reason or children[-1].interrupt_error);break
        row['verified']=store.verify_measurement(outcome.measurement_id)
        row['diagnostics']=diagnostic_summary(store.measurement(outcome.measurement_id))
        assert json.loads((directory/'process.json').read_text())['child_exit_code']==0,'graceful native exit required'
        profile=json.loads(active_profile.read_text())
        assert profile['diagnostic_synchronization'] is True and profile['records']
        row['profile']={'path':str(active_profile),'sha256':file_sha256(active_profile),'size_bytes':active_profile.stat().st_size}
        atomic_json(root/'profiles-report.json',report)
    else:
        report['status']='PROFILED-AGGREGATE'
    assert file_sha256(root/'controls.sqlite3')==report['controls_database_sha256']
except (AssertionError,ValueError,OSError,KeyError,TypeError) as error:
    report.update(status='PROFILE-STOP',reason=str(error))
finally:
    native.lower_launch=base_lower;sampler.close();atomic_json(root/'profiles-report.json',report)
print(json.dumps({'status':report['status'],'runs':len(report['runs']),'reason':report.get('reason')}))
