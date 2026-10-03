import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.preflight import file_sha256
from expertflow.compiler.runner import RuntimeBinding, ServerMeasurementRunner, lower_launch
from expertflow.compiler.schema import ArtifactIdentity
from expertflow.compiler.stock import stock_candidate_matrix
from test_compiler_stock import identities


@pytest.fixture
def tiny_child(tmp_path):
    p = tmp_path / 'child.py'
    p.write_text(r'''
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
port, mode = int(sys.argv[1]), sys.argv[2]
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a): pass
 def do_GET(self):
  self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
 def do_POST(self):
  p=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  if self.path=='/tokenize': response={'tokens':[1,2]}
  else:
   if mode=='timeout': time.sleep(2)
   if mode=='slow_success': time.sleep(.3)
   response={'tokens':[] if mode=='malformed' else [10,11,12], 'timings':{'predicted_n':3,'predicted_ms':100}}
  if mode=='header_trickle' and self.path=='/completion':
   for byte in b'HTTP/1.0 200 OK\r\n': self.wfile.write(bytes([byte])); self.wfile.flush(); time.sleep(.3)
   self.wfile.write(b'Content-Length: '+str(len(json.dumps(response).encode())).encode()+b'\r\n\r\n')
  else: self.send_response(200); self.send_header('Content-Length',str(len(json.dumps(response).encode()))); self.end_headers()
  if mode=='trickle' and self.path=='/completion':
   for byte in json.dumps(response).encode(): self.wfile.write(bytes([byte])); self.wfile.flush(); time.sleep(.3)
  else: self.wfile.write(json.dumps(response).encode())
HTTPServer(('127.0.0.1',port),Handler).serve_forever()
''', encoding='utf-8')
    return p


def binding():
    p = Path(sys.executable)
    return RuntimeBinding(ArtifactIdentity(str(p), p.stat().st_size, file_sha256(p)), '{}', (), None)


def model_for_test(tmp_path):
    from dataclasses import replace
    from test_compiler_schema import model_fixture
    p=tmp_path/'model.gguf';p.write_bytes(b'GGUF'+bytes(12000))
    return replace(model_fixture(),identity=ArtifactIdentity(str(p),p.stat().st_size,file_sha256(p)))


def factory(child, mode, owned):
    def start(command, **kwargs):
        port = command[command.index('--port') + 1]
        process = subprocess.Popen([sys._base_executable, str(child), port, mode], **kwargs)
        owned.append(process)
        return process
    return start


def test_lowering_scrubs_inherited_controls_and_uses_frozen_settings(tmp_path):
    candidate = stock_candidate_matrix(identities())[1]
    model = ArtifactIdentity('model.gguf', 10, 'a' * 64)
    launch = lower_launch(candidate, model, binding(), 12345, tmp_path,
                          inherited={'PATH': 'path', 'EXPERTFLOW_TEST': '1',
                                     'LLAMA_ARG_CPU_MOE': '1', 'GGML_CUDA_DISABLE_GRAPHS': '1', 'KEEP': 'yes'})
    assert '--cpu-moe' in launch.argv
    assert launch.argv[launch.argv.index('-c') + 1] == '4096'
    assert launch.argv[launch.argv.index('-ctk') + 1] == 'f16'
    assert launch.environment['KEEP'] == 'yes'
    assert not any(name.startswith(('EXPERTFLOW', 'LLAMA_ARG', 'GGML_')) for name in launch.environment)
    assert launch.request['return_tokens'] is True and launch.request['stream'] is False


def test_compute_idle_uses_engine_activity_not_graphics_process_listing():
    from expertflow.compiler.runner import compute_activity_by_pid
    assert compute_activity_by_pid([('pid_123_luid_0_engtype_3D', 0, 25.0)]) == {}
    assert compute_activity_by_pid([('pid_123_luid_0_engtype_Compute_0', 0, 25.0)]) == {123: 25.0}
    with pytest.raises(RuntimeError, match='unavailable'):
        compute_activity_by_pid([])


@pytest.mark.parametrize('mode,memory_good,expected', [
    ('ok', True, 'measured'), ('malformed', True, 'validation_stop'),
    ('timeout', True, 'environment_blocked'), ('trickle', True, 'environment_blocked'), ('ok', False, 'environment_blocked')])
def test_owned_real_child_health_completion_failure_and_cleanup(tmp_path, tiny_child, mode, memory_good, expected):
    from dataclasses import replace
    from expertflow.compiler.plan import PlanIdentities
    from expertflow.compiler.schema import canonical_sha256
    i = identities()
    model=model_for_test(tmp_path)
    w = replace(i.workload, health_timeout_seconds=3, completion_timeout_seconds=1)
    i = replace(i, workload=w, workload_sha256=canonical_sha256(w), runtime_sha256=binding().sha256,
                model_sha256=canonical_sha256(model))
    candidate = stock_candidate_matrix(i)[1]
    owned = []
    sample = lambda pid: ({'pid':pid,'state':'allocated' if memory_good else 'unavailable','counter_available':memory_good,
                          'dedicated_bytes':100 if memory_good else None,'device_free_bytes':512 << 20}
                         if owned and owned[0].poll() is None else
                         {'pid':pid,'state':'absent','counter_available':True,'dedicated_bytes':0})
    store = EvidenceStore(tmp_path / 'store.sqlite3')
    runner = ServerMeasurementRunner(store, process_factory=factory(tiny_child, mode, owned),
                                      memory_sampler=sample, sample_interval=0.02)
    started = time.perf_counter()
    outcome = runner.run_once(candidate, model, binding(),
                              output_dir=tmp_path / 'run', measured=True)
    if mode in {'trickle', 'timeout'}:
        assert time.perf_counter() - started < 3
    assert outcome.status == expected
    assert len(owned) == 1 and owned[0].poll() is not None
    assert (tmp_path / 'run' / 'process.json').exists()
    assert b'\r\n' not in (tmp_path / 'run' / 'process.json').read_bytes()
    if expected == 'measured':
        row = store.verify_measurement(outcome.measurement_id)
        assert row['decode_tps'] == 30
        assert row['completion_wall_ms'] > 0
    else:
        assert outcome.reason and (tmp_path / 'run' / 'failure.json').exists()


@pytest.mark.parametrize('failure', ['measurement_gap','unknown_teardown','header_deadline'])
def test_memory_gaps_teardown_unknown_and_header_deadline_fail_closed(tmp_path,tiny_child,failure):
    from dataclasses import replace
    from expertflow.compiler.schema import canonical_sha256
    i=identities();w=replace(i.workload,health_timeout_seconds=3,completion_timeout_seconds=1)
    model=model_for_test(tmp_path)
    i=replace(i,workload=w,workload_sha256=canonical_sha256(w),runtime_sha256=binding().sha256,model_sha256=canonical_sha256(model))
    owned=[];readings=0
    def sample(pid):
        nonlocal readings
        if owned[0].poll() is not None: return None
        readings+=1
        if failure=='measurement_gap' and readings>1: return None
        return {'pid':pid,'state':'allocated','counter_available':True,'dedicated_bytes':100,'device_free_bytes':512 << 20}
    store=EvidenceStore(tmp_path/'store.sqlite3')
    mode='header_trickle' if failure=='header_deadline' else 'slow_success'
    runner=ServerMeasurementRunner(store,process_factory=factory(tiny_child,mode,owned),memory_sampler=sample,sample_interval=.02,
                                   teardown_timeout_seconds=.5)
    started=time.monotonic()
    outcome=runner.run_once(stock_candidate_matrix(i)[1],model,binding(),
                            output_dir=tmp_path/'run',measured=True)
    assert outcome.status=='environment_blocked'
    assert owned[0].poll() is not None
    if failure=='header_deadline': assert time.monotonic()-started < 3
