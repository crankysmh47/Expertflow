from dataclasses import replace
import json
import time
import uuid
from pathlib import Path

import pytest

from expertflow.compiler.evidence import EvidenceArtifact, EvidenceStore, MeasurementKey, MeasurementRecord
from expertflow.compiler.pipeline import CompilationRequest, CompilerInputs, compile_phase3
from expertflow.compiler.plan import CandidatePlan, PlanIdentities, RuntimeSettings
from expertflow.compiler.preflight import file_sha256
from expertflow.compiler.runner import MeasurementOutcome, RuntimeBinding
from expertflow.compiler.schema import ArtifactIdentity, HardwareIR, WorkloadIR, canonical_payload, canonical_sha256
from test_compiler_schema import model_fixture
from test_compiler_preflight import evidence_root, tiny_external
from test_compiler_gemma4_adapter import inventory_fixture, descriptor_fixture


def inputs(tmp_path):
    model = model_fixture()
    model_path=tmp_path/'model.gguf';model_path.write_bytes(b'GGUF'+bytes(12000))
    model=replace(model,identity=ArtifactIdentity(str(model_path),model_path.stat().st_size,file_sha256(model_path)))
    w = WorkloadIR('hello', 4096, 3)
    hardware = HardwareIR('GPU-test','RTX','12.0', 16000 << 20, 14000 << 20,'616.92','12.8','b' * 64)
    exe = tmp_path / 'runtime'
    exe.write_bytes(b'fixture')
    artifact = ArtifactIdentity(str(exe), exe.stat().st_size, file_sha256(exe))
    stock = RuntimeBinding(artifact, '{"patches":[]}', (), None)
    fork = RuntimeBinding(artifact, '{"patches":["fixture-patch"]}', (), None)
    profile_rows = tuple({'layer_id': i, 'total_us': 1000, 'backend':'CPU', 'profile_id':str(j)}
                         for i in (0,1) for j in range(3))
    return CompilerInputs(model, hardware, w, stock, fork, profile_rows, {})


def request(tmp_path, recorded=None):
    return CompilationRequest(*(tmp_path / name for name in ('descriptor','inventory','hardware','workload','runtime')),
                              (), tmp_path / 'store.sqlite3', tmp_path / 'output', recorded)


class FakeRunner:
    """Search-control fixture generating complete tiny native evidence records."""
    def __init__(self, store, fail=False, replay_tps=30):
        self.store, self.fail, self.replay_tps = store, fail, replay_tps
        self.calls = []

    def run_once(self, candidate, model, binding, *, output_dir, measured, stage='initial',
                 numerical_path='stock_same_runtime', comparison_ids=(), host_environment=None):
        self.calls.append((candidate.candidate_id, stage, candidate.settings.static))
        if self.fail:
            return MeasurementOutcome('environment_blocked',None,'counter unavailable',None,str(output_dir))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True)
        w = candidate.identities.workload
        tps = self.replay_tps if stage == 'sealed_replay' else 30
        prompt = [1,2]
        from expertflow.compiler.runner import lower_launch
        launch=lower_launch(candidate,model.identity,binding,12345,output_dir,inherited={'PATH':'path'})
        run_id=str(uuid.uuid4());creation=time.time_ns()//100
        sample={'pid':123,'state':'allocated','counter_available':True,'dedicated_bytes':1000,
                'device_free_bytes':512 << 20,'phase':'measurement'}
        payloads = {
            'completion':{'tokens':[10,11,12],'timings':{'predicted_n':3,'predicted_ms':3000/tps}},
            'tokenize':{'tokens':prompt},
            'tokenize-request':{'content':w.prompt,'add_special':True},
            'request':{'prompt':prompt,'n_predict':w.predict_tokens,'seed':w.seed,'temperature':w.temperature,
                       'ignore_eos':True,'cache_prompt':False,'return_tokens':True,'stream':False},
            'memory':{'samples':[sample],'observations':[sample],
                      'teardown_reading':{'pid':123,'state':'absent','counter_available':True,'dedicated_bytes':0}},
            'process':{'pid':123,'exited':True,'cleanup':True,'exit_code':0,'memory_settled':True,
                       'run_id':run_id,'creation_time_100ns':creation},
            'run-start':{'pid':123,'run_id':run_id,'creation_time_100ns':creation,
                         'creation_source':'GetProcessTimes','started_monotonic_ns':time.monotonic_ns()},
            'completion-wall':{'started_monotonic_ns':100000000,'finished_monotonic_ns':200000000,'elapsed_ms':100},
            'launch':{'candidate_id':candidate.candidate_id,'settings_sha256':canonical_sha256(candidate.settings),
                      'runtime_binding':canonical_payload(binding),
                      'model_ir':canonical_payload(model),'argv':launch.argv,'environment':launch.environment,
                      'runtime_sha256':candidate.identities.runtime_sha256,'model_sha256':candidate.identities.model_sha256,
                      'workload_sha256':candidate.identities.workload_sha256},
        }
        artifacts=[]
        if host_environment is not None:
            payloads['launch']['host_environment'] = canonical_payload(host_environment)
        for role,payload in payloads.items():
            p=(output_dir/f'{role}.json').resolve()
            p.write_text(json.dumps(payload))
            artifacts.append(EvidenceArtifact(role,ArtifactIdentity(str(p),p.stat().st_size,file_sha256(p))))
        record=MeasurementRecord(MeasurementKey.from_candidate(candidate),candidate.candidate_id,
            json.dumps(canonical_payload(candidate.identities)),json.dumps(canonical_payload(candidate.settings)),
            tuple(artifacts),json.dumps({'decode_tps':tps}),
            json.dumps({'exact_tokens':True,'memory':True,'cleanup':True}),0,measured,stage,numerical_path,comparison_ids)
        mid=self.store.append_measurement(record)
        self.store.verify_measurement(mid)
        # auto/all/99 resolve identically; CPU-MoE remains a distinct placement.
        return MeasurementOutcome('measured',mid,None,str(candidate.settings.cpu_moe),str(output_dir))


def test_live_pipeline_confirms_and_replays_selected_stock_not_static(tmp_path, monkeypatch):
    inp = inputs(tmp_path)
    monkeypatch.setattr('expertflow.compiler.pipeline.load_compiler_inputs', lambda *a,**k:inp)
    store = EvidenceStore(tmp_path/'store.sqlite3')
    runner = FakeRunner(store)
    result = compile_phase3(request(tmp_path), runner, store)
    assert result.status == 'PASS-STOCK-FALLBACK'
    assert result.execution_plan.candidate.settings.static is None
    assert result.execution_plan.fallback is None
    assert len([c for c in runner.calls if c[1]=='confirmation']) == 10
    assert runner.calls[-1][:2] == (result.execution_plan.candidate.candidate_id,'sealed_replay')
    assert result.report['rejected_candidates']
    assert result.report['duplicate_placements']
    assert (tmp_path/'output/execution-plan.json').is_file()


@pytest.mark.parametrize('failure,replay_tps,status', [(True,30,'ENVIRONMENT-BLOCKED'), (False,20,'VALIDATION-STOP')])
def test_required_runner_failure_or_selected_replay_failure_emits_no_validated_plan(tmp_path, monkeypatch, failure,replay_tps,status):
    inp = inputs(tmp_path)
    monkeypatch.setattr('expertflow.compiler.pipeline.load_compiler_inputs',lambda *a,**k:inp)
    store = EvidenceStore(tmp_path/'store.sqlite3')
    result=compile_phase3(request(tmp_path),FakeRunner(store,fail=failure,replay_tps=replay_tps),store)
    assert result.status == status
    assert result.execution_plan is None
    assert not (tmp_path/'output/execution-plan.json').exists()
    assert (tmp_path/'output/explanation.json').is_file()


def test_disk_inputs_verify_model_runtime_cuda_and_workload(tiny_external, monkeypatch):
    from expertflow.compiler.pipeline import load_compiler_inputs
    root, model, stock, fork, cuda = tiny_external
    monkeypatch.chdir(root)
    model.write_bytes(b'GGUF' + bytes(12000))
    identity = ArtifactIdentity(str(model), model.stat().st_size, file_sha256(model))
    inventory = inventory_fixture()
    inventory['model'] = {'path':identity.path,'bytes':identity.size_bytes,'sha256':identity.sha256}
    inventory_path = root/'inventory.json'; inventory_path.write_text(json.dumps(inventory))
    descriptor_path=root/'descriptor.json';descriptor_path.write_text(json.dumps(canonical_payload(descriptor_fixture())))
    hardware=HardwareIR('GPU-test','RTX','12.0',16000 << 20,14000 << 20,'616.92','12.8',file_sha256(cuda))
    hardware_path=root/'hardware.json';hardware_path.write_text(json.dumps(canonical_payload(hardware)))
    runtime_path=root/'runtime.json';runtime_path.write_text(json.dumps({'schema_version':'1.0.0','model':canonical_payload(identity),
        'stock':{'manifest_path':str(root/'configs/compiler/runtime-stock.json'),'binary_dir':str(stock),'cuda_runtime':str(cuda)},
        'fork':{'manifest_path':str(root/'configs/compiler/runtime-fork.json'),'binary_dir':str(fork),'cuda_runtime':str(cuda)}}))
    req=CompilationRequest(descriptor_path,inventory_path,hardware_path,root/'configs/compiler/gemma4-q6-single-request.json',
        runtime_path,(),root/'store.sqlite3',root/'output')
    loaded=load_compiler_inputs(req,live=False)
    assert loaded.model.identity == identity
    assert loaded.workload.context_size == 4096
    assert loaded.stock.sha256 != loaded.fork.sha256
    model.write_bytes(b'GGUF' + bytes(11999) + b'x')
    with pytest.raises(ValueError,match='model artifact identity'):
        load_compiler_inputs(req,live=False)


def test_committed_historical_replay_is_diagnostic_and_never_runs_or_seals(tmp_path, monkeypatch):
    inp = inputs(tmp_path)
    monkeypatch.setattr('expertflow.compiler.pipeline.load_compiler_inputs',lambda *a,**k:inp)
    store=EvidenceStore(tmp_path/'store.sqlite3')
    runner=FakeRunner(store,fail=True)
    result=compile_phase3(request(tmp_path, Path('docs/evidence/q6-placement-final/results.json')),runner,store)
    assert result.status == 'RECORDED-DIAGNOSTIC'
    assert result.report['historical_cli']['mean_decode_tps'] == {'off':22.28,'on':28.13}
    assert result.report['earlier_strongest_stock_decode_tps'] == 22.966667
    assert runner.calls == [] and result.execution_plan is None
    assert not (tmp_path/'output/execution-plan.json').exists()


def test_memory_unsafe_stock_candidate_is_rejected_without_discarding_safe_floor(tmp_path, monkeypatch):
    inp=inputs(tmp_path)
    monkeypatch.setattr('expertflow.compiler.pipeline.load_compiler_inputs',lambda *a,**k:inp)
    store=EvidenceStore(tmp_path/'store.sqlite3')
    class UnsafeCudaRunner(FakeRunner):
        def run_once(self,candidate,model,binding,**kwargs):
            if not candidate.settings.cpu_moe:
                return MeasurementOutcome('validation_stop',None,'VRAM reserve violated',None,str(kwargs['output_dir']))
            return super().run_once(candidate,model,binding,**kwargs)
    result=compile_phase3(request(tmp_path),UnsafeCudaRunner(store),store)
    assert result.status == 'PASS-STOCK-FALLBACK'
    assert result.execution_plan.candidate.settings.cpu_moe is True
    assert any(r.get('reason') == 'vram_budget' for r in result.report['rejected_candidates'])
