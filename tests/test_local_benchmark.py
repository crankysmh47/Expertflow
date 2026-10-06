import pytest


def test_measurement_rejects_missing_token_ids_or_bad_timings():
    from expertflow.product.benchmark import parse_completion
    with pytest.raises(ValueError, match="token"):
        parse_completion({"content":"text","timings":{"predicted_n":1,"predicted_per_second":10}}, "test")
    with pytest.raises(ValueError, match="timing|TPS"):
        parse_completion({"tokens":[1],"timings":{"predicted_n":1,"predicted_per_second":float('nan')}}, "test")


def test_measurement_preserves_tokens_prompt_identity_and_latency():
    from expertflow.product.benchmark import parse_completion
    result=parse_completion({"tokens":[4,5],"wall_seconds":0.2,"timings":{"predicted_n":2,"predicted_per_second":100,"predicted_ms":20,"prompt_n":4,"prompt_ms":10}}, "test")
    assert result["tokens"] == [4,5]
    assert result["decode_tps"] == 100
    assert result["prompt_tokens"] == 4
    assert result["wall_seconds"] == 0.2
    assert result["prompt_sha256"] == "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"


@pytest.mark.parametrize("mutation", ["prompt", "tokens", "generation", "timing"])
def test_incomplete_fixed_workload_cannot_be_measured(mutation):
    from expertflow.product.benchmark import parse_completion
    result={"tokens":[4,5],"wall_seconds":0.2,"timings":{"predicted_n":2,"predicted_per_second":100,"predicted_ms":20,"prompt_n":4,"prompt_ms":10}}
    if mutation == "prompt":del result["timings"]["prompt_n"]
    elif mutation == "tokens":result["tokens"]=[4]
    elif mutation == "generation":result["timings"]["predicted_n"]=1;result["tokens"]=[4]
    else:result["timings"]["prompt_ms"]=float('nan')
    with pytest.raises(ValueError,match="count|tokens|timing|generation"):
        parse_completion(result,"test",expected_predict=2)


def test_busy_gpu_blocks_measurement_without_closing_apps(tmp_path,monkeypatch):
    import subprocess
    from expertflow.product.benchmark import capture_performance_host
    gpu={"index":0,"name":"test","driver_version":"test","memory_total_mib":16384,"memory_free_mib":12000}
    monkeypatch.setattr("expertflow.doctor.collect_doctor_report",lambda root:{"gpus":[gpu],"system_ram_bytes":32*1024**3})
    monkeypatch.setattr("expertflow.compiler.preflight.capture_host_environment",lambda:{})
    monkeypatch.setattr("expertflow.product.benchmark.time.sleep",lambda seconds:None)
    monkeypatch.setattr("expertflow.product.benchmark.subprocess.run",lambda argv,**kw:subprocess.CompletedProcess(argv,0,"99\n",""))
    profile={"model":{"path":str(tmp_path / "model.gguf")},"host":{"gpus":[{k:v for k,v in gpu.items() if k != "memory_free_mib"}]}}
    with pytest.raises(ValueError,match="busy|idle"):
        capture_performance_host(profile)


def test_malformed_timing_object_has_actionable_input_error():
    from expertflow.product.benchmark import parse_completion
    with pytest.raises(ValueError,match="timing"):
        parse_completion({"tokens":[1],"timings":[]},"test")


def test_recent_owned_activity_can_settle_before_measurement(tmp_path,monkeypatch):
    import subprocess
    from expertflow.product.benchmark import capture_performance_host
    gpu={"index":0,"name":"test","driver_version":"test","memory_total_mib":16384,"memory_free_mib":12000}
    monkeypatch.setattr("expertflow.doctor.collect_doctor_report",lambda root:{"gpus":[gpu],"system_ram_bytes":32*1024**3})
    monkeypatch.setattr("expertflow.compiler.preflight.capture_host_environment",lambda:{})
    monkeypatch.setattr("expertflow.product.benchmark.time.sleep",lambda seconds:None)
    values=iter([99,0,0,0])
    monkeypatch.setattr("expertflow.product.benchmark.subprocess.run",lambda argv,**kw:subprocess.CompletedProcess(argv,0,str(next(values))+"\n",""))
    profile={"model":{"path":str(tmp_path / "model.gguf")},"host":{"gpus":[{k:v for k,v in gpu.items() if k != "memory_free_mib"}]}}
    assert capture_performance_host(profile)["gpus"] == profile["host"]["gpus"]
