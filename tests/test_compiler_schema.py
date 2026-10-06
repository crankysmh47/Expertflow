from dataclasses import FrozenInstanceError, replace
import json

import pytest

from expertflow.compiler.schema import (
    ArtifactIdentity, ExactnessPolicy, HardwareIR, ModelIR, MoELayerIR,
    Objective, WorkloadIR, canonical_payload, canonical_sha256,
)


def model_fixture():
    return ModelIR('1.0.0', ArtifactIdentity('C:/model.gguf', 10, 'a' * 64),
                   'gemma4', 'gemma4-moe', 'Q6_K', 128, 8,
                   (MoELayerIR(1, 128, 8, 30, 3840), MoELayerIR(0, 128, 8, 20, 2560)),
                   'standard', 'none')


def workload_fixture():
    return WorkloadIR(prompt='hello\r\n', context_size=4096, predict_tokens=512)


def test_canonical_order_round_trip_and_immutable_model():
    model = model_fixture()
    assert [x.layer_id for x in model.moe_layers] == [0, 1]
    payload = canonical_payload(model)
    assert json.loads(json.dumps(payload)) == payload
    assert canonical_sha256(model) == canonical_sha256(payload)
    assert canonical_sha256(replace(model, moe_layers=tuple(reversed(model.moe_layers)))) == canonical_sha256(model)
    with pytest.raises(FrozenInstanceError):
        model.family = 'other'
    payload['moe_layers'].clear()
    assert len(model.moe_layers) == 2


@pytest.mark.parametrize('change', [
    {'sha256': 'x' * 64}, {'sha256': 'A' * 64}, {'size_bytes': 0}, {'size_bytes': True}, {'path': ''},
])
def test_artifact_rejects_invalid_identity(change):
    with pytest.raises(ValueError):
        replace(model_fixture().identity, **change)


@pytest.mark.parametrize('change', [
    {'expert_bundle_bytes': 0}, {'routed_expert_bank_bytes': -1},
    {'expert_top_k': 129}, {'layer_id': -1}, {'expert_count': True},
])
def test_invalid_layer(change):
    with pytest.raises(ValueError):
        replace(model_fixture().moe_layers[0], **change)


def test_duplicate_or_inconsistent_layers():
    model = model_fixture()
    with pytest.raises(ValueError, match='duplicate'):
        replace(model, moe_layers=(model.moe_layers[0],) * 2)
    with pytest.raises(ValueError, match='inconsistent'):
        replace(model, moe_layers=(MoELayerIR(0, 64, 8, 10, 640),))


@pytest.mark.parametrize('change', [
    {'concurrency': 2}, {'temperature': 0.1}, {'kv_type_k': 'q8_0'},
    {'cache_prompt': True}, {'ignore_eos': False}, {'runtime_interface': 'historical_cli'},
    {'microbatch_size': 4096}, {'maximum_cv_pct': float('nan')},
    {'objective': 'latency'}, {'policy': 'mystery'}, {'prompt': '  '},
    {'context_size': True}, {'measured_runs': 0}, {'approximate_quality_budget': 0.01},
])
def test_invalid_or_approximate_exact_workload(change):
    with pytest.raises(ValueError):
        replace(workload_fixture(), **change)


def test_workload_identity_covers_controls_and_raw_prompt_bytes():
    w = workload_fixture()
    assert w.policy is ExactnessPolicy.EXACT
    assert w.objective is Objective.DECODE_TPS
    for change in ({'prompt': 'hello\n'}, {'context_size': 8192}, {'seed': 43},
                   {'cuda_graphs': 'off'}, {'confirmation_pairs': 11}):
        assert canonical_sha256(w) != canonical_sha256(replace(w, **change))


def test_hardware_identity_and_nonfinite_values():
    hardware = HardwareIR(gpu_uuid='GPU-test', gpu_name='RTX', compute_capability='12.0',
                          total_vram_bytes=16 << 30, usable_vram_bytes=14 << 30,
                          driver_version='616.92', cuda_version='12.8',
                          cuda_runtime_sha256='b' * 64)
    assert canonical_sha256(hardware) != canonical_sha256(replace(hardware, driver_version='other'))
    with pytest.raises(ValueError):
        replace(hardware, usable_vram_bytes=17 << 30)
    with pytest.raises(ValueError):
        canonical_payload({'bad': float('inf')})
    with pytest.raises(ValueError):
        canonical_payload({'bad': object()})
