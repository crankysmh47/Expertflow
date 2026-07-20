import pytest

from expertflow.analysis.q6_inventory import summarize_tensor_inventory


def test_summarizes_complete_routed_expert_bundles() -> None:
    tensors = [
        {"name": "blk.0.ffn_gate_up_exps.weight", "type": "Q6_K", "shape": [2816, 1408, 128], "bytes": 1280},
        {"name": "blk.0.ffn_down_exps.weight", "type": "Q6_K", "shape": [704, 2816, 128], "bytes": 640},
        {"name": "blk.0.ffn_down_exps.scale", "type": "F32", "shape": [128], "bytes": 512},
        {"name": "blk.0.attn_q.weight", "type": "Q6_K", "shape": [10, 10], "bytes": 100},
    ]

    result = summarize_tensor_inventory(tensors, expert_count=128)

    assert result["routed_layer_ids"] == [0]
    assert result["expert_component_names"] == [
        "ffn_down_exps.scale",
        "ffn_down_exps.weight",
        "ffn_gate_up_exps.weight",
    ]
    assert result["layers"][0]["complete_expert_bundle_bytes"] == 19
    assert result["total_routed_expert_bank_bytes"] == 2432
    assert result["non_expert_tensor_bytes"] == 100


def test_rejects_nondivisible_expert_tensor_bytes() -> None:
    tensors = [{"name": "blk.0.ffn_down_exps.weight", "type": "Q6_K", "shape": [1, 1, 128], "bytes": 129}]

    with pytest.raises(ValueError, match="divisible"):
        summarize_tensor_inventory(tensors, expert_count=128)
