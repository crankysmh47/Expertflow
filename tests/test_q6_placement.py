from expertflow.analysis.q6_placement import build_placement_manifest


def test_cpu_moe_keeps_only_routed_experts_and_embedding_cpu_backed() -> None:
    inventory = {
        "tensors": [
            {"name": "token_embd.weight", "bytes": 100, "routed_expert_tensor": False},
            {"name": "blk.0.attn_q.weight", "bytes": 20, "routed_expert_tensor": False},
            {"name": "blk.0.ffn_down_exps.weight", "bytes": 80, "routed_expert_tensor": True},
        ]
    }

    result = build_placement_manifest(inventory)
    cpu_moe = {row["name"]: row["backend"] for row in result["modes"]["cpu_moe"]}
    full_cuda = {row["name"]: row["backend"] for row in result["modes"]["full_cuda_intent"]}

    assert cpu_moe == {
        "token_embd.weight": "CPU_Mapped",
        "blk.0.attn_q.weight": "CUDA0",
        "blk.0.ffn_down_exps.weight": "CPU_Mapped",
    }
    assert full_cuda["blk.0.ffn_down_exps.weight"] == "CUDA0"
    assert result["modes_summary"]["cpu_moe"]["CPU_Mapped"] == 180
    assert result["modes_summary"]["cpu_moe"]["CUDA0"] == 20
