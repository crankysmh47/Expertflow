import pytest

from expertflow.analysis.expert_layout import (
    ExpertTensor,
    build_expert_inventory,
    expert_tensor_from_gguf_metadata,
)


def real_shape_tensors(layer_id: int, base_offset: int) -> list[ExpertTensor]:
    return [
        ExpertTensor(
            layer_id=layer_id,
            component="down_weight",
            tensor_name=f"blk.{layer_id}.ffn_down_exps.weight",
            tensor_type="Q4_0",
            shape=(704, 2816, 2),
            expert_count=2,
            encoded_bytes=1_115_136 * 2,
            data_offset=base_offset,
            ne0=704,
            quant_block_elements=32,
            quant_block_bytes=18,
        ),
        ExpertTensor(
            layer_id=layer_id,
            component="gate_up_weight",
            tensor_name=f"blk.{layer_id}.ffn_gate_up_exps.weight",
            tensor_type="Q4_0",
            shape=(2816, 1408, 2),
            expert_count=2,
            encoded_bytes=2_230_272 * 2,
            data_offset=base_offset + 10_000_000,
            ne0=2816,
            quant_block_elements=32,
            quant_block_bytes=18,
        ),
        ExpertTensor(
            layer_id=layer_id,
            component="down_scale",
            tensor_name=f"blk.{layer_id}.ffn_down_exps.scale",
            tensor_type="F32",
            shape=(2,),
            expert_count=2,
            encoded_bytes=8,
            data_offset=base_offset + 20_000_000,
            ne0=1,
            quant_block_elements=None,
            quant_block_bytes=None,
        ),
    ]


def test_reconciles_encoded_spans_row_padding_and_aligned_slots() -> None:
    report = build_expert_inventory(
        real_shape_tensors(0, 1_000) + real_shape_tensors(1, 50_000_000),
        alignment=128,
        matrix_row_padding=512,
        component_order=("down_weight", "gate_up_weight", "down_scale"),
        target_layer_ids=(0, 1),
        capacity_per_layer=96,
    )

    assert report["layer_count"] == 2
    assert report["expert_count_per_layer"] == 2
    assert report["object_count"] == 4
    assert report["all_objects_same_encoded_bytes"] is True
    assert report["all_objects_same_projected_slot_bytes"] is True
    first = report["objects"][0]
    second = report["objects"][1]
    assert first["encoded_packed_bytes"] == 3_345_412
    assert first["projected_slot_bytes"] == 3_346_048
    assert [item["cuda_row_padding_bytes"] for item in first["components"]] == [
        180,
        144,
        0,
    ]
    assert first["components"][0]["source_start"] == 1_000
    assert second["components"][0]["source_start"] == 1_000 + 1_115_136
    assert first["slot_end_padding_bytes"] == 124
    assert report["projected_cache_bytes"] == 3_346_048 * 2 * 96


def test_rejects_tensor_bytes_not_divisible_by_experts() -> None:
    tensor = real_shape_tensors(0, 0)[0]
    invalid = ExpertTensor(
        **{**tensor.__dict__, "encoded_bytes": tensor.encoded_bytes + 1}
    )

    with pytest.raises(ValueError, match="divisible"):
        build_expert_inventory(
            [invalid],
            alignment=128,
            matrix_row_padding=512,
            component_order=("down_weight",),
            target_layer_ids=(0,),
            capacity_per_layer=1,
        )


def test_maps_pinned_gguf_tensor_metadata_to_inventory_contract() -> None:
    tensor = expert_tensor_from_gguf_metadata(
        name="blk.7.ffn_gate_up_exps.weight",
        shape=(2816, 1408, 128),
        tensor_type="Q4_0",
        encoded_bytes=285_474_816,
        data_offset=791_746_016,
    )

    assert tensor == ExpertTensor(
        layer_id=7,
        component="gate_up_weight",
        tensor_name="blk.7.ffn_gate_up_exps.weight",
        tensor_type="Q4_0",
        shape=(2816, 1408, 128),
        expert_count=128,
        encoded_bytes=285_474_816,
        data_offset=791_746_016,
        ne0=2816,
        quant_block_elements=32,
        quant_block_bytes=18,
    )
    assert (
        expert_tensor_from_gguf_metadata(
            name="blk.7.attn_q.weight",
            shape=(10, 10),
            tensor_type="Q4_0",
            encoded_bytes=100,
            data_offset=0,
        )
        is None
    )
