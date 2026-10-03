from dataclasses import replace

import pytest

from expertflow.compiler.adapters import Gemma4Adapter
from expertflow.compiler.static_placement import generate_static_candidates, rank_layer_benefits
from expertflow.compiler.schema import MoELayerIR
from test_compiler_schema import model_fixture


def rows():
    return [{'layer_id': 0, 'total_us': 1000, 'backend': 'CPU', 'profile_id': 'one'},
            {'layer_id': 1, 'total_us': 500, 'backend': 'CPU', 'profile_id': 'one'}]


def test_generic_ranking_alignment_and_deterministic_prefixes():
    model = replace(model_fixture(), moe_layers=(MoELayerIR(0,128,8,10,1280, (641,639)),
                                                 MoELayerIR(1,128,8,10,1280, (1280,))))
    benefits = rank_layer_benefits(model, rows())
    assert [b.layer_id for b in benefits] == [0,1]
    assert benefits[0].arena_bytes == 1536
    candidates = generate_static_candidates(benefits, available_bytes=4000, reserve_bytes=1000)
    assert candidates[0].layer_ids == ()
    assert [c.layer_ids for c in candidates[1:] if c.kind == 'greedy'] == [(0,), (0,1)]
    assert all(c.arena_bytes <= 3000 for c in candidates if not c.rejection_reasons or c.rejection_reasons == ('numerical_path_change',))
    assert all('numerical_path_change' in c.rejection_reasons for c in candidates if c.layer_ids)


@pytest.mark.parametrize('mutation', [
    lambda r: r.append(r[0]), lambda r: r.pop(),
    lambda r: r[0].update(backend='CUDA0'), lambda r: r[0].update(total_us=float('nan')),
])
def test_invalid_profiles(mutation):
    r = rows()
    mutation(r)
    with pytest.raises(ValueError):
        rank_layer_benefits(model_fixture(), r)


def test_memory_and_runtime_caps_diagnostic_candidate_and_no_fit():
    base = model_fixture()
    model = replace(base, moe_layers=tuple(MoELayerIR(i,128,8,10,1280) for i in range(30)))
    benefits = rank_layer_benefits(model, [{'layer_id': i, 'total_us': 1000-i, 'backend': 'CPU'} for i in range(30)])
    candidates = generate_static_candidates(benefits, available_bytes=100000, reserve_bytes=1000,
                                            diagnostic_layers=tuple(range(19)))
    assert max(len(c.layer_ids) for c in candidates if c.kind == 'greedy') == 12
    assert 'runtime_layer_cap' in candidates[-1].rejection_reasons
    assert len(candidates) <= 14
    limited = generate_static_candidates(benefits, available_bytes=100000, reserve_bytes=1000, max_static_shadows=4)
    assert max(len(c.layer_ids) for c in limited if c.kind == 'greedy') == 1
    no_fit = generate_static_candidates(benefits, available_bytes=1000, reserve_bytes=1000)
    assert no_fit[0].layer_ids == ()
    assert any('vram_budget' in c.rejection_reasons for c in no_fit[1:])
    historical = generate_static_candidates(benefits, available_bytes=100000, reserve_bytes=1000)
    assert historical[-1].layer_ids == (0,1,2,3,4,5,6,7,8,9,15,20)


def test_gemma_profile_names_are_normalized_only_in_adapter():
    raw = {'diagnostic_synchronization': True, 'records': [
        {'backend':'CPU','first_node':'ffn_moe_gate_up-0','total_us':1000},
        {'backend':'CUDA0','first_node':'ffn_norm-0','total_us':123},
        {'backend':'CPU','first_node':'ffn_moe_gate_up-1','total_us':500}]}
    normalized = Gemma4Adapter().normalize_profile(raw, profile_id='one')
    assert normalized == rows()
    assert len(rank_layer_benefits(model_fixture(), normalized)) == 2
