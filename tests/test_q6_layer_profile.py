from __future__ import annotations

import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "q6_layer_profile", Path("scripts/analyze_q6_layer_profile.py")
)
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


def test_rank_profiles_uses_layer_median_and_time_per_mib(tmp_path: Path) -> None:
    paths = []
    for index, totals in enumerate(((100, 300), (200, 500))):
        path = tmp_path / f"profile-{index}.json"
        path.write_text(
            '{"records":['
            f'{{"first_node":"ffn_moe_gate_up-0","backend":"CPU","total_us":{totals[0]},"compute_submit_us":90,"input_boundary_us":10,"completion_us":0}},'
            f'{{"first_node":"ffn_moe_gate_up-1","backend":"CPU","total_us":{totals[1]},"compute_submit_us":280,"input_boundary_us":20,"completion_us":0}}'
            ']}'
        )
        paths.append(path)

    result = profile.rank_profiles(paths, selected_layers={0}, shadow_bytes=685_933_056)

    assert result["profile_count"] == 2
    assert result["layers"][0]["layer"] == 1
    assert result["layers"][0]["total_us_median"] == 400.0
    assert result["layers"][1]["selected_initially"] is True
    assert result["layers"][0]["shadow_bytes"] == 685_933_056
    assert result["layers"][0]["score_us_per_mib"] > result["layers"][1]["score_us_per_mib"]
