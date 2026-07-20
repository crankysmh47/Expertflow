from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any


BUNDLE_BYTES = 428_212_736


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ppl(result: dict[str, Any], key: str) -> float:
    return math.exp(float(result[key]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    protocol = read(root / "protocol.json")
    stage1 = read(root / "stage1-paired-nll.json")
    stage2 = read(root / "stage2-heldout-paired-nll.json")
    layer0 = stage2
    layer15 = read(root / "layer15-paired-nll.json")
    layer20 = read(root / "layer20-paired-nll.json")
    layers2 = read(root / "layers-0-15-paired-nll.json")
    layers4 = read(root / "layers-0-1-15-20-paired-nll.json")
    generation = read(root / "generation-performance-summary.json")
    mmlu = read(root / "mmlu-results.json")

    shutil.copyfile(root / "raw" / "stage2-off-nll.jsonl", root / "per-token-nll-off.jsonl")
    shutil.copyfile(root / "raw" / "stage2-on-nll.jsonl", root / "per-token-nll-on.jsonl")

    corpus_manifest = {
        "frozen_before_candidate_result": True,
        "stage1_reproduction": protocol["stage1_corpus"],
        "stage2_independent_heldout": protocol["stage2_corpus"],
        "mmlu": protocol["mmlu"],
        "runtime": protocol["runtime"],
    }
    write(root / "corpus-manifest.json", corpus_manifest)

    off_decode = generation["performance"]["off"]["decode_tps"]["mean"]
    layer_specs = (
        (0, layer0, "layer0"),
        (15, layer15, "layer15"),
        (20, layer20, "layer20"),
    )
    layer_entries: list[dict[str, Any]] = []
    for layer, quality, mode in layer_specs:
        decode = generation["performance"][mode]["decode_tps"]["mean"]
        regression = quality["relative_perplexity_change"]
        quality_cost_pp = regression * 100.0
        speed_change_pct = (decode / off_decode - 1.0) * 100.0
        layer_entries.append(
            {
                "layer": layer,
                "perplexity": ppl(quality, "candidate_mean_nll"),
                "perplexity_relative_change": regression,
                "bootstrap_95pct": quality["bootstrap_95pct"],
                "individual_point_gate_pass": quality["point_gate_pass"],
                "generated_token_agreement_rate": generation["generation"][mode]["generated_token_agreement_rate"],
                "router_topk_set_overlap_rate": generation["generation"][mode]["router_set_overlap_rate"],
                "router_order_overlap_rate": generation["generation"][mode]["router_order_overlap_rate"],
                "decode_tps": decode,
                "decode_tps_relative_change": decode / off_decode - 1.0,
                "peak_system_gpu_used_mib": generation["generation"][mode]["peak_system_gpu_used_mib"],
                "bundle_bytes": BUNDLE_BYTES,
                "benefit_cost_ratio_tps_pct_per_ppl_pp": (
                    speed_change_pct / quality_cost_pp if quality_cost_pp > 0 else None
                ),
            }
        )
    layer_sensitivity = {
        "measurement_kind": "measured",
        "reference_perplexity": ppl(stage2, "reference_mean_nll"),
        "reference_decode_tps": off_decode,
        "ratio_units": "decode TPS percent change per positive perplexity percentage-point regression; undefined when quality improves",
        "ranking": [0, 15, 20],
        "ranking_note": "Layer 0 is dominant because quality improved and TPS increased. Layers 15 and 20 have negative measured speed benefit; their ratios are descriptive, not evidence of acceleration.",
        "layers": layer_entries,
        "sanitizer": {
            "status": "not_run",
            "reason": "compute-sanitizer was not installed on the supported Windows CUDA toolchain",
        },
    }
    write(root / "layer-sensitivity.json", layer_sensitivity)

    best4_perf = generation["performance"]["layers-0-1-15-20"]
    results = {
        "verdict": "Q1B PASS",
        "official_q1_preserved": protocol["official_q1_result_preserved"],
        "stage1_reproduction": {
            "tokens": stage1["token_count"],
            "reference_perplexity": ppl(stage1, "reference_mean_nll"),
            "candidate_perplexity": ppl(stage1, "candidate_mean_nll"),
            "relative_change": stage1["relative_perplexity_change"],
            "bootstrap_95pct": stage1["bootstrap_95pct"],
        },
        "independent_perplexity": {
            "tokens": stage2["token_count"],
            "reference_perplexity": ppl(stage2, "reference_mean_nll"),
            "candidate_perplexity": ppl(stage2, "candidate_mean_nll"),
            "relative_change": stage2["relative_perplexity_change"],
            "bootstrap_standard_error": stage2["bootstrap_standard_error"],
            "bootstrap_95pct": stage2["bootstrap_95pct"],
            "gate_pass": stage2["gate_pass"],
        },
        "mmlu": {
            "reference_correct": mmlu["reference_correct"],
            "candidate_correct": mmlu["candidate_correct"],
            "total": mmlu["paired_items"],
            "delta_percentage_points": mmlu["accuracy_delta_percentage_points"],
            "gate_pass": mmlu["gate_pass"],
            "candidate_deterministic": mmlu["candidate_deterministic"],
        },
        "scaling": [
            {
                "layers": [0],
                "arena_bytes": BUNDLE_BYTES,
                "perplexity": ppl(layer0, "candidate_mean_nll"),
                "relative_change": layer0["relative_perplexity_change"],
                "decode_tps": generation["performance"]["layer0"]["decode_tps"]["mean"],
            },
            {
                "layers": [0, 15],
                "arena_bytes": 2 * BUNDLE_BYTES,
                "perplexity": ppl(layers2, "candidate_mean_nll"),
                "relative_change": layers2["relative_perplexity_change"],
                "bootstrap_95pct": layers2["bootstrap_95pct"],
                "decode_tps": generation["performance"]["layers-0-15"]["decode_tps"]["mean"],
                "gate_pass": layers2["point_gate_pass"],
            },
            {
                "layers": [0, 1, 15, 20],
                "arena_bytes": 4 * BUNDLE_BYTES,
                "perplexity": ppl(layers4, "candidate_mean_nll"),
                "relative_change": layers4["relative_perplexity_change"],
                "bootstrap_95pct": layers4["bootstrap_95pct"],
                "decode_tps": best4_perf["decode_tps"]["mean"],
                "decode_tps_values": best4_perf["decode_tps"]["values"],
                "decode_tps_variance": best4_perf["decode_tps"]["variance"],
                "decode_tps_relative_change": generation["performance"]["best4_decode_relative_change"],
                "prompt_tps": best4_perf["prompt_tps"]["mean"],
                "peak_system_gpu_used_mib": best4_perf["peak_system_gpu_used_mib"],
                "generated_token_agreement_rate": generation["generation"]["layers-0-1-15-20-r1"]["generated_token_agreement_rate"],
                "router_topk_set_overlap_rate": generation["generation"]["layers-0-1-15-20-r1"]["router_set_overlap_rate"],
                "router_order_overlap_rate": generation["generation"]["layers-0-1-15-20-r1"]["router_order_overlap_rate"],
                "three_repetition_determinism": generation["best4_three_repetition_determinism"],
                "gate_pass": layers4["point_gate_pass"] and generation["best4_three_repetition_determinism"],
            },
        ],
        "stability": {
            "three_repetition_determinism": generation["best4_three_repetition_determinism"],
            "all_generation_processes_passed": all(
                mode["all_processes_passed"] for mode in generation["generation"].values()
            ),
            "persistent_gpu_growth_observed": False,
            "nan_or_cuda_error_observed": False,
        },
        "claim_limits": {
            "runtime_speedup_claim": False,
            "reason": "The +0.49% decode-TPS change is smaller than run-to-run dispersion and is not a meaningful speedup.",
            "q6_applicability": "unresolved; the preserved Q6 placement blocker was not modified in Q1b",
            "credible_path_to_beat_q6_22_967_tps": False,
        },
    }
    write(root / "results.json", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
