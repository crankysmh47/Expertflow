from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_tokens(prefix: Path) -> dict[str, Any]:
    return json.loads(prefix.with_suffix(".tokens.json").read_text(encoding="utf-8"))


def _load_trace(prefix: Path) -> list[dict[str, Any]]:
    events = []
    for line in prefix.with_suffix(".trace.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        event.pop("observed_at_ns", None)
        events.append(event)
    return events


def _first_sequence_difference(reference: list[Any], candidate: list[Any]) -> dict[str, Any] | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return {"index": index, "reference": left, "candidate": right}
    if len(reference) != len(candidate):
        index = min(len(reference), len(candidate))
        return {
            "index": index,
            "reference": reference[index] if index < len(reference) else None,
            "candidate": candidate[index] if index < len(candidate) else None,
        }
    return None


def compare_runs(reference_prefix: str | Path, candidate_prefix: str | Path) -> dict[str, Any]:
    reference_tokens = _load_tokens(Path(reference_prefix))
    candidate_tokens = _load_tokens(Path(candidate_prefix))
    reference_trace = _load_trace(Path(reference_prefix))
    candidate_trace = _load_trace(Path(candidate_prefix))

    generated_difference = _first_sequence_difference(
        reference_tokens["generated_token_ids"], candidate_tokens["generated_token_ids"]
    )
    router_difference = _first_sequence_difference(reference_trace, candidate_trace)
    first_router_divergence = None
    if router_difference is not None:
        reference_event = router_difference["reference"]
        candidate_event = router_difference["candidate"]
        first_router_divergence = {
            "event_index": router_difference["index"],
            "phase": reference_event.get("phase") if reference_event else None,
            "token_index": reference_event.get("token_index") if reference_event else None,
            "layer_id": reference_event.get("layer_id") if reference_event else None,
            "reference": reference_event.get("selected_expert_ids") if reference_event else None,
            "candidate": candidate_event.get("selected_expert_ids") if candidate_event else None,
        }

    return {
        "prompt_token_parity": reference_tokens["prompt_token_ids"] == candidate_tokens["prompt_token_ids"],
        "generated_token_parity": generated_difference is None,
        "first_generated_divergence": generated_difference,
        "router_parity": router_difference is None,
        "first_router_divergence": first_router_divergence,
        "reference_event_count": len(reference_trace),
        "candidate_event_count": len(candidate_trace),
    }
