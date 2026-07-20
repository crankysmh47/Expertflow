from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Mapping, Sequence


_LETTERS = "ABCD"


def mmlu_gate_pass(delta_percentage_points: float, threshold: float = -1.0) -> bool:
    return delta_percentage_points >= threshold - 1e-12


def build_zero_shot_prompt(item: Mapping[str, Any]) -> str:
    choices = list(item["choices"])
    if len(choices) != 4:
        raise ValueError("MMLU item must have exactly four choices")
    rendered_choices = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(_LETTERS, choices, strict=True)
    )
    return (
        "The following is a multiple choice question. Select the single best answer.\n\n"
        f"Question: {item['question']}\n{rendered_choices}\nAnswer:"
    )


def parse_prediction(content: str) -> int:
    normalized = content.strip()
    if len(normalized) != 1 or normalized not in _LETTERS:
        raise ValueError(f"expected a single A-D prediction, got {content!r}")
    return _LETTERS.index(normalized)


def run_mmlu(
    items: Sequence[Mapping[str, Any]], endpoint: str, *, timeout_seconds: float = 300.0
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        payload = {
            "prompt": build_zero_shot_prompt(item),
            "n_predict": 1,
            "temperature": 0.0,
            "seed": 42,
            "cache_prompt": False,
            "grammar": "root ::= [ABCD]",
            "n_probs": 4,
            "post_sampling_probs": True,
            "return_tokens": True,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.load(response)
        wall_seconds = time.perf_counter() - started
        prediction = parse_prediction(str(body["content"]))
        answer = int(item["answer"])
        results.append(
            {
                "index": index,
                "subject": item["subject"],
                "row_id": item["row_id"],
                "selection_sha256": item["selection_sha256"],
                "answer": answer,
                "prediction": prediction,
                "correct": prediction == answer,
                "content": body["content"],
                "tokens": body.get("tokens", []),
                "completion_probabilities": body.get("completion_probabilities", []),
                "timings": body.get("timings", {}),
                "wall_seconds": wall_seconds,
            }
        )
    correct = sum(bool(result["correct"]) for result in results)
    return {
        "protocol": {
            "shot_count": 0,
            "choice_constraint": "root ::= [ABCD]",
            "temperature": 0.0,
            "seed": 42,
            "cache_prompt": False,
        },
        "correct": correct,
        "total": len(results),
        "accuracy": correct / len(results),
        "items": results,
    }
