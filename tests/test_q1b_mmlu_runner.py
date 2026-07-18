from __future__ import annotations

import pytest

from expertflow.quality.mmlu import build_zero_shot_prompt, mmlu_gate_pass, parse_prediction


def test_build_zero_shot_prompt_is_frozen_and_unambiguous():
    prompt = build_zero_shot_prompt(
        {
            "question": "What is 2 + 2?",
            "choices": ["1", "2", "4", "8"],
        }
    )

    assert prompt == (
        "The following is a multiple choice question. Select the single best answer.\n\n"
        "Question: What is 2 + 2?\n"
        "A. 1\nB. 2\nC. 4\nD. 8\n"
        "Answer:"
    )


@pytest.mark.parametrize(("content", "expected"), [("A", 0), (" B", 1), ("C\n", 2), ("D", 3)])
def test_parse_prediction_accepts_only_one_constrained_choice(content, expected):
    assert parse_prediction(content) == expected


def test_parse_prediction_rejects_invalid_output():
    with pytest.raises(ValueError, match="single A-D"):
        parse_prediction("The answer is C")


def test_mmlu_gate_includes_exact_negative_one_point_boundary():
    delta = (0.41 - 0.42) * 100.0
    assert delta == -1.0000000000000009
    assert mmlu_gate_pass(delta) is True
