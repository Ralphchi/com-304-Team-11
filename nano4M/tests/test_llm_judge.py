"""Unit tests for the Qwen LLM-judge wrapper.

We mock the underlying transformers model so the test suite doesn't have to
download Qwen3-8B (~16 GB). The integration test on a real Qwen instance
runs on SCITAS via `--phases C` once eval kicks off.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from nanofm.evaluation.metrics import caption_llm_judge


class FakeJudge:
    """In-memory stand-in for `LLMJudge` that returns canned scores."""

    def __init__(self, responses):
        # responses is a list of (alignment, parse_error) tuples
        self.responses = list(responses)
        self.calls = 0

    def score_batch(self, originals, generateds):
        out = []
        for _ in originals:
            alignment, parse_error = self.responses[self.calls]
            self.calls += 1
            out.append({
                "alignment": alignment,
                "parse_error": parse_error,
                "missing_objects": [],
                "extra_objects": [],
                "wrong_attributes": [],
            })
        return out


def test_caption_llm_judge_perfect_alignment():
    judge = FakeJudge([(1.0, False), (1.0, False), (1.0, False)])
    out = caption_llm_judge(["a", "b", "c"], ["a", "b", "c"], judge)
    assert out["llm_alignment"] == 1.0
    assert out["llm_perfect_rate"] == 1.0
    assert out["llm_parse_error_rate"] == 0.0


def test_caption_llm_judge_mixed_scores():
    judge = FakeJudge([(1.0, False), (0.5, False), (0.0, False)])
    out = caption_llm_judge(["a", "b", "c"], ["a", "b", "c"], judge)
    assert abs(out["llm_alignment"] - 0.5) < 1e-9
    assert abs(out["llm_perfect_rate"] - 1 / 3) < 1e-9
    assert out["llm_parse_error_rate"] == 0.0


def test_caption_llm_judge_with_parse_errors():
    judge = FakeJudge([(1.0, False), (0.0, True), (0.5, False)])
    out = caption_llm_judge(["a", "b", "c"], ["a", "b", "c"], judge)
    # Only 2 valid; mean = (1.0 + 0.5) / 2 = 0.75
    assert abs(out["llm_alignment"] - 0.75) < 1e-9
    assert abs(out["llm_perfect_rate"] - 0.5) < 1e-9
    assert abs(out["llm_parse_error_rate"] - 1 / 3) < 1e-9


def test_caption_llm_judge_all_parse_errors():
    judge = FakeJudge([(0.0, True), (0.0, True)])
    out = caption_llm_judge(["a", "b"], ["a", "b"], judge)
    assert out["llm_alignment"] != out["llm_alignment"]  # NaN
    assert out["llm_parse_error_rate"] == 1.0


def test_caption_llm_judge_empty_input():
    judge = FakeJudge([])
    out = caption_llm_judge([], [], judge)
    for k in ("llm_alignment", "llm_perfect_rate", "llm_parse_error_rate"):
        assert out[k] != out[k]  # NaN


def test_caption_llm_judge_parser_correlation_perfect():
    judge = FakeJudge([(1.0, False), (1.0, False), (0.0, False), (0.0, False)])
    parser = [1.0, 1.0, 0.0, 0.0]
    out = caption_llm_judge(["a"] * 4, ["b"] * 4, judge, parser_set_match=parser)
    assert abs(out["parser_judge_corr"] - 1.0) < 1e-9


def test_caption_llm_judge_parser_correlation_anti():
    judge = FakeJudge([(0.0, False), (0.0, False), (1.0, False), (1.0, False)])
    parser = [1.0, 1.0, 0.0, 0.0]
    out = caption_llm_judge(["a"] * 4, ["b"] * 4, judge, parser_set_match=parser)
    assert abs(out["parser_judge_corr"] - (-1.0)) < 1e-9


# -- Tests that exercise LLMJudge.score with a mocked transformers backend --

def _build_judge_with_fake_generation(responses):
    """Construct an LLMJudge whose `_generate` returns canned strings."""
    from nanofm.evaluation.llm_judge import LLMJudge

    j = LLMJudge.__new__(LLMJudge)  # bypass __init__ (no real Qwen on the laptop)
    j.tokenizer = MagicMock()
    j.model = MagicMock()
    j.device = "cpu"
    j.max_new_tokens = 200

    iterator = iter(responses)

    def fake_generate(prompt):
        try:
            return next(iterator)
        except StopIteration:
            return ""

    j._generate = fake_generate  # type: ignore
    return j


def test_llm_judge_score_perfect_json():
    j = _build_judge_with_fake_generation([
        json.dumps({
            "alignment": 1.0,
            "missing_objects": [],
            "extra_objects": [],
            "wrong_attributes": [],
        }),
    ])
    out = j.score("a red cube at (5, 7)", "a red cube at (5, 7)")
    assert out["alignment"] == 1.0
    assert out["parse_error"] is False


def test_llm_judge_score_falls_back_to_score_prompt():
    # First (primary) call returns gibberish -> backup prompt fires.
    j = _build_judge_with_fake_generation([
        "not really json",
        "SCORE: 0.42",
    ])
    out = j.score("orig", "gen")
    assert abs(out["alignment"] - 0.42) < 1e-9
    assert out["parse_error"] is False


def test_llm_judge_score_double_failure_marks_parse_error():
    j = _build_judge_with_fake_generation([
        "not json",
        "no score either",
    ])
    out = j.score("orig", "gen")
    assert out["alignment"] == 0.0
    assert out["parse_error"] is True


def test_llm_judge_score_empty_input_skips_llm():
    j = _build_judge_with_fake_generation([])
    out = j.score("", "anything")
    assert out["alignment"] == 0.0
    assert out["parse_error"] is False
    out2 = j.score("anything", "")
    assert out2["alignment"] == 0.0
    assert out2["parse_error"] is False


def test_llm_judge_score_clamps_score_range():
    j = _build_judge_with_fake_generation([
        "not json",
        "SCORE: 1.7",  # out-of-range
    ])
    out = j.score("orig", "gen")
    assert out["alignment"] == 1.0
    assert out["parse_error"] is False
