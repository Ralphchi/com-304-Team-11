"""Unit tests for the GroundingDINO-based object-detection verifier.

We mock the underlying transformers detector so the test suite doesn't need
to download the GroundingDINO weights. The integration sanity gate (recall
>= 0.8 on 20 GT (image, scene_desc) val pairs) runs on SCITAS via Phase D.
"""
from __future__ import annotations

from typing import List

import torch

from nanofm.evaluation.metrics import rgb_object_detection_score
from nanofm.evaluation.scene_parser import SceneObject


class FakeVerifier:
    """In-memory stand-in for `RGBVerifier`."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    def score(self, image, expected_objects):
        s = self.scores[self.calls]
        self.calls += 1
        return s


def _obj(color="red", shape="cube", material="metallic", x=0, y=0):
    return SceneObject(x=x, y=y, shape=shape, color=color, material=material)


def test_rgb_object_detection_score_perfect():
    verifier = FakeVerifier([
        {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_expected": 2, "n_detected": 2},
        {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_expected": 1, "n_detected": 1},
    ])
    images = [torch.zeros(3, 4, 4), torch.zeros(3, 4, 4)]
    expected = [[_obj(), _obj(color="blue")], [_obj()]]
    out = rgb_object_detection_score(images, expected, verifier)
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0
    assert out["f1"] == 1.0
    assert out["perfect_rate"] == 1.0


def test_rgb_object_detection_score_mixed():
    verifier = FakeVerifier([
        {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_expected": 2, "n_detected": 2},
        {"precision": 0.5, "recall": 0.5, "f1": 0.5, "n_expected": 2, "n_detected": 2},
    ])
    images = [torch.zeros(3, 4, 4), torch.zeros(3, 4, 4)]
    expected = [[_obj(), _obj(color="blue")], [_obj(), _obj(color="green")]]
    out = rgb_object_detection_score(images, expected, verifier)
    assert abs(out["f1"] - 0.75) < 1e-9
    assert abs(out["perfect_rate"] - 0.5) < 1e-9


def test_rgb_object_detection_score_empty_input():
    out = rgb_object_detection_score([], [], FakeVerifier([]))
    for k in ("precision", "recall", "f1", "perfect_rate"):
        assert out[k] != out[k]  # NaN


def test_rgb_object_detection_score_skips_nan_precision():
    """A sample with no detections (precision = NaN) should not pull the
    aggregated precision toward NaN; just exclude it from the precision mean."""
    verifier = FakeVerifier([
        {"precision": float("nan"), "recall": 0.0, "f1": 0.0,
         "n_expected": 1, "n_detected": 0},
        {"precision": 1.0, "recall": 1.0, "f1": 1.0,
         "n_expected": 1, "n_detected": 1},
    ])
    images = [torch.zeros(3, 4, 4), torch.zeros(3, 4, 4)]
    expected = [[_obj()], [_obj()]]
    out = rgb_object_detection_score(images, expected, verifier)
    assert out["precision"] == 1.0  # only the valid one contributed
    assert abs(out["recall"] - 0.5) < 1e-9
    assert abs(out["f1"] - 0.5) < 1e-9


def test_rgb_object_detection_score_empty_expected_list():
    """An expected list of zero objects should be handled by RGBVerifier itself
    (vacuously perfect). Aggregator just propagates whatever the verifier returns."""
    verifier = FakeVerifier([
        {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_expected": 0, "n_detected": 0},
    ])
    images = [torch.zeros(3, 4, 4)]
    expected: List[List[SceneObject]] = [[]]
    out = rgb_object_detection_score(images, expected, verifier)
    assert out["f1"] == 1.0


def test_rgb_verifier_prompt_construction():
    """Smoke-test the prompt builder without instantiating the real model."""
    from nanofm.evaluation.rgb_verifier import _prompt_for

    assert _prompt_for(_obj(color="red", material="metallic", shape="cube")) == \
        "a red metallic cube"
    # Empty fields collapse cleanly.
    obj = SceneObject(x=0, y=0, shape="", color="", material="")
    assert _prompt_for(obj) == "an object"
