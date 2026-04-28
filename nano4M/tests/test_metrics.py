"""Sanity tests for per-modality metrics.

These are not exhaustive but catch obvious regressions:
- Correct metric value for trivial identity cases
- Sensible behaviour with edge inputs (all-zero masks, etc.)

Run:
    cd nano4M && python -m pytest tests/test_metrics.py -v
"""
import math
import pytest
import torch

from nanofm.evaluation.scene_parser import SceneObject
from nanofm.evaluation.metrics import (
    depth_absrel,
    depth_rmse,
    depth_delta1,
    normals_angular_error,
    rgb_fid,
    scene_desc_per_field_accuracy,
)


# --------------------------------------------------------------- depth

def test_depth_perfect_prediction():
    gt = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    pred = gt.clone()
    assert depth_absrel(pred, gt) == pytest.approx(0.0, abs=1e-6)
    assert depth_rmse(pred, gt) == pytest.approx(0.0, abs=1e-6)
    assert depth_delta1(pred, gt) == pytest.approx(1.0)


def test_depth_constant_offset():
    gt = torch.ones(1, 4, 4) * 2.0
    pred = torch.ones(1, 4, 4) * 3.0
    # AbsRel = |3-2|/2 = 0.5
    assert depth_absrel(pred, gt) == pytest.approx(0.5)
    # RMSE = 1.0
    assert depth_rmse(pred, gt) == pytest.approx(1.0)


def test_depth_delta1_threshold():
    # Make every pixel fail δ1 (ratio >= 1.25).
    gt = torch.ones(1, 4, 4)
    pred = torch.ones(1, 4, 4) * 1.3
    assert depth_delta1(pred, gt) == pytest.approx(0.0)


# ------------------------------------------------------------- normals

def test_normals_perfect_match():
    # Every pixel: unit vector along +z.
    pred = torch.zeros(1, 3, 2, 2)
    pred[:, 2, :, :] = 1.0
    gt = pred.clone()
    err = normals_angular_error(pred, gt)
    assert err == pytest.approx(0.0, abs=1e-3)


def test_normals_90_degrees_off():
    pred = torch.zeros(1, 3, 2, 2)
    pred[:, 0, :, :] = 1.0  # +x
    gt = torch.zeros(1, 3, 2, 2)
    gt[:, 2, :, :] = 1.0  # +z
    err = normals_angular_error(pred, gt)
    assert err == pytest.approx(90.0, abs=1e-2)


# ----------------------------------------------------------------- FID

def test_fid_same_batch_near_zero():
    """FID(x, x) over a batch should be close to zero.

    Not exactly zero because sqrtm(cov(features_real) * cov(features_fake))
    introduces numerical error on small batches, but should land well below
    any meaningful FID value.
    """
    pytest.importorskip("torchmetrics")
    torch.manual_seed(0)
    # InceptionV3 resizes internally; small spatial dims keep the test fast.
    batch = torch.rand(64, 3, 64, 64)
    val = rgb_fid(batch, batch, feature_dim=64)
    assert val < 5.0, f"FID(x, x) should be near zero, got {val}"


def test_fid_divergent_batches_positive():
    """FID between clearly different distributions must be clearly positive.

    One batch is random noise, the other is a constant image. They should
    occupy disjoint regions of InceptionV3 feature space.
    """
    pytest.importorskip("torchmetrics")
    torch.manual_seed(0)
    noise = torch.rand(64, 3, 64, 64)
    constant = torch.full((64, 3, 64, 64), 0.5)
    val = rgb_fid(noise, constant, feature_dim=64)
    assert val > 10.0, f"FID between disjoint distributions should be large, got {val}"


# ----------------------------------------------------------- scene_desc

def test_scene_desc_perfect_match():
    obj = SceneObject(x=5, y=7, shape="cube", color="blue", material="metal")
    preds = [[obj]]
    gts = [[obj]]
    r = scene_desc_per_field_accuracy(preds, gts)
    assert r["position"] == 1.0
    assert r["shape"] == 1.0
    assert r["color"] == 1.0
    assert r["material"] == 1.0
    assert r["set_match"] == 1.0
    assert r["exact_sequence"] == 1.0


def test_scene_desc_all_wrong():
    preds = [[SceneObject(x=99, y=99, shape="XXX", color="YYY", material="ZZZ")]]
    gts = [[SceneObject(x=0, y=0, shape="cube", color="blue", material="metal")]]
    r = scene_desc_per_field_accuracy(preds, gts)
    assert r["position"] == 0.0
    assert r["shape"] == 0.0
    assert r["color"] == 0.0
    assert r["material"] == 0.0
    assert r["set_match"] == 0.0
    assert r["exact_sequence"] == 0.0


def test_scene_desc_extra_prediction_counts_as_miss():
    # Predicted more objects than GT -> unmatched extras are "wrong".
    preds = [[
        SceneObject(x=0, y=0, shape="cube", color="blue", material="metal"),
        SceneObject(x=99, y=99, shape="sphere", color="red", material="rubber"),
    ]]
    gts = [[
        SceneObject(x=0, y=0, shape="cube", color="blue", material="metal"),
    ]]
    r = scene_desc_per_field_accuracy(preds, gts)
    # The 1 GT object is perfectly matched; totals are per-GT so ratio is still 1.0.
    assert r["shape"] == 1.0
    # But neither whole-match metric passes (lengths differ).
    assert r["set_match"] == 0.0
    assert r["exact_sequence"] == 0.0


def test_scene_desc_set_match_passes_when_order_differs():
    # Same set of objects but emitted in reverse order: set_match=1, exact_sequence=0.
    a = SceneObject(x=0, y=0, shape="cube", color="blue", material="metal")
    b = SceneObject(x=20, y=20, shape="sphere", color="red", material="rubber")
    preds = [[b, a]]
    gts = [[a, b]]
    r = scene_desc_per_field_accuracy(preds, gts)
    assert r["set_match"] == 1.0
    assert r["exact_sequence"] == 0.0
    assert r["shape"] == 1.0
    assert r["color"] == 1.0
    assert r["material"] == 1.0


def test_scene_desc_exact_sequence_requires_ordered_fields_correct():
    # Two GT objects in the same canonical order; predicted list has the
    # right shapes per slot but the colors are swapped between slots.
    gts = [[
        SceneObject(x=0, y=0, shape="cube", color="blue", material="metal"),
        SceneObject(x=20, y=20, shape="sphere", color="red", material="rubber"),
    ]]
    preds = [[
        SceneObject(x=0, y=0, shape="cube", color="red", material="metal"),
        SceneObject(x=20, y=20, shape="sphere", color="blue", material="rubber"),
    ]]
    r = scene_desc_per_field_accuracy(preds, gts)
    # Hungarian matches by position, so set_match passes for shape only.
    assert r["shape"] == 1.0
    # But colors are wrong on every pair, so neither whole-match metric passes.
    assert r["color"] == 0.0
    assert r["set_match"] == 0.0
    assert r["exact_sequence"] == 0.0


def test_scene_desc_position_tolerance():
    # Within tolerance (±3 px default).
    preds = [[SceneObject(x=2, y=2, shape="cube", color="blue", material="metal")]]
    gts = [[SceneObject(x=0, y=0, shape="cube", color="blue", material="metal")]]
    r = scene_desc_per_field_accuracy(preds, gts, position_tolerance_px=3)
    assert r["position"] == 1.0
    # Now tighten tolerance so the same example fails position.
    r2 = scene_desc_per_field_accuracy(preds, gts, position_tolerance_px=1)
    assert r2["position"] == 0.0
    # Shape/color/material unaffected.
    assert r2["shape"] == 1.0
