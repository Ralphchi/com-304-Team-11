"""Per-modality metrics for the Team 11 nano4M extension.

All metric functions accept torch tensors (predicted, ground_truth) that have
already been decoded from tokens back into the native modality space:
- Depth: (B, H, W) or (B, 1, H, W), float, non-negative
- Normals: (B, 3, H, W), float, expected to be unit-normalized per pixel
- RGB: (B, 3, H, W), float in [0, 1]
- Scene desc: List[SceneObject] vs List[SceneObject] (per-sample)

Per-modality reductions: each function returns a float (or dict of floats)
averaged across the batch. Per-sample values can be obtained by looping.

References used while choosing metric formulations:
- AbsRel, RMSE, δ1: standard monocular-depth metrics (e.g. Eigen NYUv2 protocol)
- Mean angular error: standard surface-normal metric in degrees
- SSIM: scikit-image structural similarity
- Per-field accuracy: defined in the extension proposal Section IV
"""

from typing import Dict, List, Optional, Sequence
import math

import torch

from .scene_parser import SceneObject
from .hungarian_match import match_objects


# ------------------------------------------------------------------ depth

def _valid_mask(gt: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Mask out zero/negative ground truth (undefined depth)."""
    return gt > eps


def depth_absrel(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-3) -> float:
    """Absolute relative error: mean(|pred - gt| / gt) over valid pixels."""
    mask = _valid_mask(gt, eps)
    if not mask.any():
        return float("nan")
    return float(((pred[mask] - gt[mask]).abs() / gt[mask]).mean().item())


def depth_rmse(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-3) -> float:
    """Root mean squared error over valid pixels."""
    mask = _valid_mask(gt, eps)
    if not mask.any():
        return float("nan")
    return float(torch.sqrt(((pred[mask] - gt[mask]) ** 2).mean()).item())


def depth_delta1(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-3) -> float:
    """δ₁: fraction of pixels where max(pred/gt, gt/pred) < 1.25."""
    mask = _valid_mask(gt, eps)
    if not mask.any():
        return float("nan")
    # Clamp pred to eps so a zero-valued prediction does not create inf in
    # gt/pred. With eps=1e-3, any pred at or below the GT-validity threshold
    # is treated as the same lower bound used for GT.
    p = pred[mask].clamp(min=eps)
    ratio = torch.maximum(p / gt[mask], gt[mask] / p)
    return float((ratio < 1.25).float().mean().item())


# -------------------------------------------------------------- normals

def normals_angular_error(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean angular error in degrees between predicted and GT surface normals.

    Parameters
    ----------
    pred, gt : (B, 3, H, W) tensors of unit normals (approximately).
        Normalised internally to be safe.

    Returns
    -------
    float
        Mean angular error in degrees over the batch.
    """
    # Flatten spatial dims and normalise for safety.
    p = torch.nn.functional.normalize(pred, dim=1, eps=1e-6)
    g = torch.nn.functional.normalize(gt, dim=1, eps=1e-6)
    # Clamp dot product to [-1, 1] to avoid NaN from acos.
    cos = (p * g).sum(dim=1).clamp(-1.0, 1.0)  # (B, H, W)
    # Convert to degrees.
    err_rad = torch.acos(cos)
    err_deg = err_rad * (180.0 / math.pi)
    return float(err_deg.mean().item())


# ------------------------------------------------------------------- RGB

def rgb_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Structural similarity index, averaged across batch and channels.

    Implementation note: uses scikit-image for correctness; if performance
    becomes a concern we can switch to the pytorch-msssim library.

    Parameters
    ----------
    pred, gt : (B, 3, H, W) float tensors in [0, 1]

    Returns
    -------
    float
        SSIM averaged over the batch.
    """
    # Deferred import to keep the metrics module importable without extras.
    from skimage.metrics import structural_similarity  # type: ignore

    pred_np = pred.detach().cpu().clamp(0, 1).numpy()
    gt_np = gt.detach().cpu().clamp(0, 1).numpy()
    assert pred_np.shape == gt_np.shape, "pred and gt must have same shape"
    B = pred_np.shape[0]
    scores = []
    for i in range(B):
        # Move channels last for scikit-image, use channel_axis=2.
        p_i = pred_np[i].transpose(1, 2, 0)
        g_i = gt_np[i].transpose(1, 2, 0)
        scores.append(
            structural_similarity(g_i, p_i, channel_axis=2, data_range=1.0)
        )
    return float(sum(scores) / len(scores)) if scores else float("nan")


def rgb_fid(pred: torch.Tensor, gt: torch.Tensor, feature_dim: int = 2048) -> float:
    """Fréchet Inception Distance over a batch of images.

    Proposal Section IV reports FID as a *secondary* RGB metric — the proposal
    explicitly notes it is unreliable on synthetic CLEVR data but commits to
    reporting it for completeness.

    Parameters
    ----------
    pred, gt : (B, 3, H, W) float tensors in [0, 1]
    feature_dim : int
        InceptionV3 feature dimension. 2048 is the standard (final pool);
        smaller values (64/192/768) index earlier Inception pool layers and
        are useful for faster tests.

    Returns
    -------
    float
        FID between `pred` and `gt`. Lower is better; exactly 0 iff the two
        batches have identical InceptionV3 feature distributions. FID is
        noisy when the number of samples is small (<50); callers should
        accumulate across the entire eval set before reading the metric,
        not call once per sample.

    Notes
    -----
    - Deferred import: torchmetrics is only needed for this function, so
      `metrics.py` remains importable without it (parser / scene_desc metrics
      stay usable).
    - On first use torchmetrics downloads the InceptionV3 weights (~95 MB).
      This requires internet; on SCITAS compute nodes use a pre-warmed cache.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance  # type: ignore

    fid = FrechetInceptionDistance(feature=feature_dim, normalize=True)
    fid.update(gt.detach().cpu().clamp(0, 1), real=True)
    fid.update(pred.detach().cpu().clamp(0, 1), real=False)
    return float(fid.compute().item())


# ----------------------------------------------------------- scene_desc

def scene_desc_per_field_accuracy(
    predicted: Sequence[List[SceneObject]],
    ground_truth: Sequence[List[SceneObject]],
    position_tolerance_px: int = 3,
) -> Dict[str, float]:
    """Per-field accuracy after Hungarian-matching objects by position,
    plus two scene-level whole-match metrics.

    Per-field flow (per sample):
    1. Hungarian-match predicted to GT objects by 2D position.
    2. For each matched pair, count per-field correctness:
       - position: True if |dx| <= tol and |dy| <= tol
       - shape / color / material: exact string match
    3. Unmatched objects count as all fields wrong (totals counted per GT).

    Returns dict with five entries:
        position, shape, color, material  -- micro-averaged over GT objects
        set_match                          -- fraction of samples where the
            sets of objects match: same length AND Hungarian matches all
            pairs AND every field correct on every pair. Order-independent.
        exact_sequence                     -- fraction of samples where
            the predicted parsed list equals the GT parsed list IN ORDER:
            same length AND for every i, pred[i] matches gt[i] field-by-field
            (with the same position tolerance). Stricter than set_match.

    Reporting both lets us distinguish "got the scene contents right but
    in a different order" (set_match high, exact_sequence low) from
    "got everything right" (both high). Plan Section IV says
    "per-field accuracy ... plus exact-sequence match"; we expose set_match
    as a complementary order-independent signal for diagnostic purposes.
    """
    assert len(predicted) == len(ground_truth), "predicted/gt must align by sample"
    totals = {"position": 0, "shape": 0, "color": 0, "material": 0}
    correct = {"position": 0, "shape": 0, "color": 0, "material": 0}
    set_match_correct = 0
    exact_sequence_correct = 0

    for pred, gt in zip(predicted, ground_truth):
        result = match_objects(pred, gt)

        totals["position"] += len(gt)
        totals["shape"] += len(gt)
        totals["color"] += len(gt)
        totals["material"] += len(gt)

        is_set_match = (len(pred) == len(gt)) and (
            len(result.unmatched_pred) == 0 and len(result.unmatched_gt) == 0
        )

        for pi, gi in result.matches:
            p, g = pred[pi], gt[gi]
            pos_ok = (abs(p.x - g.x) <= position_tolerance_px
                      and abs(p.y - g.y) <= position_tolerance_px)
            if pos_ok:
                correct["position"] += 1
            if p.shape == g.shape:
                correct["shape"] += 1
            if p.color == g.color:
                correct["color"] += 1
            if p.material == g.material:
                correct["material"] += 1
            if not (pos_ok and p.shape == g.shape and p.color == g.color
                    and p.material == g.material):
                is_set_match = False

        if is_set_match:
            set_match_correct += 1

        # exact_sequence: predicted list equals GT list in order, with the
        # same per-field tolerance as the rest of the metrics.
        is_exact_sequence = len(pred) == len(gt) and all(
            abs(p.x - g.x) <= position_tolerance_px
            and abs(p.y - g.y) <= position_tolerance_px
            and p.shape == g.shape
            and p.color == g.color
            and p.material == g.material
            for p, g in zip(pred, gt)
        )
        if is_exact_sequence:
            exact_sequence_correct += 1

    def _safe_div(a: int, b: int) -> float:
        return (a / b) if b > 0 else float("nan")

    return {
        "position": _safe_div(correct["position"], totals["position"]),
        "shape": _safe_div(correct["shape"], totals["shape"]),
        "color": _safe_div(correct["color"], totals["color"]),
        "material": _safe_div(correct["material"], totals["material"]),
        "set_match": _safe_div(set_match_correct, len(predicted)),
        "exact_sequence": _safe_div(exact_sequence_correct, len(predicted)),
    }


# --------------------------------------------- LLM-judge aggregator (Phase C)

def caption_llm_judge(
    originals: Sequence[str],
    generateds: Sequence[str],
    judge,
    parser_set_match: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Aggregate Qwen-judge alignment scores over a batch of caption pairs.

    Returns:
        llm_alignment        : mean of per-sample alignment, excluding parse errors.
        llm_perfect_rate     : fraction of samples with alignment == 1.0
                               (parse-errors excluded).
        llm_parse_error_rate : fraction of samples that returned parse_error=True.
        parser_judge_corr    : Pearson correlation between parser_set_match and
                               llm_alignment per sample (NaN if parser_set_match
                               is None or all scores identical).
    """
    assert len(originals) == len(generateds)
    if not originals:
        return {
            "llm_alignment": float("nan"),
            "llm_perfect_rate": float("nan"),
            "llm_parse_error_rate": float("nan"),
            "parser_judge_corr": float("nan"),
        }

    scored = judge.score_batch(list(originals), list(generateds))
    valid = [s for s in scored if not s.get("parse_error", False)]
    n = len(scored)
    n_valid = len(valid)
    n_errors = n - n_valid

    if n_valid == 0:
        return {
            "llm_alignment": float("nan"),
            "llm_perfect_rate": float("nan"),
            "llm_parse_error_rate": 1.0,
            "parser_judge_corr": float("nan"),
        }

    alignments = [float(s["alignment"]) for s in valid]
    mean_alignment = sum(alignments) / n_valid
    perfect_rate = sum(1 for a in alignments if a >= 1.0) / n_valid

    corr = float("nan")
    if parser_set_match is not None and len(parser_set_match) == n:
        # Pearson correlation across non-error samples.
        valid_alignments = [
            float(s["alignment"]) for s in scored if not s.get("parse_error", False)
        ]
        valid_parser = [
            float(p) for p, s in zip(parser_set_match, scored)
            if not s.get("parse_error", False)
        ]
        if len(valid_alignments) >= 2:
            ax = valid_parser
            ay = valid_alignments
            mx = sum(ax) / len(ax)
            my = sum(ay) / len(ay)
            num = sum((x - mx) * (y - my) for x, y in zip(ax, ay))
            denx = sum((x - mx) ** 2 for x in ax) ** 0.5
            deny = sum((y - my) ** 2 for y in ay) ** 0.5
            corr = num / (denx * deny) if denx > 0 and deny > 0 else float("nan")

    return {
        "llm_alignment": mean_alignment,
        "llm_perfect_rate": perfect_rate,
        "llm_parse_error_rate": n_errors / n,
        "parser_judge_corr": corr,
    }


# ------------------------------------ Object-detection aggregator (Phase D)

def rgb_object_detection_score(
    images: Sequence[torch.Tensor],
    expected_lists: Sequence[Sequence[SceneObject]],
    verifier,
) -> Dict[str, float]:
    """Mean precision/recall/F1 of detected vs expected objects per sample.

    Each `images[i]` is a (3, H, W) tensor in [0, 1]; each
    `expected_lists[i]` is a list of SceneObject (parsed from the GT
    caption). Calls `verifier.score(image, expected)` per sample.
    """
    assert len(images) == len(expected_lists)
    if not images:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "perfect_rate": float("nan"),
        }

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    perfect = 0
    for img, expected in zip(images, expected_lists):
        s = verifier.score(img, list(expected))
        if s["precision"] == s["precision"]:  # not NaN
            precisions.append(float(s["precision"]))
        if s["recall"] == s["recall"]:
            recalls.append(float(s["recall"]))
        f1s.append(float(s["f1"]))
        if float(s["f1"]) >= 1.0:
            perfect += 1

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "precision": _mean(precisions),
        "recall": _mean(recalls),
        "f1": _mean(f1s),
        "perfect_rate": perfect / len(images),
    }
