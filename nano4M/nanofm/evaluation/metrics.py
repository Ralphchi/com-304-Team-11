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

from typing import Dict, List, Sequence, Tuple
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
    ratio = torch.maximum(pred[mask] / gt[mask], gt[mask] / pred[mask])
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
    """Per-field accuracy after Hungarian-matching objects by position.

    For each sample in the batch:
    1. Hungarian-match predicted to GT objects by 2D position.
    2. For each matched pair, count per-field correctness:
       - position: True if |dx| <= tol and |dy| <= tol
       - shape / color / material: True iff exact string match
    3. Unmatched objects count as all fields wrong for that object.

    Returns dict: {"position": float, "shape": float, "color": float,
                  "material": float, "exact_sequence": float}

    - "exact_sequence" requires same length AND every field correct
      on every object (proxy for sequence-match score).
    - All other entries are micro-averaged across objects across all samples.

    Parameters
    ----------
    predicted : list of list of SceneObject, one per sample
    ground_truth : list of list of SceneObject, one per sample
    position_tolerance_px : int
        Position is "correct" when |dx|<=tol AND |dy|<=tol (proposal spec).
    """
    assert len(predicted) == len(ground_truth), "predicted/gt must align by sample"
    totals = {"position": 0, "shape": 0, "color": 0, "material": 0}
    correct = {"position": 0, "shape": 0, "color": 0, "material": 0}
    exact_sequence_correct = 0

    for pred, gt in zip(predicted, ground_truth):
        result = match_objects(pred, gt)

        # Count every GT object in the totals (unmatched GT -> 0 correct).
        totals["position"] += len(gt)
        totals["shape"] += len(gt)
        totals["color"] += len(gt)
        totals["material"] += len(gt)

        all_correct = (len(pred) == len(gt)) and (
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
                all_correct = False

        if all_correct:
            exact_sequence_correct += 1

    def _safe_div(a: int, b: int) -> float:
        return (a / b) if b > 0 else float("nan")

    return {
        "position": _safe_div(correct["position"], totals["position"]),
        "shape": _safe_div(correct["shape"], totals["shape"]),
        "color": _safe_div(correct["color"], totals["color"]),
        "material": _safe_div(correct["material"], totals["material"]),
        "exact_sequence": _safe_div(exact_sequence_correct, len(predicted)),
    }
