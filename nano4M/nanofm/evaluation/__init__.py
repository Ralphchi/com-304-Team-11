"""Evaluation module for the Team 11 nano4M extension.

Provides per-modality metrics and a CLEVR scene-description parser used to
compare the four masking variants (baseline, block, span, mixed) on a
held-out validation set.

Modules
-------
scene_parser    CLEVR scene-description string -> list of (x, y, shape, color, material)
hungarian_match Align predicted objects to ground-truth by 2D position
metrics         AbsRel / RMSE / delta_1 (depth), angular error (normals),
                per-field accuracy (scene_desc), SSIM (RGB)
eval_harness    Load a checkpoint, run N validation samples, report metrics
"""

from .scene_parser import parse_scene_description
from .hungarian_match import match_objects
from .metrics import (
    depth_absrel,
    depth_rmse,
    depth_delta1,
    normals_angular_error,
    rgb_ssim,
    scene_desc_per_field_accuracy,
)

__all__ = [
    "parse_scene_description",
    "match_objects",
    "depth_absrel",
    "depth_rmse",
    "depth_delta1",
    "normals_angular_error",
    "rgb_ssim",
    "scene_desc_per_field_accuracy",
]
