"""Evaluation harness for the Team 11 nano4M extension.

Loads a trained nano4M checkpoint, runs generation on a fixed held-out
validation subset (500 samples per the proposal Section IV), decodes the
generated tokens back into per-modality predictions, and computes all
metrics defined in `metrics.py`.

This harness is the primary comparison tool across the four masking variants
(baseline, block, span, mixed). All variants are evaluated on exactly the
same held-out sample set and the same generation hyperparameters
(ROAR decoding, constant unmasking schedule, temperature 1.0, no CFG).

Usage (from `run_evaluation.py`):
    harness = EvalHarness(checkpoint_path, config_path, device)
    harness.load()
    results = harness.run(num_samples=500, seed=0)
    print(results)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

import torch

from .scene_parser import parse_scene_description, SceneObject
from .metrics import (
    depth_absrel,
    depth_rmse,
    depth_delta1,
    normals_angular_error,
    rgb_ssim,
    rgb_fid,
    scene_desc_per_field_accuracy,
)


@dataclass
class EvalResults:
    """Aggregated metrics across all evaluated samples."""
    num_samples: int = 0
    # Depth (averaged across samples)
    depth_absrel: float = float("nan")
    depth_rmse: float = float("nan")
    depth_delta1: float = float("nan")
    # Normals
    normals_angular_error: float = float("nan")
    # RGB
    rgb_ssim: float = float("nan")
    rgb_fid: float = float("nan")  # secondary; proposal Section IV
    # Scene desc
    scene_desc_position: float = float("nan")
    scene_desc_shape: float = float("nan")
    scene_desc_color: float = float("nan")
    scene_desc_material: float = float("nan")
    scene_desc_set_match: float = float("nan")
    scene_desc_exact_sequence: float = float("nan")
    # Extras (filled opportunistically)
    extras: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        d = {
            "num_samples": self.num_samples,
            "depth/absrel": self.depth_absrel,
            "depth/rmse": self.depth_rmse,
            "depth/delta1": self.depth_delta1,
            "normals/angular_error_deg": self.normals_angular_error,
            "rgb/ssim": self.rgb_ssim,
            "rgb/fid": self.rgb_fid,
            "scene_desc/position": self.scene_desc_position,
            "scene_desc/shape": self.scene_desc_shape,
            "scene_desc/color": self.scene_desc_color,
            "scene_desc/material": self.scene_desc_material,
            "scene_desc/set_match": self.scene_desc_set_match,
            "scene_desc/exact_sequence": self.scene_desc_exact_sequence,
        }
        d.update(self.extras)
        return d


class EvalHarness:
    """Thin wrapper around a trained nano4M FourM model for evaluation.

    TODO(Ralph, Week 1/2): The core `run()` method is still a stub. The
    scaffolding below fixes signatures, wiring, and metric aggregation so
    teammates can see exactly what the evaluation pipeline expects. Filling
    in the generation loop depends on:
    - Loading the trained FourM checkpoint (see nanofm.utils.checkpoint)
    - Loading the CLEVR val split via SimpleMultimodalDataset
    - Calling `model.generate_one_modality_roar` for each target modality
    - Decoding tokens back to modalities via the Cosmos-DI16x16 tokenizer
      (for image-like modalities) and the GPT-2 text tokenizer (for scene_desc)
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.device = device
        self.model = None
        self.dataset = None
        self.text_tokenizer = None
        self.image_tokenizer = None

    def load(self) -> None:
        """Load model, dataset, and tokenizers.

        TODO(Ralph, Week 2): Wire this up once teammates confirm the config
        loading flow used by `run_training.py`. Minimal skeleton:
            1) Parse YAML config (OmegaConf)
            2) Instantiate FourM model via hydra.utils.instantiate
            3) Load weights from checkpoint_path (safetensors)
            4) model.to(self.device).eval()
            5) Build val dataloader via create_multimodal_masked_dataloader(split='val')
            6) Load Cosmos-DI16x16 tokenizer for image decoding
            7) Load GPT-2 tokenizer for text decoding
        """
        raise NotImplementedError(
            "EvalHarness.load stub — to be filled in Week 2 once we confirm the "
            "exact checkpoint/config loading path used by run_training.py."
        )

    @torch.no_grad()
    def run(
        self,
        num_samples: int = 500,
        seed: int = 0,
        sample_steps: int = 8,
        temperature: float = 1.0,
    ) -> EvalResults:
        """Run evaluation on the validation set.

        Parameters
        ----------
        num_samples : int
            Number of held-out samples to evaluate (proposal spec: 500).
        seed : int
            Generation seed (the proposal specifies 3 seeds per config).
        sample_steps : int
            ROAR unmasking steps (constant schedule per proposal).
        temperature : float
            Sampling temperature (proposal: 1.0, no CFG).

        Returns
        -------
        EvalResults
            Aggregated per-modality metrics.

        TODO(Ralph, Week 2): Fill in the generation loop. Pseudocode:

            torch.manual_seed(seed)
            preds_depth, gts_depth = [], []
            preds_normals, gts_normals = [], []
            preds_rgb, gts_rgb = [], []
            preds_scene, gts_scene = [], []

            for batch in self._iter_val(num_samples):
                # For each modality to evaluate, run chained generation
                # with all other modalities as context.
                # Decode generated tokens back to modality-native space
                # and store both predicted and ground-truth tensors.

            return EvalResults(
                num_samples=num_samples,
                depth_absrel=mean([depth_absrel(p,g) for p,g in zip(preds_depth, gts_depth)]),
                ...
                **scene_desc_per_field_accuracy(preds_scene, gts_scene),
            )
        """
        raise NotImplementedError(
            "EvalHarness.run stub — to be implemented in Week 2 after training "
            "kicks off so we can validate on real checkpoints."
        )
