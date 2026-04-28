"""Evaluation harness for the Team 11 nano4M extension.

Loads a trained nano4M checkpoint, runs per-modality reconstruction on a
fixed val subset (proposal Section IV: 500 samples, ROAR with constant
schedule, temperature 1.0, no CFG), decodes generated tokens back to native
modality space, and computes the proposal metrics.

Phase A wiring covers the per-modality reconstruction loop. Cross-modal
generation (RGB↔text), LLM-as-judge, and the object-detection verifier
land in subsequent phases (see plan in /Users/ralphchidiac/.claude/plans).

Usage (from `run_evaluation.py`):
    harness = EvalHarness(checkpoint_path, config_path, device)
    harness.load()
    results = harness.run(num_samples=500, seed=0)
    print(results.as_dict())
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from .scene_parser import parse_scene_description
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
    # Depth
    depth_absrel: float = float("nan")
    depth_rmse: float = float("nan")
    depth_delta1: float = float("nan")
    # Normals
    normals_angular_error: float = float("nan")
    # RGB
    rgb_ssim: float = float("nan")
    rgb_fid: float = float("nan")
    # Scene desc
    scene_desc_position: float = float("nan")
    scene_desc_shape: float = float("nan")
    scene_desc_color: float = float("nan")
    scene_desc_material: float = float("nan")
    scene_desc_set_match: float = float("nan")
    scene_desc_exact_sequence: float = float("nan")
    scene_desc_parse_rate: float = float("nan")
    # Extras (cross-modal metrics live here in later phases)
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
            "scene_desc/parse_rate": self.scene_desc_parse_rate,
        }
        d.update(self.extras)
        return d


class EvalHarness:
    """Loads a trained FourM, iterates a deterministic val subset, and runs
    per-modality reconstruction with all-other-modalities-fully-visible context.

    Context spec (committed in the plan): for each target modality, the
    encoder receives every OTHER modality's full token sequence as visible
    context. This is the comparable spec across all variants.
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

        self.modalities: List[str] = []
        self.max_seq_lens: List[int] = []
        self.overlap_posembs: bool = True
        self.image_modalities: List[str] = []
        self.text_modalities: List[str] = []

    # ---------------------------------------------------------------- load

    def load(self) -> None:
        from nanofm.utils.checkpoint import load_model_from_safetensors

        # Model: weights + config recovered from safetensors metadata.
        self.model = load_model_from_safetensors(
            str(self.checkpoint_path), device=self.device, to_eval=True
        )

        # Modality metadata from the model itself (single source of truth).
        self.modalities = list(self.model.modalities)
        self.max_seq_lens = list(self.model.max_seq_lens)
        self.image_modalities = [m for m in self.modalities if m.startswith("tok_")]
        self.text_modalities = [m for m in self.modalities if not m.startswith("tok_")]

        # Config: only used for dataset paths and the text tokenizer settings.
        cfg = OmegaConf.load(self.config_path)
        OmegaConf.resolve(cfg)
        eval_cfg = cfg.eval_loader_config
        self.overlap_posembs = bool(eval_cfg.get("overlap_posembs", True))

        # Val dataset: no masking transform; we read raw tokens and build the
        # encoder context ourselves. sample_from_k_augmentations=1 forces the
        # deterministic center-crop version on every read so seeded runs of
        # different variants score the same val examples.
        from nanofm.data.multimodal.simple_multimodal_dataset import SimpleMultimodalDataset

        self.dataset = SimpleMultimodalDataset(
            root_dir=eval_cfg.root_dir,
            split=eval_cfg.split,
            modalities=list(eval_cfg.modalities),
            transforms=None,
            sample_from_k_augmentations=1,
            text_tokenizer_path=eval_cfg.text_tokenizer_path,
            text_max_length=int(eval_cfg.text_max_length),
        )
        self.text_tokenizer = self.dataset.text_tokenizer

        # Cosmos image decoder. Mirrors scripts/cosmos_sanity_check.py:51-65.
        self.image_tokenizer = self._load_cosmos()

    def _load_cosmos(self):
        from huggingface_hub import snapshot_download
        from cosmos_tokenizer.image_lib import ImageTokenizer

        repo_id = "nvidia/Cosmos-0.1-Tokenizer-DI16x16"
        local_dir = "/tmp/nvidia/Cosmos-0.1-Tokenizer-DI16x16"
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        return ImageTokenizer(
            checkpoint_enc=f"{local_dir}/encoder.jit",
            checkpoint_dec=f"{local_dir}/decoder.jit",
            device=self.device,
        )

    # ---------------------------------------------------------------- run

    @torch.no_grad()
    def run(
        self,
        num_samples: int = 500,
        seed: int = 0,
        sample_steps: int = 8,
        temperature: float = 1.0,
    ) -> EvalResults:
        if self.model is None:
            raise RuntimeError("EvalHarness.load() must be called before .run().")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        n = min(num_samples, len(self.dataset))
        # Per-modality predicted and GT token tensors, in val-iteration order.
        preds: Dict[str, List[torch.Tensor]] = {m: [] for m in self.modalities}
        gts: Dict[str, List[torch.Tensor]] = {m: [] for m in self.modalities}

        for sample_idx in self._iter_val_indices(n):
            sample = self.dataset[sample_idx]
            for target_mod in self.modalities:
                ctx_tokens, ctx_positions, ctx_modalities = self._build_full_context(
                    sample, target_mod
                )
                pred_tokens, _, _, _ = self.model.generate_one_modality_roar(
                    enc_input_tokens=ctx_tokens,
                    enc_input_positions=ctx_positions,
                    enc_input_modalities=ctx_modalities,
                    target_mod=target_mod,
                    num_steps=sample_steps,
                    temp=temperature,
                    top_p=0.0,
                    top_k=0.0,
                )
                preds[target_mod].append(pred_tokens.squeeze(0).cpu())
                gts[target_mod].append(sample[target_mod].cpu())

        return self._aggregate(preds, gts, n)

    # ---------------------------------------------------- val iteration

    def _iter_val_indices(self, n: int) -> List[int]:
        """Deterministic sorted indices — same first n samples every call."""
        # `SimpleMultimodalDataset._get_file_names` already sorts file names,
        # so index order matches sorted-by-filename order. Just return 0..n-1.
        return list(range(n))

    # ---------------------------------------------------- context builder

    def _build_full_context(
        self, sample: Dict[str, torch.Tensor], target_mod: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encoder context = every OTHER modality's full token sequence.

        Returns:
            tokens      : (1, N_total) long
            positions   : (1, N_total) long, per-modality 0..max_seq_len-1
                          (with shifts when overlap_posembs is False)
            modalities  : (1, N_total) long, modality index per token
        """
        tokens_list: List[torch.Tensor] = []
        positions_list: List[torch.Tensor] = []
        modalities_list: List[torch.Tensor] = []

        cumulative_shifts = self._max_seq_len_shifts()

        for mod_idx, mod in enumerate(self.modalities):
            if mod == target_mod:
                continue
            mod_tokens = sample[mod].to(self.device).long()
            n = int(mod_tokens.shape[0])
            assert n == self.max_seq_lens[mod_idx], (
                f"Modality '{mod}' had {n} tokens, expected "
                f"max_seq_lens[{mod_idx}]={self.max_seq_lens[mod_idx]}."
            )
            shift = 0 if self.overlap_posembs else cumulative_shifts[mod_idx]
            positions = torch.arange(n, device=self.device, dtype=torch.long) + shift
            mod_id = torch.full(
                (n,), mod_idx, device=self.device, dtype=torch.long
            )
            tokens_list.append(mod_tokens)
            positions_list.append(positions)
            modalities_list.append(mod_id)

        tokens = torch.cat(tokens_list).unsqueeze(0)
        positions = torch.cat(positions_list).unsqueeze(0)
        modalities = torch.cat(modalities_list).unsqueeze(0)
        return tokens, positions, modalities

    def _max_seq_len_shifts(self) -> List[int]:
        shifts = [0]
        for m in self.max_seq_lens[:-1]:
            shifts.append(shifts[-1] + int(m))
        return shifts

    # ----------------------------------------------------------- decode

    def _decode_image_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode a (B, 256) token tensor through Cosmos to (B, 3, H, W) in [0, 1]."""
        tokens = tokens.to(self.device)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        # Cosmos decoder expects the (B, h, w) integer grid of tokens; reshape
        # 256-flat to 16×16. The dataset stores tokens as flat 256.
        b, n = tokens.shape
        side = int(round(math.sqrt(n)))
        assert side * side == n, f"Expected square grid, got {n} tokens."
        grid = tokens.view(b, side, side)
        recon = self.image_tokenizer.decode(grid).float().clamp(-1.0, 1.0)
        # Cosmos returns (B, 3, H, W) in [-1, 1]. Map to [0, 1].
        return ((recon + 1.0) / 2.0).clamp(0.0, 1.0)

    def _decode_text_tokens(self, tokens: torch.Tensor) -> str:
        """GPT-2 detokenize, dropping special tokens."""
        ids = tokens.detach().cpu().tolist()
        return self.text_tokenizer.decode(ids, skip_special_tokens=True)

    # ----------------------------------------------------- aggregation

    def _aggregate(
        self,
        preds: Dict[str, List[torch.Tensor]],
        gts: Dict[str, List[torch.Tensor]],
        n: int,
    ) -> EvalResults:
        results = EvalResults(num_samples=n)

        # Image modalities: decode and compute SSIM / FID / depth / normals.
        # Decode in batches of 8 to keep Cosmos memory pressure down.
        BATCH = 8

        rgb_pred_imgs: List[torch.Tensor] = []
        rgb_gt_imgs: List[torch.Tensor] = []
        depth_pred_vals: List[torch.Tensor] = []
        depth_gt_vals: List[torch.Tensor] = []
        normals_pred_vecs: List[torch.Tensor] = []
        normals_gt_vecs: List[torch.Tensor] = []

        for mod in self.image_modalities:
            pred_stack = torch.stack(preds[mod], dim=0)  # (n, 256)
            gt_stack = torch.stack(gts[mod], dim=0)

            for start in range(0, n, BATCH):
                end = min(start + BATCH, n)
                pred_imgs = self._decode_image_tokens(pred_stack[start:end])
                gt_imgs = self._decode_image_tokens(gt_stack[start:end])

                if mod.startswith("tok_rgb"):
                    rgb_pred_imgs.append(pred_imgs.cpu())
                    rgb_gt_imgs.append(gt_imgs.cpu())
                elif mod.startswith("tok_depth"):
                    depth_pred_vals.append(self._rgb_to_depth(pred_imgs).cpu())
                    depth_gt_vals.append(self._rgb_to_depth(gt_imgs).cpu())
                elif mod.startswith("tok_normal"):
                    normals_pred_vecs.append(self._rgb_to_normals(pred_imgs).cpu())
                    normals_gt_vecs.append(self._rgb_to_normals(gt_imgs).cpu())

        if rgb_pred_imgs:
            rgb_pred = torch.cat(rgb_pred_imgs, dim=0)
            rgb_gt = torch.cat(rgb_gt_imgs, dim=0)
            results.rgb_ssim = rgb_ssim(rgb_pred, rgb_gt)
            try:
                results.rgb_fid = rgb_fid(rgb_pred, rgb_gt)
            except Exception:
                results.rgb_fid = float("nan")

        if depth_pred_vals:
            depth_pred = torch.cat(depth_pred_vals, dim=0)
            depth_gt = torch.cat(depth_gt_vals, dim=0)
            results.depth_absrel = depth_absrel(depth_pred, depth_gt)
            results.depth_rmse = depth_rmse(depth_pred, depth_gt)
            results.depth_delta1 = depth_delta1(depth_pred, depth_gt)

        if normals_pred_vecs:
            normals_pred = torch.cat(normals_pred_vecs, dim=0)
            normals_gt = torch.cat(normals_gt_vecs, dim=0)
            results.normals_angular_error = normals_angular_error(normals_pred, normals_gt)

        # Text modality: detokenize, parse, score.
        for mod in self.text_modalities:
            if mod != "scene_desc":
                # Other text modalities not part of the proposal; skip.
                continue
            pred_strs = [self._decode_text_tokens(t) for t in preds[mod]]
            gt_strs = [self._decode_text_tokens(t) for t in gts[mod]]
            pred_objs = [parse_scene_description(s) for s in pred_strs]
            gt_objs = [parse_scene_description(s) for s in gt_strs]

            scores = scene_desc_per_field_accuracy(pred_objs, gt_objs)
            results.scene_desc_position = scores["position"]
            results.scene_desc_shape = scores["shape"]
            results.scene_desc_color = scores["color"]
            results.scene_desc_material = scores["material"]
            results.scene_desc_set_match = scores["set_match"]
            results.scene_desc_exact_sequence = scores["exact_sequence"]
            # Parse rate: fraction of GENERATED captions that produced any
            # parsed object. Useful as a "did the model emit canonical format?"
            # diagnostic; falls to 0 if the model emits free-form text the
            # regex parser can't read.
            parsed_nonempty = sum(1 for objs in pred_objs if len(objs) > 0)
            gt_nonempty = sum(1 for objs in gt_objs if len(objs) > 0)
            denom = max(gt_nonempty, 1)
            results.scene_desc_parse_rate = parsed_nonempty / denom

        return results

    # ----- minimal RGB→depth / RGB→normals (Cosmos round-trip conventions)

    @staticmethod
    def _rgb_to_depth(img: torch.Tensor) -> torch.Tensor:
        """Mean-channel grayscale as the depth value.

        The dataset preprocessor renders depth as a 3-channel image before
        Cosmos tokenization, so the decoded RGB is a depth-as-RGB rendering.
        We collapse to a single-channel scalar via channel mean, matching
        the standard depth-as-grayscale convention used by 4M/nano4M.
        """
        return img.mean(dim=1)  # (B, H, W)

    @staticmethod
    def _rgb_to_normals(img: torch.Tensor) -> torch.Tensor:
        """Map [0, 1] RGB → unit normal vectors in [-1, 1].

        Standard normal-to-RGB convention: n = 2 * rgb - 1, then renormalise
        to unit length. Returns (B, 3, H, W).
        """
        n = 2.0 * img - 1.0
        norm = n.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return n / norm
