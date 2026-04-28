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
from typing import Dict, List, Optional, Tuple

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
    # Phase A — per-modality reconstruction
    depth_absrel: float = float("nan")
    depth_rmse: float = float("nan")
    depth_delta1: float = float("nan")
    normals_angular_error: float = float("nan")
    rgb_ssim: float = float("nan")
    rgb_fid: float = float("nan")
    scene_desc_position: float = float("nan")
    scene_desc_shape: float = float("nan")
    scene_desc_color: float = float("nan")
    scene_desc_material: float = float("nan")
    scene_desc_set_match: float = float("nan")
    scene_desc_exact_sequence: float = float("nan")
    scene_desc_parse_rate: float = float("nan")
    # Phase B — cross-modal RGB → scene_desc (parser side)
    cross_rgb_to_text_position: float = float("nan")
    cross_rgb_to_text_shape: float = float("nan")
    cross_rgb_to_text_color: float = float("nan")
    cross_rgb_to_text_material: float = float("nan")
    cross_rgb_to_text_set_match: float = float("nan")
    cross_rgb_to_text_exact_sequence: float = float("nan")
    cross_rgb_to_text_parse_rate: float = float("nan")
    # Phase C — LLM judge on cross-modal RGB → scene_desc
    cross_rgb_to_text_llm_alignment: float = float("nan")
    cross_rgb_to_text_llm_perfect_rate: float = float("nan")
    cross_rgb_to_text_llm_parse_error_rate: float = float("nan")
    cross_rgb_to_text_parser_judge_corr: float = float("nan")
    # Phase B — cross-modal scene_desc → RGB (image-quality side)
    cross_text_to_rgb_ssim: float = float("nan")
    cross_text_to_rgb_fid: float = float("nan")
    # Phase D — object-detection verifier on cross-modal scene_desc → RGB
    cross_text_to_rgb_obj_precision: float = float("nan")
    cross_text_to_rgb_obj_recall: float = float("nan")
    cross_text_to_rgb_obj_f1: float = float("nan")
    cross_text_to_rgb_obj_perfect_rate: float = float("nan")
    # Extras (anything not in the schema above)
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
            "cross/rgb_to_text/position": self.cross_rgb_to_text_position,
            "cross/rgb_to_text/shape": self.cross_rgb_to_text_shape,
            "cross/rgb_to_text/color": self.cross_rgb_to_text_color,
            "cross/rgb_to_text/material": self.cross_rgb_to_text_material,
            "cross/rgb_to_text/set_match": self.cross_rgb_to_text_set_match,
            "cross/rgb_to_text/exact_sequence": self.cross_rgb_to_text_exact_sequence,
            "cross/rgb_to_text/parse_rate": self.cross_rgb_to_text_parse_rate,
            "cross/rgb_to_text/llm_alignment": self.cross_rgb_to_text_llm_alignment,
            "cross/rgb_to_text/llm_perfect_rate": self.cross_rgb_to_text_llm_perfect_rate,
            "cross/rgb_to_text/llm_parse_error_rate": self.cross_rgb_to_text_llm_parse_error_rate,
            "cross/rgb_to_text/parser_judge_corr": self.cross_rgb_to_text_parser_judge_corr,
            "cross/text_to_rgb/ssim": self.cross_text_to_rgb_ssim,
            "cross/text_to_rgb/fid": self.cross_text_to_rgb_fid,
            "cross/text_to_rgb/obj_precision": self.cross_text_to_rgb_obj_precision,
            "cross/text_to_rgb/obj_recall": self.cross_text_to_rgb_obj_recall,
            "cross/text_to_rgb/obj_f1": self.cross_text_to_rgb_obj_f1,
            "cross/text_to_rgb/obj_perfect_rate": self.cross_text_to_rgb_obj_perfect_rate,
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
        phases: frozenset = frozenset({"A"}),
        cache_dir: Optional[Path] = None,
        llm_judge=None,
        rgb_verifier=None,
    ) -> EvalResults:
        if self.model is None:
            raise RuntimeError("EvalHarness.load() must be called before .run().")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        n = min(num_samples, len(self.dataset))
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Phase A: per-modality reconstruction (always runs; loads model context).
        preds: Dict[str, List[torch.Tensor]] = {m: [] for m in self.modalities}
        gts: Dict[str, List[torch.Tensor]] = {m: [] for m in self.modalities}
        samples: List[Dict[str, torch.Tensor]] = []

        for sample_idx in self._iter_val_indices(n):
            sample = self.dataset[sample_idx]
            samples.append(sample)
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

        results = self._aggregate(preds, gts, n)

        # Phase B: cross-modal generation. Produces (gt_caption, gen_caption,
        # gen_rgb) tuples per sample. C and D consume these.
        cross_outputs = None
        if "B" in phases or "C" in phases or "D" in phases:
            cross_outputs = self._run_phase_b(
                samples=samples,
                sample_steps=sample_steps,
                temperature=temperature,
                cache_dir=cache_dir,
            )
            self._fill_phase_b_metrics(results, cross_outputs)

        # Phase C: LLM-as-judge on (gt_caption, gen_caption) pairs.
        if "C" in phases:
            if llm_judge is None:
                raise RuntimeError(
                    "Phase C requires an llm_judge (LLMJudge instance)."
                )
            self._fill_phase_c_metrics(results, cross_outputs, llm_judge)

        # Phase D: object-detection verifier on generated RGB images.
        if "D" in phases:
            if rgb_verifier is None:
                raise RuntimeError(
                    "Phase D requires an rgb_verifier (RGBVerifier instance)."
                )
            self._fill_phase_d_metrics(results, cross_outputs, rgb_verifier)

        return results

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

    # ----------------------------------------------------- cross-modal context

    def _build_single_modality_context(
        self, sample: Dict[str, torch.Tensor], source_mod: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encoder context = ONLY source_mod's full token sequence.

        Returns the same (1, N) shape tensors as `_build_full_context` but
        with N = max_seq_lens[source_mod_idx]. Used by Phase B for the
        RGB-only and text-only conditioning setups.
        """
        if source_mod not in self.modalities:
            raise ValueError(f"Unknown source modality: {source_mod}")
        mod_idx = self.modalities.index(source_mod)
        cumulative_shifts = self._max_seq_len_shifts()

        mod_tokens = sample[source_mod].to(self.device).long()
        n = int(mod_tokens.shape[0])
        assert n == self.max_seq_lens[mod_idx]

        shift = 0 if self.overlap_posembs else cumulative_shifts[mod_idx]
        positions = torch.arange(n, device=self.device, dtype=torch.long) + shift
        modalities = torch.full(
            (n,), mod_idx, device=self.device, dtype=torch.long
        )

        return mod_tokens.unsqueeze(0), positions.unsqueeze(0), modalities.unsqueeze(0)

    # --------------------------------------------------- Phase B (cross-modal)

    @torch.no_grad()
    def _run_phase_b(
        self,
        samples: List[Dict[str, torch.Tensor]],
        sample_steps: int,
        temperature: float,
        cache_dir: Optional[Path],
    ) -> Dict[str, list]:
        """Generate RGB → caption and caption → RGB outputs for every sample.

        Returns a dict with five aligned lists, length n:
            gt_captions       : List[str] (decoded GT scene_desc)
            gen_captions      : List[str] (decoded model output)
            gt_rgb            : List[Tensor (3, H, W) in [0, 1]]
            gen_rgb           : List[Tensor (3, H, W) in [0, 1]]
            gt_scene_objects  : List[List[SceneObject]] (parsed GT)
        """
        rgb_mod = next(
            (m for m in self.image_modalities if m.startswith("tok_rgb")), None
        )
        if rgb_mod is None or "scene_desc" not in self.modalities:
            # Cross-modal RGB↔text only meaningful when both modalities present.
            return {
                "gt_captions": [],
                "gen_captions": [],
                "gt_rgb": [],
                "gen_rgb": [],
                "gt_scene_objects": [],
            }

        gt_captions: List[str] = []
        gen_captions: List[str] = []
        gen_rgb_tokens: List[torch.Tensor] = []
        gt_rgb_tokens: List[torch.Tensor] = []
        gt_scene_objects = []

        for sample in samples:
            # RGB → scene_desc.
            ctx_t, ctx_p, ctx_m = self._build_single_modality_context(sample, rgb_mod)
            text_pred, _, _, _ = self.model.generate_one_modality_roar(
                enc_input_tokens=ctx_t,
                enc_input_positions=ctx_p,
                enc_input_modalities=ctx_m,
                target_mod="scene_desc",
                num_steps=sample_steps,
                temp=temperature,
                top_p=0.0,
                top_k=0.0,
            )
            gen_captions.append(self._decode_text_tokens(text_pred.squeeze(0).cpu()))
            gt_captions.append(self._decode_text_tokens(sample["scene_desc"].cpu()))

            # scene_desc → RGB.
            ctx_t, ctx_p, ctx_m = self._build_single_modality_context(sample, "scene_desc")
            rgb_pred, _, _, _ = self.model.generate_one_modality_roar(
                enc_input_tokens=ctx_t,
                enc_input_positions=ctx_p,
                enc_input_modalities=ctx_m,
                target_mod=rgb_mod,
                num_steps=sample_steps,
                temp=temperature,
                top_p=0.0,
                top_k=0.0,
            )
            gen_rgb_tokens.append(rgb_pred.squeeze(0).cpu())
            gt_rgb_tokens.append(sample[rgb_mod].cpu())

        # Decode RGB tokens in batches of 8 (Cosmos memory pressure).
        gen_rgb_imgs = self._decode_in_batches(gen_rgb_tokens, batch=8)
        gt_rgb_imgs = self._decode_in_batches(gt_rgb_tokens, batch=8)

        # Parse GT captions for Phase D's expected_objects.
        gt_scene_objects = [parse_scene_description(s) for s in gt_captions]

        out = {
            "gt_captions": gt_captions,
            "gen_captions": gen_captions,
            "gt_rgb": gt_rgb_imgs,
            "gen_rgb": gen_rgb_imgs,
            "gt_scene_objects": gt_scene_objects,
        }

        if cache_dir is not None:
            self._cache_phase_b(out, cache_dir)

        return out

    def _decode_in_batches(
        self, token_list: List[torch.Tensor], batch: int = 8
    ) -> List[torch.Tensor]:
        """Decode a list of (256,) RGB token tensors via Cosmos in batches."""
        out: List[torch.Tensor] = []
        for start in range(0, len(token_list), batch):
            chunk = torch.stack(token_list[start:start + batch], dim=0)
            decoded = self._decode_image_tokens(chunk).cpu()
            for i in range(decoded.shape[0]):
                out.append(decoded[i])
        return out

    def _cache_phase_b(self, outputs: Dict[str, list], cache_dir: Path) -> None:
        """Persist generated captions and images for inspection."""
        text_dir = cache_dir / "rgb_to_text"
        rgb_dir = cache_dir / "text_to_rgb"
        text_dir.mkdir(parents=True, exist_ok=True)
        rgb_dir.mkdir(parents=True, exist_ok=True)

        for i, (gt, gen) in enumerate(zip(outputs["gt_captions"], outputs["gen_captions"])):
            (text_dir / f"sample_{i:04d}_gt.txt").write_text(gt)
            (text_dir / f"sample_{i:04d}_gen.txt").write_text(gen)

        try:
            from torchvision.utils import save_image  # type: ignore
            for i, (gt, gen) in enumerate(zip(outputs["gt_rgb"], outputs["gen_rgb"])):
                save_image(gt, rgb_dir / f"sample_{i:04d}_gt.png")
                save_image(gen, rgb_dir / f"sample_{i:04d}_gen.png")
        except ImportError:
            # torchvision missing; skip image dumping (metrics still computed).
            pass

    def _fill_phase_b_metrics(self, results: EvalResults, outputs: Dict[str, list]) -> None:
        """Score Phase B outputs with parser per-field accuracy + SSIM/FID."""
        if not outputs["gen_captions"]:
            return

        # RGB → text: parser per-field on (gen_caption, gt_caption).
        gen_objs = [parse_scene_description(s) for s in outputs["gen_captions"]]
        gt_objs = outputs["gt_scene_objects"]
        scores = scene_desc_per_field_accuracy(gen_objs, gt_objs)
        results.cross_rgb_to_text_position = scores["position"]
        results.cross_rgb_to_text_shape = scores["shape"]
        results.cross_rgb_to_text_color = scores["color"]
        results.cross_rgb_to_text_material = scores["material"]
        results.cross_rgb_to_text_set_match = scores["set_match"]
        results.cross_rgb_to_text_exact_sequence = scores["exact_sequence"]
        nonempty_gt = sum(1 for o in gt_objs if len(o) > 0)
        nonempty_gen = sum(1 for o in gen_objs if len(o) > 0)
        results.cross_rgb_to_text_parse_rate = nonempty_gen / max(nonempty_gt, 1)

        # text → RGB: SSIM + FID.
        gen_rgb = torch.stack(outputs["gen_rgb"], dim=0)
        gt_rgb = torch.stack(outputs["gt_rgb"], dim=0)
        results.cross_text_to_rgb_ssim = rgb_ssim(gen_rgb, gt_rgb)
        try:
            results.cross_text_to_rgb_fid = rgb_fid(gen_rgb, gt_rgb)
        except Exception:
            results.cross_text_to_rgb_fid = float("nan")

    # ----------------------------------------------------------------- Phase C

    def _fill_phase_c_metrics(
        self, results: EvalResults, outputs: Dict[str, list], llm_judge
    ) -> None:
        """Run Qwen LLM-judge on the cross-modal RGB → text outputs."""
        from .metrics import caption_llm_judge

        scores = caption_llm_judge(
            originals=outputs["gt_captions"],
            generateds=outputs["gen_captions"],
            judge=llm_judge,
            parser_set_match=[
                1.0 if (len(g) == len(gt) and len(g) > 0) else 0.0
                for g, gt in zip(
                    [parse_scene_description(s) for s in outputs["gen_captions"]],
                    outputs["gt_scene_objects"],
                )
            ],
        )
        results.cross_rgb_to_text_llm_alignment = scores["llm_alignment"]
        results.cross_rgb_to_text_llm_perfect_rate = scores["llm_perfect_rate"]
        results.cross_rgb_to_text_llm_parse_error_rate = scores["llm_parse_error_rate"]
        results.cross_rgb_to_text_parser_judge_corr = scores["parser_judge_corr"]

    # ----------------------------------------------------------------- Phase D

    def _fill_phase_d_metrics(
        self, results: EvalResults, outputs: Dict[str, list], rgb_verifier
    ) -> None:
        """Run object-detection verifier on generated RGB images."""
        from .metrics import rgb_object_detection_score

        scores = rgb_object_detection_score(
            images=outputs["gen_rgb"],
            expected_lists=outputs["gt_scene_objects"],
            verifier=rgb_verifier,
        )
        results.cross_text_to_rgb_obj_precision = scores["precision"]
        results.cross_text_to_rgb_obj_recall = scores["recall"]
        results.cross_text_to_rgb_obj_f1 = scores["f1"]
        results.cross_text_to_rgb_obj_perfect_rate = scores["perfect_rate"]
