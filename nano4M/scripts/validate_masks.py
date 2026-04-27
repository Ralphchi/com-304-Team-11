"""Automated mask-invariant checker for the Team 11 Week-1 gate.

Complements scripts/inspect_scene_desc.py:
  - inspect_scene_desc.py: visual rendering on real CLEVR samples (SCITAS only).
  - validate_masks.py    : programmatic invariant checks on synthetic data
    (runs anywhere — no CLEVR mount, no GPU).

Variants (revised extension plan, Section III):
  - baseline       V0 Random i.i.d.        (Gabriel)  - SimpleMultimodalMasking
  - block          V1 Block on images      (Nacer)    - BlockMasking
  - context-block  V2 Ctx-Block on images  (Gabriel)  - ContextBlockMasking
  - span           V3 Span on scene_desc   (Ricardo)  - SpanMasking

Universal invariants asserted for every variant:
  * 8 output keys: enc/dec_{tokens,positions,modalities,pad_mask}.
  * Shapes: enc_* are (max_input_tokens,); dec_* are (max_target_tokens,).
  * Dtypes: tokens/positions/modalities are torch.long; pad_masks are torch.bool.
  * dec_tokens at padded positions equal -100 (CE ignore_index).
  * pad masks are True-prefix / False-suffix (padding lives at the tail).
  * Per modality, encoder and decoder positions are disjoint.
  * All positions in [0, max_seq_lens[m]) (or shifted range when overlap_posembs=False).

V2-specific invariants asserted when --variant context-block (or "all"):
  * For each image modality (rgb/depth/normal): the visible (encoder)
    positions form a contiguous s x s block in the 16x16 grid (regime 1) or
    a subset of one (regime 2 fallback).
  * s in {4, 5, 6}.
  * The same s is shared across the three image modalities in one __call__.
  * Image decoder positions are outside the visible block (regime 1).

V3-specific invariants asserted when --variant span (or "all"):
  * scene_desc decoder positions, sorted, form contiguous runs (= spans);
    every run has length >= 1 and stays within [0, max_seq_len).
  * Image modalities' decoder positions don't form a single contiguous run
    (sanity check that random masking is preserved on image modalities).

Variants whose class is not yet shipped are SKIPPED with a clear message.
Returns exit 0 iff every non-skipped variant passes its invariants on every call.
Writes a one-line summary per variant to nano4M/tests/fixtures/validate_masks_result.txt.

Usage:
    cd nano4M && python scripts/validate_masks.py
    cd nano4M && python scripts/validate_masks.py --variant context-block --n 100 --seed 0
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent  # nano4M/
sys.path.insert(0, str(ROOT))

from nanofm.data.multimodal.masking import SimpleMultimodalMasking  # noqa: E402


# Mirrors cfgs/nano4M/multiclevr_d6-6w512.yaml so behaviour matches training.
MODALITIES = ["tok_rgb@256", "tok_depth@256", "tok_normal@256", "scene_desc"]
VOCAB_SIZES = [64000, 64000, 64000, 50304]
MAX_SEQ_LENS = [256, 256, 256, 256]
INPUT_ALPHAS = [1.0, 1.0, 1.0, 1.0]
TARGET_ALPHAS = [1.0, 1.0, 1.0, 1.0]
INPUT_TOKENS_RANGE = (1, 128)
TARGET_TOKENS_RANGE = (1, 128)

IMAGE_MODALITIES = ["tok_rgb@256", "tok_depth@256", "tok_normal@256"]
TEXT_MODALITIES = ["scene_desc"]
GRID_SIZE = 16
CONTEXT_BLOCK_SIZES = [4, 5, 6]

OUT_PATH = ROOT / "tests" / "fixtures" / "validate_masks_result.txt"


# Variant registry: maps CLI key to (class_name, module_path, extra_kwargs).
VARIANTS: Dict[str, Tuple[str, str, Dict[str, Any]]] = {
    "block": (
        "BlockMasking",
        "nanofm.data.multimodal.block_masking",
        dict(
            image_modalities=IMAGE_MODALITIES,
            text_modalities=TEXT_MODALITIES,
            grid_size=GRID_SIZE,
            block_sizes=[2, 3, 4],
        ),
    ),
    "context-block": (
        "ContextBlockMasking",
        "nanofm.data.multimodal.context_block_masking",
        dict(
            image_modalities=IMAGE_MODALITIES,
            text_modalities=TEXT_MODALITIES,
            grid_size=GRID_SIZE,
            context_block_sizes=CONTEXT_BLOCK_SIZES,
        ),
    ),
    "span": (
        "SpanMasking",
        "nanofm.data.multimodal.span_masking",
        dict(
            text_modalities=TEXT_MODALITIES,
            mean_span_length=3,
        ),
    ),
}


def _build_baseline_kwargs() -> Dict[str, Any]:
    return dict(
        modalities=MODALITIES,
        vocab_sizes=VOCAB_SIZES,
        max_seq_lens=MAX_SEQ_LENS,
        input_alphas=INPUT_ALPHAS,
        target_alphas=TARGET_ALPHAS,
        input_tokens_range=INPUT_TOKENS_RANGE,
        target_tokens_range=TARGET_TOKENS_RANGE,
        overlap_vocab=True,
        overlap_posembs=True,
        include_unmasked_data_dict=False,
    )


def build_masking(variant: str) -> Optional[Callable]:
    """Instantiate the masking transform; return None if its class isn't shipped."""
    baseline_kwargs = _build_baseline_kwargs()
    if variant == "baseline":
        return SimpleMultimodalMasking(**baseline_kwargs)

    class_name, module_path, extras = VARIANTS[variant]
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        return None
    cls = getattr(mod, class_name, None)
    if cls is None:
        return None
    return cls(**baseline_kwargs, **extras)


def _make_synthetic_data_dict(rng: torch.Generator) -> Dict[str, torch.Tensor]:
    """One synthetic sample: random token ids in [0, vocab_size) per modality."""
    out: Dict[str, torch.Tensor] = {}
    for mod, vsize, slen in zip(MODALITIES, VOCAB_SIZES, MAX_SEQ_LENS):
        out[mod] = torch.randint(0, vsize, (slen,), generator=rng, dtype=torch.long)
    return out


# ----------------------------- invariants -----------------------------

REQUIRED_KEYS = (
    "enc_tokens", "enc_positions", "enc_modalities", "enc_pad_mask",
    "dec_tokens", "dec_positions", "dec_modalities", "dec_pad_mask",
)


def assert_universal(masked: Dict[str, torch.Tensor]) -> None:
    """Universal invariants — apply to every masking variant."""
    for k in REQUIRED_KEYS:
        if k not in masked:
            raise AssertionError(f"missing key {k!r}; got {sorted(masked.keys())}")

    max_input = INPUT_TOKENS_RANGE[1]
    max_target = TARGET_TOKENS_RANGE[1]

    for k in ("enc_tokens", "enc_positions", "enc_modalities", "enc_pad_mask"):
        if masked[k].shape != (max_input,):
            raise AssertionError(f"{k} has shape {tuple(masked[k].shape)}, expected ({max_input},)")
    for k in ("dec_tokens", "dec_positions", "dec_modalities", "dec_pad_mask"):
        if masked[k].shape != (max_target,):
            raise AssertionError(f"{k} has shape {tuple(masked[k].shape)}, expected ({max_target},)")

    # Dtypes
    for k in ("enc_tokens", "enc_positions", "enc_modalities",
              "dec_tokens", "dec_positions", "dec_modalities"):
        if masked[k].dtype != torch.long:
            raise AssertionError(f"{k} has dtype {masked[k].dtype}, expected torch.long")
    for k in ("enc_pad_mask", "dec_pad_mask"):
        if masked[k].dtype != torch.bool:
            raise AssertionError(f"{k} has dtype {masked[k].dtype}, expected torch.bool")

    # Padding values
    dec_pad = ~masked["dec_pad_mask"]
    if dec_pad.any() and not (masked["dec_tokens"][dec_pad] == -100).all():
        raise AssertionError(
            "dec_tokens at padded positions must equal -100 (CE ignore_index)"
        )

    # True-prefix, False-suffix structure for pad masks.
    for k in ("enc_pad_mask", "dec_pad_mask"):
        m = masked[k].tolist()
        n_true = sum(m)
        if m[:n_true] != [True] * n_true or m[n_true:] != [False] * (len(m) - n_true):
            raise AssertionError(f"{k} is not True-prefix / False-suffix: {m[:5]}...{m[-5:]}")

    # Per-modality enc/dec position disjointness; positions in valid range.
    for mod_idx, slen in enumerate(MAX_SEQ_LENS):
        enc_sel = (masked["enc_modalities"] == mod_idx) & masked["enc_pad_mask"]
        dec_sel = (masked["dec_modalities"] == mod_idx) & masked["dec_pad_mask"]
        enc_pos = set(masked["enc_positions"][enc_sel].tolist())
        dec_pos = set(masked["dec_positions"][dec_sel].tolist())
        # overlap_posembs=True in our config → positions are in [0, slen)
        for p in enc_pos | dec_pos:
            if not (0 <= p < slen):
                raise AssertionError(
                    f"position {p} for modality_idx={mod_idx} out of range [0, {slen})"
                )
        if enc_pos & dec_pos:
            raise AssertionError(
                f"modality_idx={mod_idx}: enc/dec positions overlap at "
                f"{sorted(enc_pos & dec_pos)[:5]}"
            )


def assert_v2_context_block(masked: Dict[str, torch.Tensor]) -> None:
    """V2-specific invariants for context-block masking."""
    sampled_block_sizes: List[int] = []
    for mod_idx, mod in enumerate(MODALITIES):
        if mod not in IMAGE_MODALITIES:
            continue
        enc_sel = (masked["enc_modalities"] == mod_idx) & masked["enc_pad_mask"]
        dec_sel = (masked["dec_modalities"] == mod_idx) & masked["dec_pad_mask"]
        enc_pos = sorted(masked["enc_positions"][enc_sel].tolist())
        dec_pos = set(masked["dec_positions"][dec_sel].tolist())

        if not enc_pos:
            raise AssertionError(
                f"image modality {mod}: no visible (encoder) positions — "
                f"V2 always shows at least one visible block"
            )

        rows = [p // GRID_SIZE for p in enc_pos]
        cols = [p % GRID_SIZE for p in enc_pos]
        s_rows = max(rows) - min(rows) + 1
        s_cols = max(cols) - min(cols) + 1
        if s_rows != s_cols:
            raise AssertionError(
                f"image modality {mod}: visible bounding box is {s_rows}x{s_cols}, "
                f"expected square s x s"
            )
        s = s_rows
        if s not in CONTEXT_BLOCK_SIZES:
            raise AssertionError(
                f"image modality {mod}: derived s={s} not in {CONTEXT_BLOCK_SIZES}"
            )
        # Regime 1: visible == full block (s^2 positions).
        # Regime 2: visible is a SUBSET of an s x s block.
        if len(enc_pos) > s * s:
            raise AssertionError(
                f"image modality {mod}: visible={len(enc_pos)} > s^2={s*s}"
            )

        # Decoder positions must be outside the bounding-box block (regime 1) or
        # specifically at the spilled-into positions (regime 2).
        block_set = {
            r * GRID_SIZE + c
            for r in range(min(rows), min(rows) + s)
            for c in range(min(cols), min(cols) + s)
        }
        if len(enc_pos) == s * s:
            # Regime 1: dec must be entirely outside the block.
            overlap = dec_pos & block_set
            if overlap:
                raise AssertionError(
                    f"image modality {mod} (regime 1): dec positions inside block: "
                    f"{sorted(overlap)[:5]}"
                )
        else:
            # Regime 2: only the spilled-into positions appear in dec; visible
            # is the rest of the block.
            spilled = block_set - set(enc_pos)
            extra_dec_in_block = dec_pos & block_set
            if extra_dec_in_block != spilled:
                raise AssertionError(
                    f"image modality {mod} (regime 2): dec positions in block "
                    f"don't equal spilled set"
                )

        sampled_block_sizes.append(s)

    if len(set(sampled_block_sizes)) > 1:
        raise AssertionError(
            f"per-call invariant violated: image modalities used different s values "
            f"{sampled_block_sizes}"
        )


def assert_v3_span(masked: Dict[str, torch.Tensor]) -> None:
    """V3-specific invariants for span masking on scene_desc."""
    text_idx = MODALITIES.index("scene_desc")
    text_max_len = MAX_SEQ_LENS[text_idx]

    dec_sel = (masked["dec_modalities"] == text_idx) & masked["dec_pad_mask"]
    dec_pos = sorted(masked["dec_positions"][dec_sel].tolist())

    for p in dec_pos:
        if not (0 <= p < text_max_len):
            raise AssertionError(
                f"scene_desc dec position {p} out of range [0, {text_max_len})"
            )

    # Group sorted positions into contiguous runs (= spans). Every run must
    # have length >= 1; this is the structural definition of a span.
    if dec_pos:
        runs = []
        run_start = dec_pos[0]
        run_len = 1
        for p in dec_pos[1:]:
            if p == run_start + run_len:
                run_len += 1
            else:
                runs.append((run_start, run_len))
                run_start = p
                run_len = 1
        runs.append((run_start, run_len))
        for rs, rl in runs:
            if rl < 1:
                raise AssertionError(f"span run at {rs} has invalid length {rl}")

    # Sanity: image modalities should still look random. A single fully
    # contiguous decoder run on any image modality (with len >= 4) is
    # vanishingly unlikely under uniform random masking and indicates a bug.
    for img_mod in IMAGE_MODALITIES:
        mod_idx = MODALITIES.index(img_mod)
        dec_sel = (masked["dec_modalities"] == mod_idx) & masked["dec_pad_mask"]
        img_dec = sorted(masked["dec_positions"][dec_sel].tolist())
        if len(img_dec) >= 4:
            is_contig = all(img_dec[j + 1] == img_dec[j] + 1 for j in range(len(img_dec) - 1))
            if is_contig:
                raise AssertionError(
                    f"image modality {img_mod} dec positions form a single contiguous "
                    f"run of length {len(img_dec)}; image masking should stay random under V3"
                )


# ----------------------------- runner -----------------------------

def run_variant(
        variant: str,
        n_calls: int,
        rng: torch.Generator,
    ) -> Tuple[str, str]:
    """Returns (status, message). status in {'PASS', 'SKIP', 'FAIL'}."""
    masking = build_masking(variant)
    if masking is None:
        cls_name, mod_path, _ = VARIANTS[variant]
        return "SKIP", f"{cls_name} not found in {mod_path} (teammate hasn't shipped yet)"

    for i in range(n_calls):
        data_dict = _make_synthetic_data_dict(rng)
        try:
            masked = masking(data_dict)
            assert_universal(masked)
            if variant == "context-block":
                assert_v2_context_block(masked)
            elif variant == "span":
                assert_v3_span(masked)
        except AssertionError as e:
            return "FAIL", f"call {i} failed: {e}"
        except Exception as e:
            return "FAIL", f"call {i} raised {type(e).__name__}: {e}"

    return "PASS", f"{n_calls}/{n_calls} calls satisfy invariants"


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate mask invariants for Team 11 variants.")
    ap.add_argument(
        "--variant",
        choices=["all", "baseline", "block", "context-block", "span"],
        default="all",
        help="Which variant(s) to validate. Default: all.",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of __call__ invocations per variant (default: 50).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for synthetic data + masking RNGs.",
    )
    args = ap.parse_args()

    import random as _random
    _random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = torch.Generator()
    rng.manual_seed(args.seed)

    targets: List[str]
    if args.variant == "all":
        targets = ["baseline", "block", "context-block", "span"]
    else:
        targets = [args.variant]

    print(f"=== validate_masks: n={args.n} per variant, seed={args.seed} ===")
    results: List[Tuple[str, str, str]] = []
    overall_ok = True
    for v in targets:
        status, msg = run_variant(v, args.n, rng)
        tag = {"PASS": "[PASS]", "SKIP": "[SKIP]", "FAIL": "[FAIL]"}[status]
        print(f"  {tag} {v:<14} — {msg}")
        results.append((v, status, msg))
        if status == "FAIL":
            overall_ok = False

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Generated by nano4M/scripts/validate_masks.py\n"
        f"# n={args.n} per variant, seed={args.seed}\n"
        f"# overall_pass={overall_ok}\n\n"
    )
    body = "\n".join(f"{v:<14} {status:<5} {msg}" for v, status, msg in results)
    OUT_PATH.write_text(header + body + "\n")
    print(f"\n[info] wrote {OUT_PATH}")

    if not overall_ok:
        print("[FAIL] one or more variants violated invariants.")
        return 1
    print("[PASS] all non-skipped variants satisfy invariants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
