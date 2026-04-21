"""Mask-inspection helper for the Team 11 Week-1 gate.

Prints, for N real CLEVR val samples, the masks produced by a given variant:

- Image modalities (tok_rgb@256, tok_depth@256, tok_normal@256): rendered as a
  16x16 ASCII grid, where
      `#` = target position (masked, model must predict this)
      `.` = encoder position (visible context)
      ` ` = unused (dropped entirely by the masking budget)
- scene_desc (256 GPT-2 tokens, heavily padded): prints the underlying text
  (caption), then the indices of target / encoder positions, then the decoded
  token at each target position so teammates can sanity-check span masking.

Usage (run on SCITAS, needs the CLEVR mount):

    cd nano4M && python scripts/inspect_scene_desc.py --variant baseline --n 3
    cd nano4M && python scripts/inspect_scene_desc.py --variant block    --n 3
    cd nano4M && python scripts/inspect_scene_desc.py --variant span     --n 3
    cd nano4M && python scripts/inspect_scene_desc.py --variant mixed    --n 3

For non-baseline variants the helper looks up
`nanofm.data.multimodal.masking.{BlockMasking,SpanMasking,MixedMasking}`
by name. If a class isn't defined yet, it raises a clear error pointing
at the teammate's task.

Week-1 gate: every variant produces valid masks on a held-out batch
(proposal Section V, TEAM.md lines 94-100).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent  # nano4M/
sys.path.insert(0, str(ROOT))

from nanofm.data.multimodal.simple_multimodal_dataset import SimpleMultimodalDataset  # noqa: E402
import nanofm.data.multimodal.masking as masking_mod  # noqa: E402


# Baseline config drawn from cfgs/nano4M/multiclevr_d6-6w512.yaml so behaviour
# matches training exactly. If the reference config changes, update here too.
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

DATA_ROOT = "/work/com-304/datasets/clevr_com_304/"


def build_masking(variant: str):
    """Instantiate the masking transform for the requested variant.

    Baseline uses SimpleMultimodalMasking directly. Other variants look up
    {BlockMasking, SpanMasking, MixedMasking} in the masking module and
    construct them with the arguments documented in each variant's YAML.
    """
    baseline_kwargs = dict(
        modalities=MODALITIES,
        vocab_sizes=VOCAB_SIZES,
        max_seq_lens=MAX_SEQ_LENS,
        input_alphas=INPUT_ALPHAS,
        target_alphas=TARGET_ALPHAS,
        input_tokens_range=INPUT_TOKENS_RANGE,
        target_tokens_range=TARGET_TOKENS_RANGE,
        overlap_vocab=True,
        overlap_posembs=True,
        include_unmasked_data_dict=True,  # need unmasked tokens to render
    )

    if variant == "baseline":
        return masking_mod.SimpleMultimodalMasking(**baseline_kwargs)

    class_name = {
        "block": "BlockMasking",
        "span": "SpanMasking",
        "mixed": "MixedMasking",
    }[variant]

    cls = getattr(masking_mod, class_name, None)
    if cls is None:
        raise RuntimeError(
            f"{class_name} is not defined in nanofm.data.multimodal.masking yet.\n"
            f"  - Baseline (Gabriel) / Block (Nacer) / Span (Ricardo) / Mixed (Gabriel)\n"
            f"    — the owner of this variant must ship the class before this\n"
            f"    helper can inspect their masks. See TEAM.md lines 156-175."
        )

    # Variant classes must accept the same baseline kwargs plus their own
    # extra hyperparameters. We pass baseline kwargs and let the class surface
    # its own required extras via a TypeError if the contract drifts.
    extras: dict = {}
    if variant == "block":
        extras = dict(
            image_modalities=IMAGE_MODALITIES,
            text_modalities=TEXT_MODALITIES,
            grid_size=GRID_SIZE,
            block_sizes=[2, 3, 4],
        )
    elif variant == "span":
        extras = dict(
            text_modalities=TEXT_MODALITIES,
            mean_span_length=3,
        )
    elif variant == "mixed":
        extras = dict(
            image_modalities=IMAGE_MODALITIES,
            text_modalities=TEXT_MODALITIES,
            grid_size=GRID_SIZE,
            block_sizes=[2, 3, 4],
            mean_span_length=3,
        )

    try:
        return cls(**baseline_kwargs, **extras)
    except TypeError as e:
        raise RuntimeError(
            f"{class_name} rejected the inspector's constructor arguments: {e}\n"
            f"  The inspector passes: baseline SimpleMultimodalMasking kwargs +\n"
            f"  {list(extras.keys())}. If your class uses a different signature,\n"
            f"  update scripts/inspect_scene_desc.py or align the signature."
        ) from e


def render_image_mask(
    modality_idx: int,
    enc_positions: torch.Tensor,
    enc_modalities: torch.Tensor,
    enc_pad_mask: torch.Tensor,
    dec_positions: torch.Tensor,
    dec_modalities: torch.Tensor,
    dec_pad_mask: torch.Tensor,
    grid_size: int = GRID_SIZE,
) -> str:
    """Render a 16x16 ASCII grid showing mask coverage for one image modality.

    Positions that appear in dec_* (and match this modality & are not padding)
    are masked targets (#). Positions in enc_* are visible context (.).
    Anything else is dropped (space).
    """
    enc_mask = (enc_modalities == modality_idx) & enc_pad_mask
    dec_mask = (dec_modalities == modality_idx) & dec_pad_mask
    enc_pos_set = set(enc_positions[enc_mask].tolist())
    dec_pos_set = set(dec_positions[dec_mask].tolist())

    lines = []
    for row in range(grid_size):
        chars = []
        for col in range(grid_size):
            p = row * grid_size + col
            if p in dec_pos_set:
                chars.append("#")
            elif p in enc_pos_set:
                chars.append(".")
            else:
                chars.append(" ")
        lines.append("".join(chars))
    n_dec = len(dec_pos_set)
    n_enc = len(enc_pos_set)
    header = (
        f"    enc(visible)={n_enc:3d}   dec(masked)={n_dec:3d}   "
        f"dropped={grid_size*grid_size - n_enc - n_dec:3d}"
    )
    return header + "\n" + "\n".join(lines)


def render_scene_desc(
    modality_idx: int,
    unmasked_tokens: torch.Tensor,  # (max_seq_len,) full caption tokens
    enc_positions: torch.Tensor,
    enc_modalities: torch.Tensor,
    enc_pad_mask: torch.Tensor,
    dec_positions: torch.Tensor,
    dec_modalities: torch.Tensor,
    dec_pad_mask: torch.Tensor,
    text_tokenizer,
) -> str:
    """Print the caption, then which indices are masked and what tokens they hold."""
    enc_mask = (enc_modalities == modality_idx) & enc_pad_mask
    dec_mask = (dec_modalities == modality_idx) & dec_pad_mask
    enc_pos = sorted(enc_positions[enc_mask].tolist())
    dec_pos = sorted(dec_positions[dec_mask].tolist())

    # Full decoded caption (skip special tokens to get the human-readable text).
    caption = text_tokenizer.decode(unmasked_tokens.tolist(), skip_special_tokens=True)

    # Per-masked-position token, to make span runs visible.
    dec_token_strs = [
        text_tokenizer.decode([int(unmasked_tokens[p].item())]).replace("\n", "\\n")
        for p in dec_pos
    ]

    lines = [
        f"    enc(visible)={len(enc_pos):3d}   dec(masked)={len(dec_pos):3d}",
        f"    caption: {caption[:160]}{'…' if len(caption) > 160 else ''}",
        f"    masked indices: {dec_pos[:40]}{' …' if len(dec_pos) > 40 else ''}",
        f"    masked tokens : {dec_token_strs[:40]}{' …' if len(dec_token_strs) > 40 else ''}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect masks for a Team-11 variant.")
    ap.add_argument(
        "--variant",
        choices=["baseline", "block", "span", "mixed"],
        required=True,
        help="Which masking variant to inspect",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of val samples to visualise (default: 3)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch/python random seed for reproducible masks",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    import random as _random
    _random.seed(args.seed)

    masking = build_masking(args.variant)

    dataset = SimpleMultimodalDataset(
        root_dir=DATA_ROOT,
        split="val",
        modalities=MODALITIES,
        text_tokenizer_path="gpt2",
        text_max_length=256,
        sample_from_k_augmentations=1,
        transforms=masking,
    )

    print(f"=== variant: {args.variant}   samples: {args.n}   seed: {args.seed} ===")
    for i in range(min(args.n, len(dataset))):
        print(f"\n--- sample {i} (file={dataset.file_names[i]}) ---")
        masked = dataset[i]
        unmasked = masked["unmasked_data_dict"]

        for mi, mod in enumerate(MODALITIES):
            print(f"\n  [{mod}]")
            if mod in IMAGE_MODALITIES:
                print(
                    render_image_mask(
                        modality_idx=mi,
                        enc_positions=masked["enc_positions"],
                        enc_modalities=masked["enc_modalities"],
                        enc_pad_mask=masked["enc_pad_mask"],
                        dec_positions=masked["dec_positions"],
                        dec_modalities=masked["dec_modalities"],
                        dec_pad_mask=masked["dec_pad_mask"],
                    )
                )
            elif mod in TEXT_MODALITIES:
                print(
                    render_scene_desc(
                        modality_idx=mi,
                        unmasked_tokens=unmasked[mod],
                        enc_positions=masked["enc_positions"],
                        enc_modalities=masked["enc_modalities"],
                        enc_pad_mask=masked["enc_pad_mask"],
                        dec_positions=masked["dec_positions"],
                        dec_modalities=masked["dec_modalities"],
                        dec_pad_mask=masked["dec_pad_mask"],
                        text_tokenizer=dataset.text_tokenizer,
                    )
                )


if __name__ == "__main__":
    main()
