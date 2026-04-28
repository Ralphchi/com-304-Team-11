#!/usr/bin/env python
"""Visualize V1 block masking on synthetic batches.

Usage:
    cd nano4M
    python scripts/visualize_block_masking.py --out assets/v1_block_mask.png
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nanofm.data.multimodal.block_masking import BlockMasking  # noqa: E402


MODALITIES = ["tok_rgb@256", "tok_depth@256", "tok_normal@256", "scene_desc"]
VOCAB_SIZES = [64000, 64000, 64000, 50304]
MAX_SEQ_LENS = [256, 256, 256, 256]
IMAGE_MODS = MODALITIES[:3]
GRID = 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--block-sizes", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input-tokens-range", type=int, nargs=2, default=[1, 128])
    p.add_argument("--target-tokens-range", type=int, nargs=2, default=[1, 128])
    p.add_argument("--out", type=str, default="assets/v1_block_mask.png")
    p.add_argument("--no-figure", action="store_true",
                   help="Skip the PNG; only print per-sample summaries.")
    return p.parse_args()


def make_random_data():
    return {
        m: torch.randint(0, V, (S,), dtype=torch.long)
        for m, V, S in zip(MODALITIES, VOCAB_SIZES, MAX_SEQ_LENS)
    }


def class_grid_for(out: dict, mod_idx: int, seq_len: int) -> np.ndarray:
    """0=dropped, 1=encoder, 2=decoder."""
    grid = np.zeros(seq_len, dtype=np.int64)
    for p, m, v in zip(out["enc_positions"].tolist(),
                       out["enc_modalities"].tolist(),
                       out["enc_pad_mask"].tolist()):
        if v and int(m) == mod_idx and int(p) < seq_len:
            grid[int(p)] = 1
    for p, m, v in zip(out["dec_positions"].tolist(),
                       out["dec_modalities"].tolist(),
                       out["dec_pad_mask"].tolist()):
        if v and int(m) == mod_idx and int(p) < seq_len:
            grid[int(p)] = 2
    return grid


def render(grids: List[List[np.ndarray]], modalities: List[str], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not available; skipping figure ({out_path})")
        return

    n_samples = len(grids)
    n_mods = len(modalities)

    cmap = np.array([
        [0.20, 0.20, 0.22],
        [0.25, 0.50, 0.85],
        [0.95, 0.55, 0.20],
    ], dtype=np.float32)

    fig, axes = plt.subplots(n_samples, n_mods,
                             figsize=(2.5 * n_mods, 2.5 * n_samples),
                             squeeze=False)
    for i, row in enumerate(grids):
        for j, vec in enumerate(row):
            ax = axes[i][j]
            ax.imshow(cmap[vec.reshape(GRID, GRID)], interpolation="nearest")
            ax.set_xticks(np.arange(-0.5, GRID, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, GRID, 1), minor=True)
            ax.grid(which="minor", color="black", linewidth=0.3, alpha=0.3)
            ax.tick_params(which="both", bottom=False, left=False,
                           labelbottom=False, labelleft=False)
            n_enc = int((vec == 1).sum())
            n_dec = int((vec == 2).sum())
            title = modalities[j] if i == 0 else ""
            ax.set_title(f"{title}\nenc={n_enc} dec={n_dec}".strip(), fontsize=8)
            if j == 0:
                ax.set_ylabel(f"sample {i}", fontsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cmap[1], label="encoder"),
        plt.Rectangle((0, 0), 1, 1, color=cmap[2], label="decoder"),
        plt.Rectangle((0, 0), 1, 1, color=cmap[0], label="dropped"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("V1 block masking: blocks on images, random on scene_desc",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    masking = BlockMasking(
        modalities=MODALITIES,
        vocab_sizes=VOCAB_SIZES,
        max_seq_lens=MAX_SEQ_LENS,
        input_alphas=[1.0] * 4,
        target_alphas=[1.0] * 4,
        input_tokens_range=tuple(args.input_tokens_range),
        target_tokens_range=tuple(args.target_tokens_range),
        image_modalities=IMAGE_MODS,
        text_modalities=["scene_desc"],
        block_sizes=tuple(args.block_sizes),
        grid_size=GRID,
    )

    grids: List[List[np.ndarray]] = []
    for s_idx in range(args.num_samples):
        data = make_random_data()
        out = masking(dict(data))

        row: List[np.ndarray] = []
        summary = []
        for m_idx, mod in enumerate(MODALITIES):
            grid = class_grid_for(out, m_idx, MAX_SEQ_LENS[m_idx])
            row.append(grid)
            n_enc = int((grid == 1).sum())
            n_dec = int((grid == 2).sum())
            summary.append(f"{mod}: enc={n_enc} dec={n_dec}")
        grids.append(row)
        print(f"[sample {s_idx}] " + " | ".join(summary))

    if not args.no_figure:
        out_path = args.out
        if not os.path.isabs(out_path):
            out_path = os.path.join(REPO_ROOT, out_path)
        render(grids, MODALITIES, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
