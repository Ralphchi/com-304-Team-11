#!/usr/bin/env python
"""Phase-D recall sanity gate.

Runs GroundingDINO on 20 GT (RGB image, scene_desc) val pairs from CLEVR
and reports mean precision / recall / F1. Pass criterion: mean recall
>= 0.8 — anything lower means GroundingDINO can't see CLEVR's stylised
renders and we fall back to `CLEVRRulesSegmenter` for Phase D.

Reads from the tokenized dataset (clevr_com_304/val/) so filenames align
with scene_desc by index. RGB tokens decoded via the Cosmos-DI16x16
decoder, the same path used by the eval harness — so the verifier sees
exactly the same RGB representation eval will see at inference time.

Run on a SCITAS GPU node (the VPN is needed to reach `/work/com-304/`):

    srun --time=00:30:00 --gres=gpu:1 --partition=l40s \\
         --account=com-304 --qos=com-304 \\
         --cpus-per-task=4 --mem-per-cpu=4G \\
         python scripts/grounding_dino_recall_gate.py

Exits 0 if mean recall >= 0.8, 1 if not, 2 if no usable pairs loaded.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofm.evaluation.rgb_verifier import RGBVerifier  # noqa: E402
from nanofm.evaluation.scene_parser import parse_scene_description  # noqa: E402


DEFAULT_DATA_ROOT = "/work/com-304/datasets/clevr_com_304"
DEFAULT_SPLIT = "val"
DEFAULT_RGB_MODALITY = "tok_rgb@256"
DEFAULT_N = 20
DEFAULT_THRESHOLD = 0.8


def load_cosmos(device: str):
    import os
    from huggingface_hub import snapshot_download
    from cosmos_tokenizer.image_lib import ImageTokenizer

    repo_id = "nvidia/Cosmos-0.1-Tokenizer-DI16x16"
    # Persistent scratch so we don't re-download on every compute node.
    base = os.environ.get("COSMOS_LOCAL_DIR") or os.path.expanduser(
        "~/cosmos_tokenizer"
    )
    local_dir = f"{base}/Cosmos-0.1-Tokenizer-DI16x16"
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    return ImageTokenizer(
        checkpoint_enc=f"{local_dir}/encoder.jit",
        checkpoint_dec=f"{local_dir}/decoder.jit",
        device=device,
    )


def decode_rgb_tokens(tokenizer, tokens: torch.Tensor) -> torch.Tensor:
    """Decode (B, 256) flat token tensor through Cosmos to (B, 3, H, W) in [0, 1]."""
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    b, n = tokens.shape
    side = int(round(math.sqrt(n)))
    assert side * side == n
    grid = tokens.view(b, side, side)
    recon = tokenizer.decode(grid).float().clamp(-1.0, 1.0)
    return ((recon + 1.0) / 2.0).clamp(0.0, 1.0)


def load_pairs(
    data_root: str, split: str, rgb_modality: str, n: int
) -> List[Tuple[np.ndarray, str]]:
    """Stream first n (rgb_token_array, caption_string) pairs from the dataset."""
    rgb_dir = Path(data_root) / split / rgb_modality
    sd_dir = Path(data_root) / split / "scene_desc"

    if not rgb_dir.is_dir():
        raise FileNotFoundError(rgb_dir)
    if not sd_dir.is_dir():
        raise FileNotFoundError(sd_dir)

    pairs: List[Tuple[np.ndarray, str]] = []
    for stem in sorted(p.stem for p in rgb_dir.glob("*.npy")):
        sd_path = sd_dir / f"{stem}.json"
        if not sd_path.exists():
            continue
        tokens = np.load(rgb_dir / f"{stem}.npy")
        # Each .npy stores K augmentations; take augmentation 0 (canonical).
        tokens = tokens[0] if tokens.ndim == 2 else tokens
        with open(sd_path) as fj:
            cap = json.load(fj)
        caption = cap[0] if isinstance(cap, list) else cap
        pairs.append((tokens, caption))
        if len(pairs) >= n:
            break

    return pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--rgb-modality", default=DEFAULT_RGB_MODALITY)
    p.add_argument("--n-samples", type=int, default=DEFAULT_N)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--detector-model", default="IDEA-Research/grounding-dino-tiny")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading up to {args.n_samples} val pairs from {args.data_root} ...")
    try:
        pairs = load_pairs(args.data_root, args.split, args.rgb_modality, args.n_samples)
    except FileNotFoundError as e:
        print(f"ERROR: missing path {e}", file=sys.stderr)
        return 2

    if len(pairs) == 0:
        print("ERROR: no (image, caption) pairs loaded — check paths.", file=sys.stderr)
        return 2
    if len(pairs) < args.n_samples:
        print(f"warning: only {len(pairs)} pairs loaded (asked for {args.n_samples})",
              file=sys.stderr)

    print(f"Loading Cosmos decoder on {args.device} ...")
    cosmos = load_cosmos(args.device)

    print(f"Loading detector: {args.detector_model}")
    verifier = RGBVerifier(model_name=args.detector_model, device=args.device)

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    n_with_objects = 0

    for i, (tokens_np, caption) in enumerate(pairs):
        tokens = torch.from_numpy(tokens_np).long().to(args.device)
        # Decode this sample's tokens to a single RGB image.
        img = decode_rgb_tokens(cosmos, tokens).squeeze(0)  # (3, H, W) in [0, 1]
        objs = parse_scene_description(caption)
        score = verifier.score(img.detach().cpu(), objs)
        n_expected = score["n_expected"]
        n_detected = score["n_detected"]

        if n_expected > 0:
            n_with_objects += 1
            r, p, f1 = score["recall"], score["precision"], score["f1"]
            if r == r:
                recalls.append(float(r))
            if p == p:
                precisions.append(float(p))
            f1s.append(float(f1))
            print(f"[{i:02d}] expected={n_expected} detected={n_detected} "
                  f"P={p:.2f} R={r:.2f} F1={f1:.2f}")
        else:
            print(f"[{i:02d}] no expected objects parsed from caption (skipped)")

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    mean_p = _mean(precisions)
    mean_r = _mean(recalls)
    mean_f1 = _mean(f1s)

    print()
    print(f"Pairs with parsed expected objects: {n_with_objects}/{len(pairs)}")
    print(f"Mean precision: {mean_p:.3f}")
    print(f"Mean recall:    {mean_r:.3f}")
    print(f"Mean F1:        {mean_f1:.3f}")
    print(f"Pass threshold: recall >= {args.threshold}")

    if mean_r != mean_r:
        print("FAIL — no usable samples; check input paths.")
        return 2
    if mean_r >= args.threshold:
        print("PASS — GroundingDINO usable on CLEVR. Phase D ready to deploy.")
        return 0
    print("FAIL — fall back to nanofm.evaluation.rgb_verifier.CLEVRRulesSegmenter.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
