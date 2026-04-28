#!/usr/bin/env python
"""Phase-D recall sanity gate.

Runs GroundingDINO on 20 GT (RGB image, scene_desc) val pairs from CLEVR
and reports mean precision / recall / F1. Pass criterion: mean recall
>= 0.8 — anything lower means GroundingDINO can't see CLEVR's stylised
renders and we fall back to `CLEVRRulesSegmenter` for Phase D.

No checkpoint needed. Just GT data + the detector.

Run on a SCITAS GPU node (the VPN is needed to reach `/work/com-304/`):

    srun --time=00:30:00 --gres=gpu:1 --partition=l40s --account=com-304 --qos=com-304 --mem=16G \\
        python scripts/grounding_dino_recall_gate.py

Or via sbatch with a small wrapper. Exit code 0 if recall >= 0.8, 1 if not.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofm.evaluation.rgb_verifier import RGBVerifier  # noqa: E402
from nanofm.evaluation.scene_parser import parse_scene_description  # noqa: E402


DEFAULT_TAR = "/work/com-304/datasets/clevr_rgb.tar.gz"
DEFAULT_SCENE_DESC_DIR = "/work/com-304/datasets/clevr_com_304/val/scene_desc"
DEFAULT_N = 20
DEFAULT_THRESHOLD = 0.8
IMAGE_SIZE = 256


def load_pairs(
    tar_path: str, scene_desc_dir: str, n: int
) -> List[Tuple[Image.Image, str]]:
    """Stream val PNGs from the tarball and pair them with their scene_desc JSON."""
    pairs: List[Tuple[Image.Image, str]] = []
    sd_dir = Path(scene_desc_dir)

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".png"):
                continue
            if "/val/" not in member.name:
                continue
            stem = Path(member.name).stem  # e.g. point_1212_view_0_domain_rgb

            json_path = sd_dir / f"{stem}.json"
            if not json_path.exists():
                # Some datasets drop the "_domain_rgb" suffix on non-RGB modalities.
                alt_stem = stem.replace("_domain_rgb", "")
                alt_path = sd_dir / f"{alt_stem}.json"
                if alt_path.exists():
                    json_path = alt_path
                else:
                    continue

            f = tf.extractfile(member)
            if f is None:
                continue
            img = (
                Image.open(BytesIO(f.read()))
                .convert("RGB")
                .resize((IMAGE_SIZE, IMAGE_SIZE))
            )

            with open(json_path) as fj:
                obj = json.load(fj)
            caption = obj[0] if isinstance(obj, list) else obj

            pairs.append((img, caption))
            if len(pairs) >= n:
                break

    return pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--tar-path", default=DEFAULT_TAR,
                   help="Tarball with CLEVR RGB val images.")
    p.add_argument("--scene-desc-dir", default=DEFAULT_SCENE_DESC_DIR,
                   help="Directory with per-sample scene_desc JSONs.")
    p.add_argument("--n-samples", type=int, default=DEFAULT_N,
                   help="Number of (image, caption) pairs to evaluate.")
    p.add_argument("--device", default="cuda",
                   help="Torch device for GroundingDINO.")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Mean-recall pass threshold (proposal default 0.8).")
    p.add_argument("--detector-model", default="IDEA-Research/grounding-dino-tiny",
                   help="HuggingFace id for the detector.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading up to {args.n_samples} val (image, caption) pairs ...")
    pairs = load_pairs(args.tar_path, args.scene_desc_dir, args.n_samples)
    if len(pairs) == 0:
        print("ERROR: no (image, caption) pairs loaded — check paths.", file=sys.stderr)
        return 2
    if len(pairs) < args.n_samples:
        print(
            f"warning: only {len(pairs)} pairs loaded (asked for {args.n_samples})",
            file=sys.stderr,
        )

    print(f"Loading detector: {args.detector_model}")
    verifier = RGBVerifier(model_name=args.detector_model, device=args.device)

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    n_with_objects = 0

    for i, (img, caption) in enumerate(pairs):
        tensor = TF.to_tensor(img)  # (3, H, W) in [0, 1]
        objs = parse_scene_description(caption)
        score = verifier.score(tensor, objs)
        n_expected = score["n_expected"]
        n_detected = score["n_detected"]

        if n_expected > 0:
            n_with_objects += 1
            r = score["recall"]
            p = score["precision"]
            f1 = score["f1"]
            if r == r:
                recalls.append(float(r))
            if p == p:
                precisions.append(float(p))
            f1s.append(float(f1))
            print(
                f"[{i:02d}] expected={n_expected} detected={n_detected} "
                f"P={p:.2f} R={r:.2f} F1={f1:.2f}"
            )
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
    print(
        "FAIL — fall back to nanofm.evaluation.rgb_verifier.CLEVRRulesSegmenter "
        "(currently a stub; needs to be filled in)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
