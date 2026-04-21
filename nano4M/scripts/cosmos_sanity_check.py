"""Cosmos-DI16x16 tokenizer sanity check on 50 CLEVR val images.

Week-1 gate (revised extension plan, Section V, Risk iv):
    "Day-1 Cosmos reconstruction check establishes fidelity before training."

Measures mean SSIM between raw CLEVR val images and their Cosmos-DI16x16
tokenize -> detokenize round-trip. If mean SSIM < 0.85, training all four
variants on Cosmos-encoded tokens risks being bottlenecked by tokenizer
loss rather than by the masking strategy we are studying.

Run on SCITAS (needs the Cosmos checkpoint + raw images + GPU):

    cd nano4M && python scripts/cosmos_sanity_check.py

Writes a result record to nano4M/tests/fixtures/cosmos_sanity_result.txt
so the gate decision is reproducible from the committed history.

Raw images are streamed from /work/com-304/datasets/clevr_rgb.tar.gz
(first 50 PNGs under ./rgb/val/). We don't need to pair them with the
pre-tokenized .npy files — the fidelity check only needs arbitrary
CLEVR images.

Exits 1 if mean SSIM < 0.85 so this script can drive a SLURM job or
CI-style gate.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent  # nano4M/
TAR_PATH = "/work/com-304/datasets/clevr_rgb.tar.gz"
VAL_PREFIX = "./rgb/val/"
N_SAMPLES = 50
IMAGE_SIZE = 256
SSIM_GATE = 0.85
COSMOS_REPO = "nvidia/Cosmos-0.1-Tokenizer-DI16x16"
COSMOS_LOCAL_DIR = "/tmp/nvidia/Cosmos-0.1-Tokenizer-DI16x16"
OUT_PATH = ROOT / "tests" / "fixtures" / "cosmos_sanity_result.txt"


def load_cosmos(device: str):
    """Download (if needed) and instantiate the Cosmos-DI16x16 tokenizer.

    Follows the exact pattern used in Notebook 4 (COM304_FM_part3_nano4M.ipynb
    cell ~344) so the sanity check mirrors what training will use.
    """
    from huggingface_hub import snapshot_download  # type: ignore
    from cosmos_tokenizer.image_lib import ImageTokenizer  # type: ignore

    snapshot_download(repo_id=COSMOS_REPO, local_dir=COSMOS_LOCAL_DIR)
    return ImageTokenizer(
        checkpoint_enc=f"{COSMOS_LOCAL_DIR}/encoder.jit",
        checkpoint_dec=f"{COSMOS_LOCAL_DIR}/decoder.jit",
        device=device,
    )


def stream_val_images(tar_path: str, n: int) -> list[tuple[str, Image.Image]]:
    """Stream the first `n` val PNGs out of the CLEVR RGB tarball.

    We don't extract the whole tarball — iterate members, stop after n.
    """
    images: list[tuple[str, Image.Image]] = []
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            if not (member.name.startswith(VAL_PREFIX) and member.name.endswith(".png")):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            img = (
                Image.open(BytesIO(f.read()))
                .convert("RGB")
                .resize((IMAGE_SIZE, IMAGE_SIZE))
            )
            images.append((Path(member.name).name, img))
            if len(images) >= n:
                break
    return images


def roundtrip_ssim(
    tokenizer,
    img: Image.Image,
    device: str,
) -> float:
    """Encode a PIL image through Cosmos, decode, and return SSIM."""
    from skimage.metrics import structural_similarity  # type: ignore

    t = TF.to_tensor(img).unsqueeze(0).to(device) * 2 - 1  # [-1, 1]
    with torch.no_grad():
        tokens, _ = tokenizer.encode(t)
        reconst = tokenizer.decode(tokens).float().clamp(-1, 1)

    orig_np = ((t + 1) / 2).clamp(0, 1).squeeze(0).detach().cpu().numpy()
    recon_np = ((reconst + 1) / 2).clamp(0, 1).squeeze(0).detach().cpu().numpy()
    orig_hwc = orig_np.transpose(1, 2, 0)
    recon_hwc = recon_np.transpose(1, 2, 0)
    return float(
        structural_similarity(orig_hwc, recon_hwc, channel_axis=2, data_range=1.0)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Cosmos-DI16x16 sanity check on CLEVR val.")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu).",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=N_SAMPLES,
        help=f"Number of val images to test (default: {N_SAMPLES}).",
    )
    ap.add_argument(
        "--gate",
        type=float,
        default=SSIM_GATE,
        help=f"Mean-SSIM gate; exit 1 if below (default: {SSIM_GATE}).",
    )
    args = ap.parse_args()

    print(f"[info] loading {args.n} CLEVR val images from {TAR_PATH}")
    images = stream_val_images(TAR_PATH, args.n)
    if not images:
        print(f"[FAIL] No images found under {VAL_PREFIX} in {TAR_PATH}", file=sys.stderr)
        return 1
    if len(images) < args.n:
        print(f"[warn] only loaded {len(images)} images (requested {args.n})")

    print(f"[info] loading Cosmos-DI16x16 on {args.device}")
    tokenizer = load_cosmos(args.device)

    scores: list[float] = []
    for i, (name, img) in enumerate(images):
        score = roundtrip_ssim(tokenizer, img, args.device)
        scores.append(score)
        if i < 5 or i >= len(images) - 3 or i % 10 == 0:
            print(f"[{i:02d}] {name}  SSIM={score:.4f}")

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    smin = float(np.min(scores))
    smax = float(np.max(scores))

    print(
        f"\n[result] n={len(scores)}  mean={mean:.4f}  min={smin:.4f}  "
        f"max={smax:.4f}  std={std:.4f}  gate={args.gate}"
    )

    # Record the result so the gate decision is reproducible from history.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Generated by nano4M/scripts/cosmos_sanity_check.py\n"
        f"# source: {TAR_PATH} ({VAL_PREFIX}*.png, first {len(scores)})\n"
        f"# tokenizer: {COSMOS_REPO}\n"
        f"# image_size: {IMAGE_SIZE}x{IMAGE_SIZE}  device: {args.device}\n"
        f"# gate: mean SSIM >= {args.gate}\n"
        f"#\n"
        f"# mean={mean:.4f}  min={smin:.4f}  max={smax:.4f}  "
        f"std={std:.4f}  n={len(scores)}\n"
        f"# pass={mean >= args.gate}\n\n"
    )
    body = "\n".join(
        f"{i:03d}  {name}  SSIM={score:.6f}"
        for i, ((name, _), score) in enumerate(zip(images, scores))
    )
    OUT_PATH.write_text(header + body + "\n")
    print(f"[info] wrote {OUT_PATH}")

    if mean < args.gate:
        print(f"[FAIL] mean SSIM {mean:.4f} < gate {args.gate}. Do not start training.")
        return 1
    print(f"[PASS] mean SSIM {mean:.4f} >= gate {args.gate}. Tokenizer fidelity OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
