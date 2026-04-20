#!/usr/bin/env python
"""Entry point for evaluating a trained nano4M masking variant.

Usage:
    python run_evaluation.py \
        --ckpt outputs/nano4M/multiclevr_d6-6w512_block/checkpoint-final.safetensors \
        --config cfgs/nano4M/multiclevr_d6-6w512_block.yaml \
        --num-samples 500 \
        --seed 0

Runs the `EvalHarness` against a held-out CLEVR val subset and prints a JSON
summary of per-modality metrics. Proposal Section IV specifies this is the
canonical comparison protocol for all four variants.

TODO(Ralph, Week 2): Finish wiring `EvalHarness.load()` and `.run()`.
This CLI stub is intentionally complete so teammates (and SLURM batch
scripts) can invoke it by the same command signature once the harness
lands.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nanofm.evaluation.eval_harness import EvalHarness


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained nano4M masking variant on CLEVR val.",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to the trained checkpoint (.safetensors or .pth).",
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the training YAML used to produce the checkpoint.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of CLEVR val samples to evaluate on (proposal default: 500).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generation (proposal uses 3 seeds per config).",
    )
    p.add_argument(
        "--sample-steps",
        type=int,
        default=8,
        help="Number of ROAR unmasking steps per target modality.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (proposal: 1.0, no CFG).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to evaluate on.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If given, write JSON results to this file (otherwise stdout).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.ckpt.exists():
        print(f"Checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 1
    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    harness = EvalHarness(
        checkpoint_path=str(args.ckpt),
        config_path=str(args.config),
        device=args.device,
    )
    harness.load()
    results = harness.run(
        num_samples=args.num_samples,
        seed=args.seed,
        sample_steps=args.sample_steps,
        temperature=args.temperature,
    )

    out_json = json.dumps(results.as_dict(), indent=2)
    if args.output is not None:
        args.output.write_text(out_json + "\n")
        print(f"Wrote results to {args.output}", file=sys.stderr)
    else:
        print(out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
