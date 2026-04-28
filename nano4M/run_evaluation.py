#!/usr/bin/env python
"""Entry point for evaluating a trained nano4M masking variant.

Usage:
    python run_evaluation.py \
        --ckpt outputs/nano4M/multiclevr_d6-6w512_block/checkpoint-final.safetensors \
        --config cfgs/nano4M/multiclevr_d6-6w512_block.yaml \
        --num-samples 500 \
        --seed 0 \
        --output results/block_seed0.json

Runs the EvalHarness against a held-out CLEVR val subset and writes a JSON
summary of per-modality metrics. Phase A wires up per-modality reconstruction
(proposal Section IV); cross-modal generation, LLM-as-judge, and the
object-detection verifier land in subsequent phases.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nanofm.evaluation.eval_harness import EvalHarness


VALID_PHASES = {"A", "B", "C", "D"}


def parse_phases(arg: str) -> set:
    requested = {p.strip().upper() for p in arg.split(",") if p.strip()}
    bad = requested - VALID_PHASES
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown phase(s): {sorted(bad)}. Valid: {sorted(VALID_PHASES)}."
        )
    if "A" not in requested:
        raise argparse.ArgumentTypeError(
            "Phase A is required (it loads the model and runs per-modality "
            "reconstruction); other phases build on it."
        )
    return requested


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained nano4M masking variant on CLEVR val.",
    )
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to the trained checkpoint (safetensors).")
    p.add_argument("--config", type=Path, required=True,
                   help="Path to the training YAML used to produce the checkpoint.")
    p.add_argument("--num-samples", type=int, default=500,
                   help="Number of CLEVR val samples to evaluate on (proposal default: 500).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for generation (proposal uses 3 seeds per config).")
    p.add_argument("--sample-steps", type=int, default=8,
                   help="Number of ROAR unmasking steps per target modality.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (proposal: 1.0, no CFG).")
    p.add_argument("--device", type=str, default="cuda",
                   help="Torch device to evaluate on.")
    p.add_argument("--output", type=Path, default=None,
                   help="If given, write JSON results to this file (otherwise stdout).")
    p.add_argument("--phases", type=parse_phases, default="A",
                   help="Comma-separated phases to run (e.g. A or A,B,C,D). "
                        "Phase A always runs (per-modality reconstruction).")
    p.add_argument("--llm-judge-model", type=str, default="Qwen/Qwen3-8B",
                   choices=["Qwen/Qwen3-8B", "Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B"],
                   help="Qwen model name for the LLM judge (Phase C).")
    p.add_argument("--llm-judge-4bit", action="store_true",
                   help="Load Qwen in 4-bit quantisation (requires bitsandbytes).")
    p.add_argument("--detector-model", type=str,
                   default="IDEA-Research/grounding-dino-tiny",
                   help="HuggingFace model id for the object-detection verifier (Phase D).")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="If given, cache Phase B generated outputs (captions + RGB) "
                        "for inspection in this directory.")
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

    llm_judge = None
    if "C" in args.phases:
        from nanofm.evaluation.llm_judge import LLMJudge
        llm_judge = LLMJudge(
            model_name=args.llm_judge_model,
            device=args.device,
            load_in_4bit=args.llm_judge_4bit,
        )

    rgb_verifier = None
    if "D" in args.phases:
        from nanofm.evaluation.rgb_verifier import RGBVerifier
        rgb_verifier = RGBVerifier(
            model_name=args.detector_model,
            device=args.device,
        )

    results = harness.run(
        num_samples=args.num_samples,
        seed=args.seed,
        sample_steps=args.sample_steps,
        temperature=args.temperature,
        phases=frozenset(args.phases),
        cache_dir=args.cache_dir,
        llm_judge=llm_judge,
        rgb_verifier=rgb_verifier,
    )

    out_json = json.dumps(results.as_dict(), indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_json + "\n")
        print(f"Wrote results to {args.output}", file=sys.stderr)
    else:
        print(out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
