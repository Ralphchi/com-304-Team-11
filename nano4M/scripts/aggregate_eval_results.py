#!/usr/bin/env python
"""Aggregate per-(variant, seed) eval JSONs into the final report tables.

Loads `results/<variant>_seed<s>.json` files written by `run_evaluation.py`,
reshapes into the (variant × metric × seed) array, and emits:

    results/aggregate_table.csv         mean ± std per (variant, metric)
    results/significance_table.csv      FDR-corrected q-values (variant vs baseline)
    results/comparison_summary.md       human-readable summary

Wilcoxon is paired across the same 500 val samples + same 3 generation seeds
across every variant — `_iter_val` in EvalHarness pins the val ordering, and
the team submits `--seed {0,1,2}` for each variant.

Usage:
    python scripts/aggregate_eval_results.py \
        --results-dir results/ \
        --baseline-variant baseline \
        --output-dir results/
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from nanofm.evaluation.statistical_tests import paired_wilcoxon_with_bh_fdr


METRIC_KEYS = [
    "depth/absrel", "depth/rmse", "depth/delta1",
    "normals/angular_error_deg",
    "rgb/ssim", "rgb/fid",
    "scene_desc/position", "scene_desc/shape", "scene_desc/color",
    "scene_desc/material", "scene_desc/set_match", "scene_desc/exact_sequence",
    "scene_desc/parse_rate",
    "cross/rgb_to_text/position", "cross/rgb_to_text/shape",
    "cross/rgb_to_text/color", "cross/rgb_to_text/material",
    "cross/rgb_to_text/set_match", "cross/rgb_to_text/exact_sequence",
    "cross/rgb_to_text/parse_rate",
    "cross/rgb_to_text/llm_alignment", "cross/rgb_to_text/llm_perfect_rate",
    "cross/rgb_to_text/llm_parse_error_rate",
    "cross/rgb_to_text/parser_judge_corr",
    "cross/text_to_rgb/ssim", "cross/text_to_rgb/fid",
    "cross/text_to_rgb/obj_precision", "cross/text_to_rgb/obj_recall",
    "cross/text_to_rgb/obj_f1", "cross/text_to_rgb/obj_perfect_rate",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Directory with <variant>_seed<n>.json files.")
    p.add_argument("--baseline-variant", type=str, default="baseline",
                   help="Variant to compare each other variant against.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write the output tables (default: results-dir).")
    return p.parse_args()


def discover(results_dir: Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Return {variant: {seed: metrics_dict}}."""
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for path in sorted(results_dir.glob("*_seed*.json")):
        stem = path.stem  # e.g. "block_seed0"
        try:
            variant, seed_part = stem.rsplit("_seed", 1)
            seed = int(seed_part)
        except ValueError:
            print(f"skipping unrecognised file: {path}", file=sys.stderr)
            continue
        out.setdefault(variant, {})[seed] = json.loads(path.read_text())
    return out


def _mean(xs: List[float]) -> float:
    valid = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(float(x))]
    return sum(valid) / len(valid) if valid else float("nan")


def _std(xs: List[float]) -> float:
    valid = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(float(x))]
    if len(valid) < 2:
        return float("nan")
    m = sum(valid) / len(valid)
    return math.sqrt(sum((x - m) ** 2 for x in valid) / (len(valid) - 1))


def write_aggregate_table(
    data: Dict[str, Dict[int, Dict[str, float]]],
    out_path: Path,
) -> None:
    variants = sorted(data.keys())
    rows = ["variant," + ",".join(f"{m}_mean,{m}_std" for m in METRIC_KEYS)]
    for v in variants:
        seeds = sorted(data[v].keys())
        cells: List[str] = [v]
        for m in METRIC_KEYS:
            xs = [float(data[v][s].get(m, float("nan"))) for s in seeds]
            cells.append(f"{_mean(xs):.6f}")
            cells.append(f"{_std(xs):.6f}")
        rows.append(",".join(cells))
    out_path.write_text("\n".join(rows) + "\n")


def run_significance(
    data: Dict[str, Dict[int, Dict[str, float]]],
    baseline: str,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Per (variant ≠ baseline, metric), paired Wilcoxon + BH-FDR.

    Pairs across (seed, sample_idx) — but since we only have aggregate
    metrics per seed, we pair across seeds only. For per-sample pairing the
    eval harness would need to dump per-sample arrays; that's a future
    extension once we have real checkpoints.
    """
    if baseline not in data:
        print(f"baseline variant '{baseline}' not found", file=sys.stderr)
        return {}

    baseline_seeds = sorted(data[baseline].keys())
    other_variants = [v for v in data if v != baseline]

    # Build the comparisons dict: one entry per (variant, metric).
    comparisons: Dict[str, Tuple[List[float], List[float]]] = {}
    for v in other_variants:
        seeds = sorted(data[v].keys())
        common = sorted(set(baseline_seeds) & set(seeds))
        if len(common) < 2:
            print(
                f"skipping {v}: only {len(common)} common seeds with baseline",
                file=sys.stderr,
            )
            continue
        for m in METRIC_KEYS:
            a = [float(data[v][s].get(m, float("nan"))) for s in common]
            b = [float(data[baseline][s].get(m, float("nan"))) for s in common]
            comparisons[f"{v}|{m}"] = (a, b)

    fdr_results = paired_wilcoxon_with_bh_fdr(comparisons)

    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for label, res in fdr_results.items():
        v, m = label.split("|", 1)
        out[(v, m)] = {
            "p": res.raw_pvalue,
            "q": res.q_value,
            "stat": res.statistic,
            "significant": float(res.significant),
        }
    return out


def write_significance_table(
    sig: Dict[Tuple[str, str], Dict[str, float]], out_path: Path
) -> None:
    variants = sorted({v for v, _ in sig.keys()})
    rows = ["variant," + ",".join(f"{m}_q,{m}_sig" for m in METRIC_KEYS)]
    for v in variants:
        cells: List[str] = [v]
        for m in METRIC_KEYS:
            entry = sig.get((v, m))
            if entry is None:
                cells.extend(["nan", "0"])
            else:
                cells.append(f"{entry['q']:.6f}")
                cells.append(str(int(entry["significant"])))
        rows.append(",".join(cells))
    out_path.write_text("\n".join(rows) + "\n")


def write_summary_md(
    data: Dict[str, Dict[int, Dict[str, float]]],
    sig: Dict[Tuple[str, str], Dict[str, float]],
    baseline: str,
    out_path: Path,
) -> None:
    lines = [f"# Variant-vs-baseline (`{baseline}`) summary", ""]
    headline = [
        "depth/absrel", "normals/angular_error_deg", "rgb/ssim",
        "scene_desc/position",
        "cross/rgb_to_text/llm_alignment", "cross/text_to_rgb/obj_f1",
    ]
    lines.append("Headline metrics (q < 0.05 marked with `*`):\n")
    lines.append("| variant | " + " | ".join(headline) + " |")
    lines.append("|---" * (len(headline) + 1) + "|")
    for v in sorted(data.keys()):
        if v == baseline:
            continue
        cells = [v]
        seeds = sorted(data[v].keys())
        for m in headline:
            xs = [float(data[v][s].get(m, float("nan"))) for s in seeds]
            mean = _mean(xs)
            entry = sig.get((v, m))
            star = "*" if entry and entry["significant"] >= 1.0 else " "
            cells.append(f"{mean:.4f}{star}")
        lines.append("| " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = discover(args.results_dir)
    if not data:
        print(f"no <variant>_seed<n>.json files in {args.results_dir}", file=sys.stderr)
        return 1

    write_aggregate_table(data, out_dir / "aggregate_table.csv")
    sig = run_significance(data, args.baseline_variant)
    write_significance_table(sig, out_dir / "significance_table.csv")
    write_summary_md(data, sig, args.baseline_variant, out_dir / "comparison_summary.md")

    print(f"wrote aggregate_table.csv, significance_table.csv, comparison_summary.md to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
