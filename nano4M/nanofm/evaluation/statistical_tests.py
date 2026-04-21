"""Paired Wilcoxon + Benjamini-Hochberg FDR correction for the Team 11 extension.

The revised extension plan (Section IV, Validation) commits to:

    "Paired Wilcoxon signed-rank tests with Benjamini-Hochberg FDR correction
    across all variant-vs-baseline comparisons (4 variants x 4 modalities
    x ~5 metrics)."

This module exposes two primitives:

- `paired_wilcoxon(variant_values, baseline_values)` -> (statistic, raw_pvalue)
- `benjamini_hochberg(pvalues, alpha)` -> list of booleans (True = significant)

Plus a convenience wrapper `paired_wilcoxon_with_bh_fdr(comparisons, alpha)`
that applies the test to a dict of named comparisons and returns the
FDR-corrected q-values and significance flags in one call.

Usage (Week 3):
    from nanofm.evaluation.statistical_tests import paired_wilcoxon_with_bh_fdr

    # Per (variant, modality, metric), you have two lists of length N (e.g.
    # 500 paired per-sample metric values for variant_i and baseline_0).
    comparisons = {
        "block/depth/absrel":  (variant_absrel_per_sample, baseline_absrel_per_sample),
        "block/depth/delta1":  (variant_delta1_per_sample, baseline_delta1_per_sample),
        # ... 4 variants * 4 modalities * ~5 metrics = ~80 entries
    }
    results = paired_wilcoxon_with_bh_fdr(comparisons, alpha=0.05)
    # results["block/depth/absrel"] -> StatResult(statistic, raw_p, q_value, significant)

Why per-sample rather than per-seed pairing? With 500 held-out samples and 3
generation seeds per config, per-sample pairing (variant mean vs baseline mean
at each sample) gives 500 paired observations per test. Per-seed pairing would
give only 3, which is too few for Wilcoxon. Per-sample is the standard choice
when the samples are the same held-out items across variants (the proposal
explicitly fixes the 500-sample set "shared across variants").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, List


@dataclass(frozen=True)
class StatResult:
    """Outcome of a single paired Wilcoxon comparison after BH-FDR."""
    statistic: float       # Wilcoxon signed-rank statistic
    raw_pvalue: float      # p-value before multiple-testing correction
    q_value: float         # BH-corrected p-value across the whole comparison set
    significant: bool      # q_value < alpha


def paired_wilcoxon(
    variant_values: Sequence[float],
    baseline_values: Sequence[float],
) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank test between two same-length sequences.

    Thin wrapper around scipy.stats.wilcoxon so we can swap implementations
    later without touching the rest of the pipeline.

    Parameters
    ----------
    variant_values, baseline_values : same-length sequences of floats.
        Paired observations (e.g. per-sample metric values on the same 500
        held-out items for variant and for baseline).

    Returns
    -------
    (statistic, raw_pvalue)
        Two-sided p-value. Zero-differences are handled by scipy's default
        ("wilcox" mode in recent scipy versions).
    """
    # Deferred import so this module stays importable without scipy present.
    from scipy.stats import wilcoxon  # type: ignore

    if len(variant_values) != len(baseline_values):
        raise ValueError(
            f"paired_wilcoxon requires equal-length inputs, got "
            f"{len(variant_values)} vs {len(baseline_values)}"
        )
    if len(variant_values) == 0:
        return float("nan"), float("nan")
    result = wilcoxon(variant_values, baseline_values)
    return float(result.statistic), float(result.pvalue)


def benjamini_hochberg(
    pvalues: Sequence[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues : sequence of floats in [0, 1]
    alpha : desired FDR (default 0.05)

    Returns
    -------
    q_values : list of BH-corrected p-values (monotone, clipped to [0, 1])
    significant : list of booleans (q_value <= alpha)

    Algorithm
    ---------
    1. Sort p-values ascending: p_(1) <= ... <= p_(m).
    2. q_(i) = min_{j >= i} (m * p_(j) / j), then clip to 1.
    3. Rejection: p_(i) <= (i/m) * alpha, equivalent to q_(i) <= alpha.

    NaN p-values pass through unchanged, are never flagged significant, and
    do not affect the ranks of the remaining comparisons.
    """
    import math

    n = len(pvalues)
    if n == 0:
        return [], []

    # Separate NaNs from real p-values; FDR is computed only over the latter.
    real_indices = [i for i, p in enumerate(pvalues) if not math.isnan(p)]
    real_p = [pvalues[i] for i in real_indices]
    m = len(real_p)

    q_by_orig: List[float] = [float("nan")] * n

    if m > 0:
        # Sort ascending, keep track of original index into real_p.
        order = sorted(range(m), key=lambda i: real_p[i])
        sorted_p = [real_p[i] for i in order]

        # Raw BH formula: q_(i) = m * p_(i) / (i+1)  (1-indexed i+1 == rank).
        raw_q = [min(1.0, (m * sorted_p[i]) / (i + 1)) for i in range(m)]
        # Enforce monotonicity from the top: q_(i) = min(q_(i), q_(i+1)).
        for i in range(m - 2, -1, -1):
            if raw_q[i] > raw_q[i + 1]:
                raw_q[i] = raw_q[i + 1]

        # Map sorted q-values back to the original positions.
        for sort_rank, orig_idx_in_real in enumerate(order):
            orig_idx_in_all = real_indices[orig_idx_in_real]
            q_by_orig[orig_idx_in_all] = raw_q[sort_rank]

    significant = [
        (not math.isnan(q)) and (q <= alpha) for q in q_by_orig
    ]
    return q_by_orig, significant


def paired_wilcoxon_with_bh_fdr(
    comparisons: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    alpha: float = 0.05,
) -> Dict[str, StatResult]:
    """Run paired Wilcoxon per comparison, then BH-FDR correct across all.

    Parameters
    ----------
    comparisons : dict of name -> (variant_values, baseline_values) pairs.
        Each pair is two equal-length sequences of paired observations.
    alpha : desired FDR across all comparisons (default 0.05).

    Returns
    -------
    Dict[name, StatResult]
        Preserves the insertion order of `comparisons`.
    """
    names = list(comparisons)
    stats: List[float] = []
    raws: List[float] = []
    for name in names:
        v, b = comparisons[name]
        stat, raw = paired_wilcoxon(v, b)
        stats.append(stat)
        raws.append(raw)

    q_values, significant = benjamini_hochberg(raws, alpha=alpha)

    return {
        name: StatResult(
            statistic=stats[i],
            raw_pvalue=raws[i],
            q_value=q_values[i],
            significant=significant[i],
        )
        for i, name in enumerate(names)
    }
