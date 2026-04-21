"""Unit tests for paired Wilcoxon + Benjamini-Hochberg FDR correction.

The revised extension plan (Section IV) requires BH-FDR correction across
4 variants x 4 modalities x ~5 metrics = ~80 comparisons when running
statistical testing in Week 3. The primitives must be correct and the
pipeline stable before that week.

Run:
    cd nano4M && python -m pytest tests/test_statistical_tests.py -v
"""
import math

import pytest

from nanofm.evaluation.statistical_tests import (
    paired_wilcoxon,
    benjamini_hochberg,
    paired_wilcoxon_with_bh_fdr,
    StatResult,
)


# --------------------------------------------------- paired_wilcoxon

def test_wilcoxon_shifted_samples_produce_small_pvalue():
    """Two sequences where every variant value is smaller than the baseline
    must give a small two-sided p-value (we'd reject H0 at any reasonable alpha).
    """
    pytest.importorskip("scipy")
    variant = [0.10, 0.11, 0.09, 0.12, 0.10, 0.08, 0.11, 0.09, 0.10, 0.11]
    baseline = [0.30, 0.31, 0.29, 0.32, 0.30, 0.28, 0.31, 0.29, 0.30, 0.31]
    _stat, p = paired_wilcoxon(variant, baseline)
    assert p < 0.01, f"expected p < 0.01, got {p}"


def test_wilcoxon_rejects_length_mismatch():
    pytest.importorskip("scipy")
    with pytest.raises(ValueError):
        paired_wilcoxon([1.0, 2.0], [1.0, 2.0, 3.0])


def test_wilcoxon_empty_input_returns_nan():
    pytest.importorskip("scipy")
    stat, p = paired_wilcoxon([], [])
    assert math.isnan(stat) and math.isnan(p)


# --------------------------------------------------- benjamini_hochberg

def test_bh_all_nonsignificant_stay_nonsignificant():
    p = [0.20, 0.30, 0.40, 0.50, 0.60]
    q, sig = benjamini_hochberg(p, alpha=0.05)
    assert all(qv > 0.05 for qv in q)
    assert not any(sig)


def test_bh_single_clearly_significant_stays_significant():
    # m=20, one p-value 1e-6 and 19 non-significant p-values.
    # BH: q_(1) = 20 * 1e-6 / 1 = 2e-5, well below 0.05.
    p = [1e-6] + [0.5] * 19
    q, sig = benjamini_hochberg(p, alpha=0.05)
    assert sig[0], f"expected p=1e-6 to stay significant, got q={q[0]}"
    assert not any(sig[1:]), "expected all p=0.5 to be non-significant"


def test_bh_monotone_qvalues():
    # q-values should be monotone non-decreasing in the order of ascending p.
    p = [0.001, 0.02, 0.03, 0.5, 0.8]
    q, _sig = benjamini_hochberg(p, alpha=0.05)
    # Sort by original p and check monotonicity of q.
    pairs = sorted(zip(p, q))
    q_sorted = [qv for _, qv in pairs]
    for i in range(1, len(q_sorted)):
        assert q_sorted[i] >= q_sorted[i - 1], f"q-values not monotone at index {i}: {q_sorted}"


def test_bh_reproduces_textbook_example():
    """Standard BH example: p = [0.005, 0.009, 0.019, 0.022, 0.051].
    With m=5, alpha=0.05: reject first 4 (critical line 0.01, 0.02, 0.03, 0.04, 0.05)."""
    p = [0.005, 0.009, 0.019, 0.022, 0.051]
    q, sig = benjamini_hochberg(p, alpha=0.05)
    assert sig[:4] == [True, True, True, True], f"expected first 4 significant, got {sig}"
    assert sig[4] is False, f"expected last non-significant, got {sig}"


def test_bh_nan_inputs_pass_through_as_non_significant():
    p = [float("nan"), 0.001, float("nan"), 0.5]
    q, sig = benjamini_hochberg(p, alpha=0.05)
    assert math.isnan(q[0]) and math.isnan(q[2])
    assert sig[0] is False and sig[2] is False
    # 0.001 should be significant (only 2 real p-values, tiny one should survive BH).
    assert sig[1] is True
    assert sig[3] is False


def test_bh_empty_returns_empty():
    q, sig = benjamini_hochberg([], alpha=0.05)
    assert q == [] and sig == []


# -------------------------------- paired_wilcoxon_with_bh_fdr (end-to-end)

def test_pipeline_labels_preserved_and_results_complete():
    pytest.importorskip("scipy")
    # Two comparisons: one where variant is clearly better, one where variant
    # and baseline are indistinguishable.
    v1 = [0.1, 0.11, 0.09, 0.12, 0.10, 0.08, 0.11, 0.09, 0.10, 0.11]
    b1 = [0.3, 0.31, 0.29, 0.32, 0.30, 0.28, 0.31, 0.29, 0.30, 0.31]
    v2 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    b2 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    results = paired_wilcoxon_with_bh_fdr(
        comparisons={
            "variant_clearly_better": (v1, b1),
            "variant_same_as_baseline": (v2, b2),
        },
        alpha=0.05,
    )
    assert set(results.keys()) == {"variant_clearly_better", "variant_same_as_baseline"}
    clear = results["variant_clearly_better"]
    same = results["variant_same_as_baseline"]
    assert isinstance(clear, StatResult)
    assert clear.significant is True
    assert same.significant is False
    assert 0.0 <= clear.q_value <= 1.0
