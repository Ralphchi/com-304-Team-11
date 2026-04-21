"""Match predicted scene objects to ground-truth by 2D position.

Uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment) to find
the assignment that minimises total position distance between predicted and
ground-truth objects. Any extras on either side count as false positives /
false negatives and are reported as unmatched.

The assumption (validated by inspecting CLEVR data) is that objects are
well-separated in (x, y) space, so position-based matching is unambiguous.
If this stops holding, we would extend the cost to include shape/color/material
mismatches as tiebreakers.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .scene_parser import SceneObject


@dataclass
class MatchResult:
    """Result of matching predicted to ground-truth objects.

    Attributes
    ----------
    matches : list of (pred_idx, gt_idx)
        Pairs of indices aligning one predicted object to one GT object.
    unmatched_pred : list of int
        Indices in `predicted` that had no GT partner (false positives).
    unmatched_gt : list of int
        Indices in `ground_truth` that had no predicted partner (false negatives).
    """
    matches: List[Tuple[int, int]]
    unmatched_pred: List[int]
    unmatched_gt: List[int]


DEFAULT_POSITION_THRESHOLD: float = 5.0
"""Default Euclidean-distance cap for accepting a Hungarian pairing.

Chosen on 2026-04-21 from the 30-sample val fixture
(nano4M/tests/fixtures/scene_desc_samples.txt). Across several multi-object
scenes the minimum observed inter-object distance was ~7-10 pixels, so 5.0
sits below that floor — a predicted object more than 5 px from any GT is
almost certainly a wrong pairing and should count as a false positive rather
than be paired with the nearest (wrong) GT. Pass `position_threshold=None`
to accept every Hungarian assignment regardless of distance."""


def match_objects(
    predicted: List[SceneObject],
    ground_truth: List[SceneObject],
    position_threshold: Optional[float] = DEFAULT_POSITION_THRESHOLD,
) -> MatchResult:
    """Hungarian-match predicted objects to GT objects by 2D position.

    Parameters
    ----------
    predicted, ground_truth : list of SceneObject
        Objects extracted from the predicted and ground-truth scene
        descriptions respectively.
    position_threshold : float, optional
        If provided, matches with position distance exceeding this threshold
        are treated as unmatched (both sides become false pos/neg).
        Default is `DEFAULT_POSITION_THRESHOLD` (5.0) — see its docstring.
        Pass None to accept every Hungarian assignment regardless of distance.

    Returns
    -------
    MatchResult

    Notes
    -----
    - Uses scipy.optimize.linear_sum_assignment. Cost = Euclidean distance on (x, y).
    - When len(predicted) != len(ground_truth), extras are returned as unmatched.
    """
    # Deferred import so the module can be imported without scipy present
    # (useful for quick tests of scene_parser alone).
    from scipy.optimize import linear_sum_assignment  # type: ignore

    n_pred = len(predicted)
    n_gt = len(ground_truth)

    if n_pred == 0 and n_gt == 0:
        return MatchResult(matches=[], unmatched_pred=[], unmatched_gt=[])
    if n_pred == 0:
        return MatchResult(matches=[], unmatched_pred=[], unmatched_gt=list(range(n_gt)))
    if n_gt == 0:
        return MatchResult(matches=[], unmatched_pred=list(range(n_pred)), unmatched_gt=[])

    # Cost matrix: [n_pred, n_gt], entry (i, j) = L2 distance between pred i and gt j.
    cost = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i, p in enumerate(predicted):
        for j, g in enumerate(ground_truth):
            cost[i, j] = np.hypot(p.x - g.x, p.y - g.y)

    row_ind, col_ind = linear_sum_assignment(cost)

    matches: List[Tuple[int, int]] = []
    matched_pred = set()
    matched_gt = set()
    for i, j in zip(row_ind, col_ind):
        if position_threshold is not None and cost[i, j] > position_threshold:
            continue
        matches.append((int(i), int(j)))
        matched_pred.add(int(i))
        matched_gt.add(int(j))

    unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_gt = [j for j in range(n_gt) if j not in matched_gt]

    return MatchResult(
        matches=matches,
        unmatched_pred=unmatched_pred,
        unmatched_gt=unmatched_gt,
    )
