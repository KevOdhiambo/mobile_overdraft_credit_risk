"""Business cost-ratio threshold selection for binary classifiers.

The Bayes-optimal decline rule for a lending model is to decline when
P(default | score) >= C_fp / (C_fp + C_fn), where C_fp is the cost of a
false positive (declining a creditworthy borrower) and C_fn is the cost of
a false negative (approving a borrower who defaults).

F1 maximisation implicitly assumes equal cost between FP and FN — an
assumption that is almost never true in lending: missing a defaulter typically
costs an order of magnitude more than declining a creditworthy borrower.

Usage::

    from src.validation.cost_threshold import CostConfig, min_cost_threshold

    cost_cfg = CostConfig(cost_fp=0.20, cost_fn=1.0)
    threshold, expected_cost = min_cost_threshold(y_val, val_proba_cal, cost_cfg)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import precision_recall_curve


@dataclass(frozen=True)
class CostConfig:
    """Cost parameters for the decline / approve decision.

    Attributes:
        cost_fp: Cost of a false positive — declining a creditworthy borrower.
            Typically the foregone margin over the borrower's expected lifetime
            draw history. At ~1.5% margin per draw, cost_fp ≈ 0.20 represents
            13 repeat draws of foregone interest.
        cost_fn: Cost of a false negative — approving a borrower who defaults.
            Typically loss-given-default (LGD). At zero recovery, cost_fn = 1.0
            (full principal lost).
        description: Optional human-readable note on the cost interpretation.
    """

    cost_fp: float
    cost_fn: float
    description: str = ""


def bayes_optimal_threshold(cost_config: CostConfig) -> float:
    """Compute the Bayes-optimal decline threshold.

    Under a well-calibrated model, this is the probability above which the
    expected cost of approving exceeds the expected cost of declining:
    threshold* = C_fp / (C_fp + C_fn).

    Args:
        cost_config: Cost parameters.

    Returns:
        Threshold in (0, 1).
    """
    return cost_config.cost_fp / (cost_config.cost_fp + cost_config.cost_fn)


def min_cost_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    cost_config: CostConfig,
) -> tuple[float, float]:
    """Empirically find the threshold that minimises total expected cost.

    Searches the precision-recall curve rather than a coarse grid — this gives
    the exact threshold at which the cost function changes direction.

    Expected cost = cost_fp * n_FP + cost_fn * n_FN.

    The Bayes-optimal threshold and this empirical result agree closely when
    the model is well-calibrated. A large divergence signals miscalibration.

    Args:
        y_true: Ground-truth binary labels.
        proba: Calibrated predicted probabilities.
        cost_config: Cost parameters.

    Returns:
        (optimal_threshold, minimum_expected_cost).
    """
    y_arr = np.asarray(y_true, dtype=int)
    prob_arr = np.asarray(proba, dtype=float)
    n_pos = int(y_arr.sum())

    precisions, recalls, thresholds_pr = precision_recall_curve(y_arr, prob_arr)

    # precisions and recalls have one extra trailing point (the sentinel at
    # recall=0, precision=1). Thresholds has len(precisions) - 1 entries.
    prec = precisions[:-1]
    rec = recalls[:-1]

    tp = rec * n_pos
    # FP = TP / precision - TP, but guard against precision == 0
    n_neg = len(y_arr) - n_pos
    fp = np.where(prec > 0, tp * (1.0 / np.maximum(prec, 1e-9) - 1.0), float(n_neg))
    fn = n_pos * (1.0 - rec)

    total_cost = cost_config.cost_fp * fp + cost_config.cost_fn * fn
    best_idx = int(np.argmin(total_cost))

    return float(thresholds_pr[best_idx]), float(total_cost[best_idx])


def cost_threshold_report(
    y_true: np.ndarray,
    proba: np.ndarray,
    cost_config: CostConfig,
) -> dict[str, object]:
    """Produce a summary dict combining both threshold estimates.

    Args:
        y_true: Ground-truth binary labels.
        proba: Calibrated predicted probabilities.
        cost_config: Cost parameters.

    Returns:
        Dict with bayes_optimal_threshold, empirical_threshold, flag_rate at
        empirical threshold, and the expected cost at empirical threshold.
    """
    bayes_t = bayes_optimal_threshold(cost_config)
    emp_t, emp_cost = min_cost_threshold(y_true, proba, cost_config)
    prob_arr = np.asarray(proba, dtype=float)
    flag_rate = float((prob_arr >= emp_t).mean())
    return {
        "cost_fp": cost_config.cost_fp,
        "cost_fn": cost_config.cost_fn,
        "description": cost_config.description,
        "bayes_optimal_threshold": round(bayes_t, 4),
        "empirical_threshold": round(emp_t, 4),
        "flag_rate_at_empirical": round(flag_rate, 4),
        "expected_cost_at_empirical": round(emp_cost, 2),
    }
