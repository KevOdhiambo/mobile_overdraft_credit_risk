"""Train the overdraft default-risk (PD) model.

Produces:
  models/model.txt              LightGBM booster
  models/calibrator.pkl         Isotonic probability calibrator (val-set fit)
  models/metrics.json           AUC, PR-AUC, calibration, train/val gap, threshold metrics
  models/fairness_report.json   Disparate impact + FPR/FNR audit across proxy groups
  models/feature_importance.png
  models/sample_reason_codes.json

Run:
    python -m src.models.train --features data/processed/features.csv
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.features.build_features import FEATURE_COLUMNS, GROUP_ID_COL, TARGET_COL
from src.validation import FairnessConfig, ModelValidationConfig, validate_model
from src.validation.cost_threshold import (
    CostConfig,
    bayes_optimal_threshold,
    min_cost_threshold,
)

logger = logging.getLogger(__name__)

SHAP_SAMPLE_SIZE = 200

SplitResult: TypeAlias = tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
]


def _load_episode_metadata(features_path: Path) -> pd.DataFrame | None:
    """Load episode metadata (cust_id, draw_date) from the parallel episode_ids.csv.

    The training pipeline uses draw_date for a time-based OOT split (preferred)
    and cust_id as a fallback for group-aware splitting when dates are absent.
    Returns None so callers can fall back to stratified split without crashing.

    Args:
        features_path: Path to features.csv; episode_ids.csv is expected in the
            same directory.
    """
    ids_path = features_path.parent / "episode_ids.csv"
    if not ids_path.exists():
        logger.warning(
            "episode_ids.csv not found at %s -- falling back to stratified split",
            ids_path,
        )
        return None
    df = pd.read_csv(ids_path)
    if "draw_date" in df.columns:
        df["draw_date"] = pd.to_datetime(df["draw_date"])
    logger.info("Loaded episode metadata: %d rows, columns: %s", len(df), df.columns.tolist())
    return df


def _split_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    draw_dates: pd.Series,
    config: "TrainingConfig",
) -> SplitResult:
    """Split by wall-clock time: train < val_cutoff <= val < oot_cutoff <= OOT test.

    This is the only split that tests temporal generalisation — the property
    that actually matters for deployment. GroupShuffleSplit prevents borrower
    memorisation but says nothing about whether the model generalises across
    time periods. Random or group-based splits on a single time window
    validate "does this generalise to other borrowers in the same period,"
    not "does this generalise forward," which is what a deployed model does.

    Args:
        X: Feature matrix, RangeIndex aligned with draw_dates.
        y: Target labels, same index as X.
        draw_dates: Episode draw dates, same index as X.
        config: Provides val_cutoff and oot_cutoff date strings.
    """
    val_dt = pd.Timestamp(config.val_cutoff)
    oot_dt = pd.Timestamp(config.oot_cutoff)

    train_mask = draw_dates < val_dt
    val_mask = (draw_dates >= val_dt) & (draw_dates < oot_dt)
    test_mask = draw_dates >= oot_dt

    return (
        X[train_mask], X[val_mask], X[test_mask],
        y[train_mask], y[val_mask], y[test_mask],
    )


def _split_data(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame | None,
    config: "TrainingConfig",
) -> tuple[SplitResult, str]:
    """Dispatch to the appropriate split strategy and return (splits, mode_label).

    Priority:
    1. Temporal OOT split when draw_date is available — tests forward-in-time
       generalisation, which is what deployment actually requires.
    2. Group-aware random split (GroupShuffleSplit) when only cust_id is
       available — prevents borrower memorisation but not period-level leakage.
    3. Stratified random split — last resort; logs a warning.

    Args:
        X: Feature matrix.
        y: Target labels.
        metadata: DataFrame from episode_ids.csv (cust_id, draw_date), or None.
        config: TrainingConfig with split parameters.

    Returns:
        ((X_train, X_val, X_test, y_train, y_val, y_test), split_mode_label)
    """
    if metadata is not None and "draw_date" in metadata.columns:
        return _split_temporal(X, y, metadata["draw_date"], config), "temporal-OOT"

    if metadata is not None and GROUP_ID_COL in metadata.columns:
        groups = metadata[GROUP_ID_COL]
        temp_fraction = config.val_fraction + config.test_fraction
        relative_test = config.test_fraction / temp_fraction

        gss_outer = GroupShuffleSplit(
            n_splits=1, test_size=temp_fraction, random_state=config.random_state
        )
        train_idx, temp_idx = next(gss_outer.split(X, y, groups=groups.values))

        gss_inner = GroupShuffleSplit(
            n_splits=1, test_size=relative_test, random_state=config.random_state
        )
        temp_groups = groups.iloc[temp_idx]
        val_local, test_local = next(
            gss_inner.split(X.iloc[temp_idx], y.iloc[temp_idx], groups=temp_groups.values)
        )
        val_idx = temp_idx[val_local]
        test_idx = temp_idx[test_local]

        splits = (
            X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx],
        )
        return splits, "group-aware-random"

    logger.warning(
        "No episode metadata available -- using stratified random split; "
        "the same borrower may appear in both train and test, and temporal "
        "generalisation is untested"
    )
    temp_fraction = config.val_fraction + config.test_fraction
    relative_test = config.test_fraction / temp_fraction
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_fraction, stratify=y, random_state=config.random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, stratify=y_temp, random_state=config.random_state
    )
    return (X_train, X_val, X_test, y_train, y_val, y_test), "stratified-random"


@dataclass(frozen=True)
class LGBMParams:
    """LightGBM hyperparameters."""

    objective: str = "binary"
    learning_rate: float = 0.03
    num_leaves: int = 63
    min_data_in_leaf: int = 30
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    lambda_l1: float = 0.1
    lambda_l2: float = 0.1
    verbose: int = -1
    seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    """Full configuration for a single training run.

    threshold is a business policy parameter, not a model quality metric. It
    controls where you operate on the precision-recall curve:

    - Low (0.20-0.35): lender-conservative — maximises recall at the cost of
      a high false-positive rate. Encodes the assumption that the cost of a
      bad loan significantly exceeds the cost of declining a good borrower.
    - High (0.70-0.95): borrower-friendly — maximises precision at the cost
      of missing some defaults. Appropriate when margins are wide and customer
      acquisition is expensive.

    evaluate() reports metrics at both this threshold and a deployment_threshold
    derived from the validation set (val-set min expected cost under C_fp/C_fn).
    The API uses the deployment_threshold; this value is preserved for reporting.

    val_cutoff / oot_cutoff define the temporal split boundaries. When
    episode_ids.csv contains draw_date, the pipeline splits by time rather than
    randomly — this is mandatory for any model that will be deployed forward.
    """

    features_path: Path = Path("data/processed/features.csv")
    model_dir: Path = Path("models")
    threshold: float = 0.3
    val_cutoff: str = "2023-07-01"
    oot_cutoff: str = "2024-01-01"
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    random_state: int = 42
    # Business cost parameters for deployment threshold selection.
    # cost_fp = cost of declining a creditworthy borrower (foregone margin).
    # cost_fn = cost of approving a defaulting borrower (loss given default).
    # Bayes-optimal threshold = cost_fp / (cost_fp + cost_fn).
    # At cost_fp=0.20, cost_fn=1.0: threshold* ≈ 0.167 (operationally feasible).
    cost_fp: float = 0.20
    cost_fn: float = 1.0
    lgbm: LGBMParams = field(default_factory=LGBMParams)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: TrainingConfig,
) -> lgb.Booster:
    """Fit a LightGBM binary classifier with early stopping.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels (0/1).
        X_val: Validation feature matrix for early stopping.
        y_val: Validation labels.
        config: Full training configuration including LightGBM hyperparams.

    Returns:
        Trained LightGBM Booster at the best iteration.
    """
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    # No scale_pos_weight: the natural class distribution produces better-calibrated
    # raw scores. Post-hoc isotonic calibration corrects residual miscalibration
    # without distorting the score distribution into the extreme bimodal shape
    # that scale_pos_weight=neg/pos causes at a 17:1 imbalance ratio.
    params: dict[str, Any] = {
        "objective": config.lgbm.objective,
        "metric": ["auc", "average_precision"],
        "learning_rate": config.lgbm.learning_rate,
        "num_leaves": config.lgbm.num_leaves,
        "min_data_in_leaf": config.lgbm.min_data_in_leaf,
        "feature_fraction": config.lgbm.feature_fraction,
        "bagging_fraction": config.lgbm.bagging_fraction,
        "bagging_freq": config.lgbm.bagging_freq,
        "lambda_l1": config.lgbm.lambda_l1,
        "lambda_l2": config.lgbm.lambda_l2,
        "verbose": config.lgbm.verbose,
        "seed": config.lgbm.seed,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=config.num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds),
            lgb.log_evaluation(0),
        ],
    )
    return model


def _threshold_metrics(
    y_true: pd.Series,
    proba: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Confusion-matrix-derived metrics at a single decision threshold.

    Args:
        y_true: Ground-truth binary labels.
        proba: Predicted probabilities, same length as y_true.
        threshold: Decision boundary; proba >= threshold → positive prediction.

    Returns:
        Dict with threshold, precision, recall, f1, flag_rate, and confusion_matrix.
    """
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "threshold": float(threshold),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "flag_rate": round(float(preds.mean()), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _calibration_table(
    y_true: pd.Series,
    proba: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Reliability diagram data: mean predicted probability vs actual rate per score decile.

    The key check is whether calibrated scores track actual default rates across
    the full score range, not just on aggregate (which Brier obscures). A score
    of 0.40 should correspond to ~40% actual default rate; a score of 0.05 should
    correspond to ~5%. Any systematic divergence indicates residual miscalibration
    in a specific band.

    Args:
        y_true: Ground-truth binary labels.
        proba: Calibrated predicted probabilities, same length as y_true.
        n_bins: Number of equal-count bins (default: 10 deciles).

    Returns:
        List of dicts ordered by score bin, each with bin label, n, mean_predicted,
        and actual_default_rate.
    """
    cut_points = np.percentile(proba, np.linspace(0, 100, n_bins + 1))
    cut_points = np.unique(cut_points)
    if len(cut_points) < 2:
        return []
    labels = pd.cut(proba, bins=cut_points, include_lowest=True)
    df = pd.DataFrame({"proba": proba, "actual": y_true.values, "bin": labels})
    rows = []
    for interval, grp in df.groupby("bin", observed=True):
        rows.append(
            {
                "bin": str(interval),
                "n": int(len(grp)),
                "mean_predicted": round(float(grp["proba"].mean()), 4),
                "actual_default_rate": round(float(grp["actual"].mean()), 4),
            }
        )
    return rows


def evaluate(
    model: lgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    calibrator: IsotonicRegression | None = None,
    deployment_threshold: float | None = None,
) -> dict[str, Any]:
    """Compute evaluation metrics on the held-out test set.

    AUC and PR-AUC are threshold-free and computed purely on the test set —
    these are the only numbers here that carry no selection bias.

    Threshold-dependent metrics (precision, recall, F1, flag_rate) are reported
    at two operating points:
    - configured_threshold (0.30): business policy choice, reported for reference.
    - deployment_threshold: derived from the validation set before this call;
      the test set is never used for threshold selection.

    The test-set optimal threshold is intentionally NOT reported: picking a
    threshold by searching the test labels and then grading at that threshold
    on the same test labels is optimistic by construction — the same family of
    bug as calibration contamination, just smaller in magnitude.

    Args:
        model: Trained LightGBM booster.
        X_test: Test feature matrix.
        y_test: Test labels.
        threshold: Configured decision boundary (business policy).
        calibrator: Optional fitted IsotonicRegression.
        deployment_threshold: Max-F1 threshold from the val set.

    Returns:
        Dict with AUC, PR-AUC, Brier, and operating-point sub-dicts.
    """
    proba = model.predict(X_test, num_iteration=model.best_iteration)
    if calibrator is not None:
        proba = calibrator.transform(proba)

    configured = _threshold_metrics(y_test, proba, threshold)
    configured["rationale"] = (
        "lender-conservative: maximises recall at cost of precision; "
        "appropriate when loss_given_default >> cost_of_foregone_loan"
    )

    result: dict[str, Any] = {
        "auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "brier_score": round(float(brier_score_loss(y_test, proba)), 4),
        "default_rate_actual": round(float(y_test.mean()), 4),
        "configured_threshold": configured,
    }

    if deployment_threshold is not None:
        dep = _threshold_metrics(y_test, proba, deployment_threshold)
        dep["derivation"] = (
            "val-set min expected cost (C_fp/C_fn ratio); "
            "test set never consulted for threshold selection"
        )
        result["deployment_threshold"] = dep

    result["calibration_table"] = _calibration_table(y_test, proba)

    return result


def fairness_audit(
    model: lgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    calibrator: IsotonicRegression | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute disparate impact, FPR, and FNR across fairness-proxy groups.

    Uses tenure and CRB flag as proxies for thin-file / new-to-system borrowers.
    A real deployment must audit against actual protected-class data subject to
    local regulatory requirements — not these proxies alone.

    FPR (false positive rate) = wrong-denial rate among creditworthy borrowers.
    FNR (false negative rate) = missed-default rate among actual defaulters.
    Both matter: FPR inequality means biased denial; FNR inequality means
    unequal risk exposure to the lender across groups.

    Args:
        model: Trained LightGBM booster.
        X_test: Test feature matrix; must contain tenure_months and crb_flagged.
        y_test: Test labels.
        threshold: Decision boundary.
        calibrator: Optional fitted IsotonicRegression to apply before thresholding.

    Returns:
        Dict keyed by proxy group name with approval rates, FPR, FNR,
        disparate impact ratio, and whether the four-fifths rule is violated.
    """
    proba_raw = model.predict(X_test, num_iteration=model.best_iteration)
    proba = calibrator.transform(proba_raw) if calibrator is not None else proba_raw
    preds = (proba >= threshold).astype(int)

    df = X_test.copy()
    df["pred"] = preds
    df["actual"] = y_test.values

    report: dict[str, dict[str, Any]] = {}

    def _group_stats(mask: pd.Series) -> dict[str, Any]:
        g = df[mask]
        approve_rate = float((1 - g["pred"]).mean())
        tp = int(((g["pred"] == 1) & (g["actual"] == 1)).sum())
        fp = int(((g["pred"] == 1) & (g["actual"] == 0)).sum())
        tn = int(((g["pred"] == 0) & (g["actual"] == 0)).sum())
        fn = int(((g["pred"] == 0) & (g["actual"] == 1)).sum())
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)
        return {
            "n": int(len(g)),
            "n_true_negative": tn,
            "n_true_positive": tp,
            "approval_rate": round(approve_rate, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
        }

    thin = df["tenure_months"] < 12
    thin_stats = _group_stats(thin)
    estab_stats = _group_stats(~thin)
    di_tenure = thin_stats["approval_rate"] / max(estab_stats["approval_rate"], 1e-6)
    report["tenure_proxy"] = {
        "thin_file": thin_stats,
        "established": estab_stats,
        "disparate_impact_ratio": round(di_tenure, 4),
        "fpr_ratio": round(thin_stats["fpr"] / max(estab_stats["fpr"], 1e-6), 4),
        "fnr_ratio": round(thin_stats["fnr"] / max(estab_stats["fnr"], 1e-6), 4),
        "fails_four_fifths_rule": di_tenure < 0.8,
    }

    flagged = df["crb_flagged"] == 1
    flag_stats = _group_stats(flagged)
    clean_stats = _group_stats(~flagged)
    di_crb = flag_stats["approval_rate"] / max(clean_stats["approval_rate"], 1e-6)
    report["crb_flag_proxy"] = {
        "crb_flagged": flag_stats,
        "crb_clean": clean_stats,
        "disparate_impact_ratio": round(di_crb, 4),
        "fpr_ratio": round(flag_stats["fpr"] / max(clean_stats["fpr"], 1e-6), 4),
        "fnr_ratio": round(flag_stats["fnr"] / max(clean_stats["fnr"], 1e-6), 4),
        "fails_four_fifths_rule": di_crb < 0.8,
    }

    return report


def generate_reason_codes(
    model: lgb.Booster,
    X_sample: pd.DataFrame,
    threshold: float,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Generate SHAP-based adverse action reason codes for a sample of rows.

    This is what a regulated lender hands to a declined applicant: a specific,
    defensible explanation grounded in the actual model decision.

    Args:
        model: Trained LightGBM booster.
        X_sample: Feature rows to explain.
        threshold: Decision boundary; used to determine APPROVE/DECLINE per row.
        top_n: Number of top SHAP contributors to return per row.

    Returns:
        List of dicts, one per row, each containing probability_of_default,
        decision, and reason_codes (sorted by absolute SHAP value descending).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    proba = model.predict(X_sample, num_iteration=model.best_iteration)
    results: list[dict[str, Any]] = []

    for i in range(len(X_sample)):
        row_shap = shap_values[i]
        top_idx = np.argsort(-np.abs(row_shap))[:top_n]
        results.append(
            {
                "probability_of_default": round(float(proba[i]), 4),
                "decision": "DECLINE" if proba[i] >= threshold else "APPROVE",
                "reason_codes": [
                    {
                        "feature": X_sample.columns[j],
                        "shap_value": round(float(row_shap[j]), 4),
                        "direction": "increases risk" if row_shap[j] > 0 else "decreases risk",
                    }
                    for j in top_idx
                ],
            }
        )

    return results


def plot_feature_importance(model: lgb.Booster, out_path: Path) -> None:
    """Save a horizontal bar chart of top-15 features by gain importance.

    Args:
        model: Trained LightGBM booster.
        out_path: Destination path for the PNG file.
    """
    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=model.feature_name(),
    ).sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    importance.tail(15).plot(kind="barh")
    plt.title("Feature Importance (gain)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main() -> None:
    """CLI entry point: train, evaluate, audit, and persist all model artifacts."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(TrainingConfig().features_path))
    parser.add_argument("--threshold", type=float, default=TrainingConfig().threshold)
    args = parser.parse_args()

    config = TrainingConfig(
        features_path=Path(args.features),
        threshold=args.threshold,
    )

    config.model_dir.mkdir(exist_ok=True)

    df = pd.read_csv(config.features_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COL]

    metadata = _load_episode_metadata(config.features_path)
    (X_train, X_val, X_test, y_train, y_val, y_test), split_mode = _split_data(
        X, y, metadata, config
    )
    logger.info(
        "Split (%s) | train=%d val=%d OOT-test=%d",
        split_mode,
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # Report borrower overlap between train and OOT test — expected for repeat
    # borrowers; worth quantifying so reviewers understand what the OOT test
    # is and is not controlling for.
    if metadata is not None and GROUP_ID_COL in metadata.columns:
        train_borrowers = set(metadata.loc[X_train.index, GROUP_ID_COL])
        test_borrowers = set(metadata.loc[X_test.index, GROUP_ID_COL])
        overlap = len(train_borrowers & test_borrowers)
        logger.info(
            "Borrower overlap: %d / %d OOT borrowers also appear in training "
            "(%.1f%% -- expected for repeat borrowers, not a data error)",
            overlap,
            len(test_borrowers),
            overlap / max(len(test_borrowers), 1) * 100,
        )

    model = train_model(X_train, y_train, X_val, y_val, config)
    model.save_model(str(config.model_dir / "model.txt"))

    # Capture train/val AUC at the best early-stopping iteration.
    train_auc = model.best_score.get("train", {}).get("auc", float("nan"))
    val_auc = model.best_score.get("val", {}).get("auc", float("nan"))
    logger.info(
        "Best iteration %d | train_AUC=%.4f val_AUC=%.4f gap=%.4f",
        model.best_iteration,
        train_auc,
        val_auc,
        train_auc - val_auc,
    )

    # Isotonic calibration on the val set — corrects probability scale without
    # touching test data.
    val_proba_raw = model.predict(X_val, num_iteration=model.best_iteration)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_proba_raw, y_val)
    cal_path = config.model_dir / "calibrator.pkl"
    with open(cal_path, "wb") as fh:
        pickle.dump(calibrator, fh)
    logger.info("Isotonic calibrator fitted on val set (%d samples) -> %s", len(y_val), cal_path)

    # Deployment threshold: val-set minimum expected cost.
    # Bayes-optimal threshold = C_fp / (C_fp + C_fn); the empirical val-set
    # search finds the threshold that minimises C_fp * n_FP + C_fn * n_FN
    # directly on calibrated scores. Both should agree closely if calibration
    # is good; a large divergence signals residual miscalibration.
    # The test set is never consulted for any modeling decision.
    val_proba_cal = calibrator.transform(val_proba_raw)
    _cost_cfg = CostConfig(cost_fp=config.cost_fp, cost_fn=config.cost_fn)
    deployment_threshold, _deployment_cost = min_cost_threshold(
        y_val, val_proba_cal, _cost_cfg
    )
    _bayes_t = bayes_optimal_threshold(_cost_cfg)
    logger.info(
        "Deployment threshold (val min-cost C_fp=%.2f C_fn=%.2f): %.4f  "
        "Bayes-optimal: %.4f",
        config.cost_fp,
        config.cost_fn,
        deployment_threshold,
        _bayes_t,
    )

    metrics = evaluate(
        model,
        X_test,
        y_test,
        threshold=config.threshold,
        calibrator=calibrator,
        deployment_threshold=deployment_threshold,
    )
    metrics["train_val_check"] = {
        "split_mode": split_mode,
        "best_iteration": model.best_iteration,
        "train_auc": round(float(train_auc), 4),
        "val_auc": round(float(val_auc), 4),
        "gap": round(float(train_auc - val_auc), 4),
        "note": (
            "Small gap expected with regularisation; "
            "OOT AUC is the forward-generalisation check"
        ),
    }
    metrics["deployment_threshold"]["derivation"] = (
        f"val-set min expected cost (C_fp={config.cost_fp:.2f}, "
        f"C_fn={config.cost_fn:.2f}); test set never consulted for threshold selection"
    )
    metrics["deployment_threshold"]["cost_config"] = {
        "cost_fp": config.cost_fp,
        "cost_fn": config.cost_fn,
        "bayes_optimal_threshold": round(_bayes_t, 4),
    }
    (config.model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    dep = metrics["deployment_threshold"]
    ct = metrics["configured_threshold"]
    logger.info(
        "OOT test | AUC=%.4f PR-AUC=%.4f Brier=%.4f default_rate=%.4f",
        metrics["auc"],
        metrics["pr_auc"],
        metrics["brier_score"],
        metrics["default_rate_actual"],
    )
    logger.info(
        "Deployment  threshold=%.4f (val min-cost) | P=%.3f R=%.3f F1=%.3f flag_rate=%.1f%%",
        dep["threshold"],
        dep["precision"],
        dep["recall"],
        dep["f1"],
        dep["flag_rate"] * 100,
    )
    logger.info(
        "Configured  threshold=%.4f (lender-consv) | P=%.3f R=%.3f F1=%.3f flag_rate=%.1f%%",
        ct["threshold"],
        ct["precision"],
        ct["recall"],
        ct["f1"],
        ct["flag_rate"] * 100,
    )

    # Fairness audit uses the deployment threshold — this is what the API applies.
    fairness = fairness_audit(
        model, X_test, y_test, threshold=deployment_threshold, calibrator=calibrator
    )
    (config.model_dir / "fairness_report.json").write_text(json.dumps(fairness, indent=2))

    for group, r in fairness.items():
        if r["fails_four_fifths_rule"]:
            logger.warning(
                "FAIRNESS FLAG: %s fails the four-fifths rule (DI=%.3f) -- "
                "review feature set / threshold before any real deployment",
                group,
                r["disparate_impact_ratio"],
            )
        logger.info(
            "Fairness %s | DI=%.3f  thin/flagged FPR=%.3f FNR=%.3f  "
            "estab/clean FPR=%.3f FNR=%.3f",
            group,
            r["disparate_impact_ratio"],
            list(r.values())[0]["fpr"],
            list(r.values())[0]["fnr"],
            list(r.values())[1]["fpr"],
            list(r.values())[1]["fnr"],
        )

    # Structured model validation: AUC gate, overfitting gap, and fairness.
    # Saves a machine-readable report and logs errors without hard-failing —
    # the CI AUC and fairness gates are the hard failure points.
    _metadata_oot = X_test[["crb_flagged"]].copy()
    _metadata_oot["thin_file"] = (X_test["tenure_months"] < 12).astype(int)
    _val_config = ModelValidationConfig(
        min_auc=0.65,
        max_train_oot_gap=0.15,
        fairness_configs=[
            FairnessConfig("thin_file", group_a_label=1, group_b_label=0),
            FairnessConfig("crb_flagged", group_a_label=1, group_b_label=0),
        ],
        forbidden_feature_names=["defaulted_30d", "fee_to_principal_ratio"],
    )
    _test_proba_raw = model.predict(X_test, num_iteration=model.best_iteration)
    _test_proba_cal = calibrator.transform(_test_proba_raw)
    model_val_report = validate_model(
        y_true=y_test,
        proba=_test_proba_cal,
        threshold=deployment_threshold,
        config=_val_config,
        feature_cols=FEATURE_COLUMNS,
        metadata=_metadata_oot,
        train_auc=float(train_auc),
    )
    (config.model_dir / "validation_report.json").write_text(
        json.dumps(model_val_report.to_dict(), indent=2)
    )
    if not model_val_report.passed:
        logger.error(
            "Model validation FAILED — see %s/validation_report.json for details",
            config.model_dir,
        )

    plot_feature_importance(model, config.model_dir / "feature_importance.png")

    sample = X_test.sample(min(SHAP_SAMPLE_SIZE, len(X_test)), random_state=config.random_state)
    reason_codes = generate_reason_codes(model, sample, threshold=deployment_threshold)
    (config.model_dir / "sample_reason_codes.json").write_text(
        json.dumps(reason_codes, indent=2)
    )

    logger.info("Training complete. Artifacts saved to %s/", config.model_dir)


if __name__ == "__main__":
    main()
