"""Model validation: AUC gate, overfitting gap, fairness, and feature leakage.

Designed to run after training, before any artifact is promoted to staging.
The validator operates on pre-computed probabilities rather than requiring
the model object — this keeps it testable without a live LightGBM booster.

The module-level validate_model() function accepts the booster directly and
handles prediction internally.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.validation.schemas import CheckResult, ValidationReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FairnessConfig:
    """Configuration for a single disparate-impact / FPR / FNR fairness check.

    group_col should identify a binary column in the metadata DataFrame, where
    group_a_label is the protected / disadvantaged group.  For continuous
    features (e.g. tenure_months), pre-compute a binary column in metadata
    before passing it to validate_model:

        metadata["thin_file"] = (X_test["tenure_months"] < 12).astype(int)
        FairnessConfig("thin_file", group_a_label=1, group_b_label=0)

    Attributes:
        group_col: Column in metadata DataFrame used for group assignment.
        group_a_label: Label identifying the protected group in group_col.
        group_b_label: Label identifying the reference group in group_col.
        di_lower_bound: Minimum acceptable disparate-impact ratio (four-fifths
            rule = 0.80).
        max_fpr_ratio: Maximum acceptable ratio of group_a FPR to group_b FPR;
            > 1.0 means the protected group is denied more often in error.
        max_fnr_ratio: Maximum acceptable ratio of group_a FNR to group_b FNR.
    """

    group_col: str
    group_a_label: Any
    group_b_label: Any
    di_lower_bound: float = 0.80
    max_fpr_ratio: float = 2.0
    max_fnr_ratio: float = 2.0


@dataclass(frozen=True)
class ModelValidationConfig:
    """Thresholds for a full model quality validation run.

    Attributes:
        min_auc: Minimum acceptable OOT AUC; below this the model fails.
        max_train_oot_gap: Maximum acceptable train – OOT AUC gap; large
            gaps indicate the model is memorising the training period rather
            than learning generalisable patterns.
        fairness_configs: One config per group to audit.
        forbidden_feature_names: Feature names that must not appear in the
            model's feature list (e.g. the target itself, or known leaking
            features from a prior incident).
    """

    min_auc: float = 0.65
    max_train_oot_gap: float = 0.15
    fairness_configs: list[FairnessConfig] = field(default_factory=list)
    forbidden_feature_names: list[str] = field(default_factory=list)


class ModelValidator:
    """Validate a trained model against defined quality and fairness gates.

    Args:
        config: Validation thresholds and fairness configurations.
        feature_cols: Feature columns in the order the model was trained on.
            Used only for the forbidden-feature check.
    """

    def __init__(
        self,
        config: ModelValidationConfig,
        feature_cols: list[str] | None = None,
    ) -> None:
        self.config = config
        self.feature_cols = feature_cols or []

    def validate(
        self,
        y_true: np.ndarray,
        proba: np.ndarray,
        threshold: float,
        metadata: pd.DataFrame | None = None,
        train_auc: float | None = None,
    ) -> ValidationReport:
        """Run all model validation checks.

        Args:
            y_true: Ground-truth binary labels (OOT test set).
            proba: Calibrated predicted probabilities, same length as y_true.
            threshold: Deployment decision threshold.
            metadata: DataFrame aligned with y_true / proba; must contain the
                group columns referenced in config.fairness_configs.
            train_auc: Training-set AUC for overfitting gap check; skipped if None.

        Returns:
            ValidationReport; call .assert_passed() to raise on any errors.
        """
        checks: list[CheckResult] = []
        checks.extend(self._check_auc(y_true, proba))
        if train_auc is not None:
            oot_auc = float(roc_auc_score(y_true, proba))
            checks.extend(self._check_overfitting_gap(train_auc, oot_auc))
        if metadata is not None:
            for fc in self.config.fairness_configs:
                checks.extend(self._check_fairness(y_true, proba, threshold, metadata, fc))
        checks.extend(self._check_forbidden_features())

        passed = not any(c.severity == "error" and not c.passed for c in checks)
        report = ValidationReport(passed=passed, checks=checks)
        report.log_summary(logger)
        return report

    def _check_auc(
        self, y_true: np.ndarray, proba: np.ndarray
    ) -> list[CheckResult]:
        auc = float(roc_auc_score(y_true, proba))
        passed = auc >= self.config.min_auc
        return [CheckResult(
            name="model.auc",
            passed=passed,
            severity="error" if not passed else "info",
            message=(
                f"AUC={auc:.4f} "
                f"{'<' if not passed else '>='} "
                f"min={self.config.min_auc}"
            ),
            details={"auc": round(auc, 4), "min_auc": self.config.min_auc},
        )]

    def _check_overfitting_gap(
        self, train_auc: float, oot_auc: float
    ) -> list[CheckResult]:
        gap = train_auc - oot_auc
        passed = gap <= self.config.max_train_oot_gap
        return [CheckResult(
            name="model.train_oot_gap",
            passed=passed,
            severity="warning" if not passed else "info",
            message=(
                f"Train-OOT AUC gap={gap:.4f} "
                f"(train={train_auc:.4f}, OOT={oot_auc:.4f})"
            ),
            details={
                "train_auc": round(train_auc, 4),
                "oot_auc": round(oot_auc, 4),
                "gap": round(gap, 4),
                "max_gap": self.config.max_train_oot_gap,
            },
        )]

    def _check_fairness(
        self,
        y_true: np.ndarray,
        proba: np.ndarray,
        threshold: float,
        metadata: pd.DataFrame,
        fc: FairnessConfig,
    ) -> list[CheckResult]:
        results: list[CheckResult] = []
        if fc.group_col not in metadata.columns:
            return results

        preds = (proba >= threshold).astype(int)
        mask_a = (metadata[fc.group_col] == fc.group_a_label).values
        mask_b = (metadata[fc.group_col] == fc.group_b_label).values

        if mask_a.sum() == 0 or mask_b.sum() == 0:
            return results

        def _group_metrics(mask: np.ndarray) -> tuple[float, float, float]:
            y = y_true[mask]
            p = preds[mask]
            tp = int(((p == 1) & (y == 1)).sum())
            fp = int(((p == 1) & (y == 0)).sum())
            tn = int(((p == 0) & (y == 0)).sum())
            fn = int(((p == 0) & (y == 1)).sum())
            approval = (tn + fn) / max(len(y), 1)
            fpr = fp / max(fp + tn, 1)
            fnr = fn / max(fn + tp, 1)
            return approval, fpr, fnr

        approval_a, fpr_a, fnr_a = _group_metrics(mask_a)
        approval_b, fpr_b, fnr_b = _group_metrics(mask_b)

        di = approval_a / max(approval_b, 1e-9)
        di_passed = di >= fc.di_lower_bound
        results.append(CheckResult(
            name=f"fairness.di.{fc.group_col}",
            passed=di_passed,
            severity="error" if not di_passed else "info",
            message=(
                f"DI({fc.group_a_label}/{fc.group_b_label})={di:.4f} "
                f"{'<' if not di_passed else '>='} {fc.di_lower_bound}"
            ),
            details={
                "di": round(di, 4),
                "approval_a": round(approval_a, 4),
                "approval_b": round(approval_b, 4),
                "di_lower_bound": fc.di_lower_bound,
            },
        ))

        fpr_ratio = fpr_a / max(fpr_b, 1e-9)
        fpr_passed = fpr_ratio <= fc.max_fpr_ratio
        results.append(CheckResult(
            name=f"fairness.fpr_ratio.{fc.group_col}",
            passed=fpr_passed,
            severity="warning" if not fpr_passed else "info",
            message=(
                f"FPR ratio({fc.group_a_label}/{fc.group_b_label})="
                f"{fpr_ratio:.4f} (a={fpr_a:.4f}, b={fpr_b:.4f})"
            ),
            details={
                "fpr_ratio": round(fpr_ratio, 4),
                "fpr_a": round(fpr_a, 4),
                "fpr_b": round(fpr_b, 4),
                "max_fpr_ratio": fc.max_fpr_ratio,
            },
        ))

        fnr_ratio = fnr_a / max(fnr_b, 1e-9)
        fnr_passed = fnr_ratio <= fc.max_fnr_ratio
        results.append(CheckResult(
            name=f"fairness.fnr_ratio.{fc.group_col}",
            passed=fnr_passed,
            severity="warning" if not fnr_passed else "info",
            message=(
                f"FNR ratio({fc.group_a_label}/{fc.group_b_label})="
                f"{fnr_ratio:.4f} (a={fnr_a:.4f}, b={fnr_b:.4f})"
            ),
            details={
                "fnr_ratio": round(fnr_ratio, 4),
                "fnr_a": round(fnr_a, 4),
                "fnr_b": round(fnr_b, 4),
                "max_fnr_ratio": fc.max_fnr_ratio,
            },
        ))
        return results

    def _check_forbidden_features(self) -> list[CheckResult]:
        """Verify no forbidden feature names appear in the model's feature list."""
        results: list[CheckResult] = []
        present = [f for f in self.config.forbidden_feature_names if f in self.feature_cols]
        if present:
            for fname in present:
                results.append(CheckResult(
                    name=f"leakage.forbidden_feature.{fname}",
                    passed=False,
                    severity="error",
                    message=(
                        f"Forbidden feature '{fname}' is present in the "
                        "model's feature list"
                    ),
                    details={"feature": fname},
                ))
        elif self.config.forbidden_feature_names:
            results.append(CheckResult(
                name="leakage.forbidden_features",
                passed=True,
                severity="info",
                message="No forbidden features present in model feature list",
            ))
        return results


def validate_model(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    threshold: float,
    config: ModelValidationConfig,
    feature_cols: list[str] | None = None,
    metadata: pd.DataFrame | None = None,
    train_auc: float | None = None,
) -> ValidationReport:
    """Validate model quality, fairness, and feature safety.

    Convenience wrapper around ModelValidator for one-shot calls. Does not
    require the model object — pass pre-computed calibrated probabilities.

    Args:
        y_true: Ground-truth binary labels.
        proba: Calibrated predicted probabilities.
        threshold: Deployment decision threshold.
        config: Validation config with AUC floor, fairness specs, etc.
        feature_cols: Feature column names in model's training order; used
            for forbidden-feature check.
        metadata: DataFrame aligned with y_true containing group columns.
        train_auc: Training-set AUC for the overfitting gap check.

    Returns:
        ValidationReport; call .assert_passed() to raise on any errors.
    """
    validator = ModelValidator(config=config, feature_cols=feature_cols)
    return validator.validate(
        y_true=np.asarray(y_true),
        proba=np.asarray(proba),
        threshold=threshold,
        metadata=metadata,
        train_auc=train_auc,
    )
