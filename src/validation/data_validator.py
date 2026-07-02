"""Data validation for incoming feature DataFrames.

Two distinct failure modes:
- Schema / range / allowed-value violations: always errors.
- Drift (PSI) beyond alert threshold: error. PSI in warn band: warning.
- Leakage (feature nearly perfectly correlated with target): error.
- Point-in-time consistency (rolling counts can't exceed total episodes): error.

All checks run unconditionally; the report accumulates the full picture.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.monitoring.drift import compute_psi
from src.validation.schemas import CheckResult, ColumnSchema, ValidationReport

logger = logging.getLogger(__name__)

_PSI_WARN_DEFAULT = 0.10
_PSI_ALERT_DEFAULT = 0.25
_LEAKAGE_CORR_DEFAULT = 0.95


@dataclass
class DataValidator:
    """Validate an incoming feature DataFrame against schema and reference distribution.

    Args:
        schema: Expected column schemas.
        reference_df: Training-time feature distribution for drift comparison.
            If None, drift checks are skipped.
        target_col: Target column name; enables leakage and target-rate checks
            when present in the DataFrame.
        psi_warn_threshold: PSI above which a warning is emitted.
        psi_alert_threshold: PSI above which drift fails as an error.
        leakage_corr_threshold: |Pearson r| above which a feature is flagged as
            potentially encoding the target.
    """

    schema: list[ColumnSchema]
    reference_df: pd.DataFrame | None = None
    target_col: str = "defaulted_30d"
    psi_warn_threshold: float = _PSI_WARN_DEFAULT
    psi_alert_threshold: float = _PSI_ALERT_DEFAULT
    leakage_corr_threshold: float = _LEAKAGE_CORR_DEFAULT

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Run all configured data checks.

        Args:
            df: Incoming feature DataFrame. May include the target column.

        Returns:
            ValidationReport with per-check results and an overall passed flag.
        """
        checks: list[CheckResult] = []
        checks.extend(self._check_schema(df))
        checks.extend(self._check_missing(df))
        if self.reference_df is not None:
            checks.extend(self._check_drift(df))
        if self.target_col in df.columns:
            checks.extend(self._check_leakage(df))
            checks.extend(self._check_target_rate(df))
        checks.extend(self._check_pit_consistency(df))

        passed = not any(c.severity == "error" and not c.passed for c in checks)
        report = ValidationReport(passed=passed, checks=checks)
        report.log_summary(logger)
        return report

    def _check_schema(self, df: pd.DataFrame) -> list[CheckResult]:
        results: list[CheckResult] = []

        expected = {s.name for s in self.schema}
        missing = sorted(expected - set(df.columns))
        if missing:
            results.append(CheckResult(
                name="schema.missing_columns",
                passed=False,
                severity="error",
                message=f"Missing required columns: {missing}",
                details={"missing": missing},
            ))
        else:
            results.append(CheckResult(
                name="schema.missing_columns",
                passed=True,
                severity="info",
                message="All required columns present",
            ))

        for col_schema in self.schema:
            if col_schema.name not in df.columns:
                continue
            col = df[col_schema.name]

            if col_schema.min_val is not None or col_schema.max_val is not None:
                lo = col_schema.min_val if col_schema.min_val is not None else -np.inf
                hi = col_schema.max_val if col_schema.max_val is not None else np.inf
                n_out = int(((col < lo) | (col > hi)).sum())
                check_name = f"schema.range.{col_schema.name}"
                if n_out > 0:
                    results.append(CheckResult(
                        name=check_name,
                        passed=False,
                        severity="error",
                        message=(
                            f"{col_schema.name}: {n_out} value(s) outside "
                            f"[{lo}, {hi}]"
                        ),
                        details={
                            "column": col_schema.name,
                            "n_out_of_range": n_out,
                            "min": lo,
                            "max": hi,
                        },
                    ))
                else:
                    results.append(CheckResult(
                        name=check_name,
                        passed=True,
                        severity="info",
                        message=f"{col_schema.name}: all values in [{lo}, {hi}]",
                    ))

            if col_schema.allowed_values is not None:
                n_invalid = int((~col.isin(col_schema.allowed_values)).sum())
                check_name = f"schema.allowed_values.{col_schema.name}"
                if n_invalid > 0:
                    results.append(CheckResult(
                        name=check_name,
                        passed=False,
                        severity="error",
                        message=(
                            f"{col_schema.name}: {n_invalid} value(s) not in "
                            f"allowed set {col_schema.allowed_values}"
                        ),
                        details={
                            "column": col_schema.name,
                            "n_invalid": n_invalid,
                        },
                    ))
                else:
                    results.append(CheckResult(
                        name=check_name,
                        passed=True,
                        severity="info",
                        message=f"{col_schema.name}: all values in allowed set",
                    ))

        return results

    def _check_missing(self, df: pd.DataFrame) -> list[CheckResult]:
        results: list[CheckResult] = []
        schema_map = {s.name: s for s in self.schema}
        any_fail = False
        for col in df.columns:
            if col not in schema_map:
                continue
            col_schema = schema_map[col]
            miss_rate = float(df[col].isna().mean())
            if miss_rate > col_schema.max_missing_rate:
                any_fail = True
                sev = "error" if miss_rate > 0.01 else "warning"
                results.append(CheckResult(
                    name=f"missing.{col}",
                    passed=False,
                    severity=sev,
                    message=(
                        f"{col}: missing rate {miss_rate:.3%} exceeds "
                        f"allowed {col_schema.max_missing_rate:.3%}"
                    ),
                    details={
                        "column": col,
                        "missing_rate": round(miss_rate, 6),
                        "limit": col_schema.max_missing_rate,
                    },
                ))
        if not any_fail:
            results.append(CheckResult(
                name="missing.all_columns",
                passed=True,
                severity="info",
                message="All columns within expected missingness bounds",
            ))
        return results

    def _check_drift(self, df: pd.DataFrame) -> list[CheckResult]:
        """PSI per numeric feature between reference and current distribution."""
        results: list[CheckResult] = []
        numeric_cols = [
            s.name
            for s in self.schema
            if s.name in df.columns
            and s.name in self.reference_df.columns  # type: ignore[union-attr]
            and pd.api.types.is_numeric_dtype(df[s.name])
            and s.name != self.target_col
        ]
        for col in numeric_cols:
            psi = compute_psi(self.reference_df[col], df[col])  # type: ignore[index]
            if np.isnan(psi):
                continue
            if psi >= self.psi_alert_threshold:
                results.append(CheckResult(
                    name=f"drift.psi.{col}",
                    passed=False,
                    severity="error",
                    message=(
                        f"{col}: PSI={psi:.4f} exceeds ALERT threshold "
                        f"{self.psi_alert_threshold}"
                    ),
                    details={"column": col, "psi": psi, "threshold": self.psi_alert_threshold},
                ))
            elif psi >= self.psi_warn_threshold:
                results.append(CheckResult(
                    name=f"drift.psi.{col}",
                    passed=False,
                    severity="warning",
                    message=(
                        f"{col}: PSI={psi:.4f} exceeds WARN threshold "
                        f"{self.psi_warn_threshold}"
                    ),
                    details={"column": col, "psi": psi, "threshold": self.psi_warn_threshold},
                ))
            else:
                results.append(CheckResult(
                    name=f"drift.psi.{col}",
                    passed=True,
                    severity="info",
                    message=f"{col}: PSI={psi:.4f} (stable)",
                    details={"column": col, "psi": psi},
                ))
        return results

    def _check_leakage(self, df: pd.DataFrame) -> list[CheckResult]:
        """Flag features that are suspiciously correlated with the target.

        Any feature with |Pearson r| >= leakage_corr_threshold is likely
        encoding the outcome — either directly (target included as a feature)
        or indirectly (a derived field that uses information from the future,
        such as accumulated fees that encode repayment duration).
        """
        results: list[CheckResult] = []
        y = df[self.target_col]
        feature_names = [
            s.name
            for s in self.schema
            if s.name in df.columns
            and s.name != self.target_col
            and pd.api.types.is_numeric_dtype(df[s.name])
        ]
        leaked: list[tuple[str, float]] = []
        for col in feature_names:
            try:
                r = float(df[col].corr(y))
                if not np.isnan(r) and abs(r) >= self.leakage_corr_threshold:
                    leaked.append((col, r))
            except Exception:
                pass

        if leaked:
            for col, r in leaked:
                results.append(CheckResult(
                    name=f"leakage.correlation.{col}",
                    passed=False,
                    severity="error",
                    message=(
                        f"{col}: |Pearson r|={abs(r):.4f} with target — "
                        f"possible feature leakage (threshold "
                        f"{self.leakage_corr_threshold})"
                    ),
                    details={
                        "column": col,
                        "pearson_r": round(r, 4),
                        "threshold": self.leakage_corr_threshold,
                    },
                ))
        else:
            results.append(CheckResult(
                name="leakage.correlation",
                passed=True,
                severity="info",
                message=(
                    f"No features correlated with target above "
                    f"{self.leakage_corr_threshold}"
                ),
            ))
        return results

    def _check_target_rate(self, df: pd.DataFrame) -> list[CheckResult]:
        """Default rate plausibility gate (0.5% – 50%)."""
        rate = float(df[self.target_col].mean())
        if not (0.005 <= rate <= 0.50):
            return [CheckResult(
                name="target.rate",
                passed=False,
                severity="error",
                message=(
                    f"Default rate {rate:.4f} outside plausible range "
                    "[0.005, 0.50]"
                ),
                details={"default_rate": round(rate, 4)},
            )]
        return [CheckResult(
            name="target.rate",
            passed=True,
            severity="info",
            message=f"Default rate {rate:.4f} within plausible range",
            details={"default_rate": round(rate, 4)},
        )]

    def _check_pit_consistency(self, df: pd.DataFrame) -> list[CheckResult]:
        """Point-in-time audit on rolling prior-history count columns.

        For each episode, cleared-within-24h and rolled-past-30d are
        non-overlapping subsets of the total prior overdraw count.  Any
        violation means future outcomes leaked into the history counters.
        """
        results: list[CheckResult] = []
        required = {
            "prior_overdraw_count",
            "prior_cleared_within_24h_count",
            "prior_rolled_past_30d_count",
        }
        if not required.issubset(df.columns):
            return results

        checks = [
            (
                "pit.prior_count_negative",
                int((df["prior_overdraw_count"] < 0).sum()),
                "prior_overdraw_count < 0",
            ),
            (
                "pit.cleared_exceeds_total",
                int(
                    (df["prior_cleared_within_24h_count"] > df["prior_overdraw_count"]).sum()
                ),
                "prior_cleared_within_24h_count > prior_overdraw_count",
            ),
            (
                "pit.rolled_exceeds_total",
                int(
                    (df["prior_rolled_past_30d_count"] > df["prior_overdraw_count"]).sum()
                ),
                "prior_rolled_past_30d_count > prior_overdraw_count",
            ),
            (
                "pit.outcomes_exceed_total",
                int(
                    (
                        df["prior_cleared_within_24h_count"]
                        + df["prior_rolled_past_30d_count"]
                        > df["prior_overdraw_count"]
                    ).sum()
                ),
                "cleared + rolled > prior_overdraw_count (non-overlapping subsets)",
            ),
        ]

        for check_name, n_violations, description in checks:
            if n_violations > 0:
                results.append(CheckResult(
                    name=check_name,
                    passed=False,
                    severity="error",
                    message=(
                        f"Point-in-time violation: {n_violations} row(s) have "
                        f"{description}"
                    ),
                    details={"n_violations": n_violations},
                ))
            else:
                results.append(CheckResult(
                    name=check_name,
                    passed=True,
                    severity="info",
                    message=f"PIT check passed: no {description} violations",
                ))
        return results


def validate_data(
    df: pd.DataFrame,
    schema: list[ColumnSchema],
    reference_df: pd.DataFrame | None = None,
    target_col: str = "defaulted_30d",
    psi_warn_threshold: float = _PSI_WARN_DEFAULT,
    psi_alert_threshold: float = _PSI_ALERT_DEFAULT,
    leakage_corr_threshold: float = _LEAKAGE_CORR_DEFAULT,
) -> ValidationReport:
    """Validate an incoming feature DataFrame against schema and reference distribution.

    Convenience wrapper around DataValidator for one-shot calls.

    Args:
        df: DataFrame to validate.
        schema: Column schema definitions (use build_features.FEATURE_SCHEMA).
        reference_df: Training-time distribution for drift checks; skipped if None.
        target_col: Name of the target column.
        psi_warn_threshold: PSI at which drift becomes a warning.
        psi_alert_threshold: PSI at which drift becomes an error.
        leakage_corr_threshold: |Pearson r| above which a feature is flagged.

    Returns:
        ValidationReport; call .assert_passed() to raise on any errors.
    """
    validator = DataValidator(
        schema=schema,
        reference_df=reference_df,
        target_col=target_col,
        psi_warn_threshold=psi_warn_threshold,
        psi_alert_threshold=psi_alert_threshold,
        leakage_corr_threshold=leakage_corr_threshold,
    )
    return validator.validate(df)
