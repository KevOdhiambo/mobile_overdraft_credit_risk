"""Tests for the data and model validation harness.

Three mandatory scenarios that the harness must catch:
1. A deliberately leaked feature (nearly perfectly correlated with target).
2. A deliberately drifted distribution (PSI >> alert threshold).
3. A deliberately unfair threshold (disparate-impact ratio < 0.80).

Plus correctness checks: clean data passes, PIT violations caught, AUC gate
triggers, forbidden features flagged, report.assert_passed() raises correctly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import FEATURE_SCHEMA
from src.validation import (
    ColumnSchema,
    DataValidator,
    FairnessConfig,
    ModelValidationConfig,
    ModelValidator,
    ValidationError,
    validate_data,
    validate_model,
)

RNG = np.random.default_rng(42)
N = 500
DEFAULT_RATE = 0.10


def _base_df(n: int = N, default_rate: float = DEFAULT_RATE) -> pd.DataFrame:
    """Minimal valid feature DataFrame for validation tests."""
    rng = np.random.default_rng(0)
    n_pos = int(n * default_rate)
    return pd.DataFrame(
        {
            "tenure_months": rng.integers(6, 60, n),
            "monthly_txn_count": rng.integers(1, 100, n),
            "avg_txn_value_kes": rng.uniform(100, 5000, n),
            "send_receive_ratio": rng.uniform(0.1, 3, n),
            "unique_counterparties_30d": rng.integers(0, 30, n),
            "agent_cashin_freq_30d": rng.integers(0, 10, n),
            "airtime_bundle_freq_30d": rng.integers(0, 10, n),
            "savings_activity_score": rng.uniform(0, 1, n),
            "crb_flagged": rng.integers(0, 2, n),
            "voice_data_spend_idx": rng.uniform(0, 1, n),
            "income_regularity": rng.uniform(0, 1, n),
            "assigned_limit_kes": rng.uniform(500, 50000, n),
            "draw_amount_kes": rng.uniform(50, 20000, n),
            "utilization_rate": rng.uniform(0, 1, n),
            "overdraw_to_inflow_ratio": rng.uniform(0, 10, n),
            "prior_overdraw_count": rng.integers(0, 10, n),
            "prior_cleared_within_24h_count": rng.integers(0, 5, n),
            "prior_rolled_past_30d_count": rng.integers(0, 3, n),
            "defaulted_30d": [1] * n_pos + [0] * (n - n_pos),
        }
    )


class TestLeakageDetection:
    def test_leaked_feature_is_caught(self) -> None:
        """Feature that is nearly perfectly correlated with target raises error."""
        df = _base_df()
        # Add a feature that encodes the outcome with tiny noise — simulates
        # a derived field accidentally including future outcome information.
        df["leaked_feature"] = df["defaulted_30d"] * 100 + RNG.normal(0, 0.01, N)
        schema = FEATURE_SCHEMA + [ColumnSchema("leaked_feature")]
        report = validate_data(df, schema=schema, target_col="defaulted_30d")

        assert not report.passed
        leaked_checks = [c for c in report.errors if "leaked_feature" in c.name]
        assert len(leaked_checks) >= 1
        assert "leakage" in leaked_checks[0].name

    def test_clean_features_pass_leakage_check(self) -> None:
        """No feature in the standard schema exceeds the correlation threshold."""
        df = _base_df()
        report = validate_data(df, schema=FEATURE_SCHEMA, target_col="defaulted_30d")
        leakage_errors = [c for c in report.errors if c.name.startswith("leakage.")]
        assert leakage_errors == []

    def test_forbidden_feature_in_model_is_caught(self) -> None:
        """A feature name that is forbidden raises an error if present in feature list."""
        config = ModelValidationConfig(
            forbidden_feature_names=["defaulted_30d", "fee_to_principal_ratio"]
        )
        # defaulted_30d accidentally included in feature list
        feature_cols = ["tenure_months", "overdraw_to_inflow_ratio", "defaulted_30d"]
        validator = ModelValidator(config=config, feature_cols=feature_cols)

        y = np.array([0, 1, 0, 1, 0] * 20, dtype=int)
        proba = np.clip(RNG.uniform(0.05, 0.50, 100), 0, 1)
        report = validator.validate(y_true=y, proba=proba, threshold=0.20)

        assert not report.passed
        forbidden_errors = [c for c in report.errors if "forbidden" in c.name]
        assert len(forbidden_errors) >= 1
        assert "defaulted_30d" in forbidden_errors[0].details["feature"]


class TestDriftDetection:
    def test_drifted_distribution_is_caught(self) -> None:
        """PSI >> 0.25 on a heavily shifted feature triggers an error."""
        reference = _base_df(n=1000)
        # Shift overdraw_to_inflow_ratio mean from ~5 to ~50 — massive drift
        serving = _base_df(n=500)
        serving["overdraw_to_inflow_ratio"] = RNG.uniform(45, 55, 500)

        schema = [ColumnSchema("overdraw_to_inflow_ratio", min_val=0.0)]
        report = validate_data(
            serving,
            schema=schema,
            reference_df=reference,
            target_col="defaulted_30d",
        )

        drift_errors = [
            c for c in report.checks
            if "drift" in c.name and not c.passed and c.severity == "error"
        ]
        assert len(drift_errors) >= 1
        assert "overdraw_to_inflow_ratio" in drift_errors[0].name

    def test_stable_distribution_passes_drift_check(self) -> None:
        """Same distribution as reference produces zero drift errors."""
        rng = np.random.default_rng(1)
        reference = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
        serving = pd.DataFrame({"x": rng.normal(0, 1, 500)})

        schema = [ColumnSchema("x")]
        report = validate_data(serving, schema=schema, reference_df=reference)

        drift_errors = [c for c in report.checks if "drift" in c.name and not c.passed]
        assert drift_errors == []


class TestFairnessDetection:
    def test_unfair_threshold_is_caught(self) -> None:
        """Threshold producing DI < 0.80 for the protected group raises an error."""
        rng = np.random.default_rng(7)
        n = 1000
        y_true = np.array([0] * 900 + [1] * 100)

        # Group A (protected) gets high predicted probabilities → most declined
        # Group B (reference) gets low predicted probabilities → most approved
        proba_a = rng.uniform(0.55, 0.90, 500)
        proba_b = rng.uniform(0.05, 0.30, 500)
        proba = np.concatenate([proba_a, proba_b])

        group_col = pd.Series([1] * 500 + [0] * 500, name="group")
        metadata = group_col.to_frame()

        config = ModelValidationConfig(
            fairness_configs=[
                FairnessConfig(
                    "group",
                    group_a_label=1,
                    group_b_label=0,
                    di_lower_bound=0.80,
                )
            ]
        )
        validator = ModelValidator(config=config)
        report = validator.validate(
            y_true=y_true, proba=proba, threshold=0.50, metadata=metadata
        )

        di_errors = [c for c in report.errors if "fairness.di" in c.name]
        assert len(di_errors) >= 1
        assert report.passed is False

    def test_fair_threshold_passes(self) -> None:
        """Balanced approval rates across groups produce no DI errors."""
        rng = np.random.default_rng(8)
        n = 1000
        # Mix of classes so AUC is defined
        y_true = np.array([0] * 900 + [1] * 100)
        # Both groups get very similar probability distribution
        proba = np.clip(rng.normal(0.15, 0.05, n), 0.01, 0.99)
        metadata = pd.DataFrame({"group": [0, 1] * (n // 2)})

        config = ModelValidationConfig(
            fairness_configs=[
                FairnessConfig("group", group_a_label=1, group_b_label=0)
            ]
        )
        validator = ModelValidator(config=config)
        report = validator.validate(
            y_true=y_true, proba=proba, threshold=0.50, metadata=metadata
        )
        di_errors = [c for c in report.errors if "fairness.di" in c.name]
        assert di_errors == []


class TestSchemaValidation:
    def test_missing_column_is_caught(self) -> None:
        df = _base_df().drop(columns=["tenure_months"])
        report = validate_data(df, schema=FEATURE_SCHEMA)

        assert not report.passed
        missing_errors = [c for c in report.errors if "missing_columns" in c.name]
        assert len(missing_errors) == 1
        assert "tenure_months" in missing_errors[0].details["missing"]

    def test_out_of_range_value_is_caught(self) -> None:
        df = _base_df()
        df.loc[0, "utilization_rate"] = 1.5  # above [0, 1]

        report = validate_data(df, schema=FEATURE_SCHEMA)
        range_errors = [
            c for c in report.errors if "schema.range.utilization_rate" in c.name
        ]
        assert len(range_errors) == 1

    def test_invalid_allowed_value_is_caught(self) -> None:
        df = _base_df()
        df.loc[0, "crb_flagged"] = 5  # not in {0, 1}

        report = validate_data(df, schema=FEATURE_SCHEMA)
        av_errors = [
            c for c in report.errors if "allowed_values.crb_flagged" in c.name
        ]
        assert len(av_errors) == 1

    def test_valid_data_passes_all_schema_checks(self) -> None:
        df = _base_df()
        report = validate_data(df, schema=FEATURE_SCHEMA)
        schema_errors = [c for c in report.errors if c.name.startswith("schema.")]
        assert schema_errors == []


class TestPITConsistency:
    def test_cleared_exceeds_overdraw_count_is_caught(self) -> None:
        """Cleared count > total prior overdraw count flags a PIT violation."""
        # Use a DataFrame with no pre-existing violations to get an exact count.
        rng_inner = np.random.default_rng(99)
        n_inner = 50
        prior_od = rng_inner.integers(5, 15, n_inner)
        df_pit = pd.DataFrame(
            {
                "prior_overdraw_count": prior_od,
                "prior_cleared_within_24h_count": rng_inner.integers(0, 3, n_inner),
                "prior_rolled_past_30d_count": rng_inner.integers(0, 2, n_inner),
            }
        )
        # Inject exactly 3 violations
        df_pit.loc[[0, 1, 2], "prior_overdraw_count"] = 2
        df_pit.loc[[0, 1, 2], "prior_cleared_within_24h_count"] = 5

        schema_pit: list[ColumnSchema] = [
            ColumnSchema("prior_overdraw_count", min_val=0.0),
            ColumnSchema("prior_cleared_within_24h_count", min_val=0.0),
            ColumnSchema("prior_rolled_past_30d_count", min_val=0.0),
        ]
        report = validate_data(df_pit, schema=schema_pit)
        pit_errors = [c for c in report.errors if "pit.cleared_exceeds_total" in c.name]
        assert len(pit_errors) == 1
        assert pit_errors[0].details["n_violations"] == 3

    def test_outcomes_exceed_total_is_caught(self) -> None:
        """Cleared + rolled > total prior overdraw count flags a PIT violation."""
        df = _base_df()
        df.loc[0, "prior_overdraw_count"] = 3
        df.loc[0, "prior_cleared_within_24h_count"] = 2
        df.loc[0, "prior_rolled_past_30d_count"] = 2  # 2+2=4 > 3

        report = validate_data(df, schema=FEATURE_SCHEMA)
        pit_errors = [c for c in report.errors if "pit.outcomes_exceed_total" in c.name]
        assert len(pit_errors) == 1

    def test_clean_pit_passes(self) -> None:
        """Consistent rolling counts produce no PIT violations."""
        df = pd.DataFrame(
            {
                "prior_overdraw_count": [5, 10, 0, 3],
                "prior_cleared_within_24h_count": [3, 5, 0, 1],
                "prior_rolled_past_30d_count": [1, 4, 0, 2],
            }
        )
        schema: list[ColumnSchema] = [
            ColumnSchema("prior_overdraw_count", min_val=0.0),
            ColumnSchema("prior_cleared_within_24h_count", min_val=0.0),
            ColumnSchema("prior_rolled_past_30d_count", min_val=0.0),
        ]
        report = validate_data(df, schema=schema)
        pit_errors = [c for c in report.errors if c.name.startswith("pit.")]
        assert pit_errors == []


class TestAUCGate:
    def test_auc_below_minimum_is_caught(self) -> None:
        """AUC near chance level (0.5) fails the model AUC gate."""
        rng = np.random.default_rng(3)
        n = 200
        y_true = np.array([0, 1] * (n // 2))
        # Near-random predictions → AUC ≈ 0.5, well below min=0.65
        proba = rng.uniform(0, 1, n)

        config = ModelValidationConfig(min_auc=0.65)
        validator = ModelValidator(config=config)
        report = validator.validate(y_true=y_true, proba=proba, threshold=0.50)

        auc_errors = [c for c in report.errors if c.name == "model.auc"]
        assert len(auc_errors) == 1
        assert report.passed is False

    def test_auc_above_minimum_passes(self) -> None:
        """Perfectly ordered predictions (AUC=1.0) pass the AUC gate."""
        n = 200
        y_true = np.array([0] * 100 + [1] * 100)
        proba = np.linspace(0, 1, n)  # perfectly correlated → AUC = 1.0

        config = ModelValidationConfig(min_auc=0.65)
        validator = ModelValidator(config=config)
        report = validator.validate(y_true=y_true, proba=proba, threshold=0.50)

        auc_errors = [c for c in report.errors if c.name == "model.auc"]
        assert auc_errors == []


class TestValidationReport:
    def test_assert_passed_does_not_raise_on_clean(self) -> None:
        df = _base_df()
        report = validate_data(df, schema=FEATURE_SCHEMA)
        # Should not raise
        schema_errors = [c for c in report.errors if c.name.startswith("schema.")]
        assert schema_errors == []

    def test_assert_passed_raises_on_error_severity(self) -> None:
        """assert_passed() raises ValidationError when error-severity checks fail."""
        df = _base_df().drop(columns=["tenure_months"])
        report = validate_data(df, schema=FEATURE_SCHEMA)
        assert not report.passed
        with pytest.raises(ValidationError, match="Validation failed"):
            report.assert_passed()

    def test_report_to_dict_is_serialisable(self) -> None:
        """to_dict() produces a JSON-serialisable dict."""
        import json

        df = _base_df()
        report = validate_data(df, schema=FEATURE_SCHEMA)
        d = report.to_dict()
        serialised = json.dumps(d)  # should not raise
        assert isinstance(serialised, str)
        assert "passed" in d
        assert "checks" in d
