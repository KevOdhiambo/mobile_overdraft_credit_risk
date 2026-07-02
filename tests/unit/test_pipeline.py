"""Unit tests for data validation, feature building, and PSI monitoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import DataValidationError, build_features, validate_raw
from src.monitoring.drift import compute_psi

DEFAULT_RATE = 0.1
FIXTURE_ROWS = 200


def _valid_df(n: int = FIXTURE_ROWS, default_rate: float = DEFAULT_RATE) -> pd.DataFrame:
    """Build a minimal valid feature DataFrame for use in tests.

    Args:
        n: Number of rows.
        default_rate: Fraction of rows with defaulted_30d == 1. Must be
            within the validator's plausible range (0.005, 0.5).

    Returns:
        DataFrame containing all FEATURE_COLUMNS plus the target column.
    """
    rng = np.random.default_rng(0)
    n_default = int(n * default_rate)
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
            "defaulted_30d": [1] * n_default + [0] * (n - n_default),
        }
    )


def test_valid_data_passes() -> None:
    validate_raw(_valid_df())


def test_missing_column_fails() -> None:
    df = _valid_df().drop(columns=["tenure_months"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_raw(df)


def test_utilization_out_of_bounds_fails() -> None:
    df = _valid_df()
    df.loc[0, "utilization_rate"] = 1.5
    with pytest.raises(DataValidationError, match="utilization_rate out of"):
        validate_raw(df)


def test_implausible_default_rate_fails() -> None:
    df = _valid_df(default_rate=0.0)
    with pytest.raises(DataValidationError, match="Default rate"):
        validate_raw(df)


def test_build_features_shapes() -> None:
    df = _valid_df()
    X, y = build_features(df)
    assert len(X) == len(y) == len(df)
    assert "defaulted_30d" not in X.columns


def test_psi_zero_for_identical_distributions() -> None:
    s = pd.Series(np.random.default_rng(1).normal(size=500))
    assert compute_psi(s, s) == 0.0


def test_psi_detects_shift() -> None:
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(0, 1, 1000))
    shifted = pd.Series(rng.normal(3, 1, 1000))
    assert compute_psi(ref, shifted) > PSI_ALERT_THRESHOLD


# Import the constant rather than embed the magic number in the assertion.
from src.monitoring.drift import PSI_ALERT_THRESHOLD  # noqa: E402
