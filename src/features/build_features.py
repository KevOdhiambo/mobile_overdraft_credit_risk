"""Feature engineering and data validation for the overdraft PD pipeline."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.validation.schemas import ColumnSchema, ValidationError

logger = logging.getLogger(__name__)

TARGET_COL = "defaulted_30d"
GROUP_ID_COL = "cust_id"

# Raw columns expected directly from the data source.
RAW_FEATURE_COLUMNS: list[str] = [
    "tenure_months",
    "monthly_txn_count",
    "avg_txn_value_kes",
    "send_receive_ratio",
    "unique_counterparties_30d",
    "agent_cashin_freq_30d",
    "airtime_bundle_freq_30d",
    "savings_activity_score",
    "crb_flagged",
    "voice_data_spend_idx",
    "income_regularity",
    "assigned_limit_kes",
    "draw_amount_kes",
    "utilization_rate",
    "overdraw_to_inflow_ratio",
    "prior_overdraw_count",
    "prior_cleared_within_24h_count",
    "prior_rolled_past_30d_count",
    # fee_to_principal_ratio intentionally excluded: fees accrue until the loan
    # clears, encoding days_to_clear (and therefore the outcome) rather than
    # being available at scoring time. Including it inflates AUC to ~1.0.
]

# Computed from RAW_FEATURE_COLUMNS — rates normalise for borrowers who have
# different numbers of prior draws. A borrower with 3 draws who cleared all 3
# within 24h is a better credit signal than one with 7 draws who cleared 3.
DERIVED_FEATURE_COLUMNS: list[str] = [
    "prior_cleared_rate",  # prior_cleared_within_24h / (prior_overdraw + 1)
    "prior_roll_rate",     # prior_rolled_past_30d / (prior_overdraw + 1)
]

FEATURE_COLUMNS: list[str] = RAW_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS

# Proxy columns used for fairness audit only — not model features.
# In a real deployment, audit against actual protected-class data subject to
# local regulatory requirements (Kenya DPA, CBK Digital Credit Providers Act).
FAIRNESS_PROXY_COLUMNS: list[str] = ["tenure_months", "crb_flagged"]

NULL_THRESHOLD = 0.01
DEFAULT_RATE_MIN = 0.005
DEFAULT_RATE_MAX = 0.5

# Schema used by validate_data() in the validation harness.
# Defines the bounds for each raw feature column; derived columns are
# validated transitively through their input columns.
FEATURE_SCHEMA: list[ColumnSchema] = [
    ColumnSchema("tenure_months", min_val=0.0, max_val=240.0),
    ColumnSchema("monthly_txn_count", min_val=0.0),
    ColumnSchema("avg_txn_value_kes", min_val=0.0),
    ColumnSchema("send_receive_ratio", min_val=0.0),
    ColumnSchema("unique_counterparties_30d", min_val=0.0),
    ColumnSchema("agent_cashin_freq_30d", min_val=0.0),
    ColumnSchema("airtime_bundle_freq_30d", min_val=0.0),
    ColumnSchema("savings_activity_score", min_val=0.0, max_val=1.0),
    ColumnSchema("crb_flagged", allowed_values=[0, 1]),
    ColumnSchema("voice_data_spend_idx", min_val=0.0, max_val=1.0),
    ColumnSchema("income_regularity", min_val=0.0, max_val=1.0),
    ColumnSchema("assigned_limit_kes", min_val=0.0),
    ColumnSchema("draw_amount_kes", min_val=0.0),
    ColumnSchema("utilization_rate", min_val=0.0, max_val=1.0),
    ColumnSchema("overdraw_to_inflow_ratio", min_val=0.0),
    ColumnSchema("prior_overdraw_count", min_val=0.0),
    ColumnSchema("prior_cleared_within_24h_count", min_val=0.0),
    ColumnSchema("prior_rolled_past_30d_count", min_val=0.0),
]


class DataValidationError(ValidationError):
    """Raised when the input DataFrame fails a data quality invariant."""


def validate_raw(df: pd.DataFrame) -> None:
    """Validate the raw feature DataFrame against expected invariants.

    Accumulates all violations before raising so the caller gets the full
    picture rather than fixing one error at a time.

    Args:
        df: Raw DataFrame containing at minimum RAW_FEATURE_COLUMNS + TARGET_COL.

    Raises:
        DataValidationError: If any required column is missing (immediate) or
            if any data quality checks fail (accumulated).
    """
    required = RAW_FEATURE_COLUMNS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    errors: list[str] = []

    if df[TARGET_COL].isna().any():
        errors.append(f"{df[TARGET_COL].isna().sum()} rows have a null target")

    null_frac = df[RAW_FEATURE_COLUMNS].isna().mean()
    bad_null_cols = null_frac[null_frac > NULL_THRESHOLD].to_dict()
    if bad_null_cols:
        errors.append(f"Columns exceed {NULL_THRESHOLD:.0%} null threshold: {bad_null_cols}")

    if (df["utilization_rate"] < 0).any() or (df["utilization_rate"] > 1).any():
        errors.append("utilization_rate out of [0, 1] bounds")

    if (df["draw_amount_kes"] <= 0).any():
        errors.append("Non-positive draw_amount_kes found")

    if not set(df["crb_flagged"].unique()).issubset({0, 1}):
        errors.append("crb_flagged contains values outside {0, 1}")

    default_rate = float(df[TARGET_COL].mean())
    if not (DEFAULT_RATE_MIN <= default_rate <= DEFAULT_RATE_MAX):
        errors.append(
            f"Default rate {default_rate:.4f} outside plausible "
            f"[{DEFAULT_RATE_MIN}, {DEFAULT_RATE_MAX}] range"
        )

    if errors:
        raise DataValidationError("Data validation failed:\n- " + "\n- ".join(errors))

    logger.info("Validation passed | %d rows | default_rate=%.4f", len(df), default_rate)


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rate-normalised repayment history features.

    The +1 denominator avoids division-by-zero for first-time borrowers with
    zero prior draws while introducing only negligible smoothing for borrowers
    with several prior episodes.

    Args:
        df: DataFrame containing RAW_FEATURE_COLUMNS.

    Returns:
        Copy of df with prior_cleared_rate and prior_roll_rate appended.
    """
    out = df.copy()
    denom = out["prior_overdraw_count"] + 1
    out["prior_cleared_rate"] = out["prior_cleared_within_24h_count"] / denom
    out["prior_roll_rate"] = out["prior_rolled_past_30d_count"] / denom
    return out


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate df, derive engineered features, and return (X, y).

    Args:
        df: Raw DataFrame. Must pass validate_raw.

    Returns:
        (X, y) where X contains FEATURE_COLUMNS and y is the integer target.

    Raises:
        DataValidationError: Propagated from validate_raw on any violation.
    """
    validate_raw(df)
    df = _add_derived_features(df)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].astype(int)
    return X, y


def save_schema(out_path: str | Path) -> None:
    """Write the feature schema (columns + target + fairness proxies) to JSON.

    Args:
        out_path: Destination file path. Parent directory must exist.
    """
    schema = {
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "derived_feature_columns": DERIVED_FEATURE_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "target_col": TARGET_COL,
        "fairness_proxy_columns": FAIRNESS_PROXY_COLUMNS,
    }
    Path(out_path).write_text(json.dumps(schema, indent=2))
    logger.info("Saved feature schema to %s", out_path)


def save_episode_metadata(df: pd.DataFrame, out_path: str | Path) -> None:
    """Write borrower IDs and draw dates to a parallel CSV for train splitting.

    The training pipeline uses draw_date for a time-based OOT split (preferred)
    and cust_id as a fallback for group-aware splitting when dates are absent.

    Args:
        df: Raw DataFrame, typically read from overdraft_lending_data.csv.
        out_path: Destination CSV, same row order as the features.csv written
            by the same run.
    """
    if GROUP_ID_COL not in df.columns:
        logger.warning(
            "%s column not found -- episode_ids.csv not written; "
            "training will fall back to stratified split",
            GROUP_ID_COL,
        )
        return
    cols = [GROUP_ID_COL]
    if "draw_date" in df.columns:
        cols.append("draw_date")
    df[cols].to_csv(out_path, index=False)
    logger.info("Saved episode metadata (%s) to %s (%d rows)", cols, out_path, len(df))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default="data/raw/overdraft_lending_data.csv")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.in_path)
    X, y = build_features(raw)
    X.assign(**{TARGET_COL: y}).to_csv(out_dir / "features.csv", index=False)
    save_schema(out_dir / "feature_schema.json")
    save_episode_metadata(raw, out_dir / "episode_ids.csv")
