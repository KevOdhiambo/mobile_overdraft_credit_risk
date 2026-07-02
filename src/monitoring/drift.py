"""PSI-based drift monitoring for the overdraft PD model.

Intentionally dependency-light (no Evidently AI etc.) so it runs on a
free-tier Lambda without pulling heavy packages. To upgrade, swap this
module for Evidently AI — the public interface (compute_psi /
generate_drift_report) is kept stable so the swap is a drop-in.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PSI_WARN_THRESHOLD = 0.1
PSI_ALERT_THRESHOLD = 0.25
PSI_BINS = 10


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = PSI_BINS) -> float:
    """Compute Population Stability Index between two distributions.

    PSI < 0.1: no significant shift.
    PSI 0.1–0.25: moderate shift, investigate.
    PSI > 0.25: significant shift, retraining likely needed.

    Args:
        reference: Training-time feature distribution (used to define bin edges).
        current: Live feature distribution to compare against.
        bins: Number of quantile-based bins.

    Returns:
        PSI value rounded to 4 decimal places, or float('nan') if either
        series is empty after dropping nulls.
    """
    reference = reference.dropna()
    current = current.dropna()
    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
    breakpoints[0] -= 1e-6
    breakpoints[-1] += 1e-6
    breakpoints = np.unique(breakpoints)

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 4)


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, float | str]]:
    """Compute per-feature PSI and classify each as stable / warn / ALERT.

    Logs a warning if any feature crosses the ALERT threshold; intended to
    trigger a CloudWatch alarm or equivalent in production.

    Args:
        reference_df: Training-time feature distribution.
        current_df: Live scoring window to compare against.
        columns: Feature columns to include. Columns absent from either
            DataFrame are silently skipped.

    Returns:
        Dict keyed by feature name, each value containing psi and status.
    """
    report: dict[str, dict[str, float | str]] = {}

    for col in columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = compute_psi(reference_df[col], current_df[col])
        if psi >= PSI_ALERT_THRESHOLD:
            status = "ALERT"
        elif psi >= PSI_WARN_THRESHOLD:
            status = "warn"
        else:
            status = "stable"
        report[col] = {"psi": psi, "status": status}

    alerts = [c for c, r in report.items() if r["status"] == "ALERT"]
    if alerts:
        logger.warning(
            "Drift ALERT on features: %s -- retraining should be evaluated", alerts
        )

    return report


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/processed/features.csv")
    parser.add_argument("--current", default="data/processed/features.csv")
    parser.add_argument("--out", default="models/drift_report.json")
    args = parser.parse_args()

    ref = pd.read_csv(args.reference)
    cur = pd.read_csv(args.current)
    cols = [c for c in ref.columns if c != "defaulted_30d"]

    report = generate_drift_report(ref, cur, cols)
    Path(args.out).write_text(json.dumps(report, indent=2))
    logger.info("Drift report saved to %s", args.out)
