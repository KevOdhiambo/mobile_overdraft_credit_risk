"""FastAPI service for the overdraft default-risk model.

POST /score       probability of default + decision + SHAP reason codes
GET  /health      liveness / readiness
GET  /model-info  model version and training metrics

Run locally:
    uvicorn src.api.main:app --reload --port 8000

A real bank-grade serving layer would add authentication, request signing,
a model registry, and a durable audit store. Those are infra/process concerns
beyond the scope of this portfolio project and are noted in the README
limitations section rather than faked here.
"""
from __future__ import annotations

import json
import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.isotonic import IsotonicRegression

from src.features.build_features import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/model.txt")
METRICS_PATH = Path("models/metrics.json")
CALIBRATOR_PATH = Path("models/calibrator.pkl")
# Fallback only — superseded at startup by deployment_threshold from metrics.json.
_FALLBACK_THRESHOLD = 0.3
MODEL_VERSION = "v1.0.0"

_model: lgb.Booster | None = None
_explainer: shap.TreeExplainer | None = None
_calibrator: IsotonicRegression | None = None
_metrics: dict[str, Any] = {}
_threshold: float = _FALLBACK_THRESHOLD


def _load_model() -> None:
    """Load the LightGBM booster, SHAP explainer, calibrator, and deployment threshold.

    Called once at application startup. If the model file is absent the app
    starts anyway and returns HTTP 503 from the /score endpoint until training
    has been run and the service restarted.
    """
    global _model, _explainer, _calibrator, _metrics, _threshold
    if not MODEL_PATH.exists():
        logger.warning("Model file not found at %s -- run training first", MODEL_PATH)
        return
    _model = lgb.Booster(model_file=str(MODEL_PATH))
    _explainer = shap.TreeExplainer(_model)
    if CALIBRATOR_PATH.exists():
        with open(CALIBRATOR_PATH, "rb") as fh:
            _calibrator = pickle.load(fh)
        logger.info("Isotonic calibrator loaded from %s", CALIBRATOR_PATH)
    else:
        logger.warning("Calibrator not found at %s -- raw model scores will be used", CALIBRATOR_PATH)
    if METRICS_PATH.exists():
        _metrics = json.loads(METRICS_PATH.read_text())
        dep = _metrics.get("deployment_threshold", {})
        if "threshold" in dep:
            _threshold = float(dep["threshold"])
            logger.info(
                "Deployment threshold loaded from metrics.json: %.4f (derivation: %s)",
                _threshold,
                dep.get("derivation", "unknown"),
            )
        else:
            logger.warning(
                "deployment_threshold not found in metrics.json -- using fallback %.2f",
                _FALLBACK_THRESHOLD,
            )
    logger.info("Model loaded successfully")


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    _load_model()
    yield


app = FastAPI(
    title="Mobile Overdraft Credit Risk API",
    description=(
        "Scores a mobile-money overdraft episode for 30-day default risk "
        "using alternative behavioral data. Trained on fully synthetic data; "
        "not affiliated with or trained on any real provider's data."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)


class ScoreRequest(BaseModel):
    """Feature payload for a single overdraft episode scoring request.

    All monetary fields (avg_txn_value_kes, assigned_limit_kes, draw_amount_kes)
    are typed as float at this boundary. FLAG: the code-standards mandate Decimal
    for monetary values; applying that here would change JSON parsing behaviour
    and require explicit float conversion before model inference. Deferred pending
    a deliberate API contract decision.
    """

    tenure_months: int = Field(..., ge=0)
    monthly_txn_count: int = Field(..., ge=0)
    avg_txn_value_kes: float = Field(..., gt=0)
    send_receive_ratio: float = Field(..., gt=0)
    unique_counterparties_30d: int = Field(..., ge=0)
    agent_cashin_freq_30d: int = Field(..., ge=0)
    airtime_bundle_freq_30d: int = Field(..., ge=0)
    savings_activity_score: float = Field(..., ge=0, le=1)
    crb_flagged: int = Field(..., ge=0, le=1)
    voice_data_spend_idx: float = Field(..., ge=0, le=1)
    income_regularity: float = Field(..., ge=0, le=1)
    assigned_limit_kes: float = Field(..., gt=0)
    draw_amount_kes: float = Field(..., gt=0)
    utilization_rate: float = Field(..., ge=0, le=1)
    overdraw_to_inflow_ratio: float = Field(..., ge=0)
    prior_overdraw_count: int = Field(..., ge=0)
    prior_cleared_within_24h_count: int = Field(..., ge=0)
    prior_rolled_past_30d_count: int = Field(..., ge=0)


class ReasonCode(BaseModel):
    feature: str
    shap_value: float
    direction: str


class ScoreResponse(BaseModel):
    probability_of_default_30d: float
    decision: str
    threshold_used: float
    top_reason_codes: list[ReasonCode]
    model_version: str


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness and readiness probe."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """Return model metadata and training metrics."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": "LightGBM",
        "features": FEATURE_COLUMNS,
        "active_threshold": _threshold,
        "test_metrics": _metrics,
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    """Score a single overdraft episode for 30-day default risk.

    Returns a probability, an APPROVE/DECLINE decision, and the top-3 SHAP
    reason codes driving the decision — required for adverse action notice
    under most regulated lending frameworks.

    Args:
        req: Feature payload for one episode.

    Raises:
        HTTPException: 503 if the model has not been loaded.
    """
    if _model is None or _explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training pipeline first.")

    # Compute derived features from the request fields — mirrors build_features._add_derived_features.
    row_dict = req.model_dump()
    denom = row_dict["prior_overdraw_count"] + 1
    row_dict["prior_cleared_rate"] = row_dict["prior_cleared_within_24h_count"] / denom
    row_dict["prior_roll_rate"] = row_dict["prior_rolled_past_30d_count"] / denom

    row = pd.DataFrame([row_dict])[FEATURE_COLUMNS]
    proba_raw = float(_model.predict(row, num_iteration=_model.best_iteration)[0])
    proba = float(_calibrator.transform([proba_raw])[0]) if _calibrator is not None else proba_raw
    decision = "DECLINE" if proba >= _threshold else "APPROVE"
    model_version = _metrics.get("model_version", MODEL_VERSION)

    # Audit log: every prediction driving a financial decision is recorded with
    # its full input features, model version, and outcome for regulatory traceability.
    logger.info(
        "score decision=%s pd=%.4f threshold=%.4f model_version=%s features=%s",
        decision,
        proba,
        _threshold,
        model_version,
        req.model_dump_json(),
    )

    shap_values = _explainer.shap_values(row)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    row_shap = shap_values[0]
    top_idx = np.argsort(-np.abs(row_shap))[:3]

    reasons = [
        ReasonCode(
            feature=row.columns[i],
            shap_value=round(float(row_shap[i]), 4),
            direction="increases risk" if row_shap[i] > 0 else "decreases risk",
        )
        for i in top_idx
    ]

    return ScoreResponse(
        probability_of_default_30d=round(proba, 4),
        decision=decision,
        threshold_used=round(_threshold, 4),
        top_reason_codes=reasons,
        model_version=str(model_version),
    )
