"""Reusable data and model validation harness for the overdraft PD pipeline.

Public API::

    from src.validation import (
        validate_data,
        validate_model,
        ValidationReport,
        CheckResult,
        ColumnSchema,
        ValidationError,
        DataValidator,
        ModelValidator,
        ModelValidationConfig,
        FairnessConfig,
        CostConfig,
        bayes_optimal_threshold,
        min_cost_threshold,
    )
"""
from src.validation.cost_threshold import (
    CostConfig,
    bayes_optimal_threshold,
    cost_threshold_report,
    min_cost_threshold,
)
from src.validation.data_validator import DataValidator, validate_data
from src.validation.model_validator import (
    FairnessConfig,
    ModelValidationConfig,
    ModelValidator,
    validate_model,
)
from src.validation.schemas import (
    CheckResult,
    ColumnSchema,
    ValidationError,
    ValidationReport,
)

__all__ = [
    "CostConfig",
    "bayes_optimal_threshold",
    "cost_threshold_report",
    "min_cost_threshold",
    "DataValidator",
    "validate_data",
    "FairnessConfig",
    "ModelValidationConfig",
    "ModelValidator",
    "validate_model",
    "CheckResult",
    "ColumnSchema",
    "ValidationError",
    "ValidationReport",
]
