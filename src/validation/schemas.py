"""Shared data types for the validation harness.

CheckResult, ValidationReport, ColumnSchema, and ValidationError are the only
types the rest of the codebase needs to import from this module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when one or more error-severity validation checks fail."""


@dataclass(frozen=True)
class CheckResult:
    """Result of a single named validation check."""

    name: str
    passed: bool
    severity: Literal["error", "warning", "info"]
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    """Structured pass/fail report produced by any validate_* call.

    Attributes:
        passed: False if any error-severity check failed.
        checks: All checks run, ordered by execution.
        run_at: ISO-8601 UTC timestamp.
    """

    passed: bool
    checks: list[CheckResult]
    run_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    def assert_passed(self) -> None:
        """Raise ValidationError listing all error-severity failures.

        Raises:
            ValidationError: If any check has severity='error' and passed=False.
        """
        if not self.passed:
            lines = [
                f"  [{c.severity.upper()}] {c.name}: {c.message}"
                for c in self.errors
            ]
            raise ValidationError(
                f"Validation failed with {len(self.errors)} error(s):\n"
                + "\n".join(lines)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "run_at": self.run_at,
            "n_checks": len(self.checks),
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "checks": [c.to_dict() for c in self.checks],
        }

    def log_summary(self, logger_instance: logging.Logger | None = None) -> None:
        """Log a one-line summary plus any failures at WARNING/ERROR level."""
        lg = logger_instance or logger
        status = "PASSED" if self.passed else "FAILED"
        lg.info(
            "Validation %s | %d checks | %d errors | %d warnings",
            status,
            len(self.checks),
            len(self.errors),
            len(self.warnings),
        )
        for check in self.checks:
            if not check.passed:
                level = logging.ERROR if check.severity == "error" else logging.WARNING
                lg.log(level, "  %s: %s", check.name, check.message)


@dataclass(frozen=True)
class ColumnSchema:
    """Expected schema constraints for a single column.

    Attributes:
        name: Column name.
        nullable: Whether NaN is allowed (defaults to False).
        min_val: Inclusive lower bound for numeric values.
        max_val: Inclusive upper bound for numeric values.
        allowed_values: Exhaustive set of valid values for categorical columns.
        max_missing_rate: Maximum fraction of NaN allowed (0.0 means no nulls).
    """

    name: str
    nullable: bool = False
    min_val: float | None = None
    max_val: float | None = None
    allowed_values: list[Any] | None = None
    max_missing_rate: float = 0.0
