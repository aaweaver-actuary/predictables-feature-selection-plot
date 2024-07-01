"""Define an interface for logistic regression models."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from enum import StrEnum

__all__ = ["BaseLogisticRegression", "ResidualType"]


class ResidualType(StrEnum):
    """Enum class for residual types."""

    DEVIANCE = "deviance"
    PEARSON = "pearson"
    STANDARDIZED = "standardized"


@dataclass
class BaseLogisticRegression(Protocol):
    """Base class for logistic regression models."""

    @property
    def coefficient_values(self) -> pd.Series:
        """Return the model coefficients."""
        ...

    @property
    def coefficient_standard_errors(self) -> pd.Series:
        """Return the standard errors of the model coefficients."""
        ...

    @property
    def coefficient_p_values(self) -> pd.DataFrame:
        """Return the p-values of the model coefficients."""
        ...

    @property
    def intercept(self) -> float:
        """Return the model intercept."""
        ...

    @property
    def summary(self) -> pd.DataFrame:
        """Return the model summary statistics."""
        ...

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return the predicted labels for the input data."""
        ...
