"""Define a logistic regression model."""

from __future__ import annotations
from typing import Protocol
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from dataclasses import dataclass

__all__ = ["LogisticRegressionInterface", "LogisticRegression"]


@dataclass
class LogisticRegressionInterface(Protocol):
    """Interface for a logistic regression model."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model to the input data."""
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predicted labels for the input data."""
        ...

    @property
    def coef_(self) -> np.ndarray:
        """Return the model coefficients."""
        ...

    @property
    def intercept_(self) -> float:
        """Return the model intercept."""
        ...


@dataclass
class LogisticRegression(LogisticRegressionInterface):
    """A logistic regression model."""

    model: _LogisticRegression | None = None

    def __post_init__(self):
        """Initialize the logistic regression model."""
        self.model = _LogisticRegression(penalty=None, solver="lbfgs")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model to the input data."""
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predicted labels for the input data."""
        return self.model.predict(X)

    @property
    def coef(self) -> np.ndarray:
        """Return the model coefficients."""
        if not hasattr(self.model, "coef_"):
            raise ValueError("Model has not been fitted.")
        return self.model.coef_

    @property
    def intercept(self) -> float:
        """Return the model intercept."""
        if not hasattr(self.model, "intercept_"):
            raise ValueError("Model has not been fitted.")
        return self.model.intercept_
