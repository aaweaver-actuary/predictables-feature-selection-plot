"""Utility functions for predictables_feature_selection_plot module."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import List, Tuple
import logging

__all__ = ["validate_columns", "generate_decision_boundary"]


def validate_columns(data: pd.DataFrame, columns: List[str]) -> None:
    """Validate that the specified columns are present in the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input data.
    columns : List[str]
        List of column names to validate.

    Raises
    ------
    ValueError
        If any column is not found in the data.
    """
    for col in columns:
        if col not in data.columns:
            logging.error(f"Column '{col}' not found in data columns.")
            raise ValueError(f"Column '{col}' not found in data columns.")


def generate_decision_boundary(
    data: pd.DataFrame,
    model: BaseEstimator,
    features: list[str],
    coefficients: list[str],
    score: str,
    feature_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate decision boundary predictions for a grid of feature values.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing the features.
    model : BaseEstimator
        The fitted classification model.
    features : list[str]
        The feature column name or names to generate the boundary for.
    coefficients : list[str]
        The coefficients of the model.
    score : str
        The score column name for grid range.
    feature_name : str
        The name of the feature column

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays of x_range, y_range, and predicted probabilities for the grid.
    """
    try:
        x_min, x_max = (data["preds"].min(), data["preds"].max())
        y_min, y_max = data[score].min(), data[score].max()
        x_range = np.linspace(x_min, x_max, 250)
        y_range = np.linspace(y_min, y_max, 250)
        xx, yy = np.meshgrid(x_range, y_range)

        grid_points = pd.DataFrame(
            {feature_name.replace("[", "").replace("]", ""): xx.ravel()}
        )
        grid_points["decision_boundary"] = model.predict_proba(grid_points)[:, 1]

        return x_range, y_range, grid_points
    except Exception as e:
        logging.error(f"Error generating decision boundary: {e}")
        raise
