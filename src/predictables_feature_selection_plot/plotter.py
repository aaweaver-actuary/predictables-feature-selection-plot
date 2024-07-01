"""Defines the ProbabilityPlotter class for plotting the probability distribution of a fitted fitted_model."""

from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import BaseEstimator
from typing import List

from sklearn.linear_model import LogisticRegression
from predictables_feature_selection_plot.utils import (
    validate_columns,
    generate_decision_boundary,
)
import logging

__all__ = ["ProbabilityPlotter"]


class ProbabilityPlotter:
    """Class for plotting the probability distribution of a fitted fitted_model.

    Attributes
    ----------
    data : pd.DataFrame
        The input data containing the features, scores, and target.
    fitted_model : BaseEstimator
        The fitted classification model.
    feature : str
        The feature column name to plot on the x-axis.
    score : str
        The score column name containing the predicted probabilities.
    target : str
        The target column name containing the actual binary target values.
    hover_data : Optional[List[str]]
        Additional columns to include in the hover information.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        fitted_model: BaseEstimator,
        features: str | list[str],
        score: str,
        target: str,
        feature_name: str = "feature",
        test_model: BaseEstimator = LogisticRegression,
        hover_data: List[str] | None = None,
    ):
        self.data = data
        self.fitted_model = fitted_model
        self.test_model = test_model()
        self.features = [features] if isinstance(features, str) else features
        self.feature_name = feature_name
        self.score = score
        self.target = target
        self.hover_data = hover_data
        self._validate_input()

        self.is_fit = False

    def _fit_test_model(self) -> BaseEstimator:
        """Fit the blank model to the feature columns."""
        try:
            fitted = self.test_model.fit(
                self.data[self.features], self.data[self.target]
            )

            self.is_fit = True
            self.data["decision_boundary"] = fitted.predict_proba(
                self.data[self.features]
            )[:, 1]
            return fitted
        except Exception as e:
            logging.error(f"Error fitting blank model: {e}")
            raise

    def _get_fitted_coefficients(self) -> pd.Series:
        """Return the fitted coefficients for the test model."""
        try:
            if not self.is_fit:
                self.test_model = self._fit_test_model()

            return pd.Series(
                self.test_model.coef_[0],
                index=self.features,
                name="Fitted Coefficients",
            )
        except Exception as e:
            logging.error(f"Error getting fitted coefficients: {e}")
            raise

    def _validate_input(self) -> None:
        """Validate that the specified columns are present in the DataFrame."""
        try:
            validate_columns(
                self.data,
                self.features + [self.score, self.target] + (self.hover_data or []),
            )
        except ValueError as e:
            logging.error(f"Validation error: {e}")
            raise

    def plot(self) -> go.Figure:
        """Create the probability distribution plot.

        Returns
        -------
        go.Figure
            The plotly scatter plot object with shaded decision boundary and hover information.
        """
        try:
            fig = self._create_scatter_plot()
            return self._add_decision_boundary_shading(fig)
        except Exception as e:
            logging.error(f"Error creating plot: {e}")
            raise

    def _make_feature_name(self) -> str:
        """Create the feature name for the grid plot.

        Returns
        -------
        str
            The formatted feature name.
        """
        if len(self.features) == 1:
            return f"[{self.features[0]}]"
        else:
            return "[" + "] + [".join(self.features) + "]"

    def _make_title(self) -> str:
        """Create the plot title based on the feature column name.

        Returns
        -------
        str
            The plot title.
        """
        return f"Probability Distribution for {self._make_feature_name()}"

    def _make_feature(self) -> pd.Series:
        """Create the feature column as the linear combination of the input features and modeled coefficients."""
        return pd.Series(
            self.data[self.features].dot(self._get_fitted_coefficients()).to_numpy(),
            name=self._make_feature_name(),
        )

    def _create_scatter_plot(self) -> go.Figure:
        """Create a scatter plot with hover information.

        Returns
        -------
        go.Figure
            The plotly scatter plot object.
        """
        beta_str = ""
        for f in self.features:
            beta_str += f"{self._get_fitted_coefficients()[f]:.3f} * [{f}]" + (
                " + " if f != self.features[-1] else ""
            )

        try:
            return px.scatter(
                self.data.assign(tempx=self._make_feature()).rename(
                    columns={
                        self.score: "Modeled Probability",
                        self.target: "Actual Hit Count",
                        "tempx": self._make_feature_name(),
                    }
                ),
                x=self._make_feature_name(),
                y="Modeled Probability",
                color="Actual Hit Count",
                hover_data=self.hover_data,
                labels={"x": self._make_feature_name(), "y": "Modeled Probability"},
                title=f"<b>{self._make_title()}</b><br>{beta_str}",
                range_y=[0, 1],
            )
        except Exception as e:
            logging.error(f"Error creating scatter plot: {e}")
            raise

    def _add_decision_boundary_shading(self, fig: go.Figure) -> go.Figure:
        """Add background shading to the plot based on the fitted_model's decision boundary.

        Parameters
        ----------
        fig (go.Figure)
            The plotly scatter plot object.

        Returns
        -------
        go.Figure
            The plotly scatter plot object with the decision boundary shading added.
        """
        # Fit the test model to generate the decision boundary
        self._fit_test_model()

        df = pd.concat(
            [
                self.data,
                pd.Series(
                    self.data[self.features].dot(self._get_fitted_coefficients()),
                    name="preds",
                ),
            ],
            axis=1,
        )

        try:
            x_range, y_range, boundary_df = generate_decision_boundary(
                df,
                self.test_model,
                self.features,
                self._get_fitted_coefficients(),
                self.score,
                self._make_feature_name(),
            )
            xx, yy = np.meshgrid(x_range, y_range)

            fig.add_trace(
                go.Contour(
                    x=x_range,
                    y=y_range,
                    z=boundary_df["decision_boundary"].to_numpy().reshape(xx.shape),
                    showscale=False,
                    colorscale=[
                        [0, "rgba(0,0,255,0.2)"],
                        [0.5, "rgba(0,0,255,0.2)"],
                        [0.5, "rgba(255,0,0,0.2)"],
                        [1, "rgba(255,0,0,0.2)"],
                    ],
                    hoverinfo="skip",
                )
            )

            

            return fig
        except Exception as e:
            logging.error(f"Error adding decision boundary shading: {e}")
            raise
