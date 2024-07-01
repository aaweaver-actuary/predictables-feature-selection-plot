"""Define a logistic regression model."""

from __future__ import annotations

from predictables_feature_selection_plot.model.base_model import BaseLogisticRegression

from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf