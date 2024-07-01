import pytest
import pandas as pd
import numpy as np

from predictables_feature_selection_plot.utils.add_glm_prediction import _fit_glm
from predictables_feature_selection_plot.model.logistic_regression import (
    LogisticRegression,
)


@pytest.fixture
def sample_data():
    """Fixture for creating a sample feature matrix and target variable."""
    X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]})
    y = pd.Series([0, 1, 0, 1, 0])
    return X, y


@pytest.fixture
def data_with_nan():
    """Fixture for creating data with NaN values."""
    X = pd.DataFrame(
        {"feature1": [1, 2, np.nan, 4, 5], "feature2": [5, np.nan, 3, 2, 1]}
    )
    y = pd.Series([0, 1, 0, 1, np.nan])
    return X, y


@pytest.fixture
def data_with_non_numeric():
    """Fixture for creating data with non-numeric values."""
    X = pd.DataFrame(
        {"feature1": ["A", "B", "C", "D", "E"], "feature2": ["F", "G", "H", "I", "J"]}
    )
    y = pd.Series([0, 1, 0, 1, 0])
    return X, y


def test_fit_glm_sample_data(sample_data):
    X, y = sample_data
    model = _fit_glm(X, y)
    assert isinstance(
        model, LogisticRegression
    ), f"Expected model to be of type LogisticRegression, got {type(model)}"
    assert hasattr(
        model, "params"
    ), f"Expected model to have params attribute, but it only has {dir(model)}"


def test_fit_glm_data_with_nan(data_with_nan):
    X, y = data_with_nan
    with pytest.raises(Exception):  # noqa: B017
        _fit_glm(X, y)


def test_fit_glm_data_with_non_numeric(data_with_non_numeric):
    X, y = data_with_non_numeric
    with pytest.raises(ValueError):
        _fit_glm(X, y)


def test_fit_glm_empty_data():
    X = pd.DataFrame({"feature1": [], "feature2": []})
    y = pd.Series([])
    with pytest.raises(ValueError):
        _fit_glm(X, y)


def test_fit_glm_mismatched_lengths():
    X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]})
    y = pd.Series([0, 1, 0, 1])  # Mismatched length
    with pytest.raises(ValueError):
        _fit_glm(X, y)


def test_fit_glm_constant_feature():
    X = pd.DataFrame(
        {
            "feature1": [1, 1, 1, 1, 1],  # Constant feature
            "feature2": [5, 4, 3, 2, 1],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0])
    model = _fit_glm(X, y)
    assert isinstance(
        model, LogisticRegression
    ), f"Expected model to be of type GLM, got {type(model)}"
    assert hasattr(
        model, "params"
    ), f"Expected model to have params attribute, but it only has {dir(model)}"


def test_fit_glm_single_feature():
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5]  # Single feature
        }
    )
    y = pd.Series([0, 1, 0, 1, 0])
    model = _fit_glm(X, y)
    assert isinstance(
        model, LogisticRegression
    ), f"Expected model to be of type LogisticRegression, got {type(model)}"
    assert hasattr(
        model, "params"
    ), f"Expected model to have params attribute, but it only has {dir(model)}"
