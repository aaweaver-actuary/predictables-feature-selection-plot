import pytest
import pandas as pd
import numpy as np

from predictables_feature_selection_plot.utils.add_glm_prediction import (
    _prepare_features__categorical,
    _fit_glm,
    _generate_predictions__categorical,
)
from predictables_feature_selection_plot.model.logistic_regression import (
    LogisticRegression,
)


@pytest.fixture
def sample_categorical_data():
    """Fixture for creating a sample dataframe with categorical data."""
    data = {
        "category": ["A", "B", "A", "C", "B", "C", "A", "C"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def fitted_glm(sample_categorical_data):
    """Fixture for creating a fitted GLM model on the sample categorical data."""
    df = sample_categorical_data
    X = _prepare_features__categorical(df, "category")
    y = df["target"]
    return _fit_glm(X, y)


def test_generate_predictions_categorical(sample_categorical_data, fitted_glm):
    df = sample_categorical_data
    glm = fitted_glm
    feature_name = "category"
    predictions = _generate_predictions__categorical(df, feature_name, glm)

    assert (
        predictions.shape[0] == df[feature_name].nunique()
    ), f"Expected {df[feature_name].nunique()} rows, got {predictions.shape[0]}"
    assert all(
        predictions["fitted_values"].notna()
    ), f"Expected all values in 'fitted_values' to be non-null, got {predictions['fitted_values'].isna().sum()}"


def test_generate_predictions_categorical_empty():
    df = pd.DataFrame({"category": [], "target": []})
    feature_name = "category"
    glm = LogisticRegression()

    with pytest.raises(ValueError):
        glm.fit(pd.DataFrame(), pd.Series())
        _generate_predictions__categorical(df, feature_name, glm)


def test_generate_predictions_categorical_with_nan(sample_categorical_data, fitted_glm):
    df = sample_categorical_data.copy()
    df.loc[0, "category"] = np.nan  # Introduce NaN value
    glm = fitted_glm
    feature_name = "category"

    predictions = _generate_predictions__categorical(df.dropna(), feature_name, glm)

    assert (
        df[feature_name].nunique(dropna=True) == predictions.shape[0]
    ), f"Expected {df[feature_name].nunique(dropna=True)} rows, got {predictions.shape[0]}"
    assert all(
        predictions["fitted_values"].notna()
    ), f"Expected all values in 'fitted_values' to be non-null, got {predictions['fitted_values'].isna().sum()}"


def test_generate_predictions_categorical_unseen_category(
    sample_categorical_data, fitted_glm
):
    df = sample_categorical_data.copy()
    df.loc[len(df)] = ["D", 1]  # Add unseen category
    glm = fitted_glm
    feature_name = "category"

    with pytest.raises(ValueError):
        _generate_predictions__categorical(df, feature_name, glm)
