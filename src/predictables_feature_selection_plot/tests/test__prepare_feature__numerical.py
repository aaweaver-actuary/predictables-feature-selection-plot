import pytest
import pandas as pd
import numpy as np

from predictables_feature_selection_plot.utils.add_glm_prediction import (
    _prepare_features__numerical,
)


@pytest.fixture
def sample_numerical_dataframe():
    """Fixture for creating a sample dataframe with numerical data."""
    data = {"numeric_feature": [1, 2, 3, 4, 5, 6, 7, 8]}
    return pd.DataFrame(data)


@pytest.fixture
def dataframe_with_nan_numerical():
    """Fixture for creating a dataframe with numerical data and NaN values."""
    data = {"numeric_feature": [1, 2, np.nan, 4, 5, np.nan, 7, 8]}
    return pd.DataFrame(data)


@pytest.fixture
def empty_numerical_dataframe():
    """Fixture for creating an empty dataframe with numerical data."""
    data = {"numeric_feature": []}
    return pd.DataFrame(data)


def test_prepare_features_numerical(
    sample_numerical_dataframe, dataframe_with_nan_numerical, empty_numerical_dataframe
):
    test_cases = [
        (sample_numerical_dataframe, "numeric_feature", (8, 1)),
        (dataframe_with_nan_numerical, "numeric_feature", (8, 1)),
        (empty_numerical_dataframe, "numeric_feature", (0, 1)),
    ]

    for df, feature_name, expected_shape in test_cases:
        result = _prepare_features__numerical(df, feature_name)
        assert (
            result.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {result.shape}"
        assert (
            result[feature_name].dtype == float
        ), f"Expected dtype float, got {result[feature_name].dtype}"


def test_prepare_features_numerical_invalid_column(sample_numerical_dataframe):
    with pytest.raises(KeyError):
        _prepare_features__numerical(sample_numerical_dataframe, "non_existent_column")


def test_prepare_features_numerical_all_nan():
    df = pd.DataFrame({"numeric_feature": [np.nan, np.nan, np.nan]})
    result = _prepare_features__numerical(df, "numeric_feature")
    assert result.shape == (3, 1), f"Expected shape (3, 1), got {result.shape}"
    assert (
        result["numeric_feature"].isna().all()
    ), f"Expected all NaN values, got {result['numeric_feature']}"
