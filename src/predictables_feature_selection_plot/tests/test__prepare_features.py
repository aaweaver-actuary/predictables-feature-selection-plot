import pytest
import pandas as pd
import numpy as np
from typing import Tuple

from predictables_feature_selection_plot.utils.add_glm_prediction import (
    _prepare_features__categorical,
    _prepare_features__numerical,
    prepare_features,
)


@pytest.fixture
def sample_dataframe():
    """Fixture for creating a sample dataframe with both categorical and numerical data."""
    data = {
        "category": ["A", "B", "A", "C", "B", "C", "A", "C"],
        "numeric_feature": [1, 2, 3, 4, 5, 6, 7, 8],
    }
    return pd.DataFrame(data)


@pytest.fixture
def dataframe_with_nan():
    """Fixture for creating a dataframe with NaN values."""
    data = {
        "category": ["A", "B", np.nan, "C", "B", "C", "A", np.nan],
        "numeric_feature": [1, 2, np.nan, 4, 5, np.nan, 7, 8],
    }
    return pd.DataFrame(data)


@pytest.fixture
def empty_dataframe():
    """Fixture for creating an empty dataframe."""
    data = {"category": [], "numeric_feature": []}
    return pd.DataFrame(data)


def test_prepare_features_categorical(
    sample_dataframe, dataframe_with_nan, empty_dataframe
):
    test_cases = [
        (sample_dataframe, "category", ["B", "C"], (8, 2)),
        (dataframe_with_nan, "category", ["B", "C"], (8, 2)),
        (empty_dataframe, "category", [], (0, 0)),
    ]

    for df, feature_name, expected_columns, expected_shape in test_cases:
        result = _prepare_features__categorical(df, feature_name)
        assert (
            list(result.columns) == expected_columns
        ), f"Expected columns {expected_columns}, got {list(result.columns)}"
        assert (
            result.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {result.shape}"


def test_prepare_features_numerical(
    sample_dataframe, dataframe_with_nan, empty_dataframe
):
    test_cases = [
        (sample_dataframe, "numeric_feature", (8, 1)),
        (dataframe_with_nan, "numeric_feature", (8, 1)),
        (empty_dataframe, "numeric_feature", (0, 1)),
    ]

    for df, feature_name, expected_shape in test_cases:
        result = _prepare_features__numerical(df, feature_name)
        assert (
            result.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {result.shape}"
        assert (
            result[feature_name].dtype == float
        ), f"Expected dtype float, got {result[feature_name].dtype}"


def test_prepare_features_invalid_column(sample_dataframe):
    with pytest.raises(KeyError):
        _prepare_features__categorical(sample_dataframe, "non_existent_column")

    with pytest.raises(KeyError):
        _prepare_features__numerical(sample_dataframe, "non_existent_column")


def test_prepare_features_all_nan():
    df = pd.DataFrame(
        {
            "category": [np.nan, np.nan, np.nan],
            "numeric_feature": [np.nan, np.nan, np.nan],
        }
    )

    result = _prepare_features__categorical(df, "category")
    assert result.empty, f"Expected empty dataframe, got {result}"

    result = _prepare_features__numerical(df, "numeric_feature")
    assert result.shape == (3, 1), f"Expected shape (3, 1), got {result.shape}"
    assert (
        result["numeric_feature"].isna().all()
    ), f"Expected all NaN values, got {result['numeric_feature']}"


def test_prepare_features(sample_dataframe, dataframe_with_nan, empty_dataframe):
    test_cases = [
        (sample_dataframe, "category", (8, 2), True),
        (sample_dataframe, "numeric_feature", (8, 1), False),
        (dataframe_with_nan, "category", (8, 2), True),
        (dataframe_with_nan, "numeric_feature", (8, 1), False),
        (empty_dataframe, "category", (0, 1), False),
        (empty_dataframe, "numeric_feature", (0, 1), False),
    ]

    for counter, (df, feature_name, expected_shape, is_categorical) in enumerate(
        test_cases
    ):
        result, cat_flag = prepare_features(df, feature_name)
        assert (
            result.shape == expected_shape
        ), f"For {feature_name}, counter {counter + 1}, expected shape {expected_shape}, got {result.shape}"
        assert (
            cat_flag == is_categorical
        ), f"For {feature_name}, counter {counter + 1}, expected categorical flag {is_categorical}, got {cat_flag}"


def test_prepare_features_invalid_column_main(sample_dataframe):
    with pytest.raises(KeyError):
        prepare_features(sample_dataframe, "non_existent_column")
