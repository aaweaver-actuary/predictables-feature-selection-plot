import pytest
import pandas as pd
import numpy as np

from predictables_feature_selection_plot.utils.add_glm_prediction import (
    _prepare_features__categorical,
)


@pytest.fixture
def sample_dataframe():
    """Fixture for creating a sample dataframe."""
    data = {"category": ["A", "B", "A", "C", "B", "C", "A", "C"]}
    return pd.DataFrame(data)


@pytest.fixture
def dataframe_with_nan():
    """Fixture for creating a dataframe with NaN values."""
    data = {"category": ["A", "B", np.nan, "C", "B", "C", "A", np.nan]}
    return pd.DataFrame(data)


@pytest.fixture
def empty_dataframe():
    """Fixture for creating an empty dataframe."""
    data = {"category": []}
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    "df_fixture, feature_name, expected_columns",
    [
        ("sample_dataframe", "category", ["B", "C"]),
        ("dataframe_with_nan", "category", ["B", "C"]),
        ("empty_dataframe", "category", []),
    ],
)
def test_prepare_features_categorical(
    df_fixture, feature_name, expected_columns, request
):
    df = request.getfixturevalue(df_fixture)
    result = _prepare_features__categorical(df, feature_name)
    assert list(result.columns) == expected_columns
    assert result.shape[0] == df.shape[0]


@pytest.mark.parametrize(
    "df_fixture, feature_name, expected_shape",
    [
        ("sample_dataframe", "category", (8, 2)),
        ("dataframe_with_nan", "category", (8, 2)),
        ("empty_dataframe", "category", (0, 0)),
    ],
)
def test_prepare_features_categorical_shape(
    df_fixture, feature_name, expected_shape, request
):
    df = request.getfixturevalue(df_fixture)
    result = _prepare_features__categorical(df, feature_name)
    assert result.shape == expected_shape


def test_prepare_features_categorical_invalid_column(sample_dataframe):
    with pytest.raises(KeyError):
        _prepare_features__categorical(sample_dataframe, "non_existent_column")


def test_prepare_features_categorical_all_nan():
    df = pd.DataFrame({"category": [np.nan, np.nan, np.nan]})
    result = _prepare_features__categorical(df, "category")
    assert result.empty
