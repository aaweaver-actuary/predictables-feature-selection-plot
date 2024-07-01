"""Add a GLM prediction to a plotly figure."""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple
import logging
from predictables_feature_selection_plot.model.logistic_regression import (
    LogisticRegression,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="./add_glm_prediction.log",
    filemode="w",
)

logger = logging.getLogger(__name__)


def _prepare_features__categorical(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """
    Prepare categorical features for GLM.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    feature_name : str
        The name of the feature column in the dataframe.

    Returns
    -------
    pd.DataFrame
        The prepared feature matrix.
    """
    logger.debug(f"Preparing categorical feature for GLM: {feature_name}")
    X = pd.get_dummies(df[feature_name], drop_first=True).reset_index(drop=True)
    logger.debug(f"Since categorical, dummifying feature {feature_name}:\n{X.head()}")
    return X


def _prepare_features__numerical(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """
    Prepare numerical features for GLM.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    feature_name : str
        The name of the feature column in the dataframe.

    Returns
    -------
    pd.DataFrame
        The prepared feature matrix.
    """
    logger.debug(f"Preparing numerical feature for GLM: {feature_name}")
    X = df[[feature_name]].astype(float).reset_index(drop=True)
    logger.debug(
        f"Since numerical, converting feature {feature_name} to float:\n{X.head()}"
    )
    return X


def prepare_features(df: pd.DataFrame, feature_name: str) -> Tuple[pd.DataFrame, bool]:
    """
    Prepare features for GLM, handling categorical and numerical data.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    feature_name : str
        The name of the feature column in the dataframe.

    Returns
    -------
    Tuple[pd.DataFrame, bool]
        A tuple containing the prepared feature dataframe and a boolean indicating if the feature is categorical.
    """
    logger.debug(f"Preparing features for GLM: {feature_name}")
    is_categorical = df[feature_name].dtype in ["object", "category"]

    logger.debug(f"Feature {feature_name} is categorical: {is_categorical}")

    X = (
        _prepare_features__categorical(df, feature_name)
        if is_categorical
        else _prepare_features__numerical(df, feature_name)
    )

    return X, is_categorical


def _fit_glm(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """
    Fit a Generalized Linear Model (GLM) to the data.

    Parameters
    ----------
    X : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The target variable.

    Returns
    -------
    sm.GLM
        The fitted GLM model.
    """
    logger.debug(f"Attempting to fit GLM model on features:\n{X.head()}")
    try:
        # Ensure input is float
        X = X.astype(float)
        glm = LogisticRegression()
        glm.fit(X, y)
        logger.debug("GLM model fitted successfully.")
        return glm
    except Exception as e:
        logger.error(f"Error fitting GLM model: {e}")
        raise e


def _generate_predictions__categorical(
    df: pd.DataFrame, feature_name: str, glm: LogisticRegression
) -> pd.DataFrame:
    """
    Generate predictions from the GLM model for a categorical feature.

    Parameters
    ----------
    df : pd.DataFrame
        The original dataframe.
    feature_name : str
        The name of the feature column.
    glm : LogisticRegression
        The fitted GLM model.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the feature values and the corresponding fitted values.

    Raises
    ------
    ValueError
        If the dataframe is empty.
    ValueError
        If the feature values contain NaN values.
    ValueError
        If the glm is not already fitted.
    """
    if df.empty:
        raise ValueError("Dataframe is empty.")

    if df[feature_name].isna().sum() > 0:
        raise ValueError("Feature values contain NaN values.")

    if not glm:
        raise ValueError("GLM model is not fitted.")

    if not isinstance(glm, LogisticRegression):
        # if not isinstance(glm, sm.regression.linear_model.RegressionResultsWrapper):
        raise ValueError("GLM model is not a fitted model type.")

    df_unique = df[feature_name].drop_duplicates().sort_values().reset_index(drop=True)
    X_new = pd.get_dummies(df_unique, drop_first=True, dtype="int")

    df_unique["fitted_values"] = glm.predict(X_new)
    return df_unique


def _generate_predictions__numerical(
    df: pd.DataFrame, feature_name: str, glm: LogisticRegression
) -> pd.DataFrame:
    """
    Generate predictions from the GLM model for a numerical feature.

    Parameters
    ----------
    df : pd.DataFrame
        The original dataframe.
    feature_name : str
        The name of the feature column.
    glm : LogisticRegression
        The fitted GLM model.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the feature values and the corresponding fitted values.
    """
    df_unique = pd.DataFrame(
        {feature_name: np.linspace(df[feature_name].min(), df[feature_name].max(), 100)}
    )
    X_new = df_unique

    df_unique["fitted_values"] = glm.predict(X_new)
    return df_unique


def generate_predictions(
    df: pd.DataFrame, feature_name: str, glm: LogisticRegression, is_categorical: bool
) -> pd.DataFrame:
    """
    Generate predictions from the GLM model.

    Parameters
    ----------
    df : pd.DataFrame
        The original dataframe.
    feature_name : str
        The name of the feature column.
    glm : LogisticRegression
        The fitted GLM model.
    is_categorical : bool
        A boolean indicating if the feature is categorical.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the feature values and the corresponding fitted values.
    """
    params = {"df": df, "feature_name": feature_name, "glm": glm}
    df_unique = (
        _generate_predictions__categorical(**params)
        if is_categorical
        else _generate_predictions__numerical(**params)
    )

    df_unique["fitted_values"] = glm.predict(df_unique)
    return df_unique


def add_glm_trace(fig: go.Figure, df: pd.DataFrame, feature_name: str) -> go.Figure:
    """
    Add the GLM prediction trace to the Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to which the GLM prediction will be added.
    df : pd.DataFrame
        The dataframe containing the feature values and fitted values.
    feature_name : str
        The name of the feature column in the dataframe.

    Returns
    -------
    go.Figure
        The Plotly figure with the added GLM prediction trace.
    """
    fig.add_scatter(
        x=df[feature_name],
        y=df["fitted_values"],
        name="Fitted GLM",
        line={"color": "black", "dash": "dash", "width": 4},
    )
    return fig


def add_glm_prediction(
    fig: go.Figure, feature_name: str, target_name: str, df: pd.DataFrame
) -> go.Figure:
    """
    Add a GLM prediction to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to which the GLM prediction will be added.
    feature_name : str
        The name of the feature column in the dataframe.
    target_name : str
        The name of the target column in the dataframe.
    df : pd.DataFrame
        The dataframe containing the data.

    Returns
    -------
    go.Figure
        The Plotly figure with the added GLM prediction trace.
    """
    # Ensure binary variables are converted to numeric
    df[target_name] = df[target_name].astype(float)
    if df[feature_name].dtype == object:
        df[feature_name] = df[feature_name].apply(
            lambda x: 1.0 if x == "1" else 0.0 if x == "0" else x
        )

    X, is_categorical = prepare_features(df, feature_name)
    y = df[target_name].astype(float).reset_index(drop=True)

    glm = _fit_glm(X, y)
    predictions = generate_predictions(df, feature_name, glm, is_categorical)

    fig = add_glm_trace(fig, predictions, feature_name)
    return fig


# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import statsmodels.api as sm
# from statsmodels.genmod.families.family import Binomial


# def add_glm_prediction(
#     fig: go.Figure, feature_name: str, target_name: str, df: pd.DataFrame
# ) -> go.Figure:
#     """Add a GLM prediction to a plotly figure."""
#     is_categorical = df[feature_name].dtype in ["object", "category"]

#     X0 = (
#         pd.concat(
#             [df[feature_name], pd.get_dummies(df[feature_name], dtype="bool")], axis=1
#         )
#         if is_categorical
#         else df[feature_name].astype(float)
#     ).reset_index(drop=True)

#     y = df.copy()[target_name].astype(float).reset_index(drop=True)

#     # for c in X0.columns.tolist():
#     #     if c in [target_name]:
#     #         X0 = X0.drop(columns=[c])

#     _ = X0.columns.tolist()[1:] if is_categorical else [0]

#     glm = sm.GLM(y, X0.iloc[:, 1:] if is_categorical else X0, family=Binomial()).fit()

#     if is_categorical:
#         df1 = X0.copy().drop_duplicates().sort_values(feature_name)
#         df1["fitted_values"] = glm.predict(df1.iloc[:, 1:])

#         try:
#             df1.iloc[:, 0] = df1.iloc[:, 0].astype(float)
#             fig.add_scatter(
#                 x=df1.iloc[:, 0],
#                 y=df1["fitted_values"],
#                 name="Fitted GLM",
#                 line={"color": "black", "dash": "dash", "width": 4},
#             )

#         except Exception as _:
#             fig.add_scatter(
#                 x=df1.iloc[:, 0],
#                 y=df1["fitted_values"],
#                 name="Fitted GLM",
#                 line={"color": "black", "dash": "dash", "width": 4},
#             )
#     else:
#         df1 = pd.DataFrame(
#             {
#                 feature_name: np.linspace(
#                     min(df[feature_name].astype(float).min(), 0),
#                     1.05 * df[feature_name].astype(float).max(),
#                     100,
#                 )
#             }
#         )

#         df1["fitted_values"] = glm.predict(df1[feature_name].astype(float))

#         fig.add_scatter(
#             x=df1[feature_name].astype(float),
#             y=df1["fitted_values"],
#             name="Fitted GLM",
#             line={"color": "black", "dash": "dash", "width": 4},
#         )

#     return fig
