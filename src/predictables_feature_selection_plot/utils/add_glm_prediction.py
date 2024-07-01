"""Add a GLM prediction to a plotly figure."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.genmod.families.family import Binomial


def add_glm_prediction(
    fig: go.Figure, feature_name: str, df: pd.DataFrame
) -> go.Figure:
    """Add a GLM prediction to a plotly figure."""
    is_categorical = df[feature_name].dtype in ["object", "category"]

    X0 = (
        pd.concat(
            [df[feature_name], pd.get_dummies(df[feature_name], dtype="bool")], axis=1
        )
        if is_categorical
        else df[feature_name].astype(float)
    ).reset_index(drop=True)

    y = df.copy()["hit_count"].astype(float).reset_index(drop=True)

    for c in X0.columns.tolist():
        if c in ["hit_count"]:
            X0 = X0.drop(columns=[c])

    _ = X0.columns.tolist()[1:] if is_categorical else [0]

    glm = sm.GLM(y, X0.iloc[:, 1:] if is_categorical else X0, family=Binomial()).fit()

    if is_categorical:
        df1 = X0.copy().drop_duplicates().sort_values(feature_name)
        df1["fitted_values"] = glm.predict(df1.iloc[:, 1:])

        try:
            df1.iloc[:, 0] = df1.iloc[:, 0].astype(float)
            fig.add_scatter(
                x=df1.iloc[:, 0],
                y=df1["fitted_values"],
                name="Fitted GLM",
                line={"color": "black", "dash": "dash", "width": 4},
            )

        except Exception as _:
            fig.add_scatter(
                x=df1.iloc[:, 0],
                y=df1["fitted_values"],
                name="Fitted GLM",
                line={"color": "black", "dash": "dash", "width": 4},
            )
    else:
        df1 = pd.DataFrame(
            {
                feature_name: np.linspace(
                    min(df[feature_name].astype(float).min(), 0),
                    1.05 * df[feature_name].astype(float).max(),
                    100,
                )
            }
        )

        df1["fitted_values"] = glm.predict(df1[feature_name].astype(float))

        fig.add_scatter(
            x=df1[feature_name].astype(float),
            y=df1["fitted_values"],
            name="Fitted GLM",
            line={"color": "black", "dash": "dash", "width": 4},
        )

    return fig
