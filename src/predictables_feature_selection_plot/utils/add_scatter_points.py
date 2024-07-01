"""Add scatter points to a plotly figure."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def add_scatter_points(
    fig: go.Figure,
    df: pd.DataFrame,
    feature_name: str,
    color: str,
    label_filter: str,
    jitter_fct: float = 1,
    label_title: str = "Actual Hit Count",
    label_name: str = "hit_count",
    preds_name: str = "preds",
) -> go.Figure:
    """Add scatter points to a plotly figure."""
    jitter = np.abs(np.random.default_rng(42).normal(0, 0.1, df.shape[0]))

    if df[feature_name].dtype == "category":
        xx = df[feature_name].cat.codes.to_numpy().astype(float) + (jitter * jitter_fct)
    else:
        xx = df[feature_name].to_numpy().astype(float) + (jitter * jitter_fct)

    fig.add_scatter(
        x=xx,
        y=df[preds_name].to_numpy(),
        mode="markers",
        name=label_filter,
        legendgroup=label_name,
        legendgrouptitle={"text": label_title},
        marker={
            "color": color,
            "opacity": 0.6,
            "line": {"color": "black", "width": 0.5},
        },
        showlegend=True,
    )

    return fig
