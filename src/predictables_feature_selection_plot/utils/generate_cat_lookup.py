"""Utility function to generate a lookup dictionary for categorical features."""

import pandas as pd

__all__ = ["generate_cat_lookup"]


def generate_cat_lookup(df: pd.DataFrame, feature_name: str) -> dict:
    """Generate a lookup dictionary for categorical features."""
    x_ = (
        pd.DataFrame(
            {"categories": df[feature_name], "codes": df[feature_name].cat.codes}
        )
        .drop_duplicates()
        .sort_values("codes")
        .reset_index(drop=True)
    )

    return dict(zip(x_["codes"], x_["categories"]))
