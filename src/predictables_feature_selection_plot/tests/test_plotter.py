# import unittest
# import pytest
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from predictables_feature_selection_plot.plotter import ProbabilityPlotter


# @pytest.fixture
# def data():
#     return pd.DataFrame(
#         {
#             "feature1": [0, 1, 2, 3, 4, 5],
#             "feature2": [5, 4, 3, 2, 1, 0],
#             "score": [0.1, 0.4, 0.35, 0.8, 0.65, 0.2],
#             "target": [0, 1, 0, 1, 0, 1],
#             "extra_info": ["A", "B", "C", "D", "E", "F"],
#         }
#     )


# @pytest.fixture
# def model(data):
#     return LogisticRegression().fit(data[["feature1"]], data["target"])


# def test_plot_probability_distribution(data, model):
#     plotter = ProbabilityPlotter(
#         data, model, "feature1", "score", "target", ["extra_info"]
#     )

#     fig = plotter.plot()
#     assert fig is not None


# def test_invalid_feature(data, model):
#     with pytest.raises(ValueError) as exc:
#         ProbabilityPlotter(data, model, "invalid_feature", "score", "target")
#     assert "Column 'invalid_feature' not found in data columns." in str(exc.value)


# def test_invalid_score(data, model):
#     with pytest.raises(ValueError) as exc:
#         ProbabilityPlotter(data, model, "feature1", "invalid_score", "target")
#     assert "Column 'invalid_score' not found in data columns." in str(exc.value)


# def test_invalid_target(data, model):
#     with pytest.raises(ValueError) as exc:
#         ProbabilityPlotter(data, model, "feature1", "score", "invalid_target")
#     assert "Column 'invalid_target' not found in data columns." in str(exc.value)


# def test_invalid_hover_data(data, model):
#     with pytest.raises(ValueError) as exc:
#         ProbabilityPlotter(
#             data, model, "feature1", "score", "target", ["invalid_column"]
#         )
#     assert "Column 'invalid_column' not found in data columns." in str(exc.value)
