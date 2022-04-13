"""
This is a boilerplate pipeline 'dt_grid_search'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import grid_cv
from ...General.metric_cal import metric_cal


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=grid_cv,
            inputs=["train", "test", "params:target_rename", "dt_object", "params:dt_grid_search_params"],
            outputs=["dtg_predicted_train", "dtg_predicted_test", "dt_grid_train_predict", "dt_grid_test_predict",
                     "dt_gridCV_object", "dt_grid_best_params"],
            name="DecisionTreeGridSearch"
        ),
        node(
            func=metric_cal,
            inputs=["train", "test", "params:target_rename", "dtg_predicted_train", "dtg_predicted_test"],
            outputs="dt_gridCV_metrics",
            name="DecisionTreeGridMetrics"
        )
    ])
