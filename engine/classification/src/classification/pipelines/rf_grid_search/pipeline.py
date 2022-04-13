"""
This is a boilerplate pipeline 'rf_grid_search'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import grid_cv
from ...General.metric_cal import metric_cal


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=grid_cv,
            inputs=["train", "test", "params:target_rename", "rf_object", "params:rf_grid_search_params"],
            outputs=["rfg_predicted_train", "rfg_predicted_test", "rf_grid_train_predict", "rf_grid_test_predict",
                     "rf_gridCV_object", "rf_grid_best_params"],
            name="RandomForestGridSearch"
        ),
        node(
            func=metric_cal,
            inputs=["train", "test", "params:target_rename", "rfg_predicted_train", "rfg_predicted_test"],
            outputs="rf_gridCV_metrics",
            name="RandomForestGridMetrics"
        )
    ])
