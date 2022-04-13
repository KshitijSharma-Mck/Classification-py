"""
This is a boilerplate pipeline 'random_forest'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import rf_base
from ...General.metric_cal import metric_cal


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=rf_base,
            inputs=["train", "test", "params:target_rename", "params:rf_hyperparameters"],
            outputs=["rf_predicted_train", "rf_predicted_test", "rf_train_predict", "rf_test_predict",
                     "rf_object", "rf_feature_imp"],
            name="RandomForest"
        ),
        node(
            func=metric_cal,
            inputs=["train", "test", "params:target_rename", "rf_predicted_train", "rf_predicted_test"],
            outputs="rf_base_metrics",
            name="RandomForestMetrics"
        )
    ])
