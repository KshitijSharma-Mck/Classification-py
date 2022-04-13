"""
This is a boilerplate pipeline 'logbase'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import log_reg_base
from ...General.metric_cal import metric_cal


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=log_reg_base,
            inputs=["train", "test", "params:target_rename"],
            outputs=["lr_train_predict", "lr_test_predict", "log_base_object", "lr_predicted_train",
                     "lr_predicted_test"],
            name="logRegBase"
        ),
        node(
            func=metric_cal,
            inputs=["train", "test", "params:target_rename", "lr_predicted_train", "lr_predicted_test"],
            outputs="log_base_metrics",
            name="logRegBaseMetricCal"
        )
    ])
