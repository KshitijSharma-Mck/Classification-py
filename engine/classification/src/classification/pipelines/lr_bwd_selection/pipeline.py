"""
This is a boilerplate pipeline 'lr_bwd_selection'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import lr_bwd_selection, log_reg_base
from ...General.metric_cal import metric_cal

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=lr_bwd_selection,
            inputs=["train", "test", "params:target_rename", "log_base_object", "params:lr_FwdBwd_no_of_features",
                    "params:lr_FwdBwd_score_criteria"],
            outputs=["filt_features_bwd", "filt_train_bwd", "filt_test_bwd"],
            name="featureSelectbNode"
        ),
        node(
            func=log_reg_base,
            inputs=["filt_train_bwd", "filt_test_bwd", "params:target_rename"],
            outputs=["lrb_train_predict", "lrb_test_predict", "lr_bwd_object","lr_bwd_predicted_train","lr_bwd_predicted_test"],
            name="logRegBack"
        ),
        node(
            func=metric_cal,
            inputs=["train", "test", "params:target_rename", "lr_bwd_predicted_train", "lr_bwd_predicted_test"],
            outputs="lr_bwd_metrics",
            name="logRegBackMetricCal"

        )
    ])
