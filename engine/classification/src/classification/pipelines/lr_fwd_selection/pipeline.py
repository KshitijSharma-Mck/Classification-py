"""
This is a boilerplate pipeline 'lr_fwd_selection'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import lr_fb_selection, log_reg_base
from ...General.metric_cal import metric_cal


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=lr_fb_selection,
            inputs=["train", "test", "params:target_rename", "log_base_object", "params:lr_FwdBwd_no_of_features",
                    "params:lr_FwdBwd_score_criteria"],
            outputs=["filt_features_fwd", "filt_train_fwd", "filt_test_fwd"],
            name="featureSelectfNode"
        ),
        node(
            func=log_reg_base,
            inputs=["filt_train_bwd", "filt_test_bwd", "params:target_rename"],
            outputs=["lrf_train_predict", "lrf_test_predict", "lr_fwd_object","lr_fwd_predicted_train", "lr_fwd_predicted_test"],
            name="logRegFwd"
        ),
        node(
            func=metric_cal,
            inputs=["train", "test", "params:target_rename", "lr_fwd_predicted_train", "lr_fwd_predicted_test"],
            outputs="lr_fwd_metrics",
            name="logRegFwdMetricCal"
        )
    ])
