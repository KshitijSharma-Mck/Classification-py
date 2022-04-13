"""
This is a boilerplate pipeline 'decision_tree'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import decision_tree
from ...General.metric_cal import metric_cal

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=decision_tree,
            inputs=["train", "test", "params:target_rename", "params:dt_hyperparameter"],
            outputs=["dt_object","dt_feature_imp","dt_predicted_train","dt_predicted_test"],
            name="DecisionTree"
        ),
        node(
            func=metric_cal,
            inputs=["train", "test", "params:target_rename", "dt_predicted_train", "dt_predicted_test"],
            outputs="decision_tree_metric",
            name="DecisionTreeMetricCal"

        )

    ])
