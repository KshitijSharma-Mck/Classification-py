"""
This is a boilerplate pipeline 'dt_random_search'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import dt_randomsearch
from ...General.metric_cal import metric_cal


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=dt_randomsearch,
            inputs=["train", "test", "params:target_rename", "params:dt_random_search_params"],
            outputs=["dt_randomsearch_object", "dt_randomsearch_hyper_params", "dt_rd_predicted_train",
                     "dt_rd_predicted_test","dt_rd_train_predict","dt_rd_test_predict"],
            name="dtRandomSearch"
        ),
        node(
            func=metric_cal,
            inputs=["train", "test", "params:target_rename", "dt_rd_predicted_train", "dt_rd_predicted_test"],
            outputs="dt_randomsearch_metric",
            name="DecisionTreeRandomSearchMetrics"

        )

    ])
