"""
This is a boilerplate pipeline 'rf_random_search'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import rf_randomsearch
from ...General.metric_cal import metric_cal

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=rf_randomsearch,
            inputs=["train", "test", "params:target_rename", "params:rf_random_search_params"],
            outputs=["rf_randomsearch_object", "rf_randomsearch_hyper_params", "rf_rd_predicted_train",
                     "rf_rd_predicted_test", "rf_rd_train_predict", "rf_rd_test_predict"],
            name="rfRandomSearch"
        ),
        node(
            func=metric_cal,
            inputs=["train", "test", "params:target_rename", "rf_rd_predicted_train", "rf_rd_predicted_test"],
            outputs="rf_randomsearch_metric",
            name="RandomForestRandomSearchMetrics"

        )
    ])
