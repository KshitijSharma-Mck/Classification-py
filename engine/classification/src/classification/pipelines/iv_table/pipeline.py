"""
This is a boilerplate pipeline 'iv_table'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import iv_table,post_iv_TrainTest

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=iv_table,
            inputs=["pre_iv_train", "params:target_rename", "params:top_iv", "params:top_iv_range", "params:bottom_iv_range"],
            outputs=["iv_table", "imp_iv_table","iv_col"],
            name="VariableSelection"
        ),
        node(
            func=post_iv_TrainTest,
            inputs=["pre_iv_train", "iv_col", "params:target_rename"],
            outputs="train",
            name="post_iv_train"
        ),
        node(
            func=post_iv_TrainTest,
            inputs=["pre_iv_test", "iv_col", "params:target_rename"],
            outputs="test",
            name="post_iv_test"
        )

    ])
