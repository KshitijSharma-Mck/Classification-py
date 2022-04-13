"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_data, remove_duplicate, fill_rate, split_train_test, save_uid, missing_value_imputation, \
    test_missing_value_imputation, standardize, test_standardize, one_hot_encoding,col_rename,test_col_rename


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_data,
            inputs=["data_input", "params:target", "params:dropCols"],
            outputs="clean_input",
            name="cleanDataNode"
        ),
        node(
            func=remove_duplicate,
            inputs="clean_input",
            outputs="int_data",
            name="removeDuplicateNode"
        ),
        node(
            func=fill_rate,
            inputs="int_data",
            outputs="fill_rate",
            name="fillRateNode"
        ),
        node(
            func=split_train_test,
            inputs=["int_data", "params:splitRatio"],
            outputs=["int_train", "int_test"],
            name="trainTestSplitNode"
        ),
        node(
            func=save_uid,
            inputs=["int_train", "int_test"],
            outputs=["train_uid", "test_uid"],
            name="saveUidNode"
        ),
        node(
            func=missing_value_imputation,
            inputs="int_train",
            outputs=["int_train_mv", "missing_val_dict", "missing_imputations"],
            name="missingValImputeNode"
        ),
        node(
            func=test_missing_value_imputation,
            inputs=["int_test", "missing_val_dict"],
            outputs="int_test_mv",
            name="testMissingValImputeNode"
        ),
        node(
            func=standardize,
            inputs=["int_train_mv", "params:target"],
            outputs=["int_train_sd", "compute_stats"],
            name="standardizeNode"
        ),
        node(
            func=test_standardize,
            inputs=["int_test_mv", "params:target", "compute_stats"],
            outputs="int_test_sd",
            name="testStandardizeNode"
        ),
        node(
            func=one_hot_encoding,
            inputs="int_train_sd",
            outputs="train_one_hot_encoding",
            name="trainOHENode"
        ),
        node(
            func=one_hot_encoding,
            inputs="int_test_sd",
            outputs="test_one_hot_encoding",
            name="testOHENode"
        ),
        node(
            func=col_rename,
            inputs=["train_one_hot_encoding", "params:target"],
            outputs=["pre_iv_train", "backup_train","col_rename"],
            name="trainColRename"
        ),
        node(
            func=test_col_rename,
            inputs=["test_one_hot_encoding", "params:target_actual", "col_rename"],
            outputs=["pre_iv_test","backup_test"],
            name="testColRename"
        )
    ])
