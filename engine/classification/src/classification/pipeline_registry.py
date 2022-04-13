"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline, pipeline
import yaml
from .pipelines import data_processing as dp
from .pipelines import logbase as lb
from .pipelines import lr_fwd_selection as lrf
from .pipelines import lr_bwd_selection as lrb
from .pipelines import iv_table as ivt
from .pipelines import decision_tree as dt
from .pipelines import random_forest as rf
from .pipelines import rf_random_search as rfrs
from .pipelines import dt_random_search as dtrs
from .pipelines import rf_grid_search as rfgs
from .pipelines import dt_grid_search as dtgs


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    val = []
    with open("./conf/base/parameters/01_parameters.yml", 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            print(parsed_yaml)
        except yaml.YAMLError as exc:
            print(exc)

    data_processing_pipeline = dp.create_pipeline()
    iv_table_pipeline = ivt.create_pipeline()
    logbase_pipeline = lb.create_pipeline()
    logfwd_pipeline = lrf.create_pipeline()
    logbwd_pipeline = lrb.create_pipeline()
    decision_tree_pipeline = dt.create_pipeline()

    random_forest_pipeline = rf.create_pipeline()
    rf_grid_search_pipeline = rfgs.create_pipeline()
    dt_grid_search_pipeline = dtgs.create_pipeline()
    rf_random_search_pipeline = rfrs.create_pipeline()
    dt_random_search_pipeline = dtrs.create_pipeline()

    if parsed_yaml["create_data_processing"]:
        val.append(data_processing_pipeline)

    if parsed_yaml["create_iv_table"]:
        val.append(iv_table_pipeline)

    if parsed_yaml["create_log_base"]:
        val.append(logbase_pipeline)

    if parsed_yaml["create_log_fwd"]:
        val.append(logfwd_pipeline)

    if parsed_yaml["create_log_bwd"]:
        val.append(logbwd_pipeline)

    if parsed_yaml["create_decision_tree"]:
        val.append(decision_tree_pipeline)

    if parsed_yaml["create_decision_tree_grid_search"]:
        val.append(dt_grid_search_pipeline)

    if parsed_yaml["create_decision_tree_random_search"]:
        val.append(dt_random_search_pipeline)

    if parsed_yaml["create_random_forest"]:
        val.append(random_forest_pipeline)

    if parsed_yaml["create_random_forest_grid_search"]:
        val.append(rf_grid_search_pipeline)

    if parsed_yaml["create_random_forest_random_search"]:
        val.append(rf_random_search_pipeline)

    pipeline_to_run = val[0]
    for i in range(1, len(val)):
        pipeline_to_run += val[i]

    return {
        "__default__": pipeline_to_run,
        "dp": data_processing_pipeline,
        "ivt": iv_table_pipeline,
        "lb": logbase_pipeline,
        "lrf": logfwd_pipeline,
        "lrb": logbwd_pipeline,
        "dt": decision_tree_pipeline,
        "dtrs": dt_random_search_pipeline,
        "rf": random_forest_pipeline,
        "rfgs": rf_grid_search_pipeline,
        "dtgs": dt_grid_search_pipeline,
        "rfrs": rf_random_search_pipeline
    }
