"""
This is a boilerplate pipeline 'lr_fwd_selection'
generated using Kedro 0.17.7
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from ...General.metric_cal import metric_cal


def lr_bwd_selection(train, test, target_var, model_object, no_of_features, score_criteria)\
        -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """This function selects features based on forward selection

        Parameters:
            train (pd.DataFrame): input train dataset.
            test (pd.DataFrame): input test dataset.
            target_var (str): input the name of the target variable.
            model_object: input the model object for which feature selection is required.
            no_of_features(int): input the number of features that need to be selected.
            score_criteria(str): input the criteria score on which the features will be selected.
        Returns:
            Dataframe containing various calculated metrics.
    """

    x_train = train.drop(target_var, axis=1)

    feature_names = list(x_train.columns)
    x_train = train.drop(target_var, axis=1)
    y_train = train[target_var]

    sfs_backward = SequentialFeatureSelector(model_object,
                                             n_features_to_select=no_of_features,
                                             scoring=score_criteria,
                                             direction="backward").fit(x_train, y_train)

    selection_inst = (sfs_backward.get_support())
    filtered = np.array(feature_names)[selection_inst]
    filtered_train = pd.concat([train[target_var], train[filtered]], axis=1)
    filtered_test = pd.concat([test[target_var], test[filtered]], axis=1)
    filtered = pd.DataFrame(data=filtered, columns=[f"backward_selection"])

    return filtered, filtered_train, filtered_test


def log_reg_base(train, test, target_var) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, LogisticRegression):
    """This function runs logistic regression model.

        Parameters:
            train (pd.DataFrame): input train dataset.
            test (pd.DataFrame): input train dataset.
            target_var (str):  input the name of the target variable.

        Returns:
            Data frame of the metric values, predicted train values, predicted test values and the model object.
    """

    x_train = train.drop(target_var, axis=1)
    y_train = train[target_var]

    x_test = test.drop(target_var, axis=1)
    y_test = test[target_var]

    logmodel = LogisticRegression()
    logmodel.fit(x_train, y_train)

    predicted_test = logmodel.predict_proba(x_test)
    predicted_train = logmodel.predict_proba(x_train)

    predicted_train_df = pd.DataFrame(data=predicted_train, columns=['1', 'Predicted'])
    predicted_test_df = pd.DataFrame(data=predicted_test, columns=['1', 'Predicted'])

    predicted_train_df = predicted_train_df["Predicted"]
    predicted_test_df = predicted_test_df["Predicted"]

    y_train.rename("Actual", inplace=True)
    y_test.rename("Actual", inplace=True)

    test_uids = pd.read_csv("data/02_intermediate/save_uid/test_uids.csv")
    train_uids = pd.read_csv("data/02_intermediate/save_uid/train_uids.csv")

    train_uids.drop("Sno", axis=1, inplace=True)
    test_uids.drop("Sno", axis=1, inplace=True)

    lr_train_predict = pd.concat([train_uids, y_train, predicted_train_df], axis=1)
    lr_test_predict = pd.concat([test_uids, y_test, predicted_test_df], axis=1)

    predicted_test = logmodel.predict(x_test)
    predicted_train = logmodel.predict(x_train)

    #metrics_df = metric_cal(train, test, target_var, predicted_train, predicted_test)

    return lr_train_predict, lr_test_predict, logmodel, predicted_train,predicted_test
