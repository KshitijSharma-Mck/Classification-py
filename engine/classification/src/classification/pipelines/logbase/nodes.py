"""
This is a boilerplate pipeline 'logbase'
generated using Kedro 0.17.7
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from ...General.metric_cal import metric_cal


def log_reg_base(train, test, target_var) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, LogisticRegression):
    """This function runs logistic regression model.

        Parameters:
            train (pd.DataFrame): input train dataset.
            test (pd.DataFrame): input train dataset.
            target_var (str):  input the name of the target variable.

        Returns:
            df-data frame with the metric values, Predicted train values, Predicted test values
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

    return lr_train_predict, lr_test_predict, logmodel,predicted_train,predicted_test
