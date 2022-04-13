"""
This is a boilerplate pipeline 'rf_grid_search'
generated using Kedro 0.17.7
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np


def grid_cv(train, test, target_var, model_object, param_grid)-> \
        (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, RandomForestClassifier):
    """This function selects features based on grid search cross validation.

        Parameters:
            train (pd.DataFrame): input train dataset.
            test (pd.DataFrame): input test dataset.
            target_var (str): input the name of the target variable.
            model_object(pickle): input the model object for which feature selection is required.
            param_grid(dict): input search parameters.
        Returns:
            Dataframe of train, test predictions and best hyper parameters and the model object.
    """
    x_train = train.drop(target_var, axis=1)
    y_train = train[target_var]

    x_test = test.drop(target_var, axis=1)
    y_test = test[target_var]

    forest_cv = GridSearchCV(model_object, param_grid, cv=5)
    forest_cv.fit(x_train, y_train)

    best_params_dict = forest_cv.best_params_
    best_hyper_params = pd.DataFrame(best_params_dict, index=[0])

    rf_best_estimator = forest_cv.best_estimator_
    rf_best_estimator.fit(x_train, y_train)

    predicted_test = rf_best_estimator.predict_proba(x_test)
    predicted_train = rf_best_estimator.predict_proba(x_train)

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

    rf_train_predict = pd.concat([train_uids, y_train, predicted_train_df], axis=1)
    rf_test_predict = pd.concat([test_uids, y_test, predicted_test_df], axis=1)

    predicted_test = rf_best_estimator.predict(x_test)
    predicted_train = rf_best_estimator.predict(x_train)

    return predicted_train, predicted_test, rf_train_predict, rf_test_predict, rf_best_estimator, best_hyper_params

