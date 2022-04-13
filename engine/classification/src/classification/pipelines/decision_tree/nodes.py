"""
This is a boilerplate pipeline 'decision_tree'
generated using Kedro 0.17.7
"""
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def decision_tree(train, test, target_var, dt_hyperparams)->(DecisionTreeClassifier,pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """This function selects features based on random search cross validation.
            Parameters:
                train (pd.DataFrame): input train dataset.
                test (pd.DataFrame): input test dataset.
                target_var (str): input the name of the target variable.
                dt_hyperparams(dict): value for hyperparameters.
            Returns:
                Dataframe of train, test predictions and best hyper parameters and the model object.
     """
    x_train = train.drop(target_var, axis=1)
    y_train = train[target_var]

    x_test = test.drop(target_var, axis=1)
    y_test = test[target_var]

    dt_classifier = DecisionTreeClassifier(criterion=dt_hyperparams["criterion"],
                                           class_weight=dt_hyperparams["class_weight"],
                                           max_depth=dt_hyperparams["max_depth"],
                                           max_features=dt_hyperparams["max_features"],
                                           max_leaf_nodes=dt_hyperparams["max_leaf_nodes"],
                                           min_impurity_decrease=dt_hyperparams["min_impurity_decrease"],
                                           min_samples_leaf=dt_hyperparams["min_samples_leaf"],
                                           min_samples_split=dt_hyperparams["min_samples_split"],
                                           min_weight_fraction_leaf=dt_hyperparams["min_weight_fraction_leaf"],
                                           random_state=np.random.seed(42),
                                           splitter=dt_hyperparams["splitter"])
    dt_classifier.fit(x_train, y_train)

    test_prediction = dt_classifier.predict(x_test)
    train_prediction = dt_classifier.predict(x_train)

    original_cols = []
    for (col, col_data) in train.iteritems():
        original_cols.append(col)
    dt_feature_imp_dict = {'Cols Name': original_cols[1:],
                           'Important Feature': dt_classifier.feature_importances_}
    dt_feature_imp_df = pd.DataFrame(dt_feature_imp_dict)
    return dt_classifier, dt_feature_imp_df, train_prediction, test_prediction
