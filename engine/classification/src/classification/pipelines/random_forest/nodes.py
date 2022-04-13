"""
This is a boilerplate pipeline 'random_forest'
generated using Kedro 0.17.7
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def rf_base(train, test, target_var, rf_hyperparams) -> (pd.DataFrame, pd.Series, pd.Series):
    """This function runs base random forrest algorithm.

        Parameters:
            train (pd.DataFrame): input train dataset.
            test (pd.DataFrame): input train dataset.
            target_var (str): input the name of the target variable.
            rf_hyperparams(dict): input the hyper parameters.

        Returns:
            Data frame predicted train values, predicted test values and features' importance and the model object.
    """

    x_train = train.drop(target_var, axis=1)
    y_train = train[target_var]

    x_test = test.drop(target_var, axis=1)
    y_test = test[target_var]

    classifier_rf = RandomForestClassifier(bootstrap=rf_hyperparams['bootstrap'],
                                           ccp_alpha=rf_hyperparams['ccp_alpha'],
                                           class_weight=rf_hyperparams['class_weight'],
                                           criterion=rf_hyperparams['criterion'],
                                           max_depth=rf_hyperparams['max_depth'],
                                           max_features=rf_hyperparams['max_features'],
                                           max_leaf_nodes=rf_hyperparams['max_leaf_nodes'],
                                           max_samples=rf_hyperparams['max_samples'],
                                           min_impurity_decrease=rf_hyperparams['min_impurity_decrease'],
                                           min_samples_leaf=rf_hyperparams['min_samples_leaf'],
                                           min_samples_split=rf_hyperparams['min_samples_split'],
                                           min_weight_fraction_leaf=rf_hyperparams['min_weight_fraction_leaf'],
                                           n_estimators=rf_hyperparams['n_estimators'],
                                           n_jobs=rf_hyperparams['n_jobs'],
                                           oob_score=rf_hyperparams['oob_score'],
                                           random_state=rf_hyperparams['random_state'],
                                           verbose=rf_hyperparams['verbose'],
                                           warm_start=rf_hyperparams['warm_start'])

    classifier_rf.fit(x_train, y_train)

    predicted_test = classifier_rf.predict_proba(x_test)
    predicted_train = classifier_rf.predict_proba(x_train)

    print(classifier_rf.feature_importances_)

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

    predicted_test = classifier_rf.predict(x_test)
    predicted_train = classifier_rf.predict(x_train)

    original_cols = []
    for (col, col_data) in train.iteritems():
        original_cols.append(col)
    rf_feat_imp_dict = {'Cols Name': original_cols[1:],
                        'Important Feature': classifier_rf.feature_importances_}
    rf_feature_imp_df = pd.DataFrame(rf_feat_imp_dict)

    return predicted_train, predicted_test, rf_train_predict, rf_test_predict, classifier_rf, rf_feature_imp_df


