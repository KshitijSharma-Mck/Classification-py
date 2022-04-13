import logging
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix

log = logging.getLogger(__name__)


def metric_cal(train, test, target_var, predicted_train, predicted_test) -> pd.DataFrame:
    """This function calculates various metrics for the given model.

        Parameters:
            train (pd.DataFrame): input train dataset.
            test (pd.DataFrame): input test dataset.
            target_var (str):  input the name of the target variable.
            predicted_train(pd.Series): input the prediction of model on the train dataset.
            predicted_test(pd.Series): input the prediction of model on the test dataset.

        Returns:
            Dataframe containing various calculated metrics.
    """

    x_train = train.drop(target_var, axis=1)
    y_train = train[target_var]

    x_test = test.drop(target_var, axis=1)
    y_test = test[target_var]

    auc_train = np.round(metrics.roc_auc_score(y_train, predicted_train), 3)
    auc_test = np.round(metrics.roc_auc_score(y_test, predicted_test), 3)

    gini_train = 2 * auc_train - 1
    gini_test = 2 * auc_test - 1

    mcc_train = metrics.matthews_corrcoef(y_train, predicted_train)
    mcc_test = metrics.matthews_corrcoef(y_test, predicted_test)

    f1_train = metrics.f1_score(y_train, predicted_train)
    f1_test = metrics.f1_score(y_test, predicted_test)

    precision_train = metrics.precision_score(y_train, predicted_train)
    precision_test = metrics.precision_score(y_test, predicted_test)

    recall_train = metrics.recall_score(y_train, predicted_train)
    recall_test = metrics.recall_score(y_test, predicted_test)

    accuracy_train = metrics.accuracy_score(predicted_train, y_train)
    accuracy_test = metrics.accuracy_score(predicted_test, y_test)

    cm_train = confusion_matrix(y_train, predicted_train)
    cm_test = confusion_matrix(y_test, predicted_test)

    specificity_train = cm_train[1, 1] / (cm_train[1, 0] + cm_train[1, 1])
    specificity_test = cm_test[1, 1] / (cm_test[1, 0] + cm_test[1, 1])

    train_row = [auc_train, gini_train, mcc_train, f1_train, precision_train, recall_train, specificity_train,
                 accuracy_train]
    test_row = [auc_test, gini_test, mcc_test, f1_test, precision_test, recall_test, specificity_test, accuracy_test]

    for i in cm_train:
        for val in i:
            train_row.append(val)

    for i in cm_test:
        for val in i:
            test_row.append(val)

    header = ['auc', 'gini', 'mcc', 'f1score', 'precision', 'recall', 'specificity', 'accuracy', 'true.0.pred.0',
              'true.1.pred.0', 'true.0.pred.1', 'true.1.pred.1']

    metrics_dataframe = pd.DataFrame(data=[train_row, test_row], columns=[header])

    return metrics_dataframe
