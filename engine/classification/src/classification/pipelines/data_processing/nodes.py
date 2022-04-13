"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

import logging
from scipy import stats
from sklearn.model_selection import train_test_split
import scipy.stats.stats as stats
from heapq import nlargest
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


def clean_data(data, dep_var, var_to_remove) -> pd.DataFrame:
    """This function cleans and arranges the data; removes completely empty columns and columns with 0 variance.

        Parameters:
            data (pd.DataFrame): input data to be cleaned.
            dep_var (str): name of the dependent variable.
            var_to_remove (list): variable(s) which are to be removed from the dataset.

        Returns:
                cleaned dataset.

    """

    dep_var_series = data[dep_var]
    to_drop = [dep_var] + var_to_remove
    data.drop(to_drop, inplace=True, axis=1)

    data = pd.concat([dep_var_series, data], axis=1)
    nan_value = float("NaN")
    data.replace("", nan_value, inplace=True)

    data.dropna(how='all', axis=1, inplace=True)

    data_var = data.var(axis=0)

    for (columnName, columnData) in data_var.iteritems():
        if columnData == 0:
            data.drop(columnName, inplace=True, axis=1)

    log.info("dataset is cleansed.")

    return data


def remove_duplicate(data) -> pd.DataFrame:
    """This function removes duplicate rows in the dataset.

        Parameters:
            data (pd.DataFrame) : pandas data frame before removing duplicates.
        Returns:
            data frame after removing duplicates.

    """

    data = data.drop_duplicates(keep='first', inplace=False)

    log.info("Duplicate rows have been removed")

    return data


def fill_rate(data) -> pd.DataFrame:
    """This function calculates the fill rate of columns.

        Parameters:
            data (pd.DataFrame): input dataset with a set of columns.

        Returns:
            Data frame with the fill rates

    """
    col = list(data)
    data = data.notna().sum() / len(data) * 100
    val = data.values.tolist()
    fill_rate_dict = {'Fill_Rate': val, 'Column': col}
    fill_rate_data = pd.DataFrame.from_dict(fill_rate_dict)
    # data.set_axis(["Fill_Rate"], axis=1, inplace=True)

    log.info("fill percentages of columns is calculated and stored.")
    return fill_rate_data


def split_train_test(df, split_ratio) -> (pd.DataFrame, pd.DataFrame):
    """This function splits the data set to train and test sets based on the split ratio and combining them back into a dataframe

        Parameters:
            df (pd.DataFrame) : pandas dataframe for the dataset.
            split_ratio (Float) : Consists of the ratio for splitting train and test (0.8).

        Returns:
            train and test data

    """

    train, test = train_test_split(df, train_size=split_ratio, random_state=0)

    log.info("Data has been split into train and test, ratio of split: %f", split_ratio)
    return train, test


def save_uid(train, test) -> (pd.DataFrame, pd.DataFrame):
    """This function Maps the unique ids of data points.

        Parameters:
            train (pd.DataFrame): input the train split dataset.
            test (pd.DataFrame): input the test split dataset.

        Returns:
            train and test data

    """

    train_uid_list = []
    test_uid_list = []

    for i, row in train.iterrows():
        train_uid_list.append(i)

    train_data = {'Sno': [i for i in range(1, len(train.index) + 1)],
                  'train_uid': train_uid_list}

    train_uid = pd.DataFrame(train_data)

    for i, row in test.iterrows():
        test_uid_list.append(i)

    test_data = {'Sno': [i for i in range(1, len(test.index) + 1)],
                 'test_uid': test_uid_list}

    test_uid = pd.DataFrame(test_data)

    log.info("train and test id mapping csv files have been generated")

    return train_uid, test_uid


def missing_value_imputation(data) -> (pd.DataFrame, dict, pd.DataFrame):
    """This function fills the missing numerical or categorical variables with median and mode respectively for the
    train data.

        Parameters:
                data(pandas dataframe): Train dataset.

        Returns:
                train data, dictionary of missing value imputations, dataframe of missing value imputations.
    """
    keys = []
    values = []
    col = list(data)
    mis_val_dict = {}
    for i in col:
        if data.dtypes[i] != np.object:
            x = data[i].median()
            data.fillna(x, inplace=True)
            mis_val_dict[i] = x
        else:
            x = data[i].mode()
            data[i].fillna(x[0], inplace=True)
            mis_val_dict[i] = x[0]

    for (key, val) in mis_val_dict.items():
        keys.append(key)
        values.append(val)

    missing_imputation_dict = {'Column': keys, 'Mean/Median': values}
    missing_value_df = pd.DataFrame.from_dict(missing_imputation_dict)

    log.info("missing values of train data has been filled with median and mode and written to file")
    return data, mis_val_dict, missing_value_df


def test_missing_value_imputation(data, mis_val_dict) -> pd.DataFrame:
    """This function fills the missing numerical or categorical variables with median and mode respectively for the
    test data based on the values calculated from the missing_value_imputation function.

        Parameters:
            data (pd.DataFrame): Test dataset.
            mis_val_dict (dict): Dictionary which consist of median and mode values for the respective columns.

        Returns:
            data frame containing the updated test dataset.
    """

    col = list(data)
    for i in col:
        x = mis_val_dict[i]
        data.fillna(x, inplace=True)

    log.info("missing values of test data has been filled with median and mode ")
    return data


def standardize(data, target_var) -> (pd.DataFrame, pd.DataFrame):
    """This function performs standardization on the set of external variable.

        Parameters:
            data (df): input dataset including the target variable in first column.
            target_var (str):  input the name of the target variable.

        Returns:
            data frame with standardized variables, data frame containing the computed mean and standard
        deviation values.

    """

    compute_stats_dict = {'type': ['mean', 'std']}

    for (col, col_data) in data.iteritems():
        if not data.dtypes[col] == 'object':
            compute_stats_dict[col] = [col_data.mean(), col_data.std()]
    compute_stats = pd.DataFrame(compute_stats_dict)
    compute_stats.drop(target_var, axis=1, inplace=True)

    for col, col_data in data.iteritems():
        if not col == target_var:
            data[col] = stats.zscore(data[col])

    log.info("train dataset's variables are standardized")

    return data, compute_stats


def test_standardize(data, target_var, get_stats) -> pd.DataFrame:
    """This function performs standardization on the set of external variable.

        Parameters:
            data (pd.DataFrame): input test dataset including the target variable in the first column.
            target_var (str):  input the name of the target variable.
            get_stats (pd.DataFrame): pandas dataframe containing the computed stats.

        Returns:
            dataframe with standardized variables.

    """
    try:
        data.drop('index', axis=1, inplace=True)
    except KeyError:
        pass

    for (col, col_data) in data.iteritems():
        if not col == target_var:
            try:
                mean = get_stats[col][0]
                std_dev = get_stats[col][1]
            except KeyError:
                pass
            else:
                data[col] = (data[col] - mean) / std_dev

    log.info("test dataset's variables are standardized")

    return data


def one_hot_encoding(data) -> pd.DataFrame:
    """This function performs one hot encoding for multiple categorical variables.

        Parameters:
            data (pd.DataFrame): Train dataset

        Returns:
            data frame containing the updated Train dataset with one hot encoded values.
    """

    col = list(data)
    one_hot_encoded_data = data
    # dict = {}
    for i in col:
        if data.dtypes[i] == np.object:
            one_hot_encoded_data = pd.get_dummies(data, columns=[i])

    log.info("One hot encoding has been performed for data")

    return one_hot_encoded_data


def col_rename(train, target_var) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """This function is used to rename the column names of a data frame by removing special characters and spaces
    thus making it usable in multiple models.

        Parameters:
            train (pd.DataFrame): input train dataset including the target variable in first column.
            target_var (str): input the name of the target variable.

        Returns:
             data frame containing the renamed train data set and the dat frame containing the original and renamed column.

    """
    original_cols = []

    train.rename({target_var: target_var.replace("_", ".")}, axis=1, inplace=True)

    for (col, col_data) in train.iteritems():
        original_cols.append(col)

    renamed_cols = [f"x{n}" for n in range(1, len(original_cols) + 1)]
    train.set_axis(renamed_cols, axis=1, inplace=True)

    col_rename_dict = {'Original_cols': original_cols,
                       'Renamed_cols': renamed_cols}
    col_rename_df = pd.DataFrame(col_rename_dict)

    log.info("train dataset's columns are renamed")
    datatypes = train.dtypes
    # print(datatypes)
    return train, train,col_rename_df


def test_col_rename(test, target_actual, get_col_mapping) -> (pd.DataFrame,pd.DataFrame):
    """This function is used to rename the column names of a data frame by removing special characters and spaces thus
    making it usable in multiple models.

       Parameters:
           test (pd.DataFrame): input test dataset including the target variable in first column.
           get_col_mapping(pd.DataFrame): data frame which contains the col_rename value.
           target_actual(int): input 0(false) or 1(true) to decide whether to rename the target variable or not.

       Returns:
            data frame containing the renamed test data set.

    """

    try:
        test.drop('index', axis=1, inplace=True)
    except KeyError:
        pass

    rename_test_cols = get_col_mapping['Renamed_cols'].tolist()

    if target_actual == 0:
        rename_test_cols[0] = get_col_mapping['Original_cols'].tolist()[0]

    test.set_axis(rename_test_cols, axis=1, inplace=True)

    log.info("test dataset's columns are renamed")

    return test,test


