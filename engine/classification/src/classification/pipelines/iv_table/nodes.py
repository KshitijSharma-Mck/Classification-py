"""
This is a boilerplate pipeline 'iv_table'
generated using Kedro 0.17.7
"""

import logging

from heapq import nlargest
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


def iv_table(df, target, top, top_iv_range, bottom_iv_range) -> (pd.DataFrame, pd.DataFrame, pd.Series):
    """This function is used to calculate the IV values for each column excluding the target and also to calculate
    the top n IV values and store them in a csv file.

       Parameters:
           df (pd.dataframe): input train data for calculating the iv values.
           target (str): input the target_rename value,i.e, the target column after the col_rename func.
           top (int/str): input the number of top records that needs to be extracted or if set to auto will extract the iv values within the top and bottom range.
           top_iv_range (int): top iv range for extracting the data when set to auto.
           bottom_iv_range (int): bottom iv range for extracting the data when set to auto.

       Returns:
            pandas dataframes that consist of all the IV values and the column names and another data frame
            consisting of the top n IV values with the respective column name.

    """
    iv_val = []
    col_iv = {}
    col = []
    lst = []
    for feature in df.columns:
        if feature != target:
            for i in range(df[feature].nunique()):
                val = list(df[feature].unique())[i]
                lst.append({
                    'Value': val,
                    'count': df[df[feature] == val].count()[feature],
                    'Event': df[(df[feature] == val) & (df[target] == 0)].count()[feature],
                    'NonEvent': df[(df[feature] == val) & (df[target] == 1)].count()[feature]
                })

            dset = pd.DataFrame(lst)
            dset['Distr_Event'] = dset['Event'] / dset['Event'].sum()
            dset['Distr_NonEvent'] = dset['NonEvent'] / dset['NonEvent'].sum()
            dset['WoE'] = np.log(dset['Distr_Event'] / dset['Distr_NonEvent'])
            dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
            dset['IV'] = (dset['Distr_Event'] - dset['Distr_NonEvent']) * dset['WoE']
            iv = dset['IV'].sum()
            col_iv[feature] = iv
            iv_val.append(iv)
            col.append(feature)

    iv_dict = {'Col': col, 'IV': iv_val}
    iv_data = pd.DataFrame.from_dict(iv_dict)

    iv_value = []
    iv_col = []
    # if set to auto will take iv values in that range
    if top == "auto":

        for keys, val in col_iv.items():
            if bottom_iv_range <= val <= top_iv_range:
                iv_col.append(keys)
                iv_value.append(val)

    # else will take the top n values that is set
    else:
        iv_col = nlargest(top, col_iv, key=col_iv.get)
        for val in iv_col:
            iv_value.append(col_iv.get(val))

    imp_iv_dict = {"Col": iv_col, "IV": iv_value}
    imp_iv_data = pd.DataFrame.from_dict(imp_iv_dict)

    return iv_data, imp_iv_data, iv_col


def post_iv_TrainTest(df, iv_col, target_rename) -> pd.DataFrame:
    """This function is used generate the train or test file after the important columns are selected from iv table.

           Parameters:
               df (pd.dataframe): input pre_iv_train or pre_iv_test data .
               target_rename (str): input the target_rename value,i.e, the target column after the col_rename func.
               iv_col (pd.Series): input the column names generated from iv table.


           Returns:
                pandas train or test dataframes that consist of only target column and important columns generated from iv table.

        """
    col = [target_rename]
    col.extend(iv_col)
    post_iv_df = df[col]

    return post_iv_df
