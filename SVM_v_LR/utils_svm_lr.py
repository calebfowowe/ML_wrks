import numpy as np
import pandas as pd
from pathlib import Path
import pandas_ta as ta
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression

#For hyperparmeter tuning
from functools import partial
import optuna

# file logger
from loguru import logger
import sys


def rnd_state(seed=30):
    return seed


#Function to create a folder where outputs of the projects are stored within the local drive
def getpath(name=None):
    if name is not None:
        #Ouput Files paths
        PATH = Path() / "SVM_lr_output"/ f"{name}"
        PATH.mkdir(parents=True, exist_ok=True)
    else:
        # defined path within path
        PATH = Path() / "SVM_lr_output"
        PATH.mkdir(parents=True, exist_ok=True)
    return PATH


#Class-weights imbalance
def cwts(dfs):
    c0, c1 = np.bincount(dfs['predict'])
    w0 = (1 / c0) * (len(dfs)) / 2
    w1 = (1 / c1) * (len(dfs)) / 2
    return {0: w0, 1: w1}

class LoadData:
    """
    This class is used in loading the data from where it is saved, checking for missing data,
    and method for filling missing data.

    It takes as input:
    dictionary of the filenames as keys, and dateformat as values.
    The first filename and format should be for the stock data

    It also has the option to provide the argument for how nan columns should be filled.
    """
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    @staticmethod
    def adjust_to_friday(date):
        #If date is on a weekend, adjust it to previous business day
        if date.weekday() == 5: #5 = Saturday
            date = date - pd.Timedelta(days=1)
        elif date.weekday() == 6: #6 = Sunday
            date = date - pd.Timedelta(days=2)
        return date

    def getData(self, filename):
        df = pd.read_csv(f'./data/{filename}.csv')
        #Check if the current index is a datetime type
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            match df.columns.any():
                case 'date':
                    df['date'] = pd.to_datetime(df['date'], format='mixed')  # convert to datetime
                    df.set_index('date', inplace=True) #set 'date' column as the index
                case 'dates':
                    df['dates'] = pd.to_datetime(df['dates'], format='mixed')  # convert to datetime
                    df.set_index('dates', inplace=True) #set 'dates' column as the index
                case 'Date':
                    df['Date'] = pd.to_datetime(df['Date'], format='mixed')  # convert to datetime
                    df.set_index('Date', inplace=True) #set 'Date' column as the index
                case 'Dates':
                    df['Dates'] = pd.to_datetime(df['Dates'], format='mixed')  # convert to datetime
                    df.set_index('Dates', inplace=True) #set 'Dates' column as the index
                case 'datetime':
                    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')  # convert to datetime
                    df.set_index('datetime', inplace=True) #set 'datetime' column as the index
                case 'Datetime':
                    df['Datetime'] = pd.to_datetime(df['Datetime'], format='mixed')  # convert to datetime
                    df.set_index('Datetime', inplace=True) #set 'Datetime' column as the index
                case _default:
                    raise KeyError("No datetime index or 'date' column found in the DataFrame")
        df.index = pd.to_datetime(df.index, format='%Y-%d-%m')
        df.sort_index(ascending=True, inplace=True)
        df.index = df.index.map(self.adjust_to_friday) #Adjust non_weekday dates to end of month
        return df

    def joinData(self):
        global add_file, fname
        file_dict = self.kwargs  #unpack the keyword arguments

        #Stock data information must be the first in the frame and should be extracted to the dataframe first
        df_merged = self.getData(file_dict['files'][0])  #Extract historical stock-price data from file list

        # Check if additional data is provided and iterate through to get the data and merge to the stock data
        if len(file_dict['files']) > 1:
            # iterate through other the values and merge to dataframe
            for key, value_list in file_dict.items():
                for fname in value_list[1:]:
                    add_file = self.getData(fname)
                    df_merged = df_merged.join(add_file, how='left', rsuffix=f'_{fname}')

        # Select the time_range specified to be used for the analysis from the finally merged dataframe
        if len(self.args) != 0:
            df_merged = df_merged[self.args[0]: self.args[1]]
        else:
            df_merged = df_merged.copy()
        #Remove duplicated dates.
        df_merged = df_merged[~df_merged.index.duplicated(keep='first')] #remove duplicated dates
        self.df_new = df_merged.copy()

        return self.df_new



def get_TA_features(data) -> pd.DataFrame:
    try:
        df = data.copy() #make a copy of the provided data

        df['days'] = df.index.day_name()# create days of the week feature

        # create all technical indicator strategies from pandas-ta library.
        df.ta.study("All", lookahead=False, talib=False)

        data = df.copy()#making a copy of the dataframe with the technical indicators features.

        # drop unwanted features columns
        data.drop(
            ['QQEl_14_5_4.236', 'QQEs_14_5_4.236', 'PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSARaf_0.02_0.2',
             'HILOs_13_21','HILOl_13_21', 'PSARr_0.02_0.2', 'SUPERTl_7_3.0', 'SUPERTs_7_3.0', 'SUPERTd_7_3.0',
             'SUPERT_7_3.0', 'ZIGZAGs_5.0%_10', 'ZIGZAGv_5.0%_10', 'ZIGZAGd_5.0%_10', 'VIDYA_14', 'VHM_610'],
            axis=1, inplace=True)


        # drop #ohlcv data
        data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)

        # drop columns with infinity as values within them
        specific_value = np.inf
        specific_value2 = -np.inf
        columns_to_drop = [col for col in data.columns if specific_value in data[col].values
                           or specific_value2 in data[col].values]
        data.drop(columns=columns_to_drop, axis=1, inplace=True)

        # drop the first set of 100rows because of features with initial observation window requiring over 100days of data
        df2 = data.copy()[100:]

        # backfill columns to address missing values
        df2 = df2.bfill(axis=1)
        df2 = df2[:-1]
        return df2

    except:
        logger.error("TA_features not created successfully, check setup")