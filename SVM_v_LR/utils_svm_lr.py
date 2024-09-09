import numpy as np
import pandas as pd

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