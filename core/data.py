"""
Module containing logic for getting and persisting data, and feature engineering.
"""

from scipy.stats import skew
from collections import defaultdict
import numpy as np
import pandas as pd

TIME_FEATURES = ['hour', 'month', 'year', 'quarter', 'dayofweek', 'dayofmonth', 'dayofyear']


def add_temporal_features(df):
    """Given a DataFrame with a datetime index, return a copy with
    added temporal feature columns."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['dayofweek'] = df.index.dayofweek
    df['dayofmonth'] = df.index.day
    df['dayofyear'] = df.index.dayofyear
    return df


def cap_column_outliers(df, column, low, high):
    """Cap the values of the given dataframe's column between the given low
    and high threshold values."""
    df = df.copy()
    print(f'Input data skew: {skew(df.D.dropna())}')
    df.loc[df.D < low, 'D'] = low
    df.loc[df.D > high, 'D'] = high
    print(f'Output data skew: {skew(df.D.dropna())}')
    return df


def impute_null_demand_values(df):
    """Given an hour-resolution EIA dataframe with a 'D' column, impute null values
    as the average value for the given month and hour."""
    df = df.copy()
    print(f'Null demand values: {sum(df.D.isnull())}')

    # Create a map from month,hour to average demand value
    avg_D_by_month_hour = defaultdict(dict)
    for month in np.arange(1, 12+1):
        for hour in np.arange(24):
            month_mask = df.index.month == month
            hour_mask = df.index.hour == hour
            avg_D_by_month_hour[month][hour] = df[month_mask & hour_mask].D.mean()

    # Impute null values
    def impute_null_demand_value(row):
        month, hour = row.name.month, row.name.hour
        return avg_D_by_month_hour[month][hour]

    df.D = df.apply(
        lambda row: impute_null_demand_value(row) if pd.isnull(row['D']) else row['D'],
        axis=1,
    )
    return df
