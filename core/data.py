"""
Module containing logic for getting and persisting data, and feature engineering.
"""


def add_temporal_features(df):
    """Given a DataFrame with a datetime column 'utc_ts', return a copy with
    added temporal feature columns.
    """
    df = df.copy()
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    df['quarter'] = df.index.quarter
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    return df
