"""
Module containing logic for getting and persisting data, and feature engineering.
"""

TIME_FEATURES = ['hour', 'month', 'year', 'quarter', 'dayofweek', 'dayofmonth', 'dayofyear']


def add_temporal_features(df):
    """Given a DataFrame with a datetime index, return a copy with
    added temporal feature columns.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['dayofweek'] = df.index.dayofweek
    df['dayofmonth'] = df.index.day
    df['dayofyear'] = df.index.dayofyear
    return df
