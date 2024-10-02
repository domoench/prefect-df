"""
Module containing logic for getting and persisting data, and feature engineering.
"""

from scipy.stats import skew
from collections import defaultdict
from core.consts import EIA_MAX_REQUEST_ROWS
from core.types import DVCDatasetInfo
from core.gx.gx import run_gx_checkpoint
from prefect.blocks.system import Secret
import numpy as np
import pandas as pd
import dvc.api
import io
import os
import requests


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


def request_EIA_data(start_ts, end_ts, offset, length=EIA_MAX_REQUEST_ROWS):
    print(f'Fetching API page. offset:{offset}. length:{length}')
    url = ('https://api.eia.gov/v2/electricity/rto/region-data/data/?'
           'frequency=hourly&data[0]=value&facets[respondent][]=PJM&'
           'sort[0][column]=period&sort[0][direction]=asc')

    eia_api_key_block = Secret.load('eia-api-key')
    eia_api_key = eia_api_key_block.get()

    # Use list of tuples instead of dict to allow duplicate params
    params = [
      ('offset', offset),
      ('length', length),
      ('api_key', eia_api_key),
      ('start', start_ts.strftime('%Y-%m-%dT%H')),
      ('end', end_ts.strftime('%Y-%m-%dT%H')),
      ('facets[type][]', 'D'),
    ]

    r = requests.get(url, params=params)
    r.raise_for_status()
    return r


"""
DVC
"""


def get_dvc_remote_repo_url(github_PAT: str = None) -> str:
    if github_PAT is None:
        if os.getenv('DF_ENVIRONMENT') == 'prod':
            github_PAT = Secret.load('dvc-git-repo-pat-prod').get()
        else:
            github_PAT = os.getenv('DVC_GIT_REPO_PAT')
    github_username = os.getenv('DVC_GIT_USERNAME')
    github_reponame = os.getenv('DVC_GIT_REPONAME')
    return f'https://{github_username}:{github_PAT}@github.com/{github_username}/{github_reponame}.git'


def get_dvc_dataset_as_df(dvc_dataset_info: DVCDatasetInfo) -> pd.DataFrame:
    data_bytes = dvc.api.read(
        path=dvc_dataset_info.path,
        repo=dvc_dataset_info.repo,
        rev=dvc_dataset_info.rev,
        mode='rb'
    )
    data_file_like = io.BytesIO(data_bytes)
    df = pd.read_parquet(data_file_like)

    # Validate data pulled from DVC data warehouse
    run_gx_checkpoint('etl', df)

    return df


def get_dvc_dataset_url(ddi: DVCDatasetInfo):
    return dvc.api.get_url(path=ddi.path, repo=ddi.repo, rev=ddi.rev)
