"""
Module containing logic for getting and persisting data, and feature engineering.
"""

from scipy.stats import skew
from collections import defaultdict
from core.consts import EIA_MAX_REQUEST_ROWS
from core.types import DVCDatasetInfo, validate_call
from core.gx.gx import gx_validate_df
from core.holidays import is_holiday
from core.utils import merge_intervals, has_full_hourly_index
from prefect.blocks.system import Secret
from datetime import datetime
from git import Repo as GitRepo
from dvc.repo import Repo as DvcRepo
import numpy as np
import pandas as pd
import dvc.api
import io
import os
import requests


def add_time_meaning_features(df):
    """Given a DataFrame with a datetime index, return a copy with added
    context on the meaning of a timestamp - e.g. month, day of week etc."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['dayofweek'] = df.index.dayofweek
    df['dayofmonth'] = df.index.day
    df['dayofyear'] = df.index.dayofyear
    return df


def add_time_lag_features(df):
    """Add lag timeseries features to the given dataframe based on its index.

    Note: This produces many NaN values in the lag columns. For example, you can't
    add lagged values to the oldest timestamp.
    TODO: Does xgboost do ok with these null values? I assume so, but confirm.
    """
    df = df.copy()
    ts_to_D = df.D.to_dict()
    # Trick: Offset by 364 days => lagged day is same day of week
    df['lag_1y'] = (df.index - pd.Timedelta('364 days')).map(ts_to_D)
    df['lag_2y'] = (df.index - pd.Timedelta('728 days')).map(ts_to_D)
    df['lag_3y'] = (df.index - pd.Timedelta('1092 days')).map(ts_to_D)
    return df


def calculate_lag_backfill_ranges(df):
    """For the given datetime-indexed dataframe, return a list of (start_ts, end_ts)
    range tuples defining the same date range for each of the past 3 years"""
    start_ts, end_ts = df.index.min(), df.index.max()
    ranges = []
    for lag_y in [3, 2, 1]:
        # TODO: Ok to be ignorant of leap years like this?
        lag_start_ts = start_ts - pd.Timedelta(days=364*lag_y)
        lag_end_ts = end_ts - pd.Timedelta(days=364*lag_y)
        # No point in allowing lag end to overlap with original data range
        lag_end_ts = min(start_ts, lag_end_ts)
        ranges.append((lag_start_ts, lag_end_ts))
    # When df's time range spans more than a year, the lags ranges will overlap.
    ranges = merge_intervals(ranges)
    return ranges


def calculate_chunk_index(df: pd.DataFrame) -> pd.DataFrame:
    """Index the given dataframe's time interval by standard calendar (not fiscal) quarter
    chunks/intervals. Each chunk in the index has a boolean flag to specify whether
    it is complete (has data for every hour) or not."""
    assert has_full_hourly_index(df)

    start_ts, end_ts = df.index.min(), df.index.max()

    # Beginning and end of logical (quarter) chunks
    q_start_ts = pd.Timestamp(start_ts).to_period('Q').start_time.tz_localize('UTC')
    q_end_ts = pd.Timestamp(end_ts).to_period('Q').end_time.tz_localize('UTC')

    first_chunk_complete = start_ts == q_start_ts
    last_chunk_complete = end_ts == q_end_ts

    chunks = []
    chunk_start_ts = q_start_ts
    while chunk_start_ts < q_end_ts:
        chunk_end_ts = chunk_start_ts + pd.offsets.QuarterBegin(startingMonth=1) - pd.Timedelta(hours=1)
        chunk_name = f'{chunk_start_ts.year}_Q{chunk_start_ts.quarter}' \
                     f"_from_{chunk_start_ts.strftime('%Y-%m-%d-%H')}" \
                     f"_to_{chunk_end_ts.strftime('%Y-%m-%d-%H')}"
        # The actual data may or may not fill the whole logical chunk
        data_start_ts = max(start_ts, chunk_start_ts)
        data_end_ts = min(end_ts, chunk_end_ts)
        chunks.append({
            'year': chunk_start_ts.year,
            'quarter': chunk_start_ts.quarter,
            'start_ts': chunk_start_ts,  # Start of the logical chunk
            'end_ts': chunk_end_ts,  # End of the logical chunk
            'data_start_ts': data_start_ts,  # Start of actual data for this chunk
            'data_end_ts': data_end_ts,  # End of actual data for this chunk
            'name': chunk_name,
            'complete': chunk_start_ts == data_start_ts and chunk_end_ts == data_end_ts,
            })
        chunk_start_ts += pd.offsets.QuarterBegin(startingMonth=1)

    chunks[0]['complete'] = first_chunk_complete
    chunks[-1]['complete'] = last_chunk_complete
    chunk_df = pd.DataFrame(chunks)
    return chunk_df


def add_holiday_feature(df):
    """Add an (integer) flag specifying whether this hour falls on a US national
    holiday"""
    df = df.copy()
    df['is_holiday'] = pd.Series(df.index.date, index=df.index).apply(is_holiday)
    # Convert bool to numeric for xgboost
    df['is_holiday'] = df['is_holiday'].astype(int)
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


@validate_call
def request_EIA_data(start_ts: datetime, end_ts: datetime, offset, length=EIA_MAX_REQUEST_ROWS):
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


def get_local_dvc_git_repo():
    """Ensure the dvc git repo is on the local filsystem in a clean state."""
    local_dvc_repo_path = os.getenv('DVC_LOCAL_REPO_PATH')
    directory = os.path.dirname(local_dvc_repo_path)
    # Create the directory and clone the repo if necessary
    git_repo = None
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        git_repo_url = get_dvc_remote_repo_url()
        git_repo = GitRepo.clone_from(git_repo_url, local_dvc_repo_path)
    else:
        git_repo = GitRepo(local_dvc_repo_path)
        # Ensure no unstaged changes
        assert not git_repo.is_dirty(untracked_files=True) # TODO something better
        print('DVC git repo is clean')
    return git_repo


def get_DvcRepo():
    """Return a DvcRepo instance to interact with the local dvc repo."""
    local_dvc_repo_path = os.getenv('DVC_LOCAL_REPO_PATH')
    return DvcRepo(local_dvc_repo_path)


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
    gx_validate_df('etl', df)

    return df


def get_dvc_dataset_url(ddi: DVCDatasetInfo):
    return dvc.api.get_url(path=ddi.path, repo=ddi.repo, rev=ddi.rev)
