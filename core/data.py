"""
Module containing logic for getting and persisting data, and feature engineering.
"""

from scipy.stats import skew
from collections import defaultdict
from core.consts import EIA_MAX_REQUEST_ROWS, EIA_MIN_D_VAL, EIA_MAX_D_VAL
from core.types import DVCDatasetInfo, validate_call, ChunkIndex, EIADataUnavailableException
from core.gx.gx import gx_validate_df
from core.holidays import is_holiday
from core.utils import (
    merge_intervals, has_full_hourly_index, interval_intersection,
    concat_time_indexed_dfs, remove_rows_with_duplicate_indices,
    create_timeseries_df_1h, df_summary
)
from prefect.blocks.system import Secret
from prefect import task
from prefect.tasks import task_input_hash
from datetime import datetime
from git import Repo as GitRepo
from dvc.repo import Repo as DvcRepo
from pathlib import Path
from pprint import pp
import numpy as np
import pandas as pd
import dvc.api
import io
import os
import requests


@validate_call
def preprocess_data(df: pd.DataFrame):
    """Data cleaning and feature engineering.

    Args:
        df: The full length (train + test time window) raw data set from the warehouse
    Returns:
        df: Training dataset (test set removed)
    """
    # Feature Engineering
    df = clean_data(df)
    df = features(df)
    return df


@validate_call
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers and impute null values"""
    # Cap threshold values
    df = cap_column_outliers(df, 'D', EIA_MIN_D_VAL, EIA_MAX_D_VAL)
    df = impute_null_demand_values(df)
    return df


@validate_call
def features(df: pd.DataFrame) -> pd.DataFrame:
    # Add temporal features
    df = add_time_meaning_features(df)
    df = add_time_lag_features(df)
    df = add_holiday_feature(df)
    print(df)
    return df


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


def add_lag_backfill_data(df: pd.DataFrame) -> pd.DataFrame:
    """For the given datetime-indexed dataframe, fetch the same date range for
    the past 3 years, and return a dataframe with those rows prefixed. """
    df = df.copy()
    lag_dfs: list[pd.DataFrame] = []
    for (lag_start_ts, lag_end_ts) in calculate_lag_backfill_ranges(df):
        lag_df = get_range_from_dvc_as_df(lag_start_ts, lag_end_ts)
        lag_dfs.append(lag_df)
    return concat_time_indexed_dfs(lag_dfs + [df])


def calculate_chunk_index(start_ts: pd.Timestamp | None, end_ts: pd.Timestamp | None) -> ChunkIndex:
    """Index the time interval into standard calendar (not fiscal) quarter
    chunks/intervals. Each chunk in the index has a boolean flag to specify whether
    it is complete (has data for every hour) or not.

    If both timestamps are set to None, returns a zero-row chunk index.
    """
    # Handle empty index case
    if all([start_ts is None, end_ts is None]):
        dtypes = {
            'year': 'int64',
            'quarter': 'int64',
            'start_ts': 'datetime64[ns, UTC]',
            'end_ts': 'datetime64[ns, UTC]',
            'data_start_ts': 'datetime64[ns, UTC]',
            'data_end_ts': 'datetime64[ns, UTC]',
            'name': 'object',
            'complete': 'bool',
        }
        df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtypes.items()})
        return ChunkIndex(df)

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
    return ChunkIndex(chunk_df)


def get_chunk_index() -> ChunkIndex:
    # Ensure dvc repo exists on local filesystem
    _ = get_dvc_GitRepo_client()

    local_dvc_repo_path = Path(os.getenv('DVC_LOCAL_REPO_PATH'))
    chunk_idx_path = local_dvc_repo_path / 'v1/chunk_idx.parquet'
    chunk_idx_df = pd.read_parquet(chunk_idx_path)
    return ChunkIndex(chunk_idx_df)


@validate_call
def chunk_index_intersection(chunk_idx: ChunkIndex, start_ts: datetime, end_ts: datetime):
    """Given a chunk index and a (start_ts, end_ts) requested range, determine the intersection
    between the contiguous range covered by the index and the requested range. Return 2 range
    tuples: one describing the 'hit' range, and the other the 'miss' range."""
    # Handly empty chunk index case
    if len(chunk_idx) == 0:
        return (None, (start_ts, end_ts))

    cache_start, cache_end = (chunk_idx.iloc[0].data_start_ts, chunk_idx.iloc[-1].data_end_ts)
    req_start, req_end = (start_ts, end_ts)

    if req_start < cache_start:
        raise NotImplementedError('Current assumption is that oldest available data is in the index.')

    hit_range = interval_intersection((cache_start, cache_end), (req_start, req_end))
    miss_range = (cache_end + pd.Timedelta(hours=1), req_end) if req_end > cache_end else None
    return (hit_range, miss_range)


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
def request_EIA_data(
    start_ts: pd.Timestamp, end_ts: pd.Timestamp, offset, length=EIA_MAX_REQUEST_ROWS
):
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

# Maintain a singleton GitRepo client
git_repo = None


def get_dvc_GitRepo_client():
    """Return a GitRepo client."""
    global git_repo
    if git_repo is not None:
        return git_repo

    # Else, we must initialize
    local_dvc_repo_path = os.getenv('DVC_LOCAL_REPO_PATH')
    directory = os.path.dirname(local_dvc_repo_path)
    # Create the directory and clone the repo if necessary
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        git_repo_url = get_dvc_remote_repo_url()
        git_repo = GitRepo.clone_from(git_repo_url, local_dvc_repo_path)
    # Else the directory exists already
    else:
        git_repo = GitRepo(local_dvc_repo_path)
        # Ensure no unstaged changes
        assert not git_repo.is_dirty(untracked_files=True)  # TODO something better
        print('DVC git repo is clean')
    return git_repo


def get_DvcRepo_client():
    """Return a DvcRepo instance to interact with the local dvc repo."""
    local_dvc_repo_path = os.getenv('DVC_LOCAL_REPO_PATH')
    return DvcRepo(local_dvc_repo_path)


def get_dvc_remote_repo_url(github_PAT: str | None = None) -> str:
    if github_PAT is None:
        if os.getenv('DF_ENVIRONMENT') == 'prod':
            github_PAT = Secret.load('dvc-git-repo-pat-prod').get()
        else:
            github_PAT = os.getenv('DVC_GIT_REPO_PAT')
    github_username = os.getenv('DVC_GIT_USERNAME')
    github_reponame = os.getenv('DVC_GIT_REPONAME')
    return f'https://{github_username}:{github_PAT}@github.com/' \
           f'{github_username}/{github_reponame}.git'


def get_dvc_ref_for_chunk(
    start_ts: pd.Timestamp, chunk_name: str, git_rev: str
) -> DVCDatasetInfo:
    path = (Path('v1/data') / chunk_name).with_suffix('.parquet')
    dvc_ref = DVCDatasetInfo(
        path=str(path),
        repo=get_dvc_remote_repo_url(),
        rev=git_rev,
    )
    return dvc_ref


def get_dvc_dataset_as_df(dvc_dataset_info: DVCDatasetInfo) -> pd.DataFrame:
    data_bytes = dvc.api.read(
        path=dvc_dataset_info.path,
        repo=dvc_dataset_info.repo,
        rev=dvc_dataset_info.rev,
        mode='rb'
    )
    data_file_like = io.BytesIO(data_bytes)
    df = pd.read_parquet(data_file_like)

    return df


def get_current_dvc_commit_hash():
    git_repo = get_dvc_GitRepo_client()
    return str(git_repo.head.commit)


@task
@validate_call
def concurrent_fetch_EIA_data(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    print(f'Requesting demand data from EIA. start:{start_ts}. end:{end_ts}')
    # Quantize the start and end timestamps to midnight. The EIA API sometimes freaks
    # out and returns nothing otherwise.
    orig_start_ts, orig_end_ts = start_ts, end_ts
    start_ts = start_ts.normalize()
    end_ts = (end_ts + pd.Timedelta(days=1)).normalize()
    print('Quantizing fetch interval.\n'
          f' Original: start:{orig_start_ts}. end:{orig_end_ts}\n'
          f'Quantized: start:{start_ts}. end:{end_ts}\n')

    time_span = end_ts - start_ts
    hours = int(time_span.total_seconds() / 3600)

    # Metadata Query: determine exactly how many EIA records exist that match
    # our time range
    r = request_EIA_data(start_ts, end_ts, 0)
    num_total_records = int(r.json()['response']['total'])
    if num_total_records <= hours:
        # TODO should we fail here instead?
        print(f'Warning: EIA does not have all the data we are requesting. '
              f'Records requested: {hours}. Records available: {num_total_records}')

    # Calculate how many paginated API requests will be required to fetch all
    # the timeseries data
    num_full_requests = num_total_records // EIA_MAX_REQUEST_ROWS
    final_request_length = num_total_records % EIA_MAX_REQUEST_ROWS
    print((f'Fetching {hours} hours of data: {num_total_records} records.\n',
          f'Start: {start_ts}. End: {end_ts}'))
    print((f'Will make {num_full_requests} {EIA_MAX_REQUEST_ROWS}-length requests '
           f'and one {final_request_length}-length request.'))

    # Make the requests concurrently
    result_df_futures = []

    # Initiate the full-length requests
    for i in range(num_full_requests):
        offset = i * EIA_MAX_REQUEST_ROWS
        future = get_eia_data_as_df.submit(start_ts, end_ts, offset)
        result_df_futures.append(future)

    # Initiate the final request for the remainder records
    offset = num_full_requests * EIA_MAX_REQUEST_ROWS
    future = get_eia_data_as_df.submit(start_ts, end_ts, offset, final_request_length)
    result_df_futures.append(future)

    result_dfs = [future.result() for future in result_df_futures]
    api_df = pd.concat(result_dfs)

    # Trim back to original request time range
    request_range_mask = (api_df.index >= orig_start_ts) & (api_df.index <= orig_end_ts)
    num_extra_records = (~request_range_mask).sum()
    if num_extra_records > 0:
        print(f'Trimming off {num_extra_records} extra records.')
    api_df = api_df[request_range_mask]
    return api_df


@task
@validate_call
def get_eia_data_as_df(
    start_ts: pd.Timestamp, end_ts: pd.Timestamp, offset: int = 0, length=EIA_MAX_REQUEST_ROWS
) -> pd.DataFrame:
    """Fetch a single page of EIA power demand data for the given time range and API page offset.
    Return a dataframe with a datetime index."""
    r = request_EIA_data(start_ts, end_ts, offset, length)
    df = pd.DataFrame(r.json()['response']['data'])
    # Immediately cast timestamps to proper datetime type, as the result of this
    # fetch is used in multiple contexts - all of which assume pandas datetime object
    # format.
    df['utc_ts'] = pd.to_datetime(df['period'], utc=True)
    df = df.set_index('utc_ts')
    df = df.drop(columns='period')
    return df


@task
@validate_call
def fetch_weather_data(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    print(f'Requesting weather data from OpenMeteo. start:{start_ts}. end:{end_ts}')

    # Quantize fetch interval - OpenMeteo only accepts dates
    orig_start_ts, orig_end_ts = start_ts, end_ts
    start_ts = start_ts.normalize()
    end_ts = (end_ts + pd.Timedelta(days=1)).normalize()
    print('Quantizing fetch interval.\n'
          f' Original: start:{orig_start_ts}. end:{orig_end_ts}\n'
          f'Quantized: start:{start_ts}. end:{end_ts}\n')

    # Request data. TODO probably move this into a request_weather_data function
    url = ('https://archive-api.open-meteo.com/v1/era5?'
           'hourly=temperature_2m,cloud_cover')
    params = {
        # Using Reading PA as a single representative temp for PJM region
        'latitude': 40.407293,
        'longitude': -75.968453,
        'start_date': start_ts.strftime('%Y-%m-%d'),
        'end_date': end_ts.strftime('%Y-%m-%d')
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    df = pd.DataFrame(r.json()['hourly'])

    df['utc_ts'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('utc_ts')
    df = df.drop(columns='time')

    # TODO: Trim data back to original requested range
    request_range_mask = (df.index >= orig_start_ts) & (df.index <= orig_end_ts)
    num_extra_records = (~request_range_mask).sum()
    if num_extra_records > 0:
        print(f'Trimming off {num_extra_records} extra records.')
    df = df[request_range_mask]
    print(df_summary(df, 'weather api result'))
    return df


@validate_call
def fetch_data(
    start_ts: pd.Timestamp, end_ts: pd.Timestamp,
    use_dvc: bool = True,
) -> pd.DataFrame | None:
    """Fetch EIA-demand and weather timeseries data (in DVC warehouse format)
    for the given time range. Abstracts away the data source: Either DVC
    (cached) or live API.

    Args:
        start_ts: Start of data range to fetch
        end_ts: End of data range to fetch
        use_dvc: If use_dvc is true, we will fetch what we can from DVC and the
            rest from the live EIA API. Otherwise, all data will be fetched
            from the API.
    """
    # Decide which data sources will serve the request range
    if use_dvc:
        chunk_idx = get_chunk_index()
        hit_range, miss_range = chunk_index_intersection(chunk_idx, start_ts, end_ts)
    else:
        hit_range, miss_range = None, (start_ts, end_ts)

    # Fetch cached data from DVC
    dvc_df = get_range_from_dvc_as_df(*hit_range) if hit_range is not None else None

    # Fetch data from live APIs
    eia_df = concurrent_fetch_EIA_data(*miss_range) if miss_range is not None else None

    # TODO: TEMPORARY REMOVE. The first time we run this, we'll use the DVC cache for
    # EIA data (to reach back to 2015) but not for weather. After that, replace the
    # following call with fetch_weather_data(*miss_range)
    weather_df = fetch_weather_data(start_ts, end_ts) if miss_range is not None else None # TODO restore

    if miss_range is not None and len(eia_df) == 0:
        raise EIADataUnavailableException

    # We must apply the ETL transform to the raw api-fetched data before merging
    # with DVC-fetched data
    api_df = transform_api_data_to_dvc_form(eia_df, weather_df)
    df = concat_time_indexed_dfs([dvc_df, api_df])

    # Logging
    dvc_fetch_range = f'start:{dvc_df.index.min()}. end:{dvc_df.index.max()}' if dvc_df is not None else 'None'
    api_fetch_range = f'start:{api_df.index.min()}. end:{api_df.index.max()}' if api_df is not None else 'None'
    print(
        '\nfetch_data complete:\n'
        f'  Requested range: start:{start_ts}. end:{end_ts}.\n'
        f'  Fetched from DVC: {dvc_fetch_range}.\n'
        f'  Fetched from API: {api_fetch_range}.\n'
        f'  Fetched range: start:{df.index.min()}. end:{df.index.max()}.'
    )

    return df


@validate_call
def transform_api_data_to_dvc_form(
    eia_df: pd.DataFrame | None,
    weather_df: pd.DataFrame | None
) -> pd.DataFrame:
    """Convert types, drop duplicates, add D column, ensure every hour."""
    if eia_df is None and weather_df is None:
        return None
    assert all([eia_df is not None, weather_df is not None])

    print('Transforming raw API-fetched timeseries to DVC warehouse format.')

    # TRANSFORM EIA DATA

    # Convert types
    eia_df['value'] = pd.to_numeric(eia_df['value'])

    # EIA results can have duplicates (at the boundaries of the pages)
    # And such behavior seems to be non-deterministic.
    # Remove those duplicates
    eia_df = remove_rows_with_duplicate_indices(eia_df)

    # In the EIA API response, for any given hour, there are between 0 and 1 records:
    # 1 record for D value, or 0 if EIA has no D record. Units are MWh.
    demand_df = eia_df[eia_df.type == 'D'] # TODO can i drop this line now that we aren't fetching DF records?

    # TRANSFORM WEATHER DATA

    # Convert types
    weather_df['temperature_2m'] = pd.to_numeric(weather_df['temperature_2m'])
    weather_df['cloud_cover'] = pd.to_numeric(weather_df['cloud_cover'])

    # Create base dataframe with a timestamp for every hour in the range
    start_ts = eia_df.index.min()
    end_ts = eia_df.index.max()
    dt_df = create_timeseries_df_1h(start_ts, end_ts)

    # Merge in the EIA demand timeseries
    merge_df = pd.merge(
        dt_df,
        demand_df[['value']].rename(columns={'value': 'D'}),
        left_index=True,
        right_index=True,
        how='left',
    )

    # Merge in the weather timerseries
    merge_df = pd.merge(
        merge_df,
        weather_df,
        left_index=True,
        right_index=True,
        how='left',
    )

    print(df_summary(merge_df, 'transform_api_data result'))
    return merge_df


def get_range_from_dvc_as_df(start_ts: pd.Timestamp, end_ts: pd.Timestamp):
    """Fetch data from DVC covering the given time range."""
    print(f'Requesting data from DVC. start:{start_ts}. end:{end_ts}')
    rev = get_current_dvc_commit_hash()

    # Get logical chunk start timestamps for requested range
    req_chunk_idx = calculate_chunk_index(start_ts, end_ts)

    # Fetch chunks from DVC
    chunk_dfs = []
    for _, row in req_chunk_idx.iterrows():
        chunk_name = row['name']
        dvc_data_ref = get_dvc_ref_for_chunk(start_ts, chunk_name, rev)
        chunk_df = get_dvc_dataset_as_df(dvc_data_ref)
        chunk_dfs.append(chunk_df)

    # Concatenate data chunks
    df = concat_time_indexed_dfs(chunk_dfs)

    # Trim data down to the requested range
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]

    print(f'Fetched data from DVC. start:{df.index.min()}. end:{df.index.max()}')
    return df


def get_dvc_dataset_url(ddi: DVCDatasetInfo):
    return dvc.api.get_url(path=ddi.path, repo=ddi.repo, rev=ddi.rev)


def commit_df_to_dvc_in_chunks(df: pd.DataFrame, overwrite_index: bool):
    """Commit the given dataframe to DVC in quarterly indexed chunks.

    Args:
        df: The extracted data to be persisted to DVC in chunks.
        overwrite_index: If true, the local dvc index will be cleared: Chunks
            deleted and index file set to an empty dataframe parquet file. All
            chunks will then appear as appends. If false, only non-complete
            existing chunks can be updated and only recent data that is not covered
            by existing chunks can be appended.
    """
    assert has_full_hourly_index(df)
    gx_validate_df('dvc', df)
    git_repo = get_dvc_GitRepo_client()
    dvc_repo = get_DvcRepo_client()

    # Create an in-memory chunk index for the given dataset
    chunk_idx = calculate_chunk_index(df.index.min(), df.index.max())

    # We must compare the old chunk index with the new datasets chunk index
    # to distinguish between update and append writes
    old_chunk_idx = get_chunk_index()
    if overwrite_index:
        old_chunk_idx = clear_local_chunk_index()
    print_chunk_index_diff(old_chunk_idx, chunk_idx)
    update_starts, append_starts = diff_chunk_indices(old_chunk_idx, chunk_idx)

    # Create new chunk dataframes
    chunk_dfs = []
    for _, row in chunk_idx.iterrows():
        start_ts, end_ts, chunk_name = row['start_ts'], row['end_ts'], row['name']
        start_mask = df.index >= start_ts
        end_mask = df.index <= end_ts
        chunk_df = df[start_mask & end_mask]

        # Distinguish between update and append
        is_update = update_starts.eq(start_ts).any()
        update_strs = []
        if is_update:
            # Fetch existing data for chunk from DVC
            dvc_data_ref = get_dvc_ref_for_chunk(start_ts, chunk_name, str(git_repo.head.commit))
            dvc_chunk_df = get_dvc_dataset_as_df(dvc_data_ref)
            # Merge old and new data for chunk
            chunk_df = concat_time_indexed_dfs([dvc_chunk_df, chunk_df])
            update_strs.append(
                f'Updating chunk: {row.year}-Q{row.quarter}\n'
                f'  Logical bounds: {row.start_ts} to {row.end_ts}\n'
                f'  Old data range: {dvc_chunk_df.index.min()} to {dvc_chunk_df.index.max()}\n'
                f'  New data range: {chunk_df.index.min()} to {chunk_df.index.max()}\n'
            )
        chunk_dfs.append(chunk_df)

    # Write chunk files to disk
    local_dvc_repo_path = Path(os.getenv('DVC_LOCAL_REPO_PATH'))
    chunk_idx_path = local_dvc_repo_path / 'v1/chunk_idx.parquet'
    chunk_data_path = local_dvc_repo_path / 'v1/data'
    for i, chunk_df in enumerate(chunk_dfs):
        # Don't overwrite full chunks
        chunk_start_ts = chunk_idx.loc[i].start_ts
        # TODO this overwrite protection logic doesn't work in the overwrite index case
        old_chunk_is_complete = chunk_idx[chunk_idx.start_ts == chunk_start_ts].complete.item()
        if not old_chunk_is_complete:
            # Write new chunk to disk
            file_name = f"{chunk_idx.loc[i]['name']}.parquet"
            dataset_path = chunk_data_path / file_name
            chunk_df.to_parquet(dataset_path)
            # Add chunk to dvc tracking
            dvc_repo.add(dataset_path)
            # Stage file for git tracking
            git_repo.git.add(dataset_path.with_suffix('.parquet.dvc'))

    # Write new chunk index to disk
    new_data_start_ts, new_data_end_ts = df.index.min(), df.index.max()
    if overwrite_index:
        updated_chunk_idx = calculate_chunk_index(new_data_start_ts, new_data_end_ts)
    else:
        # Merge old and new chunk indices
        old_data_start_ts = old_chunk_idx.iloc[0].data_start_ts
        old_data_end_ts = old_chunk_idx.iloc[-1].data_end_ts
        if new_data_start_ts > old_data_end_ts + pd.Timedelta(hours=1):
            raise NotImplementedError('DVC index currently assumes no data gaps.')
        updated_chunk_idx = calculate_chunk_index(old_data_start_ts, new_data_end_ts)
    updated_chunk_idx.to_parquet(chunk_idx_path)
    update_strs.append(
        'Updated index:\n'
        f'  Chunk range: {updated_chunk_idx.iloc[0].year}-Q{updated_chunk_idx.iloc[0].quarter} '
        f'to {updated_chunk_idx.iloc[-1].year}-Q{updated_chunk_idx.iloc[-1].quarter}.\n'
        f'  Data range: {updated_chunk_idx.iloc[0].data_start_ts} '
        f'to  {updated_chunk_idx.iloc[-1].data_end_ts}\n'
    )

    # Git stage non-data files: gitignore + index
    git_repo.git.add(chunk_data_path / '.gitignore')
    git_repo.git.add(chunk_idx_path)

    # Commit and push to Git (tracking) and DVC (data files)
    diffs = git_repo.index.diff('HEAD')
    diff_files = list(map(lambda d: d.a_path, diffs))
    if len(diff_files) > 0:
        print('Staged files:')
        pp(diff_files)
        commit_msg = 'Update dataset.\n\n'
        commit_msg += '\n'.join(update_strs)
        commit = git_repo.index.commit(commit_msg)
        git_commit_hash = str(commit)
        print(f'Git commit hash: {git_commit_hash}')
        origin = git_repo.remote(name='origin')
        # Push commit
        origin.push()

        print('Pushing dataset to DVC remote storage')
        # Note: Push requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
        dvc_repo.push()
    else:
        print('No dataset changes.')


def clear_local_chunk_index():
    """Clear the local filesystem's chunk index"""
    print('Clearing the chunk index.')
    # Ensure current chunk index is on the filesystem
    _ = get_chunk_index()

    # Delete the chunk files
    local_dvc_repo_path = Path(os.getenv('DVC_LOCAL_REPO_PATH'))
    chunk_idx_path = local_dvc_repo_path / 'v1/chunk_idx.parquet'
    chunk_data_path = local_dvc_repo_path / 'v1/data'
    for filename in os.listdir(chunk_data_path):
        filepath = chunk_data_path / filename
        print(f'Deleting chunk file: {filepath}')
        os.unlink(filepath)

    # Overwrite the chunk index with an empty one
    empty_chunk_idx = calculate_chunk_index(None, None)
    empty_chunk_idx.to_parquet(chunk_idx_path)
    return empty_chunk_idx



def diff_chunk_indices(old_idx: ChunkIndex, new_idx: ChunkIndex):
    """Diff two chunk indices, returning 2 lists of logical chunk start
    timestamps: One indicating chunks that need updating, and the second
    indicating chunks that are to be appended (all new data)."""
    start_intersect_mask = old_idx.start_ts.isin(new_idx.start_ts)
    old_partial_mask = ~old_idx.complete
    update_starts = old_idx[start_intersect_mask & old_partial_mask].start_ts
    new_append_mask = ~new_idx.start_ts.isin(old_idx.start_ts)
    append_starts = new_idx[new_append_mask].start_ts
    return (
        update_starts.reset_index(drop=True),
        append_starts.reset_index(drop=True)
    )


def print_chunk_index_diff(old_idx: ChunkIndex, new_idx: ChunkIndex):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    cols = ['year', 'quarter', 'start_ts', 'end_ts', 'complete']
    update_starts, append_starts = diff_chunk_indices(old_idx, new_idx)
    append_df = new_idx[new_idx.start_ts.isin(append_starts)]

    # Ugly formatting below, but I can't seem to get textwrap.dedent to work
    print(
f"""
Updating {len(update_starts)} chunks.
{old_idx[old_idx.start_ts.isin(update_starts)][cols]}
Appending {len(append_df)} chunks.
{append_df[cols] if len(append_df) else ''}\
""")
