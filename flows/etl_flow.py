from prefect import flow, task
from prefect.tasks import task_input_hash
from core.consts import EIA_EARLIEST_HOUR_UTC, EIA_MAX_REQUEST_ROWS
from core.data import (
    request_EIA_data, get_dvc_remote_repo_url, get_dvc_GitRepo_client,
    get_dvc_dataset_as_df, get_DvcRepo_client, chunk_index_intersection,
    get_chunk_index, ChunkIndex, commit_df_to_dvc_in_chunks
)
from core.types import DVCDatasetInfo, validate_call, EIADataUnavailableException
from core.utils import (
    ensure_empty_dir, df_summary, utcnow_minus_buffer_ts,
    create_timeseries_df_1h, remove_rows_with_duplicate_indices,
    concat_time_indexed_dfs
)
from datetime import datetime
from pathlib import Path
import os
import pandas as pd


@task
@validate_call
def get_eia_data_as_df(
    start_ts: datetime, end_ts: datetime, offset=0, length=EIA_MAX_REQUEST_ROWS
) -> pd.DataFrame:
    """Fetch EIA power demand data for the given time range and API page offset.
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


@task(cache_key_fn=task_input_hash, refresh_cache=(os.getenv('DF_ENVIRONMENT') == 'dev'))
@validate_call
def concurrent_fetch_EIA_data(start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    time_span = end_ts - start_ts
    hours = int(time_span.total_seconds() / 3600)

    # Metadata Query: determine exactly how many EIA records exist that match
    # our time range
    r = request_EIA_data(start_ts, end_ts, 0)
    num_total_records = int(r.json()['response']['total'])
    if num_total_records != hours:
        # TODO should we fail here instead?
        print(f'Warning: EIA does not have all the data we are requesting.'
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
    print('api_df')
    print(api_df)

    # EIA results quantize the search to the day, so often extra hours are present
    # in the result.
    request_range_mask = (api_df.index >= start_ts) & (api_df.index <= end_ts)
    num_extra_records = (~request_range_mask).sum()
    if num_extra_records > 0:
        print(f'Trimming off {num_extra_records} extra records.')
    api_df = api_df[request_range_mask]
    return api_df


@task
@validate_call
def extract(start_ts: datetime, end_ts: datetime) -> pd.DataFrame | None:
    """Fetch any EIA demand timeseries data in the specified range that is not
    already in the DVC warehouse."""
    chunk_idx = get_chunk_index()
    hit_range, miss_range = chunk_index_intersection(chunk_idx, start_ts, end_ts)
    if miss_range:
        fetched_df = concurrent_fetch_EIA_data(*miss_range)

        print(f'Requested range: start:{start_ts}. end:{end_ts}')
        print(f'   Cached range: start:{chunk_idx.iloc[0].data_start_ts}.'
              f' end:{chunk_idx.iloc[-1].data_end_ts}.')
        print(f'      Hit range: start:{hit_range[0]}. end:{hit_range[1]}.')
        if miss_range:
            print(f'     Miss range: start:{miss_range[0]}. end:{miss_range[1]}.')

        if len(fetched_df) == 0:
            raise EIADataUnavailableException
        print(f'Fetched range: start:{fetched_df.index.min()}. end:{fetched_df.index.max()}')
        print('Fetched data summary:')
        print(df_summary(fetched_df))
        return fetched_df


@task
@validate_call
def transform(eia_df: pd.DataFrame) -> pd.DataFrame:
    """Convert types, drop duplicates, add D column, ensure every hour."""
    print('Transforming timeseries.')

    # Convert types
    eia_df['value'] = pd.to_numeric(eia_df['value'])

    # EIA results can have duplicates (at the boundaries of the pages)
    # And such behavior seems to be non-deterministic.
    # Remove those duplicates
    eia_df = remove_rows_with_duplicate_indices(eia_df)

    # In the EIA API response, for any given hour, there are between 0 and 1 records:
    # 1 record for D value, or 0 if EIA has no D record. Units are MWh.
    demand_df = eia_df[eia_df.type == 'D']

    # Create base dataframe with a timestamp for every hour in the range
    start_ts = eia_df.index.min()
    end_ts = eia_df.index.max()
    dt_df = create_timeseries_df_1h(start_ts, end_ts)

    # Merge in the demand timeseries
    merge_df = pd.merge(
        dt_df,
        demand_df[['value']].rename(columns={'value': 'D'}),
        left_index=True,
        right_index=True,
        how='left',
    )

    print(df_summary(merge_df))
    return merge_df


@task
def load(df: pd.DataFrame):
    """Load the data into the data warehouse."""
    commit_df_to_dvc_in_chunks(df)


@flow(log_prints=True)
def etl(
    start_ts: datetime = pd.to_datetime(EIA_EARLIEST_HOUR_UTC).to_pydatetime(),
    end_ts: datetime = utcnow_minus_buffer_ts(),
):
    """Idempotently ensures all available hourly EIA demand data between the given start
    and end timestamps are persisted in the DVC data warehouse as a parquet chunk files.
    """
    df = extract(start_ts, end_ts)
    if df is None:
        print('Requested data range already exists in DVC data warehouse.')
        return
    df = transform(df)
    load(df)
