from prefect import flow, task
from prefect.tasks import task_input_hash
from utils.storage import get_s3_client, filename_with_timestamps, df_to_parquet_buff
from utils.pandas import print_df_summary
from core.logging import get_logger
import requests
import pandas as pd
import os

# TODO: Now that I've learned I can't log to prefect UI from my core module
# is there any more benefit to this logging abstraction? Inside flow/task code
# should I use print statements or the prefect logger?
lg = get_logger()

EIA_MAX_REQUEST_ROWS = 5000


def request_EIA_data(start_ts, end_ts, offset, length=EIA_MAX_REQUEST_ROWS):
    lg.info(f'Fetching API page. offset:{offset}. length:{length}')
    url = ('https://api.eia.gov/v2/electricity/rto/region-data/data/?'
           'frequency=hourly&data[0]=value&facets[respondent][]=PJM&'
           'sort[0][column]=period&sort[0][direction]=asc')

    # Use list of tuples instead of dict to allow duplicate params
    params = [
      ('offset', offset),
      ('length', length),
      ('api_key', os.environ['EIA_API_KEY']),
      ('start', start_ts.strftime('%Y-%m-%dT%H')),
      ('end', end_ts.strftime('%Y-%m-%dT%H')),
      ('facets[type][]', 'D'),
      ('facets[type][]', 'DF'),
    ]

    r = requests.get(url, params=params)
    r.raise_for_status()
    return r


@task
def get_eia_data_as_df(start_ts, end_ts, offset, length=EIA_MAX_REQUEST_ROWS):
    r = request_EIA_data(start_ts, end_ts, offset, length)
    df = pd.DataFrame(r.json()['response']['data'])
    lg.info(f"  First row: {df.iloc[0]['period']}")
    lg.info(f"  Last row: {df.iloc[-1]['period']}")
    return df


# TODO Locally, the cache value disappears while the key remains. Leading to
# the errors like:
# ValueError: Path /root/.prefect/storage/fde6265e3819476eb8fbcb4f234dc9fc
# does not exist.
@task(cache_key_fn=task_input_hash)
def concurrent_fetch_EIA_data(start_ts, end_ts):
    time_span = end_ts - start_ts
    hours = int(time_span.total_seconds() / 3600)

    # Query EIA to determine exactly how many records match our time range
    r = request_EIA_data(start_ts, end_ts, 0)
    num_total_records = int(r.json()['response']['total'])
    lg.info(f'Total records to fetch: {num_total_records}')

    # Calculate how many paginated API requests will be required to fetch all
    # the timeseries data
    num_full_requests = num_total_records // EIA_MAX_REQUEST_ROWS
    final_request_length = num_total_records % EIA_MAX_REQUEST_ROWS
    lg.info((f'Fetching {hours} hours of data: {num_total_records} records.\n',
          f'Start: {start_ts}. End: {end_ts}'))
    lg.info((f'Will make {num_full_requests} {EIA_MAX_REQUEST_ROWS}-length requests '
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
    return api_df


@task
def extract(start_ts, end_ts):
    lg.info("Fetching EIA electricty demand timeseries.")

    # Calculate the number of rows to fetch from the API between start and end
    eia_df = concurrent_fetch_EIA_data(start_ts, end_ts)

    print_df_summary(eia_df)

    return eia_df


@task
def transform(eia_df):
    lg.info('Transforming timeseries.')

    # Convert types
    eia_df['UTC period'] = pd.to_datetime(eia_df['period'], utc=True)
    eia_df['value'] = pd.to_numeric(eia_df['value'])

    # Careful, EIA results can have duplicates (at the boundaries of the pages)
    # And such behavior seems to be non-deterministic.
    # Remove duplicates
    eia_df = eia_df.drop_duplicates(subset=['UTC period', 'value', 'type'])

    # In the EIA API response, for any given hour, there are between 0 and 2 records:
    # 1 record for D value, and another for the DF value. Update dataframe to have 1 row
    # for each hour, with 2 columns: D and DF. Units are MWh.
    demand_df = eia_df[eia_df.type == 'D']
    d_forecast_df = eia_df[eia_df.type == 'DF']

    # Create base dataframe with a timestamp for every hour in the range
    start_ts = eia_df['UTC period'].min()
    end_ts = eia_df['UTC period'].max()
    dt_df = pd.DataFrame({'utc_ts': pd.date_range(start=start_ts, end=end_ts, freq='h')})

    # Merge in the demand timeseries
    merge_df = pd.merge(
        dt_df,
        demand_df[["UTC period", "respondent", "value"]].rename(columns={"value": "D"}),
        left_on="utc_ts",
        right_on="UTC period",
        how="left",
    )

    # Merge in the demand forecast timeseries
    merge_df = pd.merge(
        merge_df,
        d_forecast_df[['UTC period', 'value']].rename(columns={'value': 'DF'}),
        left_on='utc_ts', right_on='UTC period',
        how='left'
    )

    merge_df = merge_df.drop(columns=['UTC period_x', 'UTC period_y'])

    # Set timestamp as index
    merge_df = merge_df.set_index('utc_ts')

    # TODO: Imputation? Or save that for later down the pipe?

    print_df_summary(merge_df)
    return merge_df


@task
def load(df):
    print("Loading timeseries into warehouse.")
    # Serialize
    df_buff = df_to_parquet_buff(df)

    # Encode start and end times into file name
    start_ts = df.index.min()
    end_ts = df.index.max()
    filename = filename_with_timestamps('eia_d_df', start_ts, end_ts)
    bucket = os.environ['TIMESERIES_BUCKET_NAME']

    # Write to S3
    s3 = get_s3_client()
    s3.upload_fileobj(df_buff, bucket, filename)
    print(f'Uploaded {bucket}/{filename}.')


@flow(log_prints=True)
def etl():
    """Pulls all available hourly EIA demand data up to the last XXXday and
    persists it in the S3 data warehouse as a parquet file.
    """
    # TODO parametarize the dataset time interval
    start_ts = pd.to_datetime('2015-07-01 05:00:00+00:00')
    end_ts = pd.to_datetime('2024-06-28 04:00:00+00:00')

    df = extract(start_ts, end_ts)
    df = transform(df)
    load(df)
