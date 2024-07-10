from prefect import flow, task
from prefect.tasks import task_input_hash
from utils.storage import get_s3_client
from utils.pandas import print_df_summary
import requests
import pandas as pd
import os
import io


@task()
def request_EIA_data(start, end, offset, length):
    print(f'Fetching API page. offset:{offset}. length:{length}')
    url = ('https://api.eia.gov/v2/electricity/rto/region-data/data/?'
           'frequency=hourly&data[0]=value&facets[respondent][]=PJM&'
           'facets[type][]=D&facets[type][]=DF&sort[0][column]=period&'
           'sort[0][direction]=asc')

    params = {
        'offset': offset,
        'length': length,
        'api_key': os.environ['EIA_API_KEY'],
        'start': start.strftime('%Y-%m-%dT%H'),
        'end': end.strftime('%Y-%m-%dT%H'),
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    return pd.DataFrame(r.json()['response']['data'])


@task(cache_key_fn=task_input_hash)
def concurrent_fetch_EIA_data(start, end):
    time_span = end - start
    hours = int(time_span.total_seconds() / 3600)

    # Calculate how many paginated API requests will be required to fetch all
    # the timeseries data
    MAX_REQUEST_ROWS = 5000
    num_full_requests = hours // MAX_REQUEST_ROWS
    final_request_length = hours % MAX_REQUEST_ROWS
    print(f'Fetching {hours} hours of data. Start: {start}. End: {end}')
    print((f'Will make {num_full_requests} {MAX_REQUEST_ROWS}-length requests '
           f'and one {final_request_length}-length request.'))

    # Make the requests concurrently
    result_df_futures = []

    # Initiate the full-length requests
    for i in range(num_full_requests):
        offset = i * MAX_REQUEST_ROWS
        result_df_futures.append(request_EIA_data.submit(start, end, offset,
                                                         MAX_REQUEST_ROWS))

    # Initiate the final remainder request
    result_df_futures.append(
        request_EIA_data.submit(start, end,
                                num_full_requests * MAX_REQUEST_ROWS,
                                final_request_length)
    )

    result_dfs = [future.result() for future in result_df_futures]
    api_df = pd.concat(result_dfs)
    return api_df


@task
def extract():
    print("Fetching EIA electricty demand timeseries.")

    # Calculate the number of rows to fetch from the API between start and end
    start = pd.to_datetime('2015-07-01 05:00:00+00:00')
    end = pd.to_datetime('2024-06-28 04:00:00+00:00')
    eia_df = concurrent_fetch_EIA_data(start, end)

    print_df_summary(eia_df)

    return eia_df


@task
def transform(eia_df):
    print('Transforming timeseries.')

    # Convert types
    eia_df['UTC period'] = pd.to_datetime(eia_df['period'], utc=True)
    eia_df['value'] = pd.to_numeric(eia_df['value'])

    # In the EIA API response, for any given hour, there is 1 rows for D value
    # and another for the DF value. Update dataframe to have 1 row, with 2
    # columns: D and DF. Units are MWh.
    demand_df = eia_df[eia_df.type == 'D']
    d_forecast_df = eia_df[eia_df.type == 'DF']
    eia_df = pd.merge(
        demand_df[['UTC period', 'respondent', 'value']].rename(columns={'value': 'D'}),
        d_forecast_df[['UTC period', 'value']].rename(columns={'value': 'DF'}),
        on='UTC period')

    # TODO: Imputation? Or save that for later down the pipe?

    print_df_summary(eia_df)
    return eia_df


@task
def load(df):
    print("Loading timeseries into warehouse.")
    # Convert dataframe to parquet file
    buff = io.BytesIO()
    df.to_parquet(buff)
    buff.seek(0)  # Reset buffer position to the beginning

    # Encode start and end times into file name
    start_str = df['UTC period'].min().strftime('%Y-%m-%d_%H')
    end_str = df['UTC period'].max().strftime('%Y-%m-%d_%H')
    filename = f'eia_d_df_{start_str}_{end_str}.parquet'
    bucket = os.environ['TIMESERIES_BUCKET_NAME']

    # Write to S3
    s3 = get_s3_client()
    s3.upload_fileobj(buff, bucket, filename)
    print(f'Uploaded {bucket}/{filename}.')


@flow(log_prints=True)
def etl():
    """Pulls all available hourly EIA demand data up to the last XXXday and
    persists it in the S3 data warehouse as a parquet file.
    """
    df = extract()
    df = transform(df)
    load(df)
