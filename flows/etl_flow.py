from prefect import flow, task
from utils.storage import get_s3_client
import requests
import pandas as pd
import os


@task
def extract():
    print("Downloading timeseries.")
    # TODO: Fetch EIA data
    # TODO: Demonstrate concurrent fetch of paginated API requests

    url = ('https://api.eia.gov/v2/electricity/rto/region-data/data/?'
           'frequency=hourly&data[0]=value&facets[respondent][]=PJM&'
           'facets[type][]=D&facets[type][]=DF&sort[0][column]=period&'
           'sort[0][direction]=asc')

    # Calculate the number of rows to fetch from the API between start and end
    start = pd.to_datetime('2015-07-01 05:00:00+00:00')
    end = pd.to_datetime('2024-06-28 04:00:00+00:00')
    time_span = end - start
    hours = int(time_span.total_seconds() / 3600)

    # Calculate how many paginated API requests will be required to fetch all
    # the timeseries data
    REQUEST_ROWS = 5000
    num_full_requests = hours // REQUEST_ROWS
    final_request_length = hours % REQUEST_ROWS
    print(f'Fetching {hours} hours of data. Start: {start}. End: {end}')
    print((f'Will make {num_full_requests} {REQUEST_ROWS}-length requests and '
           f'one {final_request_length}-length request.'))

    # Build a list of dataframes storing each API request (page)'s response
    response_dfs = []

    def append_EIA_page_response_df(start, end, offset, length, result_list):
        print(f'Fetching API page. offset:{offset}. length:{length}')

        params = {
            'offset': offset,
            'length': length,
            'api_key': os.environ['EIA_API_KEY'],
            'start': start.strftime('%Y-%m-%dT%H'),
            'end': end.strftime('%Y-%m-%dT%H'),
        }

        r = requests.get(url, params=params)
        r.raise_for_status()
        result_df = pd.DataFrame(r.json()['response']['data'])
        result_list.append(result_df)

    # Make the full-length requests
    for i in range(num_full_requests):
        offset = i * REQUEST_ROWS
        append_EIA_page_response_df(start, end, offset, REQUEST_ROWS,
                                    response_dfs)
    # Make the final remainder request
    append_EIA_page_response_df(start, end, num_full_requests * REQUEST_ROWS,
                                final_request_length, response_dfs)

    api_df = pd.concat(response_dfs)
    return api_df


@task
def transform(df):
    print('Transforming timeseries.')
    print(f'Dataframe length: {len(df)}')
    # Save into a dataframe
    # Convert types
    # Imputation? Or save that for later down the pipe?


@task
def load():
    print("Loading timeseries into warehouse.")
    # TODO: Convert dataframe to parquet file
    # TODO: Encode start and end times into file name
    # TODO: Write to S3
    s3_client = get_s3_client()
    buckets = s3_client.list_buckets()
    print(buckets)


@flow(log_prints=True)
def etl():
    """Pulls all available hourly EIA demand data up to the last XXXday and
    persists it in the S3 data warehouse as a parquet file.
    """
    df = extract()
    transform(df)
    load()
