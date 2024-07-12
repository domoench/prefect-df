from prefect import flow, task
from utils.storage import get_s3_client, filename_with_timestamps
from utils.pandas import print_df_summary
import pandas as pd
import io
import os


@task
def get_data(start_ts, end_ts):
    """Get training data covering every hour between start and end timestamps.
    """
    # TODO: Check data warehouse for appropriate data file. If not present,
    # kick off the ETL flow.
    # For now assume it is there.
    buff = io.BytesIO()
    s3 = get_s3_client()
    bucket = os.environ['TIMESERIES_BUCKET_NAME']
    filename = filename_with_timestamps('eia_d_df', start_ts, end_ts)
    print(f'Getting object: {bucket}/{filename}.')
    s3.download_fileobj(bucket, filename, buff)
    buff.seek(0)
    df = pd.read_parquet(buff)
    return df


@task
def features(df):
    df = df.rename(columns={
        'Datetime': 'time',
        'PJME_MW': 'load'
    })


@flow
def train_model():
    # TODO: Determine training end date

    # Get data
    start_ts = pd.Timestamp('2015-07-01 05:00:00+00:00')
    end_ts = pd.Timestamp('2024-06-28 04:00:00+00:00')
    df = get_data(start_ts, end_ts)
    print_df_summary(df)

    # TODO: Feature Engineering

    # TODO: Train/Test Split

    # TODO: Train Model
