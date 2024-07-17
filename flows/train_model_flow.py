from prefect import flow, task
from utils.storage import get_s3_client, obj_key_with_timestamps
from utils.pandas import print_df_summary
from core.data import add_temporal_features
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
    object_key = obj_key_with_timestamps('eia_d_df', start_ts, end_ts)
    print(f'Getting object: {bucket}/{object_key}.')
    s3.download_fileobj(bucket, object_key, buff)
    buff.seek(0)
    df = pd.read_parquet(buff)
    print_df_summary(df)
    return df


@task
def features(df):
    # Add temporal features
    df = add_temporal_features(df)

    # TODO: Drop the demand forecast column for now.
    # Haven't decided yet if that will be interesting, or just training my model
    # to copy EIA's model.
    df = df.drop(columns=['DF'])
    print_df_summary(df)
    return df


@flow
def train_model(log_prints=True):
    # TODO: Determine training end date (un-hardcode)
    start_ts = pd.Timestamp('2015-07-01 05:00:00+00:00')
    end_ts = pd.Timestamp('2024-06-28 04:00:00+00:00')

    # Get data
    df = get_data(start_ts, end_ts)

    # TODO: Feature Engineering
    df = features(df)

    # TODO: Train/Test Split

    # TODO: Train Model
