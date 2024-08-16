import boto3
import os
import io
import pickle
import shutil
from core.utils import compact_ts_str


def get_s3_client():
    # If dev, connect to minio
    df_env = os.getenv('DF_ENVIRONMENT')
    if df_env == 'dev':
        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MINIO_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('MINIO_ROOT_USER'),
            aws_secret_access_key=os.getenv('MINIO_ROOT_PASSWORD'),
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )
    # Production connects to s3
    elif df_env == 'prod':
        raise Exception('No such thing as production environment yet.')
    else:
        raise Exception(f'Unknown environment: {df_env}')
    return s3_client


def obj_key_with_timestamps(prefix, start_ts, end_ts):
    """Generate a parquet file name that encodes a time range"""
    start_str = compact_ts_str(start_ts)
    end_str = compact_ts_str(end_ts)
    return f'{prefix}_{start_str}_{end_str}.parquet'


def df_to_parquet_buff(df):
    """Serialize the given dataframe in parquet format in an in-memory buffer"""
    buff = io.BytesIO()
    df.to_parquet(buff)
    buff.seek(0)  # Reset buffer position to the beginning
    return buff


def model_to_pickle_buff(model):
    """Serialize the given fitted model, via pickle, into an in-memory buffer.
    Assumption: The model implements sklearn's Predictor interface."""
    buff = io.BytesIO()
    pickle.dump(model, buff)
    buff.seek(0)
    return buff


def ensure_empty_dir(dir_path):
    """Ensure the given directory exists and is empty"""
    directory = os.path.dirname(dir_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    else:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
