"""
Module containing miscellaneous logic I haven't yet defined a place for.
"""

from datetime import datetime, timezone
import boto3
import io
import os
import pickle
import pandas as pd
import shutil
from core.types import MLFlowModelSpecifier, validate_call
from core.consts import EIA_BUFFER_HOURS


class InvalidExecutionEnvironmentError(Exception):
    pass


"""
datetime utils
"""

COMPACT_TS_FORMAT = '%Y-%m-%d_%H'


def compact_ts_str(ts: datetime) -> str:
    return ts.strftime(COMPACT_TS_FORMAT)


def parse_compact_ts_str(ts: str) -> datetime:
    return datetime.strptime(ts, COMPACT_TS_FORMAT).replace(tzinfo=timezone.utc)


def utcnow_minus_buffer_ts() -> datetime:
    """Calculate the full dataset end timestamp - leaving a buffer window (before now)
    to ensure balancing authorities have reported their data to EIA"""
    return (pd.Timestamp.utcnow().round('h') - pd.Timedelta(hours=EIA_BUFFER_HOURS)).to_pydatetime()


"""
MLFlow utils
"""


def mlflow_endpoint_uri():
    port = os.getenv('MLFLOW_TRACKING_PORT')
    df_env = os.getenv('DF_ENVIRONMENT')
    if df_env == 'prod':
        service_name = os.getenv('MLFLOW_SERVICE_ENDPOINT_PRIVATE')
        return f'http://{service_name}:{port}'
    elif df_env == 'dev':
        return f'http://mlflow:{port}'
    else:
        raise InvalidExecutionEnvironmentError(df_env)


def mlflow_model_uri(ms: MLFlowModelSpecifier) -> str:
    name = ms.name
    version = ms.version
    return f'models:/{name}/{version}'


"""
Pandas
"""


def df_summary(df) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer)
    return f'Dataframe info:\n{buffer.getvalue()}\n' \
           f'Dataframe summary:\n{df}\n'


@validate_call
def create_timeseries_df_1h(start_ts, end_ts) -> pd.DataFrame:
    df = pd.DataFrame({'utc_ts': pd.date_range(start=start_ts, end=end_ts, freq='h')})
    df = df.set_index('utc_ts')
    return df


def remove_rows_with_duplicate_indices(df: pd.DataFrame):
    """Where there are rows with duplicate timestamps, remove all but the
    first."""
    dupe_mask = df.index.duplicated(keep='first')
    return df[~dupe_mask]


def concat_time_indexed_dfs(dfs) -> pd.DataFrame:
    """Concatenate the given list of DatetimeIndex-ed dataframes, ensuring any
    duplicate rows (e.g. when input dataframes have overlapping time ranges)
    are removed."""
    concat_df = pd.concat(dfs)
    concat_df = remove_rows_with_duplicate_indices(concat_df)
    concat_df = concat_df.sort_index()
    return concat_df


def has_full_hourly_index(df: pd.DataFrame) -> bool:
    start_ts, end_ts = df.index.min(), df.index.max()
    expected_range = pd.date_range(start=start_ts, end=end_ts, freq='H')
    return df.index.equals(expected_range)


"""
Persistance
"""


def minio_endpoint_url():
    port = os.getenv('MINIO_API_PORT')
    return f'http://minio:{port}'


def get_s3_client():
    # If dev, connect to minio
    df_env = os.getenv('DF_ENVIRONMENT')
    if df_env == 'dev':
        s3_client = boto3.client(
            's3',
            endpoint_url=minio_endpoint_url(),
            aws_access_key_id=os.getenv('MINIO_ROOT_USER'),
            aws_secret_access_key=os.getenv('MINIO_ROOT_PASSWORD'),
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )
    # Production connects to s3
    elif df_env == 'prod':
        # TODO this is no longer true
        raise Exception('No such thing as production environment yet.')
    else:
        raise InvalidExecutionEnvironmentError(df_env)
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
    """Ensure the given directory exists and is empty."""
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


"""
General
"""


def merge_intervals(intervals):
    """Standard iterative interval merge algorithm"""
    # Sort intervals by their start
    intervals = sorted(intervals, key=lambda x: x[0])

    merged = []
    curr_s, curr_e = intervals[0]

    for next_s, next_e in intervals[1:]:
        # CASE I: Current overlaps with next
        if curr_e >= next_s:
            # Merge the intevals: Extend the current for next iteration
            curr_e = max(curr_e, next_e)
        # CASE II: Current does not overlap with next
        else:
            merged.append((curr_s, curr_e))
            curr_s, curr_e = next_s, next_e
    # Finally append the current interval
    merged.append((curr_s, curr_e))

    return merged
