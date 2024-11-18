from prefect import flow, task
from core.consts import DVC_EARLIEST_DATA_HOUR
from core.data import (
    fetch_data, chunk_index_intersection, get_chunk_index,
    commit_df_to_dvc_in_chunks, clear_local_chunk_index
)
from core.types import validate_call, RedundantExtractionException
from core.utils import utcnow_minus_buffer_ts
from datetime import datetime
import pandas as pd


@task
@validate_call
def extract_and_transform(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame | None:
    """Fetch any timeseries data in the specified range that is not
    already in the DVC warehouse."""
    start_ts, end_ts = pd.Timestamp(start_ts), pd.Timestamp(end_ts)
    chunk_idx = get_chunk_index()

    # Short circuit to stop ETL if the requested data is already in DVC
    _, miss_range = chunk_index_intersection(chunk_idx, start_ts, end_ts)
    if miss_range is None:
        msg = 'Requested data range already fully covered by DVC data warehouse. ' \
              f'start:{start_ts}. end:{end_ts}'
        raise RedundantExtractionException(msg)

    return fetch_data(start_ts, end_ts)


@task
def load(df: pd.DataFrame):
    """Load the data into the data warehouse."""
    commit_df_to_dvc_in_chunks(df)


@flow(log_prints=True)
def etl(
    start_ts: datetime | None,
    end_ts: datetime | None,
):
    """Idempotently ensures all available hourly EIA demand data between the given start
    and end timestamps are persisted in the DVC data warehouse as a parquet chunk files.
    """
    if not start_ts:
        start_ts = pd.Timestamp(DVC_EARLIEST_DATA_HOUR)
    if not end_ts:
        end_ts = utcnow_minus_buffer_ts()
    print(f'ETL Flow requested range: start:{start_ts}. end:{end_ts}')

    try:
        df = extract_and_transform(start_ts, end_ts)
        load(df)
    except RedundantExtractionException as error:
        print(error)
        return
