from prefect import flow
from datetime import datetime
import pandas as pd

from core.consts import EIA_EARLIEST_HOUR_UTC, EIA_BUFFER_HOURS
from etl_flow import etl
from train_model_flow import train_model


def end_ts() -> datetime:
    """Calculate the dataset end timestamp that leaves a buffer window (before now)
    to ensure balancing authorities have reported their data to EIA"""
    return (pd.Timestamp.utcnow().round('h') - pd.Timedelta(hours=EIA_BUFFER_HOURS)).to_pydatetime()


@flow(log_prints=True)
def etl_and_train(
    start_ts: datetime = pd.to_datetime(EIA_EARLIEST_HOUR_UTC).to_pydatetime(),
    end_ts: datetime = end_ts(),
):
    dvc_dataset_info = etl(start_ts, end_ts)
    train_model(dvc_dataset_info=dvc_dataset_info)
