from prefect import flow
from datetime import datetime
import pandas as pd

from core.consts import EIA_EARLIEST_HOUR_UTC
from flows.etl_flow import etl, utcnow_minus_buffer_ts
from flows.train_model_flow import train_model


@flow(log_prints=True)
def etl_and_train(
    start_ts: datetime = pd.to_datetime(EIA_EARLIEST_HOUR_UTC).to_pydatetime(),
    end_ts: datetime = utcnow_minus_buffer_ts(),
):
    dvc_dataset_info = etl(start_ts, end_ts)
    train_model(dvc_dataset_info=dvc_dataset_info)
