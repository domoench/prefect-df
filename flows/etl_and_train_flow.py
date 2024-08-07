from prefect import flow
from datetime import datetime
import pandas as pd

from consts import EIA_EARLIEST_HOUR_UTC
from etl_flow import etl
from train_model_flow import train_model


@flow(log_prints=True)
def etl_and_train(
    start_ts: datetime = pd.to_datetime(EIA_EARLIEST_HOUR_UTC).to_pydatetime(),
    end_ts: datetime = (pd.Timestamp.utcnow().round('h') - pd.Timedelta(weeks=1)).to_pydatetime()
):
    df = etl(start_ts, end_ts)
    print(df.head())
    # TODO for some reason passing the df directly to the subflow hangs.
    # Instead pass a DVC reference and let train_model fetch it.
    train_model(df=None)
