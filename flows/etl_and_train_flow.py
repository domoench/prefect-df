from prefect import flow
from datetime import datetime
from core.types import ModelFeatureFlags
from flows.etl_flow import etl
from flows.train_model_flow import train_model


@flow(log_prints=True)
def etl_and_train(
    start_ts: datetime | None,
    end_ts: datetime | None,
    mlflow_tracking: bool = True,
    feature_flags: ModelFeatureFlags = ModelFeatureFlags(),
    log_prints=True
):
    # ETL to ensure training data exists in warehouse
    etl(start_ts, end_ts)
    # Train
    train_model(start_ts, end_ts, mlflow_tracking=mlflow_tracking,
                feature_flags=feature_flags, log_prints=log_prints)
