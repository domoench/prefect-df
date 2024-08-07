from prefect import flow, task, runtime
from utils.storage import (
    get_s3_client,
    model_to_pickle_buff,
    get_dvc_remote_repo_url,
)
from core.data import (
    add_temporal_features,
    cap_column_outliers,
    impute_null_demand_values,
)
from core.model import train_xgboost
import pandas as pd
import os
import mlflow
import dvc.api


@task
def get_data(start_ts, end_ts):
    """Get training data covering every hour between start and end timestamps.
    """
    # TODO start and end ts are no longer used. Decide how you want to specify dataset.
    # TODO: Check data warehouse for appropriate data file. If not present,
    # kick off the ETL flow.
    # For now assume it is there.
    with dvc.api.open(
        # TODO Dynamically pull path
        path='data/eia_d_df_2015-07-01_05_2024-07-17_14.parquet',
        repo=get_dvc_remote_repo_url(),
        mode='rb'
    ) as f:
        df = pd.read_parquet(f)
    return df


@task
def clean_data(df):
    """Cap outliers and impute null values"""
    # Cap threshold values determined from PJM demand data between 2015 and 2024
    MAX_D_VAL = 165_000
    MIN_D_VAL = 60_000
    df = cap_column_outliers(df, 'D', MIN_D_VAL, MAX_D_VAL)
    df = impute_null_demand_values(df)
    return df


@task
def features(df):
    # Add temporal features
    df = add_temporal_features(df)

    # TODO: Drop the demand forecast column for now.
    # Haven't decided yet if that will be interesting, or just training my model
    # to copy EIA's model.
    df = df.drop(columns=['DF'])
    print(df)
    return df


@task
def persist_model(model, filename):
    s3 = get_s3_client()
    model_buff = model_to_pickle_buff(model)
    s3.upload_fileobj(model_buff, 'models', filename)
    print(model)


# TODO: parameterize hyperparam tuning option. Use pydantic?
# https://docs.prefect.io/latest/concepts/flows/#parameters
@flow
def train_model(df: pd.DataFrame | None, log_prints=True):
    """Train an XGBoost timeseries forecasting model

    Args:
        df: A raw hourly demand dataset. If this is None, we will fetch the
            relevant dataset from the DVC data warehouse.
    """
    if df is None:
        # Fetch dataset

        # TODO: Parameterize dataset timeframe (un-hardcode), or grab the
        # most recent dataset from DVC?
        start_ts = pd.Timestamp('2015-07-01 05:00:00+00:00')
        end_ts = pd.Timestamp('2024-06-28 04:00:00+00:00')
        df = get_data(start_ts, end_ts)

    # Feature Engineering
    df = clean_data(df)
    df = features(df)

    # MLFlow Tracking
    mlflow.set_tracking_uri(uri=os.getenv('MLFLOW_ENDPOINT_URI'))
    mlflow.set_experiment('XGBoost Demand Forecast')
    mlflow.set_tag('prefect_flow_run', runtime.flow_run.name)
    mlflow.xgboost.autolog()

    # Cross validation training
    # TODO: Parameterize Optional Hyper param tuning
    reg = train_xgboost(df, hyperparam_tuning=False)

    persist_model(reg, 'model.pkl')
