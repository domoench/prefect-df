from prefect import flow, task, runtime
from prefect.blocks.system import Secret
from utils.storage import (
    get_s3_client,
    obj_key_with_timestamps,
    model_to_pickle_buff,
)
from core.data import (
    add_temporal_features,
    cap_column_outliers,
    impute_null_demand_values,
)
from core.model import train_xgboost
from core.logging import get_logger
import pandas as pd
import os
import mlflow
import dvc.api

lg = get_logger()


@task
def get_data(start_ts, end_ts):
    """Get training data covering every hour between start and end timestamps.
    """
    # TODO start and end ts are no longer used. Decide how you want to specify dataset.
    # TODO: Check data warehouse for appropriate data file. If not present,
    # kick off the ETL flow.
    # For now assume it is there.
    # TODO: Pull dvc repo name generation into function. Used in 2 places now.
    github_PAT = Secret.load('github-pat-dvc-dev').get()
    github_username = os.getenv('DVC_GIT_USERNAME')
    github_reponame = os.getenv('DVC_GIT_REPONAME')
    git_repo_url = f'https://{github_username}:{github_PAT}@github.com/{github_username}/{github_reponame}.git'
    with dvc.api.open(
        # TODO Dynamically pull path
        path='data/eia_d_df_2015-07-01_05_2024-07-17_14.parquet',
        repo=git_repo_url,
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
    lg.info(df)
    return df


@task
def persist_model(model, filename):
    s3 = get_s3_client()
    model_buff = model_to_pickle_buff(model)
    s3.upload_fileobj(model_buff, 'models', filename)
    lg.info(model)


# TODO: parameterize hyperparam tuning option. Use pydantic?
# https://docs.prefect.io/latest/concepts/flows/#parameters
@flow
def train_model(log_prints=True):
    # TODO: Determine training end date (un-hardcode)
    start_ts = pd.Timestamp('2015-07-01 05:00:00+00:00')
    end_ts = pd.Timestamp('2024-06-28 04:00:00+00:00')

    # Get data
    df = get_data(start_ts, end_ts)
    df = clean_data(df)

    # Feature Engineering
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
