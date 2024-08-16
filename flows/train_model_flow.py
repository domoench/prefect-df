from prefect import flow, task, runtime
from flows.utils.storage import (
    get_s3_client, model_to_pickle_buff,
)
from core.data import (
    add_temporal_features, cap_column_outliers, impute_null_demand_values,
    get_dvc_dataset_as_df, get_dvc_dataset_url,
)
from core.types import DVCDatasetInfo
from core.model import train_xgboost
from core.utils import compact_ts_str
import os
import mlflow


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


# TODO: I think this is now deprecated, as model versions are tracked by MLFlow now, right?
@task
def persist_model(model, filename):
    s3 = get_s3_client()
    model_buff = model_to_pickle_buff(model)
    s3.upload_fileobj(model_buff, 'models', filename)
    print(model)


# TODO: parameterize hyperparam tuning option. Use pydantic?
# https://docs.prefect.io/latest/concepts/flows/#parameters
@flow
def train_model(dvc_dataset_info: DVCDatasetInfo | None, log_prints=True):
    """Train an XGBoost timeseries forecasting model

    Args:
        dvc_dataset_info: Describes which dataset to pull from DVC. This timeseries
            dataset time span will cover the contiguous training and test set windows.
    """
    if dvc_dataset_info is None:
        assert False  # TODO implement.
    else:
        df = get_dvc_dataset_as_df(dvc_dataset_info)

    # Feature Engineering
    df = clean_data(df)
    df = features(df)

    # MLFlow Tracking
    mlflow.set_tracking_uri(uri=os.getenv('MLFLOW_ENDPOINT_URI'))
    mlflow.set_experiment('xgb.df.train')
    with mlflow.start_run():
        mlflow.set_tags({
            'prefect_flow_run': runtime.flow_run.name,
        })

        mlflow.log_params({
            'dvc.url': get_dvc_dataset_url(dvc_dataset_info),
            'dvc.commit': dvc_dataset_info.rev,
            'dvc.dataset.full.start': compact_ts_str(df.index.min()),
            'dvc.dataset.full.end': compact_ts_str(df.index.max()),
        })

        # Cross validation training
        # TODO: Parameterize Optional Hyper param tuning
        mlflow.xgboost.autolog()
        reg = train_xgboost(df, hyperparam_tuning=False)

    persist_model(reg, 'model.pkl')

    # TODO: Save plots as mlflow artifacts?
