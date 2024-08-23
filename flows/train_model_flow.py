from prefect import flow, task, runtime
from core.consts import EIA_TEST_SET_HOURS, EIA_MAX_D_VAL, EIA_MIN_D_VAL
from core.data import (
    add_temporal_features, cap_column_outliers, impute_null_demand_values,
    get_dvc_dataset_as_df, get_dvc_dataset_url,
)
from core.types import DVCDatasetInfo
from core.model import train_xgboost
from core.utils import compact_ts_str, mlflow_endpoint_uri
from core.gx.gx import run_gx_checkpoint
import mlflow
import pandas as pd


@task
def preprocess_data(df):
    # Feature Engineering
    df = clean_data(df)
    df = features(df)

    # Remove a TEST_SET_SIZE window at the end, so that after cross validation
    # and refit, the final training set excludes that window for use by adhoc model
    # evaluation on the most recent TEST_SET_SIZE hours.
    # TODO add expectation that rows are sorted by time
    df = df[:-EIA_TEST_SET_HOURS]
    return df


@task
def clean_data(df):
    """Cap outliers and impute null values"""
    # Cap threshold values
    df = cap_column_outliers(df, 'D', EIA_MIN_D_VAL, EIA_MAX_D_VAL)
    df = impute_null_demand_values(df)
    return df


@task
def features(df):
    # Add temporal features
    df = add_temporal_features(df)
    print(df)
    return df


@task
def mlflow_emit_tags_and_params(train_df: pd.DataFrame, dvc_dataset_info: DVCDatasetInfo):
    """Emit relevant model training tags and params for this mlflow run.

    This function assumes it will be called in an mlflow run context.
    """
    mlflow.set_tags({
        'prefect_flow_run': getattr(runtime.flow_run, 'name', None),
    })

    mlflow.log_params({
        'dvc.url': get_dvc_dataset_url(dvc_dataset_info),
        'dvc.commit': dvc_dataset_info.rev,
        'dvc.dataset.train.start': compact_ts_str(train_df.index.min()),
        'dvc.dataset.train.end': compact_ts_str(train_df.index.max()),
    })


@task
def train_xgb_with_tracking(train_df: pd.DataFrame, dvc_dataset_info: DVCDatasetInfo):
    # Validate training data
    run_gx_checkpoint('train', train_df)

    # MLFlow Tracking
    mlflow.set_tracking_uri(uri=mlflow_endpoint_uri())
    mlflow.set_experiment('xgb.df.train')
    with mlflow.start_run():
        mlflow_emit_tags_and_params(train_df, dvc_dataset_info)

        # Cross validation training
        # TODO: Parameterize Optional Hyper param tuning
        mlflow.xgboost.autolog()
        train_xgboost(train_df, hyperparam_tuning=False)


# TODO: parameterize hyperparam tuning option. Use pydantic?
# https://docs.prefect.io/latest/concepts/flows/#parameters
@flow
def train_model(dvc_dataset_info: DVCDatasetInfo, log_prints=True):
    """Train an XGBoost timeseries forecasting model

    Args:
        dvc_dataset_info: Describes which full dataset to pull from DVC. This timeseries
            dataset time span will cover the contiguous training and test set windows.
    """
    train_df = get_dvc_dataset_as_df(dvc_dataset_info)

    train_df = preprocess_data(train_df)

    train_xgb_with_tracking(train_df, dvc_dataset_info)
