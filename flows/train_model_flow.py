from prefect import flow, task, runtime
from prefect.exceptions import ObjectNotFound
from flows.etl_flow import concurrent_fetch_EIA_data, transform
from core.consts import EIA_TEST_SET_HOURS, EIA_MAX_D_VAL, EIA_MIN_D_VAL
from core.data import (
    add_time_meaning_features, add_time_lag_features, add_holiday_feature,
    calculate_lag_backfill_ranges, cap_column_outliers,
    impute_null_demand_values, get_dvc_dataset_as_df, get_dvc_dataset_url
)
from core.types import DVCDatasetInfo, ModelFeatureFlags, validate_call
from core.model import train_xgboost, get_model_features
from core.utils import (
    compact_ts_str, mlflow_endpoint_uri, concat_time_indexed_dfs
)
from core.gx.gx import gx_validate_df
import mlflow
import pandas as pd
import xgboost


@task
@validate_call
def preprocess_data(df: pd.DataFrame):
    """Data cleaning and feature engineering.

    Args:
        df: The full length (train + test time window) raw data set from the warehouse
    Returns:
        df: Training dataset (test set removed)
    """
    # Feature Engineering
    df = clean_data(df)
    df = features(df)

    # Remove a TEST_SET_SIZE window at the end, so that after cross validation
    # and refit, the final training set excludes that window for use by adhoc model
    # evaluation on the most recent TEST_SET_SIZE hours.
    # TODO Write test to confirm test set is stripped of the end
    df = df[:-EIA_TEST_SET_HOURS]
    return df


@task
@validate_call
def clean_data(df: pd.DataFrame):
    """Cap outliers and impute null values"""
    # Cap threshold values
    df = cap_column_outliers(df, 'D', EIA_MIN_D_VAL, EIA_MAX_D_VAL)
    df = impute_null_demand_values(df)
    return df


@task
@validate_call
def features(df: pd.DataFrame):
    # Add temporal features
    df = add_time_meaning_features(df)
    df = add_time_lag_features(df)
    df = add_holiday_feature(df)
    print(df)
    return df


@task
@validate_call
def mlflow_emit_tags_and_params(train_df: pd.DataFrame, dvc_dataset_info: DVCDatasetInfo):
    """Emit relevant model training tags and params for this mlflow run.

    This function assumes it will be called in an mlflow run context.
    """
    flow_run_name = None
    try:
        flow_run_name = getattr(runtime.flow_run, 'name', None)
    except ObjectNotFound:
        pass  # We are not in a prefect flow context
    mlflow.set_tags({
        'prefect_flow_run': flow_run_name,
    })

    mlflow.log_params({
        'dvc.url': get_dvc_dataset_url(dvc_dataset_info),
        'dvc.commit': dvc_dataset_info.rev,
        'dvc.dataset.train.start': compact_ts_str(train_df.index.min()),
        'dvc.dataset.train.end': compact_ts_str(train_df.index.max()),
    })


@task
@validate_call
def train_xgb_with_tracking(train_df: pd.DataFrame, features, dvc_dataset_info: DVCDatasetInfo):
    # MLFlow Tracking
    mlflow.set_tracking_uri(uri=mlflow_endpoint_uri())
    mlflow.set_experiment('xgb.df.train')
    reg = None
    with mlflow.start_run():
        mlflow_emit_tags_and_params(train_df, dvc_dataset_info)

        # Cross validation training
        # TODO: Parameterize Optional Hyper param tuning
        mlflow.xgboost.autolog()
        reg = train_xgboost(train_df, features, hyperparam_tuning=False)
    return reg


@task
def add_lag_backfill_data(df: pd.DataFrame):
    """For the given datetime-indexed dataframe, fetch the same date range for
    the past 3 years, and return a dataframe with those rows prefixed. """
    df = df.copy()
    lag_dfs = []
    for (lag_start_ts, lag_end_ts) in calculate_lag_backfill_ranges(df):
        lag_df = concurrent_fetch_EIA_data(lag_start_ts, lag_end_ts)
        lag_dfs.append(lag_df)
    # df has already had T from ETL applied, since it was fetched from the warehouse.
    # For backfill, we must apply the same transform.
    lag_df = pd.concat(lag_dfs)
    lag_df = transform(lag_df)
    lag_df = lag_df.tz_convert(df.index.tz)

    # Concatenate and drop duplicates at the boundaries of the backfill and
    # original dataset
    concat_df = concat_time_indexed_dfs([lag_df, df])
    return concat_df


@task
def get_training_data():
    # Fetch training data from DVC
    # Backfill lag data (also from DVC)
    # Preprocess
    # Strip off backfill prefix AND eval suffix (move from preprocess_data)
    # Run gx validation
    pass  # TODO implement above


# https://docs.prefect.io/latest/concepts/flows/#parameters
@flow
def train_model(
    dvc_dataset_info: DVCDatasetInfo,
    mlflow_tracking: bool = True,
    feature_flags: ModelFeatureFlags = ModelFeatureFlags(),
    log_prints=True
) -> xgboost.sklearn.XGBRegressor:
    """Train an XGBoost timeseries forecasting model

    Args:
        dvc_dataset_info: Describes which full dataset to pull from DVC. This timeseries
            dataset time span will cover the contiguous training and test set windows.
        mlflow_tracking: Flag to enable/disable mlflow tracking
    """
    # Replace following block with get_training_data
    train_df = get_dvc_dataset_as_df(dvc_dataset_info)
    start_ts = train_df.index.min()
    train_df = add_lag_backfill_data(train_df)
    train_df = preprocess_data(train_df)
    # Strip off the historical/lag prefix data
    train_df = train_df.loc[start_ts:]
    # Validate training data
    gx_validate_df('train', train_df)

    # Preprocessing adds all feature groups to the training data set.
    # The feature flags determine which features the model will make use
    # of during training.
    features = get_model_features(feature_flags)

    if mlflow_tracking:
        return train_xgb_with_tracking(train_df, features, dvc_dataset_info)
    else:
        return train_xgboost(train_df, features, hyperparam_tuning=False)
