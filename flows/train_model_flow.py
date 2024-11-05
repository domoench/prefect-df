from prefect import flow, task, runtime
from prefect.exceptions import ObjectNotFound
from core.consts import (
    EIA_TEST_SET_HOURS, EIA_MAX_D_VAL, EIA_MIN_D_VAL,
    DVC_EARLIEST_DATA_HOUR
)
from core.data import (
    add_time_meaning_features, add_time_lag_features, add_holiday_feature,
    calculate_lag_backfill_ranges, cap_column_outliers, impute_null_demand_values,
    get_chunk_index, chunk_index_intersection, get_range_from_dvc_as_df,
    get_current_dvc_commit_hash, fetch_data
)
from core.types import ModelFeatureFlags, validate_call, ChunkIndex
from core.model import train_xgboost, get_model_features
from core.utils import (
    compact_ts_str, mlflow_endpoint_uri, concat_time_indexed_dfs,
    utcnow_minus_buffer_ts, df_summary
)
from core.gx.gx import gx_validate_df
from datetime import datetime
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
    return df


@task
@validate_call
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers and impute null values"""
    # Cap threshold values
    df = cap_column_outliers(df, 'D', EIA_MIN_D_VAL, EIA_MAX_D_VAL)
    df = impute_null_demand_values(df)
    return df


@task
@validate_call
def features(df: pd.DataFrame) -> pd.DataFrame:
    # Add temporal features
    df = add_time_meaning_features(df)
    df = add_time_lag_features(df)
    df = add_holiday_feature(df)
    print(df)
    return df


@task
@validate_call
def mlflow_emit_tags_and_params(train_df: pd.DataFrame):
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
        'dvc.commit': get_current_dvc_commit_hash(),
        'dvc.dataset.train.start': compact_ts_str(train_df.index.min()),
        'dvc.dataset.train.end': compact_ts_str(train_df.index.max()),
    })


@task
@validate_call
def train_xgb_with_tracking(train_df: pd.DataFrame, features) -> xgboost.sklearn.XGBRegressor:
    # MLFlow Tracking
    mlflow.set_tracking_uri(uri=mlflow_endpoint_uri())
    mlflow.set_experiment('xgb.df.train')
    with mlflow.start_run():
        mlflow_emit_tags_and_params(train_df)

        # Cross validation training
        # TODO: Parameterize Optional Hyper param tuning
        mlflow.xgboost.autolog()
        reg = train_xgboost(train_df, features, hyperparam_tuning=False)
    return reg


@task
def add_lag_backfill_data(df: pd.DataFrame) -> pd.DataFrame:
    """For the given datetime-indexed dataframe, fetch the same date range for
    the past 3 years, and return a dataframe with those rows prefixed. """
    df = df.copy()
    lag_dfs: list[pd.DataFrame] = []
    for (lag_start_ts, lag_end_ts) in calculate_lag_backfill_ranges(df):
        lag_df = get_range_from_dvc_as_df(lag_start_ts, lag_end_ts)
        lag_dfs.append(lag_df)
    return concat_time_indexed_dfs(lag_dfs + [df])


@task
@validate_call
def get_training_data(
    start_ts: pd.Timestamp, end_ts: pd.Timestamp, chunk_idx: ChunkIndex
) -> pd.DataFrame:
    # Fetch training data from DVC
    _, miss_range = chunk_index_intersection(chunk_idx, start_ts, end_ts)
    if miss_range is not None:
        raise NotImplementedError('We can only train on data in DVC')
    train_df = fetch_data(start_ts, end_ts)

    # Backfill lag data (also from DVC)
    train_df = add_lag_backfill_data(train_df)

    # Preprocess
    train_df = preprocess_data(train_df)
    # Strip off the historical/lag prefix data
    train_df = train_df.loc[start_ts:]
    # Validate
    gx_validate_df('train', train_df)
    return train_df


# https://docs.prefect.io/latest/concepts/flows/#parameters
@flow
def train_model(
    start_ts: datetime | None,
    end_ts: datetime | None,
    mlflow_tracking: bool = True,
    feature_flags: ModelFeatureFlags = ModelFeatureFlags(),
    log_prints=True
) -> xgboost.sklearn.XGBRegressor:
    """Train an XGBoost timeseries forecasting model

    Args:
        start_ts: Beginning of training dataset timespan (UTC)
        end_ts: End of training dataset timespan (UTC)
        mlflow_tracking: Flag to enable/disable mlflow tracking
        feature_flags: Controls which of the feature groups will be used
            in model training.
    """
    chunk_idx = get_chunk_index()
    if not start_ts:
        # Pick start date that allows 3 years of lag data
        start_ts = pd.Timestamp(DVC_EARLIEST_DATA_HOUR) + pd.DateOffset(years=3)
    if not end_ts:
        end_ts = chunk_idx.iloc[-1].data_end_ts
    start_ts, end_ts = pd.Timestamp(start_ts), pd.Timestamp(end_ts)

    # Ensure the training set leaves enough hours of EIA data for the
    end_ts = min(
        end_ts, utcnow_minus_buffer_ts() - pd.Timedelta(hours=EIA_TEST_SET_HOURS)
    )
    print(f'Training set time span: {start_ts} to {end_ts}')

    train_df = get_training_data(start_ts, end_ts, chunk_idx)
    print('Training data summary:')
    print(df_summary(train_df))

    # Preprocessing adds all feature groups to the training data set.
    # The feature flags determine which features the model will make use
    # of during training.
    features = get_model_features(feature_flags)

    if mlflow_tracking:
        return train_xgb_with_tracking(train_df, features)
    else:
        return train_xgboost(train_df, features, hyperparam_tuning=False)
