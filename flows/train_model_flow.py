from prefect import flow, task, runtime
from prefect.exceptions import ObjectNotFound
from core.consts import (
    EIA_TEST_SET_HOURS, DVC_EARLIEST_DATA_HOUR, EIA_BUFFER_HOURS
)
from core.data import (
    get_chunk_index, chunk_index_intersection, get_current_dvc_commit_hash
)
from core.model import get_data_for_model_input
from core.types import ModelFeatureFlags, validate_call, ChunkIndex
from core.model import train_xgboost, get_model_features
from core.utils import (
    compact_ts_str, mlflow_endpoint_uri, utcnow_minus_buffer_ts, df_summary
)
from datetime import datetime
import mlflow
import pandas as pd
import xgboost


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
def train_xgb_with_tracking(
    train_df: pd.DataFrame, features: list, hyperparam_tuning: bool
) -> xgboost.sklearn.XGBRegressor:
    # MLFlow Tracking
    mlflow.set_tracking_uri(uri=mlflow_endpoint_uri())
    mlflow.set_experiment('xgb.df.train')
    with mlflow.start_run():
        mlflow_emit_tags_and_params(train_df)

        # Cross validation training
        # TODO: Parameterize Optional Hyper param tuning
        mlflow.xgboost.autolog()
        reg, cv_res_df = train_xgboost(train_df, features, hyperparam_tuning=hyperparam_tuning)

        # Log table of cross validation results
        filename = 'cv_results.html'
        cv_res_df.to_html(filename)
        mlflow.log_artifact(filename)
    return reg


@task
@validate_call
def get_training_data(
    start_ts: pd.Timestamp, end_ts: pd.Timestamp, chunk_idx: ChunkIndex
) -> pd.DataFrame:
    # Fetch training data from DVC
    _, miss_range = chunk_index_intersection(chunk_idx, start_ts, end_ts)
    if miss_range is not None:
        raise NotImplementedError('We can only train on data in DVC')
    return get_data_for_model_input(start_ts, end_ts)


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
        max_dvc_hour = chunk_idx.iloc[-1].data_end_ts
        max_train_hour = pd.Timestamp.utcnow().round('h') - pd.Timedelta(
            hours=EIA_BUFFER_HOURS + EIA_TEST_SET_HOURS
        )
        end_ts = min(max_dvc_hour, max_train_hour)
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

    hyperparam_tuning = False  # TODO parametrize in prefect flow run
    if mlflow_tracking:
        return train_xgb_with_tracking(train_df, features, hyperparam_tuning=hyperparam_tuning)
    else:
        return train_xgboost(train_df, features, hyperparam_tuning=hyperparam_tuning)
