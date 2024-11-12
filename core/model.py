"""
Module containing logic for ML model training.
"""

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score
import xgboost as xgb
import pandas as pd
import mlflow
from core.consts import (
    EIA_TEST_SET_HOURS, TIME_FEATURES, LAG_FEATURES, WEATHER_FEATURES,
    HOLIDAY_FEATURES, TARGET
)
from core.data import fetch_data, add_lag_backfill_data, preprocess_data
from core.types import ModelFeatureFlags, validate_call
from core.gx.gx import gx_validate_df

# Determined by hyperparam tuning on 11/12/24
DEFAULT_XGB_PARAMS = {
    'learning_rate': [0.02],
    'max_depth': [5],
    'n_estimators': [500],
    'objective': ['reg:squarederror'],
}


@validate_call
def get_model_features(feature_flags: ModelFeatureFlags = ModelFeatureFlags()) -> list:
    """Return the list of features covering the specified feature components"""
    features = TIME_FEATURES.copy()
    if feature_flags.lag:
        features += LAG_FEATURES
    if feature_flags.weather:
        features += WEATHER_FEATURES
    if feature_flags.holidays:
        features += HOLIDAY_FEATURES
    return features


@validate_call
def detect_model_features(model: mlflow.pyfunc.PyFuncModel) -> ModelFeatureFlags:
    model_input_features = set(model.metadata.get_input_schema().input_names())
    return ModelFeatureFlags(
        lag=set(LAG_FEATURES).issubset(model_input_features),
        weather=set(WEATHER_FEATURES).issubset(model_input_features),
        holidays=set(HOLIDAY_FEATURES).issubset(model_input_features),
    )


@validate_call
def model_features_str(model: mlflow.pyfunc.PyFuncModel) -> str:
    """Returns a string indicating which feature groups are present in the
    given model's input feature set.

    e.g. 'fBL__' for a model with base and lag features.
    Or 'fB__H' for a model with base and holiday features.
    Or 'fBLWH' for a model with all input features.
    """
    mff = detect_model_features(model)
    base = 'B'
    lag = 'L' if mff.lag else '_'
    weather = 'W' if mff.weather else '_'
    holidays = 'H' if mff.holidays else '_'
    s = f'f{base}{lag}{weather}{holidays}'
    return s


def train_xgboost(df, features, hyperparam_tuning=False):
    """Train an XGBoost regression estimator with a given list of features.

    - Optionally performs hyperparameter tuning
    - Performs time series cross validation
    """
    params = DEFAULT_XGB_PARAMS
    if hyperparam_tuning:
        params = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.01, 0.02, 0.04],
            'objective': ['reg:squarederror'],
        }

    hpt_str = 'hyperparameter tuning and' if hyperparam_tuning else ''
    print(f'Performing {hpt_str} cross validation')

    # Define timeseries cross validation train/test splits
    NUM_SPLITS = 8
    tss = TimeSeriesSplit(n_splits=NUM_SPLITS, test_size=EIA_TEST_SET_HOURS)

    # Perform hyperparameter tuning with time series cross validation.
    reg = GridSearchCV(xgb.XGBRegressor(eval_metric=r2_score), params, cv=tss, verbose=2)
    reg.fit(df[features], df[TARGET])

    cv_results_df = pd.DataFrame(reg.cv_results_).sort_values(by='rank_test_score')
    print(f'Cross validation results:\n{cv_results_df}')
    print(f'Best parameters:\n{reg.best_params_}')
    print(f'Best score:\n{reg.best_score_}')

    best_est = reg.best_estimator_
    print(f'Feature importances: {best_est.feature_importances_}')
    return best_est, cv_results_df


@validate_call
def get_data_for_model_input(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    df = fetch_data(start_ts, end_ts)

    # Backfill lag data (also from DVC)
    df = add_lag_backfill_data(df)

    # Preprocess
    df = preprocess_data(df)

    # Strip off the historical/lag prefix data
    df = df.loc[start_ts:]

    # Validate
    gx_validate_df('xgb_input', df)
    return df
