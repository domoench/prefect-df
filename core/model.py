"""
Module containing logic for ML model training.
"""

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
import pandas as pd

from core.data import TIME_FEATURES, TARGET
from core.logging import get_logger

lg = get_logger()

DEFAULT_XGB_PARAMS = {
    'learning_rate': [0.02],
    'max_depth': [5],
    'n_estimators': [1000],
    'objective': ['reg:squarederror'],
}


def train_xgboost(df, hyperparam_tuning=False):
    """Train an XGBoost regression estimator.

    - Optionally performs hyperparameter tuning
    - Performs time series cross validation
    """
    params = DEFAULT_XGB_PARAMS
    if hyperparam_tuning:
        params = {
            'n_estimators': [100, 1000],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.01, 0.02, 0.04],
            'objective': ['reg:squarederror'],
        }

    # Perform hyperparameter tuning with time series cross validation
    TEST_SET_SIZE = 14 * 24  # TODO parameterize
    NUM_SPLITS = 8  # TODO parameterize
    tss = TimeSeriesSplit(n_splits=NUM_SPLITS, test_size=TEST_SET_SIZE)
    reg = GridSearchCV(xgb.XGBRegressor(), params, cv=tss, verbose=2)

    FEATURES = TIME_FEATURES

    hpt_str = 'hyperparameter tuning and' if hyperparam_tuning else ''
    lg.info(f'Performing {hpt_str} cross validation')
    reg.fit(df[FEATURES], df[TARGET])

    cv_results_df = pd.DataFrame(reg.cv_results_).sort_values(by='rank_test_score')
    lg.info(f'Cross validation results:\n{cv_results_df}')
    lg.info(f'Best parameters:\n{reg.best_params_}')

    best_est = reg.best_estimator_
    lg.info(f'Feature importances: {best_est.feature_importances_}')
    return best_est
