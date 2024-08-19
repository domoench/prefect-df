"""
Module containing logic for ML model training.
"""

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
import pandas as pd

from core.data import TIME_FEATURES, TARGET
from core.consts import EIA_TEST_SET_HOURS

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

    hpt_str = 'hyperparameter tuning and' if hyperparam_tuning else ''
    print(f'Performing {hpt_str} cross validation')

    # Define timeseries cross validation train/test splits
    NUM_SPLITS = 8
    tss = TimeSeriesSplit(n_splits=NUM_SPLITS, test_size=EIA_TEST_SET_HOURS)

    # Perform hyperparameter tuning with time series cross validation.
    reg = GridSearchCV(xgb.XGBRegressor(), params, cv=tss, verbose=2)
    reg.fit(df[TIME_FEATURES], df[TARGET])

    cv_results_df = pd.DataFrame(reg.cv_results_).sort_values(by='rank_test_score')
    print(f'Cross validation results:\n{cv_results_df}')
    print(f'Best parameters:\n{reg.best_params_}')

    best_est = reg.best_estimator_
    print(f'Feature importances: {best_est.feature_importances_}')
    return best_est
