from core.data import add_time_lag_features, add_holiday_feature
from core.utils import create_timeseries_df_1h
from core.consts import LAG_FEATURES
import pandas as pd
import numpy as np
import pytest


LAG_DAYS_1Y = '364 days'
LAG_DAYS_2Y = '728 days'
LAG_DAYS_3Y = '1092 days'


@pytest.fixture
def timeseries_df():
    # Creates a dataframe indexed by hourly timestamps
    def _create_timeseries_df(start_ts=None, end_ts=None):
        if end_ts is None:
            end_ts = pd.Timestamp.utcnow().round('h')
        if start_ts is None:
            start_ts = end_ts - pd.Timedelta(days=365*3)
        df = create_timeseries_df_1h(start_ts, end_ts)
        df['D'] = np.linspace(0.0, 1.0, len(df))
        return df

    return _create_timeseries_df


class TestData:
    def test_add_time_lag_features(self, timeseries_df):
        df = add_time_lag_features(timeseries_df())

        # Lag columns were added
        for feature in LAG_FEATURES:
            assert feature in df.columns

        # Confirm, for a given row, that the lag values are correct
        # Timestamps of interest
        t = df.index.max()
        # Trick: Offset by 364 days => lagged day is same day of week
        t_lag1y = t - pd.Timedelta(LAG_DAYS_1Y)
        t_lag2y = t - pd.Timedelta(LAG_DAYS_2Y)
        t_lag3y = t - pd.Timedelta(LAG_DAYS_3Y)
        # Confirm this rows lag column values match the D value of their respective rows
        assert df.loc[t, 'lag_1y'] == df.loc[t_lag1y, 'D']
        assert df.loc[t, 'lag_2y'] == df.loc[t_lag2y, 'D']
        assert df.loc[t, 'lag_3y'] == df.loc[t_lag3y, 'D']
        # Confirm that day of week is maintained for lagged dates
        assert pd.to_datetime(t).dayofweek == pd.to_datetime(t_lag1y).dayofweek
        assert pd.to_datetime(t).dayofweek == pd.to_datetime(t_lag2y).dayofweek
        assert pd.to_datetime(t).dayofweek == pd.to_datetime(t_lag3y).dayofweek

        # Where lag is not defined, values should be nan
        t = df.index.min()
        df.loc[t, 'lag_1y'] == np.NaN
        df.loc[t, 'lag_2y'] == np.NaN
        df.loc[t, 'lag_3y'] == np.NaN

    def test_add_holiday_feature(self, timeseries_df):
        # 3 days, sandwiching christmas
        start_ts = pd.Timestamp('2024-12-24 00:00:00+00:00')
        end_ts = pd.Timestamp('2024-12-26 23:00:00+00:00')
        df = timeseries_df(start_ts, end_ts)
        df = add_holiday_feature(df)
        assert 'is_holiday' in df.columns
        # First 24 hours are 12/24
        assert (df.iloc[:24]['is_holiday'] == 0).all()
        # 2nd 24 hours are Christmas holiday
        assert (df.iloc[24:48]['is_holiday'] == 1).all()
        # 34d 24 hours are 12/26
        assert (df.iloc[48:]['is_holiday'] == 0).all()
        assert isinstance(df['is_holiday'].iloc[0], np.int64)
