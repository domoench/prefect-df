from core.data import (
    add_time_lag_features, add_holiday_feature, calculate_lag_backfill_ranges,
    calculate_chunk_index, chunk_index_intersection, print_chunk_index_diff,
    diff_chunk_indices, get_chunk_index, fetch_data
)
from core.utils import create_timeseries_df_1h
from core.types import ChunkIndex
from core.consts import LAG_FEATURES
import pandas as pd
from pandas import Timestamp, Timedelta
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
            end_ts = Timestamp.utcnow().round('h')
        if start_ts is None:
            start_ts = end_ts - Timedelta(days=365*3)
        df = create_timeseries_df_1h(start_ts, end_ts)
        df['D'] = np.linspace(0.0, 1.0, len(df))
        return df

    return _create_timeseries_df

@pytest.fixture
def chunk_index():
    return ChunkIndex(pd.DataFrame({
        'year': [2015, 2015, 2016, 2016, 2016, 2016],
        'quarter': [3, 4, 1, 2, 3, 4],
        'start_ts': [Timestamp('2015-07-01 00:00:00+0000', tz='UTC'), Timestamp('2015-10-01 00:00:00+0000', tz='UTC'), Timestamp('2016-01-01 00:00:00+0000', tz='UTC'), Timestamp('2016-04-01 00:00:00+0000', tz='UTC'), Timestamp('2016-07-01 00:00:00+0000', tz='UTC'), Timestamp('2016-10-01 00:00:00+0000', tz='UTC')],
        'end_ts': [Timestamp('2015-09-30 23:00:00+0000', tz='UTC'), Timestamp('2015-12-31 23:00:00+0000', tz='UTC'), Timestamp('2016-03-31 23:00:00+0000', tz='UTC'), Timestamp('2016-06-30 23:00:00+0000', tz='UTC'), Timestamp('2016-09-30 23:00:00+0000', tz='UTC'), Timestamp('2016-12-31 23:00:00+0000', tz='UTC')],
        'data_start_ts': [Timestamp('2015-07-01 05:00:00+0000', tz='UTC'), Timestamp('2015-10-01 00:00:00+0000', tz='UTC'), Timestamp('2016-01-01 00:00:00+0000', tz='UTC'), Timestamp('2016-04-01 00:00:00+0000', tz='UTC'), Timestamp('2016-07-01 00:00:00+0000', tz='UTC'), Timestamp('2016-10-01 00:00:00+0000', tz='UTC')],
        'data_end_ts': [Timestamp('2015-09-30 23:00:00+0000', tz='UTC'), Timestamp('2015-12-31 23:00:00+0000', tz='UTC'), Timestamp('2016-03-31 23:00:00+0000', tz='UTC'), Timestamp('2016-06-30 23:00:00+0000', tz='UTC'), Timestamp('2016-09-30 23:00:00+0000', tz='UTC'), Timestamp('2016-12-20 23:00:00+0000', tz='UTC')],
        'name': ['2015_Q3_from_2015-07-01-00_to_2015-09-30-23', '2015_Q4_from_2015-10-01-00_to_2015-12-31-23', '2016_Q1_from_2016-01-01-00_to_2016-03-31-23', '2016_Q2_from_2016-04-01-00_to_2016-06-30-23', '2016_Q3_from_2016-07-01-00_to_2016-09-30-23', '2016_Q4_from_2016-10-01-00_to_2016-12-31-23'],
        'complete': [False, True, True, True, True, False],
    }))

@pytest.fixture
def chunk_indices():
    # Return 2 partially-overlapping chunk indices
    old_chunk_idx = ChunkIndex(pd.DataFrame({
        'year': [2015, 2015, 2016, 2016, 2016, 2016],
        'quarter': [3, 4, 1, 2, 3, 4],
        'start_ts': [Timestamp('2015-07-01 00:00:00+0000', tz='UTC'), Timestamp('2015-10-01 00:00:00+0000', tz='UTC'), Timestamp('2016-01-01 00:00:00+0000', tz='UTC'), Timestamp('2016-04-01 00:00:00+0000', tz='UTC'), Timestamp('2016-07-01 00:00:00+0000', tz='UTC'), Timestamp('2016-10-01 00:00:00+0000', tz='UTC')],
        'end_ts': [Timestamp('2015-09-30 23:00:00+0000', tz='UTC'), Timestamp('2015-12-31 23:00:00+0000', tz='UTC'), Timestamp('2016-03-31 23:00:00+0000', tz='UTC'), Timestamp('2016-06-30 23:00:00+0000', tz='UTC'), Timestamp('2016-09-30 23:00:00+0000', tz='UTC'), Timestamp('2016-12-31 23:00:00+0000', tz='UTC')],
        'data_start_ts': [Timestamp('2015-07-01 05:00:00+0000', tz='UTC'), Timestamp('2015-10-01 00:00:00+0000', tz='UTC'), Timestamp('2016-01-01 00:00:00+0000', tz='UTC'), Timestamp('2016-04-01 00:00:00+0000', tz='UTC'), Timestamp('2016-07-01 00:00:00+0000', tz='UTC'), Timestamp('2016-10-01 00:00:00+0000', tz='UTC')],
        'data_end_ts': [Timestamp('2015-09-30 23:00:00+0000', tz='UTC'), Timestamp('2015-12-31 23:00:00+0000', tz='UTC'), Timestamp('2016-03-31 23:00:00+0000', tz='UTC'), Timestamp('2016-06-30 23:00:00+0000', tz='UTC'), Timestamp('2016-09-30 23:00:00+0000', tz='UTC'), Timestamp('2016-12-20 23:00:00+0000', tz='UTC')],
        'name': ['2015_Q3_from_2015-07-01-00_to_2015-09-30-23', '2015_Q4_from_2015-10-01-00_to_2015-12-31-23', '2016_Q1_from_2016-01-01-00_to_2016-03-31-23', '2016_Q2_from_2016-04-01-00_to_2016-06-30-23', '2016_Q3_from_2016-07-01-00_to_2016-09-30-23', '2016_Q4_from_2016-10-01-00_to_2016-12-31-23'],
        'complete': [False, True, True, True, True, False],
    }))
    new_chunk_idx = ChunkIndex(pd.DataFrame({
        'year': [2016, 2016, 2017, 2017],
        'quarter': [3, 4, 1, 2],
        'start_ts': [Timestamp('2016-07-01 00:00:00+0000', tz='UTC'), Timestamp('2016-10-01 00:00:00+0000', tz='UTC'), Timestamp('2017-01-01 00:00:00+0000', tz='UTC'), Timestamp('2017-04-01 00:00:00+0000', tz='UTC')],
        'end_ts': [Timestamp('2016-09-30 23:00:00+0000', tz='UTC'), Timestamp('2016-12-31 23:00:00+0000', tz='UTC'), Timestamp('2017-03-31 23:00:00+0000', tz='UTC'), Timestamp('2017-06-30 23:00:00+0000', tz='UTC')],
        'data_start_ts': [Timestamp('2016-07-15 08:00:00+0000', tz='UTC'), Timestamp('2016-10-01 00:00:00+0000', tz='UTC'), Timestamp('2017-01-01 00:00:00+0000', tz='UTC'), Timestamp('2017-04-01 00:00:00+0000', tz='UTC')],
        'data_end_ts': [Timestamp('2016-09-30 23:00:00+0000', tz='UTC'), Timestamp('2016-12-31 23:00:00+0000', tz='UTC'), Timestamp('2017-03-31 23:00:00+0000', tz='UTC'), Timestamp('2017-06-15 10:00:00+0000', tz='UTC')],
        'name': ['2016_Q3_from_2016-07-01-00_to_2016-09-30-23', '2016_Q4_from_2016-10-01-00_to_2016-12-31-23', '2017_Q1_from_2017-01-01-00_to_2017-03-31-23', '2017_Q2_from_2017-04-01-00_to_2017-06-30-23'],
        'complete': [False, True, True, False]
    }))
    return [old_chunk_idx, new_chunk_idx]


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
        t_lag1y = t - Timedelta(LAG_DAYS_1Y)
        t_lag2y = t - Timedelta(LAG_DAYS_2Y)
        t_lag3y = t - Timedelta(LAG_DAYS_3Y)
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
        start_ts = Timestamp('2024-12-24 00:00:00+00:00')
        end_ts = Timestamp('2024-12-26 23:00:00+00:00')
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

    def test_calculate_lag_backfill_ranges_lt_1y(self, timeseries_df):
        # CASE I: Time range is less than 1 year => No overlap of lag ranges
        end_ts = Timestamp.utcnow().round('h')
        start_ts = end_ts - Timedelta(f'{4*7} days')
        df = timeseries_df(start_ts, end_ts)
        ranges = calculate_lag_backfill_ranges(df)
        lag_3y_range, lag_2y_range, lag_1y_range = ranges
        assert lag_1y_range[1] - lag_1y_range[0] == end_ts - start_ts
        assert lag_2y_range[1] - lag_2y_range[0] == end_ts - start_ts
        assert lag_3y_range[1] - lag_3y_range[0] == end_ts - start_ts
        assert lag_1y_range[0] == start_ts - Timedelta('364 days')
        assert lag_2y_range[0] == start_ts - Timedelta('728 days')
        assert lag_3y_range[0] == start_ts - Timedelta('1092 days')
        assert lag_1y_range[1] == end_ts - Timedelta('364 days')
        assert lag_2y_range[1] == end_ts - Timedelta('728 days')
        assert lag_3y_range[1] == end_ts - Timedelta('1092 days')
        # Confirm that day of week is maintained for lagged dates
        assert lag_1y_range[0].dayofweek == (start_ts - Timedelta('364 days')).dayofweek
        assert lag_2y_range[0].dayofweek == (start_ts - Timedelta('728 days')).dayofweek
        assert lag_3y_range[0].dayofweek == (start_ts - Timedelta('1092 days')).dayofweek
        assert lag_1y_range[1].dayofweek == (end_ts - Timedelta('364 days')).dayofweek
        assert lag_2y_range[1].dayofweek == (end_ts - Timedelta('728 days')).dayofweek
        assert lag_3y_range[1].dayofweek == (end_ts - Timedelta('1092 days')).dayofweek

    def test_calculate_lag_backfill_ranges_gt_1y(self, timeseries_df):
        # Time range is greater than 1 year => No overlap of lag ranges
        df = timeseries_df()
        start_ts = df.index.min()
        ranges = calculate_lag_backfill_ranges(df)
        # Only 1 backfill range, because 3 years of backfill is 1 contiguous range
        # in this case.
        assert len(ranges) == 1
        lag_range = ranges[0]
        # Start of backfill is 3 years before start of data range
        assert lag_range[0] == start_ts - Timedelta('1092 days')
        # End of backfill is the start of the data range
        assert lag_range[1] == start_ts

    def test_calculate_chunk_index(self, timeseries_df):
        start_ts = Timestamp('2023-12-24 05:00:00+00:00')
        end_ts = Timestamp('2024-10-29 20:00:00+00:00')
        chunk_idx_df = calculate_chunk_index(start_ts, end_ts)
        assert len(chunk_idx_df) == 5
        assert chunk_idx_df.iloc[0].to_dict() == {
            'year': 2023,
            'quarter': 4,
            'start_ts': Timestamp('2023-10-01 00:00:00+00:00'),  # start-inclusive
            'end_ts': Timestamp('2023-12-31 23:00:00+00:00'),  # end-exlusive
            'data_start_ts': Timestamp('2023-12-24 05:00:00+00:00'),
            'data_end_ts': Timestamp('2023-12-31 23:00:00+00:00'),
            'name': '2023_Q4_from_2023-10-01-00_to_2023-12-31-23',
            'complete': False
        }
        assert chunk_idx_df.iloc[1].to_dict() == {
            'year': 2024,
            'quarter': 1,
            'start_ts': Timestamp('2024-01-01 00:00:00+00:00'),
            'end_ts': Timestamp('2024-03-31 23:00:00+00:00'),
            'data_start_ts': Timestamp('2024-01-01 00:00:00+00:00'),
            'data_end_ts': Timestamp('2024-03-31 23:00:00+00:00'),
            'name': '2024_Q1_from_2024-01-01-00_to_2024-03-31-23',
            'complete': True
        }
        assert chunk_idx_df.iloc[4].to_dict() == {
            'year': 2024,
            'quarter': 4,
            'start_ts': Timestamp('2024-10-01 00:00:00+00:00'),
            'end_ts': Timestamp('2024-12-31 23:00:00+00:00'),
            'data_start_ts': Timestamp('2024-10-01 00:00:00+00:00'),
            'data_end_ts': Timestamp('2024-10-29 20:00:00+00:00'),
            'name': '2024_Q4_from_2024-10-01-00_to_2024-12-31-23',
            'complete': False
        }

    def test_chunk_index_intersection(self):
        chunk_idx = ChunkIndex(pd.DataFrame({
            'year': [2015, 2015, 2016, 2016, 2016, 2016],
            'quarter': [3, 4, 1, 2, 3, 4],
            'start_ts': [Timestamp('2015-07-01 00:00:00+0000', tz='UTC'), Timestamp('2015-10-01 00:00:00+0000', tz='UTC'), Timestamp('2016-01-01 00:00:00+0000', tz='UTC'), Timestamp('2016-04-01 00:00:00+0000', tz='UTC'), Timestamp('2016-07-01 00:00:00+0000', tz='UTC'), Timestamp('2016-10-01 00:00:00+0000', tz='UTC')],
            'end_ts': [Timestamp('2015-09-30 23:00:00+0000', tz='UTC'), Timestamp('2015-12-31 23:00:00+0000', tz='UTC'), Timestamp('2016-03-31 23:00:00+0000', tz='UTC'), Timestamp('2016-06-30 23:00:00+0000', tz='UTC'), Timestamp('2016-09-30 23:00:00+0000', tz='UTC'), Timestamp('2016-12-31 23:00:00+0000', tz='UTC')],
            'data_start_ts': [Timestamp('2015-07-01 05:00:00+0000', tz='UTC'), Timestamp('2015-10-01 00:00:00+0000', tz='UTC'), Timestamp('2016-01-01 00:00:00+0000', tz='UTC'), Timestamp('2016-04-01 00:00:00+0000', tz='UTC'), Timestamp('2016-07-01 00:00:00+0000', tz='UTC'), Timestamp('2016-10-01 00:00:00+0000', tz='UTC')],
            'data_end_ts': [Timestamp('2015-09-30 23:00:00+0000', tz='UTC'), Timestamp('2015-12-31 23:00:00+0000', tz='UTC'), Timestamp('2016-03-31 23:00:00+0000', tz='UTC'), Timestamp('2016-06-30 23:00:00+0000', tz='UTC'), Timestamp('2016-09-30 23:00:00+0000', tz='UTC'), Timestamp('2016-12-20 23:00:00+0000', tz='UTC')],
            'name': ['2015_Q3_from_2015-07-01-00_to_2015-09-30-23', '2015_Q4_from_2015-10-01-00_to_2015-12-31-23', '2016_Q1_from_2016-01-01-00_to_2016-03-31-23', '2016_Q2_from_2016-04-01-00_to_2016-06-30-23', '2016_Q3_from_2016-07-01-00_to_2016-09-30-23', '2016_Q4_from_2016-10-01-00_to_2016-12-31-23'],
            'complete': [False, True, True, True, True, False],
        }))

        # Intersect case 1: Data to fetch at end
        start_ts = pd.Timestamp('2016-01-15 10:00:00+00:00')
        end_ts = pd.Timestamp('2024-10-30 00:00:00+00:00')
        hit_range, miss_range = chunk_index_intersection(chunk_idx, start_ts, end_ts)
        assert hit_range[0] == start_ts
        assert hit_range[1] == Timestamp('2016-12-20 23:00:00+0000', tz='UTC')
        assert miss_range[0] == Timestamp('2016-12-21 00:00:00+0000', tz='UTC')
        assert miss_range[1] == end_ts

        # Intersect case 2: No data to fetch at end
        start_ts = pd.Timestamp('2016-01-15 10:00:00+00:00')
        end_ts = pd.Timestamp('2016-10-30 00:00:00+00:00')
        hit_range, miss_range = chunk_index_intersection(chunk_idx, start_ts, end_ts)
        assert miss_range is None
        assert hit_range[0] == start_ts
        assert hit_range[1] == end_ts

    def test_diff_chunk_indices(self, chunk_indices):
        old_chunk_idx, new_chunk_idx = chunk_indices
        update_chunk_starts, append_chunk_starts = diff_chunk_indices(old_chunk_idx, new_chunk_idx)
        assert update_chunk_starts.equals(
            pd.Series([pd.Timestamp('2016-10-01 00:00:00+00:00')])
        )
        assert append_chunk_starts.equals(
            pd.Series([
                pd.Timestamp('2017-01-01 00:00:00+00:00'),
                pd.Timestamp('2017-04-01 00:00:00+00:00'),
            ])
        )

    def test_print_chunk_index_diff(self, capsys, chunk_indices):
        old_chunk_idx, new_chunk_idx = chunk_indices
        print_chunk_index_diff(old_chunk_idx, new_chunk_idx)

        captured = capsys.readouterr()
        assert captured.out == \
            '\nUpdating 1 chunks.\n' \
            '   year  quarter                  start_ts                    end_ts  complete\n' \
            '5  2016        4 2016-10-01 00:00:00+00:00 2016-12-31 23:00:00+00:00     False\n' \
            'Appending 2 chunks.\n' \
            '   year  quarter                  start_ts                    end_ts  complete\n' \
            '2  2017        1 2017-01-01 00:00:00+00:00 2017-03-31 23:00:00+00:00      True\n' \
            '3  2017        2 2017-04-01 00:00:00+00:00 2017-06-30 23:00:00+00:00     False\n'

    """ TODO: Decide if its worth mocking all this out
    def test_fetch_data_with_dvc(self, chunk_index, monkeypatch):
        def mock_get_chunk_index():
            return chunk_index
        monkeypatch.setattr('core.data.get_chunk_index', mock_get_chunk_index)
        from unittest.mock import Mock
        mock_chunk_index_intersection = Mock()
        monkeypatch.setattr('core.data.chunk_index_intersection', mock_chunk_index_intersection)
        start_ts = chunk_index.iloc[0].data_start_ts
        end_ts = chunk_index.iloc[0].data_end_ts + pd.Timedelta(hours=1)

        # CASE I:
        fetch_data(start_ts, end_ts, use_dvc=True)
        assert mock_chunk_index_intersection.called
    """
