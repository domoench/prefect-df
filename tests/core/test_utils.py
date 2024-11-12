from core.utils import (
    create_timeseries_df_1h, merge_intervals, remove_rows_with_duplicate_indices,
    concat_time_indexed_dfs, has_full_hourly_index, interval_intersection
)
import pandas as pd


class TestUtils:
    def test_create_timeseries_df_1h(self):
        num_hours = 5
        end_ts = pd.Timestamp.utcnow().round('h')
        start_ts = end_ts - pd.Timedelta(hours=num_hours)
        df = create_timeseries_df_1h(start_ts, end_ts)
        assert isinstance(df.index[0], pd.Timestamp)
        assert df.index.min() == start_ts
        assert df.index.max() == end_ts

        # End timestamp is included
        assert len(df) == num_hours + 1

    def test_merge_intervals(self):
        # No overlaps
        intervals = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
        assert merge_intervals(intervals) == [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]

        # Multiple overlaps
        intervals = [(0.0, 1.0), (0.7, 0.8), (0.1, 0.9), (0.2, 0.8)]
        assert merge_intervals(intervals) == [(0.0, 1.0)]

        # Some overlaps, some non-overlaps
        intervals = [(0.9, 1.0), (0.0, 0.5), (0.25, 0.75)]
        assert merge_intervals(intervals) == [(0.0, 0.75), (0.9, 1.0)]

    def test_remove_rows_with_duplicate_indices(self):
        # Create a df with duplicates
        num_hours = 3
        end_ts = pd.Timestamp.utcnow().round('h')
        start_ts = end_ts - pd.Timedelta(hours=num_hours)
        df = create_timeseries_df_1h(start_ts, end_ts)
        dupe_df = pd.concat([df, df])
        assert not dupe_df.index.is_unique
        dupes = dupe_df[dupe_df.index.duplicated(keep='first')]
        assert len(dupes) == num_hours + 1

        # Remove duplicates
        deduped_df = remove_rows_with_duplicate_indices(dupe_df)
        assert (deduped_df.index == df.index).all()

    def test_concat_time_indexed_dfs_overlapping(self):
        end_ts_1 = pd.Timestamp('2024-01-01 00:00:00+00:00')
        start_ts_1 = end_ts_1 - pd.Timedelta(hours=10)
        df_1 = create_timeseries_df_1h(start_ts_1, end_ts_1)

        end_ts_2 = pd.Timestamp('2024-01-01 05:00:00+00:00')
        start_ts_2 = end_ts_2 - pd.Timedelta(hours=10)
        df_2 = create_timeseries_df_1h(start_ts_2, end_ts_2)

        df = concat_time_indexed_dfs([df_2, df_1]) # Input order doesn't matter

        assert (df.index == create_timeseries_df_1h(start_ts_1, end_ts_2).index).all()

    def test_concat_time_indexed_dfs_nonoverlapping(self):
        end_ts_1 = pd.Timestamp('2024-01-01 00:00:00+00:00')
        start_ts_1 = end_ts_1 - pd.Timedelta(hours=2)
        df_1 = create_timeseries_df_1h(start_ts_1, end_ts_1)

        end_ts_2 = pd.Timestamp('2024-01-01 05:00:00+00:00')
        start_ts_2 = end_ts_2 - pd.Timedelta(hours=2)
        df_2 = create_timeseries_df_1h(start_ts_2, end_ts_2)

        df = concat_time_indexed_dfs([df_1, df_2])
        index_match = df.index == [
            pd.Timestamp('2023-12-31 22:00:00+0000', tz='UTC'),
            pd.Timestamp('2023-12-31 23:00:00+0000', tz='UTC'),
            pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC'),
            pd.Timestamp('2024-01-01 03:00:00+0000', tz='UTC'),
            pd.Timestamp('2024-01-01 04:00:00+0000', tz='UTC'),
            pd.Timestamp('2024-01-01 05:00:00+0000', tz='UTC')
        ]
        assert index_match.all()

    def test_has_full_hourly_index(self):
        # Full index
        end_ts = pd.Timestamp('2024-01-01 00:00:00+00:00')
        start_ts = end_ts - pd.Timedelta(hours=100)
        df = create_timeseries_df_1h(start_ts, end_ts)
        assert has_full_hourly_index(df)

        # Index with missing entries
        end_ts_1 = pd.Timestamp('2024-01-01 00:00:00+00:00')
        start_ts_1 = end_ts_1 - pd.Timedelta(hours=2)
        df_1 = create_timeseries_df_1h(start_ts_1, end_ts_1)
        end_ts_2 = pd.Timestamp('2024-01-01 05:00:00+00:00')
        start_ts_2 = end_ts_2 - pd.Timedelta(hours=2)
        df_2 = create_timeseries_df_1h(start_ts_2, end_ts_2)
        df = concat_time_indexed_dfs([df_1, df_2])
        assert not has_full_hourly_index(df)

    def test_interval_intersection(self):
        a = (0.0, 1.0)
        b = (0.5, 1.0)
        assert interval_intersection(a, b) == (0.5, 1.0)
        assert interval_intersection(a, b) == interval_intersection(b, a)

        a = (0.5, 1.0)
        b = (0.2, 0.7)
        assert interval_intersection(a, b) == (0.5, 0.7)

        a = (0.0, 1.0)
        b = (0.5, 0.7)
        assert interval_intersection(a, b) == (0.5, 0.7)

        a = (0.0, 1.0)
        b = (1.5, 2.0)
        assert interval_intersection(a, b) is None
