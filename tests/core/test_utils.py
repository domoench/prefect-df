from core.utils import create_timeseries_df_1h, merge_intervals
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
