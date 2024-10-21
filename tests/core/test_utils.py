from core.utils import create_timeseries_df_1h
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
