import pandas as pd
from core.holidays import is_holiday


class TestHolidays:
    def test_is_holiday(self):
        assert is_holiday(pd.Timestamp('2024-12-25', tz='UTC').date())
        assert is_holiday(pd.Timestamp('2030-12-25', tz='UTC').date())
        assert not is_holiday(pd.Timestamp('2024-10-22', tz='UTC').date())
