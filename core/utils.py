from datetime import datetime


def compact_ts_str(ts: datetime) -> str:
    return ts.strftime('%Y-%m-%d_%H')
