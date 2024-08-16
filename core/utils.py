"""
Module containing miscellaneous logic I haven't yet defined a place for.
"""

from datetime import datetime, timezone
from core.types import MLFlowModelSpecifier
from core.consts import EIA_BUFFER_HOURS
import pandas as pd


"""
datetime utils
"""

COMPACT_TS_FORMAT = '%Y-%m-%d_%H'


def compact_ts_str(ts: datetime) -> str:
    return ts.strftime(COMPACT_TS_FORMAT)


def parse_compact_ts_str(ts: str) -> datetime:
    return datetime.strptime(ts, COMPACT_TS_FORMAT).replace(tzinfo=timezone.utc)


def utcnow_minus_buffer_ts() -> datetime:
    """Calculate the full dataset end timestamp - leaving a buffer window (before now)
    to ensure balancing authorities have reported their data to EIA"""
    return (pd.Timestamp.utcnow().round('h') - pd.Timedelta(hours=EIA_BUFFER_HOURS)).to_pydatetime()


"""
MLFlow utils
"""


def mlflow_model_uri(ms: MLFlowModelSpecifier) -> str:
    name = ms.name
    version = ms.version
    return f'models:/{name}/{version}'
