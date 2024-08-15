"""
Module containing miscellaneous logic I haven't yet defined a place for.
"""

from datetime import datetime, timezone
from core.types import MLFlowModelSpecifier


COMPACT_TS_FORMAT = '%Y-%m-%d_%H'


def compact_ts_str(ts: datetime) -> str:
    return ts.strftime(COMPACT_TS_FORMAT)


def parse_compact_ts_str(ts: str) -> datetime:
    return datetime.strptime(ts, COMPACT_TS_FORMAT).replace(tzinfo=timezone.utc)


"""
MLFlow-related utils
"""


def mlflow_model_uri(ms: MLFlowModelSpecifier) -> str:
    name = ms.name
    version = ms.version
    return f'models:/{name}/{version}'
