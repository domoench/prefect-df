"""
Module containing miscellaneous logic I haven't yet defined a place for.
"""

from datetime import datetime
from core.types import MLFlowModelSpecifier


def compact_ts_str(ts: datetime) -> str:
    return ts.strftime('%Y-%m-%d_%H')


"""
MLFlow-related utils
"""


def mlflow_model_uri(ms: MLFlowModelSpecifier) -> str:
    name = ms.name
    version = ms.version
    return f'models:/{name}/{version}'
