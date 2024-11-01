import pydantic
import pandera as pa
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from mlflow.pyfunc import PyFuncModel
from mlflow.entities import Run


def validate_call(func):
    """Wrap pyantics' validate_call decorator with a version that allows
    arbitrary types"""
    return pydantic.validate_call(func, config={'arbitrary_types_allowed': True, 'strict': True})


"""
DVC Types
"""


class DVCDatasetInfo(BaseModel):
    # Git URL
    repo: str
    # Path to the dataset file in the DVC repo directory
    path: str
    # Git commit for this DVC dataset version
    rev: str


class ChunkIndexSchema(pa.DataFrameModel):
    year: int
    quarter: int
    start_ts: pd.DatetimeTZDtype(tz='UTC')  # type: ignore
    end_ts: pd.DatetimeTZDtype(tz='UTC')  # type: ignore
    data_start_ts: pd.DatetimeTZDtype(tz='UTC')  # type: ignore
    data_end_ts: pd.DatetimeTZDtype(tz='UTC')  # type: ignore
    name: str
    complete: bool


# Wrapper class for dataframe, enforcing that it follows the chunk index schema
class ChunkIndex(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ChunkIndexSchema.validate(self)


"""
Core model types
"""


# Specify which groups of features (in addition to the base features) the
# xgboost model should take as input
class ModelFeatureFlags(BaseModel):
    lag: bool = False
    weather: bool = False
    holidays: bool = False


"""
MLFlow-related types
"""


class MLFlowModelSpecifier(BaseModel):
    name: str
    version: int


class MLFlowModelInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    specifier: MLFlowModelSpecifier
    model: PyFuncModel
    run: Run


"""
Exceptions
"""

class EIADataUnavailableException(Exception):
    pass
