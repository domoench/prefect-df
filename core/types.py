import pydantic
from pydantic import BaseModel, ConfigDict
from mlflow.pyfunc import PyFuncModel
from mlflow.entities import Run


def validate_call(func):
    """Wrap pyantics' validate_call decorator with a version that allows
    arbitrary types"""
    return pydantic.validate_call(func, config={'arbitrary_types_allowed': True})


class DVCDatasetInfo(BaseModel):
    # Git URL
    repo: str
    # Path to the dataset file in the DVC repo directory
    path: str
    # Git commit for this DVC dataset version
    rev: str


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
