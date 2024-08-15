from pydantic import BaseModel
from mlflow.pyfunc import PyFuncModel
from mlflow.entities import RunInfo


class DVCDatasetInfo(BaseModel):
    repo: str
    path: str
    rev: str


"""
MLFlow-related types
"""


class MLFlowModelSpecifier(BaseModel):
    name: str
    version: int


class MLFlowModelInfo(BaseModel):
    specifier: MLFlowModelSpecifier
    model: PyFuncModel
    run_info: RunInfo

    class Config:
        arbitrary_types_allowed = True
