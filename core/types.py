from pydantic import BaseModel
from mlflow.pyfunc import PyFuncModel
from mlflow.entities import Run


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
    run: Run

    class Config:
        arbitrary_types_allowed = True
