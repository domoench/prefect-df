from pydantic import BaseModel


class DVCDatasetInfo(BaseModel):
    repo: str
    path: str
    rev: str


class MLFlowModelSpecifier(BaseModel):
    name: str
    version: int
