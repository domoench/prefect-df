from pydantic import BaseModel


class DVCDatasetInfo(BaseModel):
    repo: str
    path: str
    rev: str
