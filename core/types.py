from pydantic import BaseModel


# TODO move
class DVCDatasetInfo(BaseModel):
    repo: str
    path: str
    rev: str
