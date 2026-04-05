from pydantic import BaseModel
from typing import List

class DataAction(BaseModel):
    action_type: str

class DataObservation(BaseModel):
    data_preview: List[dict]
    issues: List[str]