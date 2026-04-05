from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class DataAction(BaseModel):
    action_type: str


class DataObservation(BaseModel):
    data_preview: List[Dict[str, Any]]
    issues: List[str]


class StepResult(BaseModel):
    observation: DataObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResult(BaseModel):
    observation: DataObservation


class GradeResult(BaseModel):
    task: str
    score: float
    passed: bool


class StateResult(BaseModel):
    observation: DataObservation
    step_count: int
    max_steps: int
    current_task: str
    issues: List[str]
    done: bool