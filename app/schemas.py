from typing import Optional, List, Dict, Literal
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    log_id: int
    request_id: str
    answer_type: Literal['generated', 'low_confidence', 'safety_refusal', 'service_unavailable']
    question: str
    answer: str
    risk_info: Dict
    low_confidence: bool
    need_human: bool
    handoff_reason: Optional[str] = None
    ticket: Optional[Dict] = None
    contexts: List[Dict]
    generation: Optional[Dict] = None
    timings: Dict


class FeedbackRequest(BaseModel):
    log_id: int
    rating: int = Field(ge=-1, le=1)
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: int
