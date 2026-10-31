from dataclasses import dataclass
from typing import Optional


@dataclass
class QueryLog:
    question: str
    answer: str
    risk_level: str
    need_human: bool
    low_confidence: bool
    latency_ms: float
    id: Optional[int] = None
    created_at: Optional[str] = None


@dataclass
class Feedback:
    log_id: int
    rating: int
    comment: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[str] = None


QUERY_LOGS_TABLE = "query_logs"
FEEDBACK_TABLE = "feedback"
