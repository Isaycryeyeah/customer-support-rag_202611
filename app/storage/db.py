import sqlite3
import json
from pathlib import Path
from typing import Optional

from app.config import DB_PATH
from app.storage.models import QUERY_LOGS_TABLE, FEEDBACK_TABLE


def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {QUERY_LOGS_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        risk_level TEXT,
        need_human INTEGER,
        low_confidence INTEGER,
        latency_ms REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Additive, idempotent migration: preserve existing query and feedback IDs.
    columns = {row[1] for row in cursor.execute(f'PRAGMA table_info({QUERY_LOGS_TABLE})')}
    for name, definition in {
        'request_id': 'TEXT', 'answer_type': 'TEXT', 'handoff_reason': 'TEXT',
        'trace_json': 'TEXT',
    }.items():
        if name not in columns:
            cursor.execute(f'ALTER TABLE {QUERY_LOGS_TABLE} ADD COLUMN {name} {definition}')

    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {FEEDBACK_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        log_id INTEGER,
        rating INTEGER,
        comment TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (log_id) REFERENCES query_logs (id)
    )
    """)

    conn.commit()
    conn.close()


def save_query_log(
    question: str,
    answer: str,
    risk_level: str,
    need_human: bool,
    low_confidence: bool,
    latency_ms: float,
    *, request_id=None, answer_type=None, handoff_reason=None, trace=None
) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"""
    INSERT INTO {QUERY_LOGS_TABLE} (
        question, answer, risk_level, need_human, low_confidence, latency_ms
    )
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        question,
        answer,
        risk_level,
        int(need_human),
        int(low_confidence),
        latency_ms
    ))

    log_id = cursor.lastrowid
    cursor.execute(
        f'UPDATE {QUERY_LOGS_TABLE} SET request_id=?, answer_type=?, handoff_reason=?, trace_json=? WHERE id=?',
        (request_id, answer_type, handoff_reason,
         json.dumps(trace or {}, ensure_ascii=False), log_id),
    )
    conn.commit()
    conn.close()

    return log_id


def save_feedback(
    log_id: int,
    rating: int,
    comment: Optional[str] = None
) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"""
    INSERT INTO {FEEDBACK_TABLE} (log_id, rating, comment)
    VALUES (?, ?, ?)
    """, (log_id, rating, comment))

    conn.commit()
    feedback_id = cursor.lastrowid
    conn.close()

    return feedback_id


def query_log_exists(log_id: int) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"SELECT 1 FROM {QUERY_LOGS_TABLE} WHERE id = ?", (log_id,))
    row = cursor.fetchone()
    conn.close()

    return row is not None
