"""SQLite-backed persistence for chat sessions and ingest jobs.

Replaces sessions.json + ingest_jobs/*.json so multiple uvicorn workers can
share state safely (file-backed SQLite with a process lock).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "datasets" / "index"
DB_PATH = INDEX_DIR / "cad_agent.db"
LEGACY_SESSIONS = INDEX_DIR / "sessions.json"
LEGACY_JOBS_DIR = INDEX_DIR / "ingest_jobs"

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

MAX_SESSIONS_DEFAULT = 200
MAX_HISTORY_DEFAULT = 40


def _connect() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is not None:
            return _conn
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(DB_PATH),
            check_same_thread=False,
            timeout=30,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _init_schema(conn)
        _migrate_legacy(conn)
        _conn = conn
        return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            drawing TEXT,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, id);
        CREATE TABLE IF NOT EXISTS ingest_jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            files_json TEXT NOT NULL DEFAULT '[]',
            stems_json TEXT NOT NULL DEFAULT '[]',
            steps_json TEXT NOT NULL DEFAULT '[]',
            error TEXT,
            mode TEXT NOT NULL DEFAULT 'auto',
            resolved_mode TEXT,
            rebuild_embeddings INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL,
            finished_at REAL
        );
        """
    )
    conn.commit()


def _migrate_legacy(conn: sqlite3.Connection) -> None:
    """One-shot import from JSON stores if the DB is empty."""
    row = conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()
    if row and row["n"] == 0 and LEGACY_SESSIONS.exists():
        try:
            payload = json.loads(LEGACY_SESSIONS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        now = time.time()
        for session_id, raw in (payload.get("sessions") or {}).items():
            conn.execute(
                "INSERT OR REPLACE INTO sessions(id, drawing, updated_at) VALUES (?,?,?)",
                (session_id, raw.get("drawing"), now),
            )
            for msg in raw.get("history") or []:
                conn.execute(
                    "INSERT INTO messages(session_id, role, content, created_at) "
                    "VALUES (?,?,?,?)",
                    (
                        session_id,
                        msg.get("role") or "user",
                        msg.get("content") or "",
                        now,
                    ),
                )
        conn.commit()

    job_row = conn.execute("SELECT COUNT(*) AS n FROM ingest_jobs").fetchone()
    if job_row and job_row["n"] == 0 and LEGACY_JOBS_DIR.is_dir():
        for path in sorted(LEGACY_JOBS_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            _upsert_job_row(conn, data)
        conn.commit()


def get_session(
    session_id: str,
    *,
    max_history: int = MAX_HISTORY_DEFAULT,
) -> dict[str, Any]:
    conn = _connect()
    with _lock:
        row = conn.execute(
            "SELECT id, drawing FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO sessions(id, drawing, updated_at) VALUES (?,?,?)",
                (session_id, None, time.time()),
            )
            conn.commit()
            return {"drawing": None, "history": []}
        msgs = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, max_history),
        ).fetchall()
        history = [
            {"role": m["role"], "content": m["content"]} for m in reversed(msgs)
        ]
        return {"drawing": row["drawing"], "history": history}


def save_session(
    session_id: str,
    *,
    drawing: str | None,
    history: list[dict],
    max_sessions: int = MAX_SESSIONS_DEFAULT,
    max_history: int = MAX_HISTORY_DEFAULT,
) -> None:
    conn = _connect()
    with _lock:
        now = time.time()
        conn.execute(
            "INSERT INTO sessions(id, drawing, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET drawing=excluded.drawing, "
            "updated_at=excluded.updated_at",
            (session_id, drawing, now),
        )
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        for msg in history[-max_history:]:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) "
                "VALUES (?,?,?,?)",
                (
                    session_id,
                    msg.get("role") or "user",
                    msg.get("content") or "",
                    now,
                ),
            )
        # LRU trim
        ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        ]
        for old_id in ids[max_sessions:]:
            # foreign_keys=ON makes ON DELETE CASCADE clear messages.
            conn.execute("DELETE FROM sessions WHERE id=?", (old_id,))
        conn.commit()


def session_count() -> int:
    conn = _connect()
    with _lock:
        row = conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()
        return int(row["n"] if row else 0)


def _upsert_job_row(conn: sqlite3.Connection, data: dict) -> None:
    conn.execute(
        """
        INSERT INTO ingest_jobs(
            id, status, files_json, stems_json, steps_json, error,
            mode, resolved_mode, rebuild_embeddings, created_at, finished_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            status=excluded.status,
            files_json=excluded.files_json,
            stems_json=excluded.stems_json,
            steps_json=excluded.steps_json,
            error=excluded.error,
            mode=excluded.mode,
            resolved_mode=excluded.resolved_mode,
            rebuild_embeddings=excluded.rebuild_embeddings,
            created_at=excluded.created_at,
            finished_at=excluded.finished_at
        """,
        (
            data["id"],
            data.get("status") or "queued",
            json.dumps(data.get("files") or [], ensure_ascii=False),
            json.dumps(data.get("stems") or [], ensure_ascii=False),
            json.dumps(data.get("steps") or [], ensure_ascii=False),
            data.get("error"),
            data.get("mode") or "auto",
            data.get("resolved_mode"),
            1 if data.get("rebuild_embeddings", True) else 0,
            data.get("created_at") or time.time(),
            data.get("finished_at"),
        ),
    )


def save_job(data: dict) -> None:
    conn = _connect()
    with _lock:
        _upsert_job_row(conn, data)
        conn.commit()
    # Keep a JSON mirror for debugging / older tooling.
    LEGACY_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    path = LEGACY_JOBS_DIR / f"{data['id']}.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_job(job_id: str) -> dict | None:
    conn = _connect()
    with _lock:
        row = conn.execute(
            "SELECT * FROM ingest_jobs WHERE id=?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        return _job_from_row(row)


def list_jobs(limit: int = 20) -> list[dict]:
    conn = _connect()
    with _lock:
        rows = conn.execute(
            "SELECT * FROM ingest_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_job_from_row(r) for r in rows]


def _job_from_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "status": row["status"],
        "files": json.loads(row["files_json"] or "[]"),
        "stems": json.loads(row["stems_json"] or "[]"),
        "steps": json.loads(row["steps_json"] or "[]"),
        "error": row["error"],
        "mode": row["mode"],
        "resolved_mode": row["resolved_mode"],
        "rebuild_embeddings": bool(row["rebuild_embeddings"]),
        "created_at": row["created_at"],
        "finished_at": row["finished_at"],
    }
