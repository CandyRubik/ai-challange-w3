from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from ..storage.chat_sessions import DEFAULT_CHAT_DB_PATH


class WeatherChatSessionNotFound(LookupError):
    pass


class SQLiteWeatherChatRepository:
    """Profileless, principal-isolated conversation history for the weather page."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self._database_path = Path(
            database_path or os.getenv("CHAT_DB_PATH") or DEFAULT_CHAT_DB_PATH
        )
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS weather_chat_sessions (
                    id TEXT PRIMARY KEY,
                    principal_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_weather_chat_principal
                    ON weather_chat_sessions(principal_id, updated_at);
                CREATE TABLE IF NOT EXISTS weather_chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES weather_chat_sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_weather_chat_messages_session
                    ON weather_chat_messages(session_id, created_at);
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create(self, principal_id: str) -> dict:
        session_id = str(uuid4())
        now = self._timestamp()
        with closing(self._connect()) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM weather_chat_sessions WHERE principal_id = ?",
                (principal_id,),
            ).fetchone()[0]
            if count >= 20:
                raise ValueError("Достигнут лимит в 20 погодных чатов для этого доступа")
            connection.execute(
                "INSERT INTO weather_chat_sessions (id, principal_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, principal_id, now, now),
            )
            connection.commit()
        return {"id": session_id, "messages": []}

    def get(self, session_id: str, principal_id: str) -> dict:
        with closing(self._connect()) as connection:
            session = connection.execute(
                "SELECT id FROM weather_chat_sessions WHERE id = ? AND principal_id = ?",
                (session_id, principal_id),
            ).fetchone()
            if session is None:
                raise WeatherChatSessionNotFound(session_id)
            rows = connection.execute(
                """SELECT id, role, content, created_at FROM weather_chat_messages
                   WHERE session_id = ? ORDER BY created_at, rowid""",
                (session_id,),
            ).fetchall()
        return {
            "id": session_id,
            "messages": [
                {
                    "id": row["id"], "role": row["role"], "content": row["content"],
                    "created_at": datetime.fromisoformat(row["created_at"]),
                    "kind": "message", "refusal": False,
                }
                for row in rows
            ],
        }

    def append_exchange(
        self, session_id: str, principal_id: str, user_content: str, assistant_content: str,
    ) -> dict:
        now = self._timestamp()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM weather_chat_sessions WHERE id = ? AND principal_id = ?",
                (session_id, principal_id),
            ).fetchone()
            if exists is None:
                raise WeatherChatSessionNotFound(session_id)
            connection.executemany(
                """INSERT INTO weather_chat_messages (id, session_id, role, content, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (str(uuid4()), session_id, "user", user_content.strip(), now),
                    (str(uuid4()), session_id, "assistant", assistant_content, now),
                ],
            )
            connection.execute(
                "UPDATE weather_chat_sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            connection.execute(
                """DELETE FROM weather_chat_messages WHERE session_id = ? AND id NOT IN (
                       SELECT id FROM weather_chat_messages WHERE session_id = ?
                       ORDER BY created_at DESC, rowid DESC LIMIT 200
                   )""",
                (session_id, session_id),
            )
            connection.commit()
        return self.get(session_id, principal_id)
