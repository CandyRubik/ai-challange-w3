"""Small persistent Checklist MCP server used by the multi-server demo."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from typing import Annotated
from uuid import uuid4

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field

from ..storage.chat_sessions import DEFAULT_CHAT_DB_PATH


class ChecklistItemView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    created_at: datetime


class ChecklistView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    items: list[ChecklistItemView]


class SQLiteChecklistRepository:
    """Persist checklist state so separate stdio calls share one flow."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self._database_path = Path(
            database_path or os.getenv("CHAT_DB_PATH") or DEFAULT_CHAT_DB_PATH,
        )
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS mcp_checklists (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mcp_checklist_items (
                    id TEXT PRIMARY KEY,
                    checklist_id TEXT NOT NULL REFERENCES mcp_checklists(id) ON DELETE CASCADE,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_mcp_checklist_items
                ON mcp_checklist_items(checklist_id, created_at);
                """,
            )
            connection.commit()

    @staticmethod
    def _item(row: sqlite3.Row) -> ChecklistItemView:
        return ChecklistItemView(
            id=row["id"],
            text=row["text"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _view(self, connection: sqlite3.Connection, checklist_id: str) -> ChecklistView:
        checklist = connection.execute(
            "SELECT id, title FROM mcp_checklists WHERE id = ?",
            (checklist_id,),
        ).fetchone()
        if checklist is None:
            raise ValueError("Чек-лист не найден")
        rows = connection.execute(
            """SELECT id, text, created_at FROM mcp_checklist_items
               WHERE checklist_id = ? ORDER BY created_at, id""",
            (checklist_id,),
        ).fetchall()
        return ChecklistView(
            id=checklist["id"],
            title=checklist["title"],
            items=[self._item(row) for row in rows],
        )

    def create(self, title: str) -> ChecklistView:
        normalized = title.strip()
        if not normalized:
            raise ValueError("Укажите название чек-листа")
        now = datetime.now(timezone.utc).isoformat()
        checklist_id = str(uuid4())
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO mcp_checklists(id, title, created_at) VALUES (?, ?, ?)",
                (checklist_id, normalized, now),
            )
            connection.commit()
            return self._view(connection, checklist_id)

    def add_item(self, checklist_id: str, text: str) -> ChecklistItemView:
        normalized = text.strip()
        if not normalized:
            raise ValueError("Укажите пункт чек-листа")
        item_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            exists = connection.execute(
                "SELECT 1 FROM mcp_checklists WHERE id = ?",
                (checklist_id,),
            ).fetchone()
            if exists is None:
                raise ValueError("Чек-лист не найден")
            count = connection.execute(
                "SELECT COUNT(*) FROM mcp_checklist_items WHERE checklist_id = ?",
                (checklist_id,),
            ).fetchone()[0]
            if count >= 100:
                raise ValueError("В чек-листе не может быть больше 100 пунктов")
            connection.execute(
                """INSERT INTO mcp_checklist_items(id, checklist_id, text, created_at)
                   VALUES (?, ?, ?, ?)""",
                (item_id, checklist_id, normalized, now),
            )
            connection.commit()
        return ChecklistItemView(id=item_id, text=normalized, created_at=datetime.fromisoformat(now))

    def get(self, checklist_id: str) -> ChecklistView:
        with closing(self._connect()) as connection:
            return self._view(connection, checklist_id)


def create_checklist_server(repository: SQLiteChecklistRepository | None = None) -> MCPServer:
    repo = repository or SQLiteChecklistRepository()
    server = MCPServer(
        "checklist",
        instructions=(
            "Use this server for an explicit request to create or update a checklist. "
            "Create the checklist first, then add items using the returned checklist id, "
            "and list it at the end to verify the complete result."
        ),
    )

    @server.tool(title="Создать чек-лист")
    async def create_checklist(
        title: Annotated[str, Field(min_length=1, max_length=200)],
    ) -> ChecklistView:
        """Создать пустой чек-лист и вернуть его идентификатор."""
        try:
            return repo.create(title)
        except ValueError as error:
            raise ToolError(str(error)) from error

    @server.tool(title="Добавить пункт в чек-лист")
    async def add_checklist_item(
        checklist_id: Annotated[str, Field(min_length=1, max_length=64)],
        text: Annotated[str, Field(min_length=1, max_length=300)],
    ) -> ChecklistItemView:
        """Добавить один пункт в существующий чек-лист."""
        try:
            return repo.add_item(checklist_id, text)
        except ValueError as error:
            raise ToolError(str(error)) from error

    @server.tool(title="Показать пункты чек-листа")
    async def list_checklist_items(
        checklist_id: Annotated[str, Field(min_length=1, max_length=64)],
    ) -> ChecklistView:
        """Вернуть чек-лист со всеми добавленными пунктами."""
        try:
            return repo.get(checklist_id)
        except ValueError as error:
            raise ToolError(str(error)) from error

    return server


mcp = create_checklist_server()


if __name__ == "__main__":
    mcp.run(transport="stdio")
