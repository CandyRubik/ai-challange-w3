from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class StoredMcpTraceStep:
    """A safe, persisted summary of one MCP call made for a chat message."""

    step: int
    server: str
    tool: str
    arguments: dict[str, Any]
    result: str


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: str
    role: str
    kind: str
    content: str
    created_at: datetime
    refusal: bool = False
    mcp_trace: tuple[StoredMcpTraceStep, ...] = ()
