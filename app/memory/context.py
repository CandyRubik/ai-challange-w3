from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class MemoryItem(TypedDict):
    category: str
    content: str


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """Explicit working and long-term memory snapshot for a single call."""

    working: tuple[MemoryItem, ...] = ()
    long_term: tuple[MemoryItem, ...] = ()
