from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from ..invariants import InvariantSettings

if TYPE_CHECKING:
    from .profiles import StoredProfile


class ProfileContext(TypedDict):
    name: str
    description: str
    language: str
    tone: str
    detail_level: str
    response_format: str
    constraints: list[str]


@dataclass(frozen=True, slots=True)
class OrchestrationContext:
    """Personalization for this orchestration call, separate from memory and state."""

    profile: ProfileContext | None = None
    invariants: InvariantSettings | None = None


def profile_context(profile: StoredProfile) -> ProfileContext:
    return {
        "name": profile.name,
        "description": profile.description,
        "language": profile.language,
        "tone": profile.tone,
        "detail_level": profile.detail_level,
        "response_format": profile.response_format,
        "constraints": list(profile.constraints),
    }
