from __future__ import annotations

import logging

from ..agents.agent import AgentMessage
from ..memory.extractor import MemoryExtractionError, MemoryExtractor
from ..orchestration.context import ProfileContext
from .context import MemoryContext
from .service import MemoryRepository


logger = logging.getLogger(__name__)


class MemoryRuntime:
    """Load and extract memory without managing profile or task transitions."""

    def __init__(
        self, repository: MemoryRepository | None = None,
        extractor: MemoryExtractor | None = None,
    ) -> None:
        self._repository = repository
        self._extractor = extractor

    def snapshot(
        self,
        session_id: str,
        profile_id: str,
    ) -> MemoryContext:
        if self._repository is None:
            return MemoryContext()
        return MemoryContext(
            tuple(
                {"category": memory.category, "content": memory.content}
                for memory in self._repository.list_working(session_id)
            ),
            tuple(
                {"category": memory.category, "content": memory.content}
                for memory in self._repository.list_long_term(profile_id)
            ),
        )

    @staticmethod
    def _memory_key(
        layer: str,
        category: str,
        content: str,
    ) -> tuple[str, str, str]:
        normalized_content = " ".join(content.split()).casefold()
        return layer, category, normalized_content

    def remember(
        self,
        *,
        session_id: str,
        profile_id: str,
        profile: ProfileContext | None,
        context: list[AgentMessage],
        content: str,
        memory: MemoryContext,
    ) -> None:
        if self._repository is None or self._extractor is None:
            return

        try:
            candidates = self._extractor.extract(
                context=context,
                current_message=content,
                profile=profile,
                working_memory=memory.working,
                long_term_memory=memory.long_term,
            )
            known = {
                self._memory_key(
                    "working",
                    entry["category"],
                    entry["content"],
                )
                for entry in memory.working
            }
            known.update(
                self._memory_key(
                    "long_term",
                    entry["category"],
                    entry["content"],
                )
                for entry in memory.long_term
            )
            for candidate in candidates:
                key = self._memory_key(
                    candidate.layer,
                    candidate.category,
                    candidate.content,
                )
                if key in known:
                    continue
                self._repository.add(
                    layer=candidate.layer,
                    category=candidate.category,
                    content=candidate.content,
                    session_id=session_id if candidate.layer == "working" else None,
                    profile_id=profile_id if candidate.layer == "long_term" else None,
                )
                known.add(key)
        except MemoryExtractionError:
            logger.warning("Automatic memory extraction failed", exc_info=True)
        except Exception:
            logger.exception("Automatic memory persistence failed")
