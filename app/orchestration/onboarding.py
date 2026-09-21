from __future__ import annotations

from dataclasses import dataclass
import logging

from ..orchestration.profile_interviewer import ProfileInterviewError, ProfileInterviewer
from .context import profile_context
from .profiles import ProfileRepository, StoredProfile


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OnboardingResult:
    profile: StoredProfile
    answer: str | None = None


class ProfileOnboarding:
    """Advance profile interviews independently of chat and task persistence."""

    def __init__(self, repository: ProfileRepository, interviewer: ProfileInterviewer) -> None:
        self._repository = repository
        self._interviewer = interviewer

    def respond(self, profile: StoredProfile, content: str) -> OnboardingResult:
        normalized = content.strip().casefold()
        if normalized in {"/skip", "skip", "пропустить", "пропусти", "по умолчанию"}:
            completed = self._repository.apply_interview_update(
                profile.id, {}, next_step=3, complete=True,
            )
            return OnboardingResult(completed)

        if profile.onboarding_step == 0:
            updated = self._repository.apply_interview_update(
                profile.id, {}, next_step=1, complete=False,
            )
            return OnboardingResult(updated, self._interviewer.questions[1])

        step = profile.onboarding_step
        try:
            inference = self._interviewer.extract(
                step=step, answer=content, profile=profile_context(profile),
            )
        except ProfileInterviewError:
            logger.warning("Automatic profile extraction failed", exc_info=True)
            return OnboardingResult(
                profile,
                "Не получилось надёжно разобрать ответ; попробуйте сформулировать "
                "ещё раз — " + self._interviewer.questions[step],
            )

        next_state = profile.onboarding.advance()
        updated = self._repository.apply_interview_update(
            profile.id, inference.values(),
            next_step=next_state.step, complete=next_state.complete,
        )
        return OnboardingResult(
            updated, None if next_state.complete else self._interviewer.questions[next_state.step],
        )
