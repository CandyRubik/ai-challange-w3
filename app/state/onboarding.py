from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OnboardingState:
    step: int = 0
    complete: bool = False

    def advance(self) -> OnboardingState:
        return OnboardingState(step=min(self.step + 1, 3), complete=self.step == 3)
