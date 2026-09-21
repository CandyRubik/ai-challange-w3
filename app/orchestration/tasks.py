from __future__ import annotations

from ..agents.agent import Agent
from ..memory.context import MemoryContext
from ..invariants import InvariantPolicy
from ..state.task import TaskContext, apply_validation, complete_step, propose_plan, require_action
from .context import OrchestrationContext


class TaskOrchestrator:
    """Choose the next model stage and apply explicit domain transitions."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def advance(
        self, ctx: TaskContext, *, orchestration: OrchestrationContext,
        memory: MemoryContext,
    ) -> tuple[TaskContext, str]:
        require_action(ctx, ctx.expected_action)
        policy = InvariantPolicy(orchestration.invariants)
        upper = orchestration.invariants is not None and orchestration.invariants.uppercase_enabled
        if ctx.expected_action == "generate_plan":
            output = self._agent.plan_task(ctx, orchestration=orchestration, memory=memory)
            updated = propose_plan(
                ctx, tuple(text.upper() if upper else text for text in output.plan),
                tuple(text.upper() if upper else text for text in output.criteria),
            )
            answer = output.summary + "\n\nПлан:\n" + "\n".join(
                f"{index}. {title}" for index, title in enumerate(output.plan, 1)
            ) + "\n\nКритерии готовности:\n" + "\n".join(
                "• " + criterion for criterion in output.criteria
            ) + "\n\nУтвердите план или укажите изменения."
            return updated, answer
        if ctx.expected_action == "execute_step":
            answer = self._agent.execute_task_step(ctx, orchestration=orchestration, memory=memory)
            answer = policy.apply(answer)
            return complete_step(ctx, answer), answer
        if ctx.expected_action == "validate":
            output = self._agent.validate_task(ctx, orchestration=orchestration, memory=memory)
            report = policy.apply(output.report)
            updated = apply_validation(
                ctx, passed=output.passed, report=report,
                repair_steps=tuple(text.upper() if upper else text for text in output.repair_steps),
            )
            answer = report
            if not output.passed:
                answer += "\n\nШаги исправления:\n" + "\n".join(output.repair_steps)
            return updated, answer
        raise AssertionError("Неизвестный этап генерации")
