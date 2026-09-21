from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
import json


class TaskState(StrEnum):
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DONE = "done"


class TaskConflict(ValueError):
    """The requested action cannot be applied to the current snapshot."""

    def __init__(self, message: str, *, ctx: TaskContext | None = None, code: str = "task_conflict") -> None:
        super().__init__(message)
        self.detail = {"code": code, "message": message}
        if ctx is not None:
            self.detail.update(
                state=ctx.state, paused=ctx.paused, expected_action=ctx.expected_action,
                allowed_actions=ctx.allowed_actions,
            )


@dataclass(frozen=True, slots=True)
class StepResult:
    title: str
    output: str


@dataclass(frozen=True, slots=True)
class TaskContext:
    task: str
    state: TaskState = TaskState.PLANNING
    step: int = 0  # Number of completed steps in the current plan.
    plan: tuple[str, ...] = ()
    done: tuple[StepResult, ...] = ()
    criteria: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    previous_results: tuple[StepResult, ...] = ()
    validation_report: str = ""
    result: str = ""
    paused: bool = False
    plan_approved: bool = False
    validation_passed: bool | None = None

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("Задача не должна быть пустой")
        if type(self.plan_approved) is not bool or (self.validation_passed is not None and type(self.validation_passed) is not bool):
            raise ValueError("Признаки утверждения и проверки должны быть логическими")
        if self.step != len(self.done) or not 0 <= self.step <= self.total:
            raise ValueError("Прогресс не соответствует сохранённым результатам")
        if self.state != TaskState.PLANNING and (not self.plan or not self.criteria):
            raise ValueError("Для выполнения нужен план и критерии готовности")
        if self.state == TaskState.PLANNING and (self.plan or self.criteria):
            raise ValueError("Сформированный план должен ожидать утверждения")
        if self.state in {TaskState.PLANNING, TaskState.AWAITING_APPROVAL}:
            if self.step or self.plan_approved or self.validation_passed is not None or self.validation_report or self.result:
                raise ValueError("Новый план ещё не утверждён и не выполнен")
        elif not self.plan_approved:
            raise ValueError("Сначала пользователь должен утвердить план")
        if self.state in {TaskState.VALIDATION, TaskState.DONE} and self.step != self.total:
            raise ValueError("Сначала нужно завершить все шаги")
        if self.state == TaskState.EXECUTION and self.step == self.total:
            raise ValueError("Все шаги выполнены — нужна проверка")
        if self.state == TaskState.DONE and (not self.result or self.paused):
            raise ValueError("Завершённая задача должна содержать итог")
        if self.state == TaskState.DONE and (self.validation_passed is not True or not self.validation_report.strip()):
            raise ValueError("Для завершения нужна успешная проверка")
        if self.state != TaskState.DONE and (self.result or self.validation_passed is True):
            raise ValueError("Итог доступен только после успешной проверки")
        if any(not text.strip() for text in (*self.plan, *self.criteria)):
            raise ValueError("Шаги и критерии не должны быть пустыми")
        if any(item.title != self.plan[index] or not item.output.strip() for index, item in enumerate(self.done)):
            raise ValueError("Результаты должны соответствовать выполненным шагам")

    @property
    def total(self) -> int:
        return len(self.plan)

    @property
    def expected_action(self) -> str:
        if self.state == TaskState.PLANNING:
            return "generate_plan"
        return {
            TaskState.AWAITING_APPROVAL: "approve_plan",
            TaskState.EXECUTION: "execute_step",
            TaskState.VALIDATION: "validate",
            TaskState.DONE: "none",
        }[self.state]

    @property
    def allowed_actions(self) -> tuple[str, ...]:
        if self.state == TaskState.DONE:
            return ()
        if self.paused:
            return ("resume",)
        action = "approve" if self.expected_action == "approve_plan" else self.expected_action
        return (action, "pause", "replan") if self.state != TaskState.VALIDATION else (action, "pause")

    @property
    def current(self) -> str:
        if self.state == TaskState.EXECUTION:
            return self.plan[self.step]
        return {
            "generate_plan": "Сформировать план",
            "approve_plan": "Утвердить план",
            "validate": "Проверить результат по критериям",
            "none": "Задача завершена",
        }[self.expected_action]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "total": self.total,
            "current": self.current,
            "expected_action": self.expected_action,
            "allowed_actions": self.allowed_actions,
        }

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, value: str) -> TaskContext:
        data = json.loads(value)
        data["state"] = TaskState(data["state"])
        # Upgrade trusted, persisted Day 13/14 snapshots without changing SQLite.
        # Explicit new fields are never overwritten, so invalid new snapshots fail.
        if "plan_approved" not in data:
            if data["state"] == TaskState.PLANNING and data["plan"]:
                data["state"] = TaskState.AWAITING_APPROVAL
            data["plan_approved"] = data["state"] in {TaskState.EXECUTION, TaskState.VALIDATION, TaskState.DONE}
        if "validation_passed" not in data:
            if data["state"] == TaskState.DONE:
                data["validation_passed"] = True
            else:
                data["validation_passed"] = False if data["validation_report"] else None
        for field in ("plan", "criteria", "notes"):
            data[field] = tuple(data[field])
        for field in ("done", "previous_results"):
            data[field] = tuple(StepResult(**item) for item in data[field])
        return cls(**data)


TRANSITIONS = {
    TaskState.PLANNING: frozenset({TaskState.AWAITING_APPROVAL}),
    TaskState.AWAITING_APPROVAL: frozenset({TaskState.EXECUTION, TaskState.PLANNING}),
    TaskState.EXECUTION: frozenset({TaskState.VALIDATION, TaskState.PLANNING}),
    TaskState.VALIDATION: frozenset({TaskState.DONE, TaskState.EXECUTION}),
    TaskState.DONE: frozenset(),
}


def require_action(ctx: TaskContext, action: str) -> str:
    """Resolve the legacy advance alias and reject actions before any model call."""
    if action == "advance" and ctx.expected_action in {"generate_plan", "execute_step", "validate"}:
        action = ctx.expected_action
    if action not in ctx.allowed_actions:
        if ctx.state == TaskState.DONE:
            reason = "Задача уже завершена"
        elif ctx.paused:
            reason = "Задача на паузе. Сначала нажмите «Продолжить»"
        elif ctx.state == TaskState.AWAITING_APPROVAL and action in {"execute_step", "validate", "advance"}:
            reason = f"План ещё не утверждён. Следующее действие: {ctx.current}"
        else:
            reason = f"Действие «{action}» сейчас недоступно. Следующее действие: {ctx.current}"
        raise TaskConflict(reason, ctx=ctx, code="task_action_forbidden")
    return action


def _transition(ctx: TaskContext, target: TaskState, **changes) -> TaskContext:
    if ctx.paused:
        raise TaskConflict("Задача на паузе. Сначала нажмите «Продолжить»")
    if target not in TRANSITIONS[ctx.state]:
        raise TaskConflict(f"Переход {ctx.state} → {target} запрещён")
    return replace(ctx, **changes, state=target)


def propose_plan(ctx: TaskContext, plan: tuple[str, ...], criteria: tuple[str, ...]) -> TaskContext:
    require_action(ctx, "generate_plan")
    return _transition(ctx, TaskState.AWAITING_APPROVAL, plan=plan, criteria=criteria)


def approve_plan(ctx: TaskContext) -> TaskContext:
    require_action(ctx, "approve")
    return _transition(ctx, TaskState.EXECUTION, plan_approved=True)


def complete_step(ctx: TaskContext, output: str) -> TaskContext:
    require_action(ctx, "execute_step")
    done = (*ctx.done, StepResult(ctx.current, output))
    changes = {"step": ctx.step + 1, "done": done}
    if ctx.step + 1 == ctx.total:
        return _transition(ctx, TaskState.VALIDATION, **changes)
    return replace(ctx, **changes)


def pause(ctx: TaskContext) -> TaskContext:
    require_action(ctx, "pause")
    return replace(ctx, paused=True)


def resume(ctx: TaskContext) -> TaskContext:
    require_action(ctx, "resume")
    return replace(ctx, paused=False)


def replan(ctx: TaskContext, note: str) -> TaskContext:
    require_action(ctx, "replan")
    if not note.strip():
        raise ValueError("Укажите, что изменить в плане")
    changes = {
        "plan": (), "criteria": (), "step": 0, "done": (),
        "notes": (*ctx.notes, note),
        "previous_results": (*ctx.previous_results, *ctx.done),
        "validation_report": "", "result": "",
        "plan_approved": False, "validation_passed": None,
    }
    if ctx.state != TaskState.PLANNING:
        return _transition(ctx, TaskState.PLANNING, **changes)
    return replace(ctx, **changes)


def apply_validation(ctx: TaskContext, *, passed: bool, report: str, repair_steps: tuple[str, ...] = ()) -> TaskContext:
    require_action(ctx, "validate")
    if type(passed) is not bool or not report.strip() or (passed and repair_steps) or (not passed and not repair_steps):
        raise ValueError("Некорректный результат проверки")
    if passed:
        return _transition(
            ctx, TaskState.DONE, result=report, validation_report=report, validation_passed=True,
        )
    return _transition(
        ctx, TaskState.EXECUTION, plan=(*ctx.plan, *repair_steps),
        validation_report=report, validation_passed=False,
    )


@dataclass(frozen=True, slots=True)
class StoredTask:
    context: TaskContext
    revision: int
    progress_revision: int
