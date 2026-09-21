from dataclasses import replace
import json

import pytest

from app.agents.task_state import (
    TaskContext, TaskConflict, TaskState, approve_plan, complete_step,
    apply_validation, pause, propose_plan, replan, require_action, resume,
)


def planned_task() -> TaskContext:
    return propose_plan(TaskContext(task="Подготовить курс"), ("Темы", "Практика"), ("Есть упражнения",))


@pytest.mark.parametrize("action", ["execute_step", "validate", "advance"])
def test_unapproved_plan_rejects_skipping_approval(action: str) -> None:
    with pytest.raises(TaskConflict, match="не утверждён") as error:
        require_action(planned_task(), action)
    assert error.value.detail["allowed_actions"] == ("approve", "pause", "replan")


def test_plan_and_completed_steps_are_required() -> None:
    with pytest.raises(TaskConflict):
        approve_plan(TaskContext(task="Курс"))
    with pytest.raises(TaskConflict):
        apply_validation(approve_plan(planned_task()), passed=True, report="Итог")
    with pytest.raises(ValueError, match="завершить"):
        replace(approve_plan(planned_task()), state=TaskState.VALIDATION)


def test_progress_and_expected_action_follow_saved_results() -> None:
    ctx = approve_plan(planned_task())
    assert ctx.current == "Темы"
    first = complete_step(ctx, "Готовые темы")
    assert first.step == 1 and first.total == 2
    assert first.current == "Практика"
    assert ctx.step == 0  # Old snapshots remain immutable.
    validated = complete_step(first, "Готовые упражнения")
    assert validated.state == TaskState.VALIDATION
    assert validated.expected_action == "validate"
    assert TaskContext.from_json(validated.to_json()) == validated


def test_pause_preserves_the_stage_step_and_expected_action() -> None:
    ctx = complete_step(approve_plan(planned_task()), "Готовые темы")
    paused = pause(ctx)
    assert paused == replace(ctx, paused=True)
    with pytest.raises(TaskConflict):
        complete_step(paused, "Нельзя выполнять")
    assert resume(paused) == ctx


def test_replanning_archives_results_and_records_new_requirements() -> None:
    ctx = complete_step(approve_plan(planned_task()), "Готовые темы")
    updated = replan(ctx, "Добавить итоговый проект")
    assert updated.state == TaskState.PLANNING
    assert updated.expected_action == "generate_plan"
    assert updated.step == 0 and updated.plan == () and updated.done == ()
    assert updated.previous_results == ctx.done
    assert updated.notes == ("Добавить итоговый проект",)
    assert not updated.plan_approved
    proposed = propose_plan(updated, ("Проект",), ("Есть проект",))
    with pytest.raises(TaskConflict):
        complete_step(proposed, "Нельзя выполнять без повторного утверждения")


def test_done_is_terminal() -> None:
    ctx = complete_step(complete_step(approve_plan(planned_task()), "Темы"), "Практика")
    done = apply_validation(ctx, passed=True, report="Итоговый курс")
    assert done.validation_passed is True and done.validation_report == done.result
    for action in ("generate_plan", "approve", "execute_step", "validate", "pause", "resume", "replan", "advance"):
        with pytest.raises(TaskConflict):
            require_action(done, action)
    with pytest.raises(TaskConflict):
        pause(done)


def test_snapshots_cannot_claim_execution_or_completion_without_evidence() -> None:
    with pytest.raises(ValueError, match="утвердить"):
        replace(planned_task(), state=TaskState.EXECUTION)
    ctx = complete_step(complete_step(approve_plan(planned_task()), "Темы"), "Практика")
    with pytest.raises(ValueError, match="успешная проверка"):
        replace(ctx, state=TaskState.DONE, result="Преждевременный финал")


@pytest.mark.parametrize(("passed", "report", "repair_steps"), [
    (True, "", ()), (True, "Итог", ("Исправить",)),
    (False, "Недостатки", ()), ("true", "Итог", ()),
    (False, "Недостатки", (" ",)),
])
def test_domain_rejects_invalid_validation(passed, report, repair_steps) -> None:
    ctx = complete_step(complete_step(approve_plan(planned_task()), "Темы"), "Практика")
    with pytest.raises(ValueError):
        apply_validation(ctx, passed=passed, report=report, repair_steps=repair_steps)


@pytest.mark.parametrize("checkpoint", range(6))
def test_legacy_snapshots_keep_progress_and_gain_control_fields(checkpoint: int) -> None:
    ctx = TaskContext(task="Подготовить курс")
    for operation in (
        lambda task: propose_plan(task, ("Темы", "Практика"), ("Есть упражнения",)),
        approve_plan, lambda task: complete_step(task, "Темы"),
        lambda task: complete_step(task, "Практика"),
        lambda task: apply_validation(task, passed=True, report="Итоговый курс"),
    )[:checkpoint]:
        ctx = operation(ctx)
    legacy = json.loads(ctx.to_json())
    del legacy["plan_approved"], legacy["validation_passed"]
    if ctx.state == TaskState.AWAITING_APPROVAL:
        legacy["state"] = "planning"
    assert TaskContext.from_json(json.dumps(legacy)) == ctx


@pytest.mark.parametrize("approval", [False, "false", 1])
def test_explicit_invalid_approval_is_never_auto_approved(approval) -> None:
    data = json.loads(approve_plan(planned_task()).to_json())
    data["plan_approved"] = approval
    with pytest.raises(ValueError):
        TaskContext.from_json(json.dumps(data))


def test_repeated_controls_and_empty_step_do_not_change_snapshot() -> None:
    ctx = approve_plan(planned_task())
    with pytest.raises(TaskConflict):
        approve_plan(ctx)
    with pytest.raises(TaskConflict):
        resume(ctx)
    with pytest.raises(TaskConflict):
        pause(pause(ctx))
    with pytest.raises(ValueError):
        complete_step(ctx, " ")
    assert ctx.step == 0
