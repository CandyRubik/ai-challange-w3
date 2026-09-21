"""Compatibility imports; task state belongs to app.state."""

from ..state.task import (
    TRANSITIONS, StepResult, TaskConflict, TaskContext, TaskState,
    apply_validation, approve_plan, complete_step, pause, propose_plan, replan, require_action, resume,
)
