from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
from threading import Event

from fastapi.testclient import TestClient
import pytest

from app.agents.agent import Agent, AgentOutputError
from app.agents.task_state import TaskConflict
from app.main import app, get_chat_session_service
from app.providers.deepseek import LlmRequestError, LlmTruncatedResponseError
from app.schemas import TaskActionRequest, UserProfileCreateRequest
from app.services.chat_sessions import ChatSessionService, SQLiteChatSessionRepository
from app.services.memory import SQLiteMemoryRepository
from app.services.profiles import SQLiteProfileRepository


PLAN = json.dumps({
    "summary": "Подготовим курс для новичков",
    "plan": ["Определить темы", "Составить упражнения"],
    "criteria": ["Два занятия с упражнениями"],
}, ensure_ascii=False)
SUCCESS = json.dumps({"passed": True, "report": "Критерий выполнен. Итог: курс с упражнениями", "repair_steps": []}, ensure_ascii=False)
FAILURE = json.dumps({"passed": False, "report": "Нужны ответы к упражнениям", "repair_steps": ["Добавить ответы"]}, ensure_ascii=False)


class TaskModel:
    def __init__(self, answers: list[str | Exception]) -> None:
        self.answers = answers
        self.snapshots: list[dict] = []
        self.prompts: list[str] = []

    def generate(self, *, messages, max_tokens=2_000) -> str:
        self.snapshots.append(json.loads(messages[-1]["content"]))
        self.prompts.append(messages[0]["content"])
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def service_at(path: Path, model: TaskModel) -> ChatSessionService:
    return ChatSessionService(SQLiteChatSessionRepository(path), Agent(model))


def act(service: ChatSessionService, session_id: str, action="advance", content=""):
    task = service.get(session_id).task
    assert task is not None
    return service.task_action(session_id, TaskActionRequest(action=action, revision=task.revision, content=content))


def start(service: ChatSessionService):
    return service.start_task(service.create().id, "Курс для новичков: два занятия с упражнениями")


def prepare(service: ChatSessionService, session_id: str, checkpoint: int) -> None:
    for action in ["advance", "approve", "advance", "advance"][:checkpoint]:
        act(service, session_id, action)


@pytest.mark.parametrize("checkpoint", range(5))
def test_pause_and_resume_after_restart_at_every_stage(tmp_path: Path, checkpoint: int) -> None:
    path = tmp_path / "tasks.sqlite3"
    original_model = TaskModel([PLAN, "Темы курса", "Упражнения курса"])
    service = service_at(path, original_model)
    session = start(service)
    prepare(service, session.id, checkpoint)
    before = service.get(session.id).task
    assert before is not None
    act(service, session.id, "pause")
    with pytest.raises(TaskConflict, match="паузе"):
        act(service, session.id)

    restarted_model = TaskModel([PLAN if checkpoint == 0 else SUCCESS if checkpoint == 4 else "Следующий результат"])
    restarted = service_at(path, restarted_model)
    paused = restarted.get(session.id).task
    assert paused is not None and paused.paused
    assert paused.state == before.state and paused.step == before.step
    assert paused.expected_action == before.expected_action and paused.done == before.done
    resumed = act(restarted, session.id, "resume").task
    assert resumed is not None and not resumed.paused
    assert restarted_model.snapshots == []  # Controls do not call the model.
    act(restarted, session.id, "approve" if checkpoint == 1 else "advance")
    if checkpoint != 1:
        snapshot = restarted_model.snapshots[0]
        assert snapshot["task"] == before.task
        assert snapshot["step"] == before.step
        assert [item["output"] for item in snapshot["done"]] == [item.output for item in before.done]


def test_validation_returns_to_execution_without_repeating_completed_steps(tmp_path: Path) -> None:
    model = TaskModel([PLAN, "Темы", "Упражнения", FAILURE, "Ответы", SUCCESS])
    service = service_at(tmp_path / "tasks.sqlite3", model)
    session = start(service)
    prepare(service, session.id, 4)
    repaired = act(service, session.id).task
    assert repaired is not None
    assert repaired.state == "execution" and repaired.step == 2 and repaired.total == 3
    assert repaired.validation_passed is False and repaired.validation_report and not repaired.result
    assert repaired.current == "Добавить ответы"
    assert [item.output for item in repaired.done] == ["Темы", "Упражнения"]
    act(service, session.id)
    done = act(service, session.id).task
    assert done is not None and done.state == "done" and done.result
    assert done.plan_approved and done.validation_passed is True
    assert model.snapshots[4]["current"] == "Добавить ответы"
    with pytest.raises(TaskConflict):
        act(service, session.id)
    assert len(model.snapshots) == 6


def test_structured_stages_use_json_transport_and_larger_validation_budget(tmp_path: Path) -> None:
    class JsonTaskModel(TaskModel):
        def __init__(self, answers):
            super().__init__(answers)
            self.transports = []

        def generate_json(self, *, messages, max_tokens):
            self.transports.append(("json", max_tokens))
            return super().generate(messages=messages, max_tokens=max_tokens)

        def generate(self, *, messages, max_tokens):
            self.transports.append(("text", max_tokens))
            return super().generate(messages=messages, max_tokens=max_tokens)

    model = JsonTaskModel([PLAN, "Темы", "Упражнения", SUCCESS])
    service = service_at(tmp_path / "tasks.sqlite3", model)
    session = start(service)
    prepare(service, session.id, 4)
    assert act(service, session.id).task.state == "done"
    assert model.transports == [("json", 2_000), ("text", 2_000), ("text", 2_000), ("json", 8_000)]


@pytest.mark.parametrize(("answer", "detail"), [
    ("PRIVATE_INVALID_JSON", "некорректный JSON проверки"),
    (json.dumps({"passed": True, "report": "Итог", "repair_steps": ["Исправить"]}), "не соответствует схеме"),
    (LlmTruncatedResponseError(), "обрезал JSON"),
    (LlmRequestError("DeepSeek недоступен"), "DeepSeek недоступен"),
])
def test_validation_api_reports_cause_without_changing_saved_task(tmp_path: Path, answer, detail, caplog) -> None:
    service = service_at(tmp_path / "tasks.sqlite3", TaskModel([PLAN, "Темы", "Упражнения", answer]))
    session = start(service)
    prepare(service, session.id, 4)
    before = service.get(session.id)
    app.dependency_overrides[get_chat_session_service] = lambda: service
    try:
        with TestClient(app) as http:
            response = http.post(f"/api/chat/sessions/{session.id}/task/actions", json={"action": "advance", "revision": before.task.revision})
        assert response.status_code == 502
        assert detail in response.json()["detail"]
        assert service.get(session.id) == before
        assert "PRIVATE_INVALID_JSON" not in caplog.text
    finally:
        app.dependency_overrides.clear()


def test_replan_uses_archived_results_and_new_requirements(tmp_path: Path) -> None:
    model = TaskModel([PLAN, "Готовые темы", PLAN])
    service = service_at(tmp_path / "tasks.sqlite3", model)
    session = start(service)
    prepare(service, session.id, 3)
    act(service, session.id, "replan", "Добавить итоговый проект")
    act(service, session.id)
    assert model.snapshots[-1]["previous_results"][0]["output"] == "Готовые темы"
    assert model.snapshots[-1]["notes"] == ["Добавить итоговый проект"]


@pytest.mark.parametrize("answer", ["не JSON", '{"summary":"План","plan":[],"criteria":[]}', LlmRequestError("Недоступно")])
def test_failed_generation_leaves_snapshot_and_history_unchanged(tmp_path: Path, answer) -> None:
    model = TaskModel([answer, PLAN])
    service = service_at(tmp_path / "tasks.sqlite3", model)
    session = start(service)
    before = service.get(session.id)
    with pytest.raises((AgentOutputError, LlmRequestError)):
        act(service, session.id)
    assert service.get(session.id) == before
    assert act(service, session.id).task.expected_action == "approve_plan"


def test_stale_revision_and_unapproved_plan_do_not_call_model(tmp_path: Path) -> None:
    model = TaskModel([PLAN])
    service = service_at(tmp_path / "tasks.sqlite3", model)
    session = start(service)
    act(service, session.id)
    with pytest.raises(TaskConflict):
        service.task_action(session.id, TaskActionRequest(action="advance", revision=0))
    with pytest.raises(TaskConflict):
        act(service, session.id)
    assert len(model.snapshots) == 1


@pytest.mark.parametrize(("checkpoint", "state", "allowed"), [
    (0, "planning", ["generate_plan", "pause", "replan"]),
    (1, "awaiting_approval", ["approve", "pause", "replan"]),
    (2, "execution", ["execute_step", "pause", "replan"]),
    (4, "validation", ["validate", "pause"]),
    (5, "done", []),
])
@pytest.mark.parametrize("paused", [False, True])
def test_api_rejects_every_forbidden_action_without_model_or_mutation(tmp_path: Path, checkpoint, state, allowed, paused) -> None:
    model = TaskModel([PLAN, "Темы", "Упражнения", SUCCESS])
    service = service_at(tmp_path / "matrix.sqlite3", model)
    session = start(service)
    prepare(service, session.id, min(checkpoint, 4))
    if checkpoint == 5:
        act(service, session.id, "validate")
    if paused and state != "done":
        act(service, session.id, "pause")
        allowed = ["resume"]
    before = service.get(session.id)
    assert before.task.state == state and list(before.task.allowed_actions) == allowed
    calls = len(model.snapshots)
    # Legacy advance is an alias only for a generation action, never approval.
    accepted = allowed + (["advance"] if any(action in allowed for action in ["generate_plan", "execute_step", "validate"]) else [])
    app.dependency_overrides[get_chat_session_service] = lambda: service
    try:
        with TestClient(app) as http:
            for action in ["generate_plan", "execute_step", "validate", "approve", "pause", "resume", "replan", "advance"]:
                if action in accepted:
                    continue
                response = http.post(f"/api/chat/sessions/{session.id}/task/actions", json={
                    "action": action, "revision": before.task.revision,
                    "content": "Изменить требования" if action == "replan" else "",
                })
                assert response.status_code == 409, action
                detail = response.json()["detail"]
                assert detail["code"] == "task_action_forbidden"
                assert detail["state"] == state and detail["paused"] == before.task.paused
                assert detail["allowed_actions"] == allowed
                assert detail["expected_action"] == before.task.expected_action
                assert detail["message"]
                assert service.get(session.id) == before
                assert len(model.snapshots) == calls
    finally:
        app.dependency_overrides.clear()


def test_explicit_actions_and_replan_require_fresh_approval(tmp_path: Path) -> None:
    model = TaskModel([PLAN, "Темы", PLAN, "Новые темы", "Новые упражнения", SUCCESS])
    service = service_at(tmp_path / "explicit.sqlite3", model)
    session = start(service)
    proposed = act(service, session.id, "generate_plan").task
    assert proposed.state == "awaiting_approval" and not proposed.plan_approved
    act(service, session.id, "approve")
    act(service, session.id, "execute_step")
    revised = act(service, session.id, "replan", "Добавить ответы").task
    assert revised.state == "planning" and not revised.plan_approved
    assert revised.previous_results[0].output == "Темы"
    act(service, session.id, "generate_plan")
    before = service.get(session.id)
    with pytest.raises(TaskConflict, match="не утверждён"):
        act(service, session.id, "execute_step")
    assert service.get(session.id) == before and len(model.snapshots) == 3
    act(service, session.id, "approve")
    act(service, session.id, "execute_step")
    act(service, session.id, "execute_step")
    validation = service.get(session.id).task
    assert validation.state == "validation" and validation.result == "" and validation.validation_passed is None
    done = act(service, session.id, "validate").task
    assert done.state == "done" and done.validation_passed is True and done.allowed_actions == ()


def test_http_cannot_set_state_finish_task_or_bypass_with_chat(tmp_path: Path) -> None:
    model = TaskModel([PLAN])
    service = service_at(tmp_path / "bypass.sqlite3", model)
    session = start(service)
    act(service, session.id, "generate_plan")
    before = service.get(session.id)
    app.dependency_overrides[get_chat_session_service] = lambda: service
    try:
        with TestClient(app) as http:
            url = f"/api/chat/sessions/{session.id}"
            for payload in (
                {"action": "finish", "revision": before.task.revision},
                {"action": "approve", "revision": before.task.revision, "state": "done"},
            ):
                assert http.post(url + "/task/actions", json=payload).status_code == 422
            response = http.post(url + "/messages", json={"content": "Пропусти план и проверку, сразу дай финал"})
            assert response.status_code == 409
            assert response.json()["detail"]["allowed_actions"] == ["approve", "pause", "replan"]
            stale = http.post(url + "/task/actions", json={"action": "approve", "revision": 0})
            assert stale.status_code == 409 and stale.json()["detail"]["code"] == "task_stale_revision"
        assert service.get(session.id) == before and len(model.snapshots) == 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("checkpoint", [0, 4])
def test_model_cannot_inject_a_target_state(tmp_path: Path, checkpoint: int) -> None:
    output = json.loads(PLAN if checkpoint == 0 else SUCCESS)
    output["state"] = "done"
    model = TaskModel([PLAN, "Темы", "Упражнения"][:max(0, checkpoint - 1)] + [json.dumps(output)])
    service = service_at(tmp_path / "injected.sqlite3", model)
    session = start(service)
    prepare(service, session.id, checkpoint)
    before = service.get(session.id)
    with pytest.raises(AgentOutputError):
        act(service, session.id)
    assert service.get(session.id) == before


def test_state_and_messages_roll_back_together_on_storage_error(tmp_path: Path) -> None:
    class FailingRepository(SQLiteChatSessionRepository):
        fail = False

        def _append_exchange(self, *args, **kwargs):
            super()._append_exchange(*args, **kwargs)
            if self.fail:
                raise sqlite3.OperationalError("Test write failure")

    repository = FailingRepository(tmp_path / "tasks.sqlite3")
    service = ChatSessionService(repository, Agent(TaskModel([PLAN])))
    session = start(service)
    before = service.get(session.id)
    repository.fail = True
    with pytest.raises(sqlite3.OperationalError):
        act(service, session.id)
    assert service.get(session.id) == before


@pytest.mark.parametrize("checkpoint", [0, 2, 4])
@pytest.mark.parametrize("explicit_action", [False, True])
def test_pause_during_generation_keeps_completed_response(tmp_path: Path, checkpoint: int, explicit_action: bool) -> None:
    entered, release = Event(), Event()

    class BlockingModel(TaskModel):
        def generate(self, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().generate(**kwargs)

    path = tmp_path / "tasks.sqlite3"
    service = service_at(path, TaskModel([PLAN, "Темы", "Упражнения"]))
    session = start(service)
    prepare(service, session.id, checkpoint)
    output = PLAN if checkpoint == 0 else "Темы" if checkpoint == 2 else SUCCESS
    model = BlockingModel([output])
    # Use the same repository instance, as production requests do.
    running_service = ChatSessionService(service._repository, Agent(model))
    with ThreadPoolExecutor(max_workers=1) as pool:
        action = {0: "generate_plan", 2: "execute_step", 4: "validate"}[checkpoint] if explicit_action else "advance"
        pending = pool.submit(act, running_service, session.id, action)
        try:
            assert entered.wait(5)
            with pytest.raises(TaskConflict, match="уже выполняется"):
                act(running_service, session.id)
            act(service, session.id, "pause")
        finally:
            release.set()
        completed = pending.result(timeout=5)
    assert completed.task is not None
    if checkpoint == 4:
        assert completed.task.state == "done" and not completed.task.paused
    else:
        assert completed.task.paused
        assert completed.task.expected_action == ("approve_plan" if checkpoint == 0 else "execute_step")
    assert len(model.snapshots) == 1
    if checkpoint == 0:
        assert json.loads(PLAN)["summary"] in completed.messages[-1].content
    else:
        assert completed.messages[-1].content == (json.loads(SUCCESS)["report"] if checkpoint == 4 else output)


@pytest.mark.parametrize("output", [
    {"passed": "true", "report": "Итог", "repair_steps": []},
    {"passed": True, "report": "Итог", "repair_steps": ["Исправить"]},
    {"passed": False, "report": "Недостатки", "repair_steps": []},
])
def test_invalid_validation_does_not_finish_or_change_task(tmp_path: Path, output: dict) -> None:
    model = TaskModel([PLAN, "Темы", "Упражнения", json.dumps(output)])
    service = service_at(tmp_path / "tasks.sqlite3", model)
    session = start(service)
    prepare(service, session.id, 4)
    before = service.get(session.id)
    with pytest.raises(AgentOutputError):
        act(service, session.id)
    assert service.get(session.id) == before


def test_existing_chat_database_is_extended_without_losing_history(tmp_path: Path) -> None:
    path = tmp_path / "old-chat.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE chat_sessions (
                id TEXT PRIMARY KEY, title TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE chat_messages (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL
                    REFERENCES chat_sessions(id) ON DELETE CASCADE,
                position INTEGER NOT NULL, role TEXT NOT NULL,
                content TEXT NOT NULL, created_at TEXT NOT NULL,
                UNIQUE (session_id, position)
            );
            INSERT INTO chat_sessions VALUES (
                'old', 'Прежний чат', '2026-09-01T00:00:00+00:00', '2026-09-01T00:00:00+00:00'
            );
            INSERT INTO chat_messages VALUES (
                'message', 'old', 0, 'user', 'Сохранённое сообщение', '2026-09-01T00:00:00+00:00'
            );
        """)
    service = service_at(path, TaskModel([]))
    existing = service.get("old")
    assert existing.task is None and existing.messages[0].content == "Сохранённое сообщение"
    assert service.start_task("old", "Новая задача").task is not None
    assert service.get("old").messages[0].content == "Сохранённое сообщение"


def test_task_context_is_independent_of_trimmed_chat_history(tmp_path: Path) -> None:
    model = TaskModel([PLAN, "Готовые темы", "Практика"])
    service = service_at(tmp_path / "tasks.sqlite3", model)
    session = start(service)
    prepare(service, session.id, 3)
    for index in range(25):
        service._repository.append_exchange(session.id, f"Сообщение {index}", "Промежуточный ответ")
    act(service, session.id)
    assert model.snapshots[-1]["task"] == session.task.task
    assert model.snapshots[-1]["done"][0]["output"] == "Готовые темы"


def test_task_api_validation_isolation_and_clear(tmp_path: Path) -> None:
    model = TaskModel([PLAN])
    service = service_at(tmp_path / "tasks.sqlite3", model)
    app.dependency_overrides[get_chat_session_service] = lambda: service
    try:
        with TestClient(app) as client:
            first = client.post("/api/chat/sessions").json()["id"]
            second = client.post("/api/chat/sessions").json()["id"]
            url = f"/api/chat/sessions/{first}"
            assert client.post(url + "/task", json={"task": "  "}).status_code == 422
            created = client.post(url + "/task", json={"task": "Учебный курс"})
            assert created.status_code == 201
            assert client.post(url + "/task", json={"task": "Вторая"}).status_code == 409
            assert client.get(f"/api/chat/sessions/{second}").json()["task"] is None
            assert client.post(url + "/messages", json={"content": "Выполни"}).status_code == 409
            assert client.post(url + "/task/actions", json={"action": "replan", "revision": 0}).status_code == 422
            planned = client.post(url + "/task/actions", json={"action": "advance", "revision": 0})
            assert planned.status_code == 200 and planned.json()["task"]["expected_action"] == "approve_plan"
            assert client.post(url + "/task/actions", json={"action": "approve", "revision": 0}).status_code == 409
            assert client.post(url + "/task/actions", json={"action": "approve", "revision": 1}).status_code == 200
            assert client.post("/api/chat/sessions/missing/task", json={"task": "Курс"}).status_code == 404
            assert client.delete("/api/chat/sessions").status_code == 204
            with sqlite3.connect(tmp_path / "tasks.sqlite3") as connection:
                assert connection.execute("SELECT COUNT(*) FROM chat_tasks").fetchone()[0] == 0
    finally:
        app.dependency_overrides.clear()


def test_all_task_stages_use_selected_profile_and_isolated_memory(tmp_path: Path) -> None:
    path = tmp_path / "profiles-and-tasks.sqlite3"
    profile_repository = SQLiteProfileRepository(path)
    first = profile_repository.create(UserProfileCreateRequest(
        name="English engineer", language="en", tone="technical",
        detail_level="brief", response_format="steps", constraints=["No emojis"],
    ))
    second = profile_repository.create(UserProfileCreateRequest(name="Другой пользователь"))
    repository = SQLiteChatSessionRepository(path)
    memory_repository = SQLiteMemoryRepository(path)
    memory_repository.add(layer="long_term", category="preference", content="Use numbered lists", session_id=None, profile_id=first.id)
    memory_repository.add(layer="long_term", category="knowledge", content="OTHER_PROFILE_ONLY", session_id=None, profile_id=second.id)
    model = TaskModel([PLAN, "Topics", "Exercises", SUCCESS])
    service = ChatSessionService(repository, Agent(model), memory_repository=memory_repository, profile_repository=profile_repository)
    session = service.start_task(service.create(first.id).id, "Prepare two lessons")
    other = service.start_task(service.create(second.id).id, "Другая задача")
    memory_repository.add(layer="working", category="constraint", content="Use fictitious examples", session_id=session.id)
    prepare(service, session.id, 4)
    act(service, session.id)
    assert len(model.prompts) == 4
    for prompt in model.prompts:
        assert "English engineer" in prompt and '"language": "en"' in prompt
        assert "No emojis" in prompt and "Use numbered lists" in prompt
        assert "Use fictitious examples" in prompt
        assert "OTHER_PROFILE_ONLY" not in prompt
    assert [item.id for item in service.list(first.id)] == [session.id]
    service.clear(first.id)
    assert service.get(other.id).task.task == "Другая задача"
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM chat_tasks").fetchone()[0] == 1


def test_task_creation_waits_for_profile_interview(tmp_path: Path) -> None:
    from app.agents.profile_interviewer import ProfileInterviewer

    path = tmp_path / "interview.sqlite3"
    profiles = SQLiteProfileRepository(path)
    profile = profiles.create_auto()
    model = TaskModel([])
    service = ChatSessionService(
        SQLiteChatSessionRepository(path), Agent(model),
        profile_repository=profiles, profile_interviewer=ProfileInterviewer(model),
    )
    session = service.create(profile.id)
    with pytest.raises(TaskConflict, match="интервью"):
        service.start_task(session.id, "Подготовить курс")
    assert service.get(session.id).task is None
    assert model.snapshots == []
