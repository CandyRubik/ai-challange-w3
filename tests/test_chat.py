from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
import pytest

from app.agents.agent import Agent, AgentInputError, AgentMessage, AgentOutputError
from app.main import app, get_chat_session_service
from app.services.chat_sessions import ChatSessionService, SQLiteChatSessionRepository


class FakeLanguageModel:
    def __init__(self, answers: list[str] | None = None) -> None:
        self.answers = answers or ["Ответ агента"]
        self.calls: list[tuple[list[AgentMessage], int]] = []

    def generate(
        self,
        *,
        messages: Sequence[AgentMessage],
        max_tokens: int = 2_000,
    ) -> str:
        self.calls.append((list(messages), max_tokens))
        return self.answers.pop(0)


def repository(tmp_path: Path) -> SQLiteChatSessionRepository:
    return SQLiteChatSessionRepository(tmp_path / "chat.sqlite3")


def test_frontend_uses_same_origin_api_by_default() -> None:
    javascript = (Path(__file__).parents[1] / "static" / "app.js").read_text()

    assert 'window.API_BASE_URL || ""' in javascript
    assert "http://localhost:8000" not in javascript


def test_frontend_keeps_composer_visible_and_sends_with_enter() -> None:
    root = Path(__file__).parents[1]
    javascript = (root / "static" / "app.js").read_text()
    styles = (root / "static" / "styles.css").read_text()
    markup = (root / "static" / "index.html").read_text()

    assert 'input.addEventListener("keydown"' in javascript
    assert 'event.key === "Enter"' in javascript
    assert "!event.shiftKey" in javascript
    assert "form.requestSubmit()" in javascript
    assert "memoryCommands" in javascript
    assert 'name: "/goal", layer: "working"' in javascript
    assert 'name: "/profile", layer: "long_term"' in javascript
    assert "parsedMemoryCommand(content)" in javascript
    assert "memorySnapshot = memory" in javascript
    assert 'id="profile-select"' in markup
    assert 'id="profile-dialog"' in markup
    assert 'id="delete-profile"' in markup
    assert 'api("/api/profiles")' in javascript
    assert 'api("/api/profiles/auto", { method: "POST" })' in javascript
    assert "profile.onboarding_complete" in javascript
    assert 'method: "DELETE"' in javascript
    assert "profile_id: currentProfileId" in javascript
    assert 'id="command-menu"' in markup
    assert 'id="memory-form"' not in markup
    assert 'id="followup-queue"' in markup
    assert "queuedMessages" in javascript
    assert "enqueueMessage(content)" in javascript
    assert 'kind: "pending"' in javascript
    assert "typing-indicator" in styles
    assert "input.disabled = value" not in javascript
    assert "grid-template-rows: auto minmax(0, 1fr) auto" in styles
    assert ".conversation" in styles and "min-height: 0" in styles
    assert 'class="memory-layers"' in markup


def test_repository_migrates_existing_messages_to_message_kind(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (session_id, position)
            );
            """,
        )

    SQLiteChatSessionRepository(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(chat_messages)")
        }
    assert "kind" in columns


def test_agent_applies_input_and_output_policies() -> None:
    model = FakeLanguageModel(["  Готово  "])
    agent = Agent(model)

    answer = agent.respond(
        [{"role": "user", "content": "Раньше"}, {"role": "assistant", "content": "Да"}],
        "  Продолжим?  ",
    )

    assert answer == "Готово"
    assert model.calls[0][0][0]["role"] == "system"
    assert model.calls[0][0][-1] == {"role": "user", "content": "Продолжим?"}


def test_agent_trims_old_context_by_message_count() -> None:
    model = FakeLanguageModel()
    agent = Agent(model)
    context = [
        {"role": "user", "content": f"Вопрос {index}"}
        for index in range(45)
    ]

    agent.respond(context, "Новый вопрос")

    sent_messages = model.calls[0][0]
    assert len(sent_messages) == 41  # system + 39 history messages + current
    assert sent_messages[1]["content"] == "Вопрос 6"
    assert sent_messages[-1]["content"] == "Новый вопрос"


def test_agent_trims_old_context_by_content_size() -> None:
    model = FakeLanguageModel()
    agent = Agent(model)
    context = [
        {"role": "user", "content": "a" * 25_000},
        {"role": "assistant", "content": "b" * 10_000},
    ]

    agent.respond(context, "c" * 10_000)

    sent_messages = model.calls[0][0]
    assert [message["content"] for message in sent_messages[1:]] == [
        "b" * 10_000,
        "c" * 10_000,
    ]


def test_agent_rejects_empty_or_oversized_values() -> None:
    model = FakeLanguageModel(["x" * 50_001, "Ответ"])
    agent = Agent(model)

    with pytest.raises(AgentInputError):
        agent.respond([], "   ")
    with pytest.raises(AgentInputError):
        agent.respond([], "x" * 40_001)
    with pytest.raises(AgentOutputError):
        agent.respond([], "Вопрос")

    assert len(model.calls) == 1


def test_sessions_keep_contexts_isolated(tmp_path: Path) -> None:
    model = FakeLanguageModel(["Ответ A", "Ответ B", "Ответ A2"])
    service = ChatSessionService(repository(tmp_path), Agent(model))
    first = service.create()
    second = service.create()

    service.send(first.id, "Вопрос A")
    service.send(second.id, "Вопрос B")
    service.send(first.id, "Ещё A")

    assert [message.content for message in service.get(first.id).messages] == [
        "Вопрос A", "Ответ A", "Ещё A", "Ответ A2",
    ]
    assert [message.content for message in service.get(second.id).messages] == [
        "Вопрос B", "Ответ B",
    ]
    assert "Вопрос B" not in [message["content"] for message in model.calls[2][0]]


def test_context_survives_backend_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "persistent-chat.sqlite3"
    first_service = ChatSessionService(
        SQLiteChatSessionRepository(database_path),
        Agent(FakeLanguageModel(["Тебя зовут Лена"])),
    )
    session = first_service.create()
    first_service.send(session.id, "Запомни: меня зовут Лена")

    restarted_model = FakeLanguageModel(["Тебя зовут Лена"])
    restarted_service = ChatSessionService(
        SQLiteChatSessionRepository(database_path),
        Agent(restarted_model),
    )
    restarted_service.send(session.id, "Как меня зовут?")

    model_context, _ = restarted_model.calls[0]
    assert model_context[-3:] == [
        {"role": "user", "content": "Запомни: меня зовут Лена"},
        {"role": "assistant", "content": "Тебя зовут Лена"},
        {"role": "user", "content": "Как меня зовут?"},
    ]
    assert [message.content for message in restarted_service.get(session.id).messages] == [
        "Запомни: меня зовут Лена",
        "Тебя зовут Лена",
        "Как меня зовут?",
        "Тебя зовут Лена",
    ]


def test_agent_applies_runtime_experiment_options() -> None:
    model = FakeLanguageModel()
    agent = Agent(
        model,
        system_prompt="Экспериментальный prompt",
        max_tokens=777,
        context_enabled=False,
    )

    agent.respond([{"role": "user", "content": "Скрытый контекст"}], "Новый вопрос")

    messages, max_tokens = model.calls[0]
    assert messages == [
        {"role": "system", "content": "Экспериментальный prompt"},
        {"role": "user", "content": "Новый вопрос"},
    ]
    assert max_tokens == 777


def test_disabled_context_hides_all_memory_layers() -> None:
    model = FakeLanguageModel()
    agent = Agent(model, context_enabled=False)

    agent.respond(
        [{"role": "user", "content": "Скрытый диалог"}],
        "Новый вопрос",
        working_memory=[{"category": "goal", "content": "Скрытая цель"}],
        long_term_memory=[{"category": "profile", "content": "Скрытый профиль"}],
    )

    system_prompt = model.calls[0][0][0]["content"]
    assert "Скрытая цель" not in system_prompt
    assert "Скрытый профиль" not in system_prompt
    assert model.calls[0][0][-1]["content"] == "Новый вопрос"


def test_chat_session_http_flow(tmp_path: Path) -> None:
    service = ChatSessionService(
        repository(tmp_path),
        Agent(FakeLanguageModel(["Привет! Чем помочь?"])),
    )
    app.dependency_overrides[get_chat_session_service] = lambda: service
    client = TestClient(app)
    try:
        created = client.post("/api/chat/sessions")
        session_id = created.json()["id"]
        sent = client.post(
            f"/api/chat/sessions/{session_id}/messages",
            json={"content": "Привет"},
        )
        loaded = client.get(f"/api/chat/sessions/{session_id}")
        sessions = client.get("/api/chat/sessions")
    finally:
        app.dependency_overrides.clear()

    assert created.status_code == 201
    assert sent.status_code == 200
    assert sent.json()["assistant_message"]["content"] == "Привет! Чем помочь?"
    assert [message["role"] for message in loaded.json()["messages"]] == [
        "user", "assistant",
    ]
    assert sessions.json()[0]["title"] == "Привет"


def test_clear_chat_database_removes_all_sessions(tmp_path: Path) -> None:
    service = ChatSessionService(
        repository(tmp_path),
        Agent(FakeLanguageModel(["Ответ"])),
    )
    session = service.create()
    service.send(session.id, "Сообщение")
    service.create()
    app.dependency_overrides[get_chat_session_service] = lambda: service
    try:
        response = TestClient(app).delete("/api/chat/sessions")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 204
    assert service.list() == []


def test_unknown_chat_session_returns_404(tmp_path: Path) -> None:
    service = ChatSessionService(repository(tmp_path), Agent(FakeLanguageModel()))
    app.dependency_overrides[get_chat_session_service] = lambda: service
    try:
        response = TestClient(app).get("/api/chat/sessions/missing")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 404


def test_agent_prompt_declares_code_isolation() -> None:
    model = FakeLanguageModel()
    Agent(model).respond([], "Что доступно?")

    system_prompt = model.calls[0][0][0]
    assert system_prompt["role"] == "system"
    assert "no access to source code" in system_prompt["content"]
