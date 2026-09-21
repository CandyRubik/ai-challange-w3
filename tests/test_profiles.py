from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import sqlite3
from typing import Any

from fastapi.testclient import TestClient
import pytest

from app.agents.agent import Agent, AgentMessage
from app.agents.memory_extractor import MemoryExtractor
from app.agents.profile_interviewer import ProfileInterviewer
from app.main import app, get_chat_session_service, get_profile_service
from app.schemas import (
    MemoryCreateRequest,
    UserProfileCreateRequest,
    UserProfileUpdateRequest,
)
from app.services.chat_sessions import (
    ChatSessionNotFound,
    ChatSessionService,
    SQLiteChatSessionRepository,
)
from app.services.memory import MemoryService, SQLiteMemoryRepository
from app.services.profiles import (
    DEFAULT_PROFILE_ID,
    ProfileDeletionError,
    ProfileNotFound,
    ProfileService,
    SQLiteProfileRepository,
)


class ProfileAwareLanguageModel:
    def __init__(self) -> None:
        self.calls: list[list[AgentMessage]] = []

    def generate(
        self,
        *,
        messages: Sequence[AgentMessage],
        max_tokens: int = 2_000,
    ) -> str:
        sent = list(messages)
        self.calls.append(sent)
        system_prompt = sent[0]["content"]
        if '"detail_level": "brief"' in system_prompt:
            return "Краткий ответ"
        return "Подробный ответ с примером"


class FakeMemoryModel:
    def __init__(self, result: str) -> None:
        self.result = result

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = 2_000,
    ) -> str:
        return self.result


class FakeInterviewModel:
    def __init__(self, results: list[str]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = 2_000,
    ) -> str:
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_format": response_format,
            "max_tokens": max_tokens,
        })
        return self.results.pop(0)


def profile_request(
    name: str,
    *,
    detail_level: str = "balanced",
    tone: str = "neutral",
) -> UserProfileCreateRequest:
    return UserProfileCreateRequest(
        name=name,
        description=f"Профиль {name}",
        language="ru",
        tone=tone,
        detail_level=detail_level,
        response_format="bullets" if detail_level == "brief" else "plain",
        constraints=["Без эмодзи"] if detail_level == "brief" else [],
    )


def test_profile_repository_creates_default_and_updates_profiles(tmp_path: Path) -> None:
    repository = SQLiteProfileRepository(tmp_path / "profiles.sqlite3")

    default = repository.get(DEFAULT_PROFILE_ID)
    created = repository.create(profile_request("Виктор", detail_level="brief"))
    updated = repository.update(
        created.id,
        UserProfileUpdateRequest(
            name="Виктор",
            description="Разработчик",
            language="ru",
            tone="technical",
            detail_level="brief",
            response_format="steps",
            constraints=["Без эмодзи", "Сначала вывод"],
        ),
    )

    assert default.name == "Основной"
    assert default.onboarding_complete is False
    assert [profile.id for profile in repository.list()] == [
        DEFAULT_PROFILE_ID,
        created.id,
    ]
    assert updated.tone == "technical"
    assert updated.onboarding_complete is True
    assert updated.constraints == ("Без эмодзи", "Сначала вывод")


def test_agent_interviews_user_then_answers_original_request(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "interview.sqlite3"
    profiles = SQLiteProfileRepository(database_path)
    sessions = SQLiteChatSessionRepository(database_path)
    session = sessions.create(DEFAULT_PROFILE_ID)
    interview_model = FakeInterviewModel([
        json.dumps({
            "name": "Виктор",
            "description": "Опытный backend-разработчик",
            "language": "ru",
            "tone": None,
            "detail_level": None,
            "response_format": None,
            "constraints": None,
        }, ensure_ascii=False),
        json.dumps({
            "name": None,
            "description": None,
            "language": None,
            "tone": None,
            "detail_level": "brief",
            "response_format": "bullets",
            "constraints": None,
        }, ensure_ascii=False),
        json.dumps({
            "name": None,
            "description": None,
            "language": None,
            "tone": "technical",
            "detail_level": None,
            "response_format": None,
            "constraints": ["Без эмодзи"],
        }, ensure_ascii=False),
    ])
    answer_model = ProfileAwareLanguageModel()
    chat = ChatSessionService(
        sessions,
        Agent(answer_model),
        None,
        None,
        profiles,
        ProfileInterviewer(interview_model),
    )

    first = chat.send(session.id, "Объясни, что такое embeddings")
    second = chat.send(session.id, "Виктор, опытный backend-разработчик")
    third = chat.send(session.id, "Кратко и списками")
    final = chat.send(session.id, "Технический тон, без эмодзи")

    assert "1/3" in first.assistant_message.content
    assert "2/3" in second.assistant_message.content
    assert "3/3" in third.assistant_message.content
    assert final.assistant_message.content == "Краткий ответ"
    assert len(answer_model.calls) == 1
    assert answer_model.calls[0][-1] == {
        "role": "user",
        "content": "Объясни, что такое embeddings",
    }
    system_prompt = answer_model.calls[0][0]["content"]
    assert '"name": "Виктор"' in system_prompt
    assert '"detail_level": "brief"' in system_prompt
    assert '"constraints": ["Без эмодзи"]' in system_prompt
    profile = profiles.get(DEFAULT_PROFILE_ID)
    assert profile.onboarding_step == 3
    assert profile.onboarding_complete is True
    assert profile.tone == "technical"
    assert [json.loads(call["user_prompt"])["interview_step"] for call in interview_model.calls] == [1, 2, 3]


def test_automatic_profile_is_blank_until_agent_interview(tmp_path: Path) -> None:
    repository = SQLiteProfileRepository(tmp_path / "auto-profile.sqlite3")

    profile = repository.create_auto()

    assert profile.name == "Пользователь 2"
    assert profile.onboarding_step == 0
    assert profile.onboarding_complete is False


def test_interview_can_be_skipped_without_losing_original_request(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "skip-interview.sqlite3"
    profiles = SQLiteProfileRepository(database_path)
    sessions = SQLiteChatSessionRepository(database_path)
    session = sessions.create()
    answer_model = ProfileAwareLanguageModel()
    chat = ChatSessionService(
        sessions,
        Agent(answer_model),
        None,
        None,
        profiles,
        ProfileInterviewer(FakeInterviewModel([])),
    )

    chat.send(session.id, "Объясни embeddings")
    response = chat.send(session.id, "/skip")

    assert response.assistant_message.content == "Подробный ответ с примером"
    assert answer_model.calls[0][-1]["content"] == "Объясни embeddings"
    assert profiles.get(DEFAULT_PROFILE_ID).onboarding_complete is True


def test_different_profiles_change_answers_and_isolate_long_term_memory(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "personalized.sqlite3"
    profiles = SQLiteProfileRepository(database_path)
    brief = profiles.create(profile_request("Виктор", detail_level="brief"))
    detailed = profiles.create(
        profile_request("Мария", detail_level="detailed", tone="friendly"),
    )
    sessions = SQLiteChatSessionRepository(database_path)
    memories = SQLiteMemoryRepository(database_path)
    memory = MemoryService(memories, sessions, profiles)
    brief_chat = sessions.create(brief.id)
    detailed_chat = sessions.create(detailed.id)
    memory.create(
        MemoryCreateRequest(
            layer="long_term",
            category="preference",
            content="Предпочитает примеры на Python",
            profile_id=brief.id,
        ),
    )
    model = ProfileAwareLanguageModel()
    chat = ChatSessionService(sessions, Agent(model), memories, None, profiles)

    brief_answer = chat.send(brief_chat.id, "Объясни embeddings")
    detailed_answer = chat.send(detailed_chat.id, "Объясни embeddings")

    assert brief_answer.assistant_message.content == "Краткий ответ"
    assert detailed_answer.assistant_message.content == "Подробный ответ с примером"
    brief_prompt = model.calls[0][0]["content"]
    detailed_prompt = model.calls[1][0]["content"]
    assert "USER_PROFILE" in brief_prompt
    assert '"name": "Виктор"' in brief_prompt
    assert '"response_format": "bullets"' in brief_prompt
    assert "Предпочитает примеры на Python" in brief_prompt
    assert '"name": "Мария"' in detailed_prompt
    assert "Предпочитает примеры на Python" not in detailed_prompt


def test_automatic_memory_is_saved_to_the_active_profile(tmp_path: Path) -> None:
    database_path = tmp_path / "automatic-profile-memory.sqlite3"
    profiles = SQLiteProfileRepository(database_path)
    first = profiles.create(profile_request("Первый"))
    second = profiles.create(profile_request("Второй"))
    sessions = SQLiteChatSessionRepository(database_path)
    memories = SQLiteMemoryRepository(database_path)
    session = sessions.create(first.id)
    extraction = json.dumps({
        "memories": [{
            "layer": "long_term",
            "category": "preference",
            "content": "Начинать с краткого вывода",
        }],
    }, ensure_ascii=False)
    chat = ChatSessionService(
        sessions,
        Agent(ProfileAwareLanguageModel()),
        memories,
        MemoryExtractor(FakeMemoryModel(extraction)),
        profiles,
    )

    chat.send(session.id, "Всегда начинай с краткого вывода")

    assert [memory.content for memory in memories.list_long_term(first.id)] == [
        "Начинать с краткого вывода",
    ]
    assert memories.list_long_term(second.id) == []


def test_profile_changes_apply_to_every_request_of_an_existing_chat(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "live-profile-update.sqlite3"
    profiles = SQLiteProfileRepository(database_path)
    created = profiles.create(profile_request("Динамический"))
    sessions = SQLiteChatSessionRepository(database_path)
    session = sessions.create(created.id)
    model = ProfileAwareLanguageModel()
    chat = ChatSessionService(sessions, Agent(model), None, None, profiles)

    before = chat.send(session.id, "Первый вопрос")
    profiles.update(
        created.id,
        UserProfileUpdateRequest(
            name="Динамический",
            description="Теперь отвечать кратко",
            language="ru",
            tone="technical",
            detail_level="brief",
            response_format="bullets",
            constraints=["Без эмодзи"],
        ),
    )
    after = chat.send(session.id, "Второй вопрос")

    assert before.assistant_message.content == "Подробный ответ с примером"
    assert after.assistant_message.content == "Краткий ответ"


def test_legacy_sessions_and_memories_migrate_to_default_profile(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy-profile.sqlite3"
    timestamp = "2026-01-01T00:00:00+00:00"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE long_term_memory (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """,
        )
        connection.execute(
            "INSERT INTO chat_sessions VALUES ('chat-1', 'Старый чат', ?, ?)",
            (timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO long_term_memory
            VALUES ('memory-1', 'preference', 'Старое предпочтение', ?, ?)
            """,
            (timestamp, timestamp),
        )

    sessions = SQLiteChatSessionRepository(database_path)
    memories = SQLiteMemoryRepository(database_path)

    assert sessions.get("chat-1").profile_id == DEFAULT_PROFILE_ID
    assert [item.content for item in memories.list_long_term()] == [
        "Старое предпочтение",
    ]


def test_deleting_profile_removes_only_its_chats_and_memory(tmp_path: Path) -> None:
    database_path = tmp_path / "delete-profile.sqlite3"
    profiles = SQLiteProfileRepository(database_path)
    removed = profiles.create(profile_request("Удаляемый"))
    kept = profiles.create(profile_request("Оставшийся"))
    sessions = SQLiteChatSessionRepository(database_path)
    memories = SQLiteMemoryRepository(database_path)
    memory = MemoryService(memories, sessions, profiles)
    removed_session = sessions.create(removed.id)
    kept_session = sessions.create(kept.id)
    sessions.append_exchange(removed_session.id, "Вопрос", "Ответ")
    memory.create(
        MemoryCreateRequest(
            layer="working",
            category="goal",
            content="Удаляемая цель",
            session_id=removed_session.id,
        ),
    )
    memory.create(
        MemoryCreateRequest(
            layer="long_term",
            category="preference",
            content="Удаляемое предпочтение",
            profile_id=removed.id,
        ),
    )
    memory.create(
        MemoryCreateRequest(
            layer="long_term",
            category="preference",
            content="Сохранённое предпочтение",
            profile_id=kept.id,
        ),
    )

    profiles.delete(removed.id)

    with pytest.raises(ProfileNotFound):
        profiles.get(removed.id)
    with pytest.raises(ChatSessionNotFound):
        sessions.get(removed_session.id)
    assert memories.list_working(removed_session.id) == []
    assert memories.list_long_term(removed.id) == []
    assert sessions.get(kept_session.id).profile_id == kept.id
    assert [item.content for item in memories.list_long_term(kept.id)] == [
        "Сохранённое предпочтение",
    ]


def test_default_profile_cannot_be_deleted(tmp_path: Path) -> None:
    profiles = SQLiteProfileRepository(tmp_path / "protected-profile.sqlite3")

    with pytest.raises(ProfileDeletionError, match="нельзя удалить"):
        profiles.delete(DEFAULT_PROFILE_ID)

    assert profiles.get(DEFAULT_PROFILE_ID).id == DEFAULT_PROFILE_ID


def test_profile_http_flow(tmp_path: Path) -> None:
    service = ProfileService(SQLiteProfileRepository(tmp_path / "api-profiles.sqlite3"))
    app.dependency_overrides[get_profile_service] = lambda: service
    client = TestClient(app)
    payload = {
        "name": "Видео-профиль",
        "description": "Для демонстрации",
        "language": "ru",
        "tone": "friendly",
        "detail_level": "detailed",
        "response_format": "steps",
        "constraints": ["Добавлять примеры"],
    }
    try:
        automatic = client.post("/api/profiles/auto")
        created = client.post("/api/profiles", json=payload)
        profile_id = created.json()["id"]
        payload["tone"] = "technical"
        updated = client.put(f"/api/profiles/{profile_id}", json=payload)
        listed = client.get("/api/profiles")
        deleted = client.delete(f"/api/profiles/{profile_id}")
        protected = client.delete(f"/api/profiles/{DEFAULT_PROFILE_ID}")
        after_delete = client.get("/api/profiles")
    finally:
        app.dependency_overrides.clear()

    assert created.status_code == 201
    assert automatic.status_code == 201
    assert automatic.json()["onboarding_complete"] is False
    assert updated.status_code == 200
    assert updated.json()["tone"] == "technical"
    assert deleted.status_code == 204
    assert protected.status_code == 409
    assert [profile["name"] for profile in listed.json()] == [
        "Основной",
        "Пользователь 2",
        "Видео-профиль",
    ]
    assert [profile["name"] for profile in after_delete.json()] == [
        "Основной",
        "Пользователь 2",
    ]


def test_chat_http_flow_binds_and_filters_by_profile(tmp_path: Path) -> None:
    database_path = tmp_path / "profile-chat-api.sqlite3"
    profiles = SQLiteProfileRepository(database_path)
    first = profiles.create(profile_request("Первый"))
    second = profiles.create(profile_request("Второй"))
    sessions = SQLiteChatSessionRepository(database_path)
    service = ChatSessionService(
        sessions,
        Agent(ProfileAwareLanguageModel()),
        None,
        None,
        profiles,
    )
    app.dependency_overrides[get_chat_session_service] = lambda: service
    client = TestClient(app)
    try:
        created = client.post(
            "/api/chat/sessions",
            json={"profile_id": first.id},
        )
        client.post("/api/chat/sessions", json={"profile_id": second.id})
        listed = client.get(
            "/api/chat/sessions",
            params={"profile_id": first.id},
        )
    finally:
        app.dependency_overrides.clear()

    assert created.status_code == 201
    assert created.json()["profile_id"] == first.id
    assert [session["profile_id"] for session in listed.json()] == [first.id]
