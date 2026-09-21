from collections.abc import Sequence
import json
from pathlib import Path

from app.agents.agent import Agent, AgentMessage
from app.memory.context import MemoryContext
from app.memory.service import SQLiteMemoryRepository
from app.orchestration.context import OrchestrationContext, ProfileContext
from app.orchestration.profiles import SQLiteProfileRepository
from app.schemas import TaskActionRequest, UserProfileCreateRequest, UserProfileUpdateRequest
from app.services.chat_sessions import ChatSessionService
from app.storage.chat_sessions import SQLiteChatSessionRepository


class RecordingModel:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls: list[list[AgentMessage]] = []

    def generate(self, *, messages: Sequence[AgentMessage], max_tokens=2_000) -> str:
        self.calls.append(list(messages))
        return self.answers.pop(0)


def profile(name: str) -> ProfileContext:
    return {
        "name": name, "description": "", "language": "ru", "tone": "technical",
        "detail_level": "brief", "response_format": "bullets", "constraints": [],
    }


def test_orchestration_and_memory_do_not_leak_into_next_call() -> None:
    model = RecordingModel(["Первый ответ", "Второй ответ"])
    agent = Agent(model)
    agent.respond(
        [], "Первый вопрос",
        orchestration=OrchestrationContext(profile=profile("Виктор")),
        memory=MemoryContext(working=({"category": "goal", "content": "Личная цель"},)),
    )
    agent.respond([], "Второй вопрос")

    assert "Виктор" in model.calls[0][0]["content"]
    assert "Личная цель" in model.calls[0][0]["content"]
    assert model.calls[1] == [
        {"role": "system", "content": Agent.default_system_prompt},
        {"role": "user", "content": "Второй вопрос"},
    ]


def test_disabled_context_hides_explicit_orchestration_and_memory() -> None:
    model = RecordingModel(["Ответ"])
    Agent(model, context_enabled=False).respond(
        [{"role": "user", "content": "Старая переписка"}], "Новый вопрос",
        orchestration=OrchestrationContext(profile=profile("Виктор")),
        memory=MemoryContext(
            working=({"category": "goal", "content": "Цель"},),
            long_term=({"category": "knowledge", "content": "Знание"},),
        ),
    )

    assert model.calls[0] == [
        {"role": "system", "content": Agent.default_system_prompt},
        {"role": "user", "content": "Новый вопрос"},
    ]


def test_memory_window_does_not_trim_orchestration_profile() -> None:
    model = RecordingModel(["Ответ"])
    Agent(model).respond(
        [], "Вопрос",
        orchestration=OrchestrationContext(profile=profile("Виктор")),
        memory=MemoryContext(working=tuple(
            {"category": "goal", "content": f"Цель-{index:02d}"}
            for index in range(35)
        )),
    )

    prompt = model.calls[0][0]["content"]
    assert "USER_PROFILE" in prompt and "Виктор" in prompt
    assert "Цель-00" not in prompt and "Цель-04" not in prompt
    assert "Цель-05" in prompt and "Цель-34" in prompt


def test_profile_change_during_task_preserves_memory_and_progress(tmp_path: Path) -> None:
    path = tmp_path / "chat.sqlite3"
    profiles = SQLiteProfileRepository(path)
    sessions = SQLiteChatSessionRepository(path)
    memories = SQLiteMemoryRepository(path)
    user = profiles.create(UserProfileCreateRequest(name="Исходный профиль"))
    model = RecordingModel([
        json.dumps({"summary": "План", "plan": ["Написать материал"], "criteria": ["Есть текст"]}),
        "Готовый материал",
    ])
    chat = ChatSessionService(sessions, Agent(model), memories, profile_repository=profiles)
    session = chat.create(user.id)
    memories.add(
        layer="working", category="constraint", content="Проверить определения",
        session_id=session.id,
    )
    memories.add(
        layer="long_term", category="knowledge", content="Пользователь изучает Python",
        session_id=None, profile_id=user.id,
    )
    chat.start_task(session.id, "Написать учебный материал")

    def act(action: str) -> None:
        task = chat.get(session.id).task
        assert task is not None
        chat.task_action(session.id, TaskActionRequest(action=action, revision=task.revision))

    act("advance")
    act("approve")
    before = chat.get(session.id).task
    working = memories.list_working(session.id)
    long_term = memories.list_long_term(user.id)
    profiles.update(user.id, UserProfileUpdateRequest(name="Новый профиль", tone="friendly"))

    assert chat.get(session.id).task == before
    assert memories.list_working(session.id) == working
    assert memories.list_long_term(user.id) == long_term
    act("advance")

    prompt = model.calls[1][0]["content"]
    assert "Новый профиль" in prompt and "Исходный профиль" not in prompt
    assert "Проверить определения" in prompt and "Пользователь изучает Python" in prompt
    snapshot = json.loads(model.calls[1][-1]["content"])
    assert snapshot["state"] == "execution" and snapshot["step"] == 0
    assert "profile" not in snapshot and "working_memory" not in snapshot
    assert chat.get(session.id).task.state == "validation"
