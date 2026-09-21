from __future__ import annotations

from ..agents.agent import Agent, AgentInputError, AgentMessage
from ..invariants import InvariantPolicy, InvariantViolation, SQLiteInvariantRepository
from ..memory.extractor import MemoryExtractor
from ..orchestration.profile_interviewer import ProfileInterviewer
from ..memory.runtime import MemoryRuntime
from ..orchestration.context import OrchestrationContext, profile_context
from ..orchestration.onboarding import ProfileOnboarding
from ..orchestration.profiles import DEFAULT_PROFILE_ID, ProfileRepository, StoredProfile
from ..orchestration.tasks import TaskOrchestrator
from ..schemas import (
    ChatMessage, ChatSendResponse, ChatSession, ChatSessionSummary,
    TaskActionRequest, TaskSummary, TaskView,
)
from ..state.task import TaskContext, TaskConflict, TaskState, approve_plan, pause, replan, require_action, resume
from ..memory.service import MemoryRepository
from ..storage.chat_sessions import (
    ChatSessionNotFound, ChatSessionRepository, DEFAULT_CHAT_DB_PATH,
    DEFAULT_DB_PATH, SQLiteChatSessionRepository, StoredMessage, StoredSession, StoredTask,
)


class ChatSessionService:
    """Coordinate storage, orchestration and memory behind the chat API."""

    def __init__(
        self,
        repository: ChatSessionRepository,
        agent: Agent,
        memory_repository: MemoryRepository | None = None,
        memory_extractor: MemoryExtractor | None = None,
        profile_repository: ProfileRepository | None = None,
        profile_interviewer: ProfileInterviewer | None = None,
        *, invariant_repository: SQLiteInvariantRepository | None = None,
    ) -> None:
        self._repository = repository
        self._invariants = invariant_repository
        self._agent = agent
        self._memory = MemoryRuntime(memory_repository, memory_extractor)
        self._tasks = TaskOrchestrator(agent)
        self._profile_repository = profile_repository
        self._onboarding = (
            ProfileOnboarding(profile_repository, profile_interviewer)
            if profile_repository is not None and profile_interviewer is not None
            else None
        )

    def _policy(self) -> InvariantPolicy:
        return InvariantPolicy(self._invariants.get() if self._invariants is not None else None)

    def _refuse(self, session_id: str, content: str, error: InvariantViolation) -> ChatSendResponse:
        return self._response(
            self._repository.append_refusal(session_id, content.strip(), error.refusal()),
        )

    @staticmethod
    def _summary(session: StoredSession) -> ChatSessionSummary:
        return ChatSessionSummary(
            id=session.id,
            profile_id=session.profile_id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            task=None if session.task is None else TaskSummary(
                state=session.task.context.state,
                step=session.task.context.step,
                total=session.task.context.total,
                current=session.task.context.current,
                paused=session.task.context.paused,
                revision=session.task.revision,
            ),
        )

    @staticmethod
    def _message(message: StoredMessage) -> ChatMessage:
        return ChatMessage(
            id=message.id,
            role=message.role,
            kind=message.kind,
            refusal=message.refusal,
            content=message.content,
            created_at=message.created_at,
        )

    @classmethod
    def _response(cls, updated: StoredSession) -> ChatSendResponse:
        return ChatSendResponse(
            session=cls._summary(updated),
            user_message=cls._message(updated.messages[-2]),
            assistant_message=cls._message(updated.messages[-1]),
        )

    @staticmethod
    def _pending_request(session: StoredSession) -> str | None:
        return next(
            (
                message.content
                for message in session.messages
                if message.role == "user" and message.kind == "message" and not message.refusal
            ),
            None,
        )

    def _finish_onboarding(
        self,
        *,
        session: StoredSession,
        content: str,
        profile: StoredProfile,
        policy: InvariantPolicy,
    ) -> ChatSendResponse:
        pending_request = self._pending_request(session)
        if pending_request is None:
            answer = (
                "Профиль готов — теперь напишите задачу, дальше я буду учитывать "
                "эти настройки автоматически."
            )
            return self._response(
                self._repository.append_exchange(session.id, content.strip(), policy.apply(answer)),
            )

        memory = self._memory.snapshot(
            session.id,
            session.profile_id,
        )
        policy.check_request(pending_request)
        active_profile = profile_context(profile)
        answer = self._agent.respond(
            [],
            pending_request,
            orchestration=OrchestrationContext(profile=active_profile, invariants=policy.settings),
            memory=memory,
        )
        updated = self._repository.append_exchange(session.id, content.strip(), policy.apply(answer))
        self._memory.remember(
            session_id=session.id,
            profile_id=session.profile_id,
            profile=active_profile,
            context=[],
            content=pending_request,
            memory=memory,
        )
        return self._response(updated)

    def _onboard(
        self, *, session: StoredSession, content: str, profile: StoredProfile, policy: InvariantPolicy,
    ) -> ChatSendResponse:
        assert self._onboarding is not None
        result = self._onboarding.respond(profile, content)
        if result.answer is None:
            return self._finish_onboarding(
                session=session, content=content, profile=result.profile, policy=policy,
            )
        return self._response(
            self._repository.append_exchange(session.id, content.strip(), policy.apply(result.answer)),
        )

    def create(self, profile_id: str = DEFAULT_PROFILE_ID) -> ChatSession:
        if self._profile_repository is not None:
            self._profile_repository.get(profile_id)
        session = self._repository.create(profile_id)
        return ChatSession(**self._summary(session).model_dump(), messages=[])

    def list(self, profile_id: str | None = None) -> list[ChatSessionSummary]:
        if profile_id is not None and self._profile_repository is not None:
            self._profile_repository.get(profile_id)
        return [
            self._summary(session)
            for session in self._repository.list(profile_id)
        ]

    def get(self, session_id: str) -> ChatSession:
        session = self._repository.get(session_id)
        return ChatSession(
            **self._summary(session).model_dump(exclude={"task"}),
            messages=[self._message(message) for message in session.messages],
            task=None if session.task is None else TaskView(
                **session.task.context.to_dict(), revision=session.task.revision,
            ),
        )

    def clear(self, profile_id: str | None = None) -> None:
        if profile_id is not None and self._profile_repository is not None:
            self._profile_repository.get(profile_id)
        self._repository.clear(profile_id)

    def send(self, session_id: str, content: str) -> ChatSendResponse:
        if not content.strip():
            raise AgentInputError("Сообщение не должно быть пустым")
        policy = self._policy()
        try:
            return self._send(session_id, content, policy)
        except InvariantViolation as error:
            return self._refuse(session_id, content, error)

    def _send(self, session_id: str, content: str, policy: InvariantPolicy) -> ChatSendResponse:
        session = self._repository.get(session_id)
        if session.task is not None and session.task.context.state != TaskState.DONE:
            ctx = session.task.context
            raise TaskConflict(
                f"Используйте действия задачи. Следующее действие: {ctx.current}", ctx=ctx,
                code="task_action_forbidden",
            )
        policy.check_request(content)
        profile = None
        if self._profile_repository is not None:
            stored_profile = self._profile_repository.get(session.profile_id)
            if (
                self._onboarding is not None
                and not stored_profile.onboarding_complete
            ):
                return self._onboard(
                    session=session,
                    content=content,
                    profile=stored_profile,
                    policy=policy,
                )
            profile = profile_context(stored_profile)
        context: list[AgentMessage] = [
            {"role": message.role, "content": message.content}
            for message in session.messages
            if message.kind == "message" and not message.refusal
        ]
        memory = self._memory.snapshot(
            session_id,
            session.profile_id,
        )
        answer = self._agent.respond(
            context,
            content,
            orchestration=OrchestrationContext(profile=profile, invariants=policy.settings),
            memory=memory,
        )
        updated = self._repository.append_exchange(session_id, content.strip(), policy.apply(answer))
        self._memory.remember(
            session_id=session_id,
            profile_id=session.profile_id,
            profile=profile,
            context=context,
            content=content,
            memory=memory,
        )
        return self._response(updated)


    def start_task(self, session_id: str, task: str) -> ChatSession:
        policy = self._policy()
        try:
            return self._start_task(session_id, task, policy)
        except InvariantViolation as error:
            self._refuse(session_id, "Задача: " + task, error)
            return self.get(session_id)

    def _start_task(self, session_id: str, task: str, policy: InvariantPolicy) -> ChatSession:
        session = self._repository.get(session_id)
        if self._profile_repository is not None and self._onboarding is not None:
            profile = self._profile_repository.get(session.profile_id)
            if not profile.onboarding_complete:
                raise TaskConflict("Сначала завершите интервью профиля или отправьте /skip в обычном чате")
        policy.check_request(task)
        self._repository.create_task(
            session_id, TaskContext(task=task.strip()),
            assistant_content=policy.apply(
                "Задача создана — нажмите «Сформировать план», затем утвердите его",
            ),
        )
        return self.get(session_id)

    def task_action(self, session_id: str, request: TaskActionRequest) -> ChatSession:
        policy = self._policy()
        try:
            if request.action in {"pause", "resume"}:
                return self._apply_task_action(session_id, request, policy)
            with self._repository.task_operation(session_id):
                return self._apply_task_action(session_id, request, policy)
        except InvariantViolation as error:
            self._refuse(session_id, request.content or "Продолжить задачу", error)
            return self.get(session_id)

    def _apply_task_action(
        self, session_id: str, request: TaskActionRequest, policy: InvariantPolicy,
    ) -> ChatSession:
        session = self._repository.get(session_id)
        stored = session.task
        if stored is None:
            raise TaskConflict("В чате нет задачи")
        if request.revision != stored.revision:
            raise TaskConflict(
                "Состояние изменилось. Обновите чат и повторите действие",
                ctx=stored.context, code="task_stale_revision",
            )
        ctx = stored.context
        action = require_action(ctx, request.action)
        if action not in {"pause", "resume"}:
            policy.check_request(request.content if action == "replan" else ctx.task + "\n" + "\n".join(ctx.notes))
        if action == "pause":
            updated, answer = pause(ctx), "Задача на паузе — этап, шаг и результаты сохранены"
        elif action == "resume":
            updated = resume(ctx)
            answer = "Продолжаем с сохранённого состояния — выполните ожидаемое действие"
        elif action == "approve":
            updated = approve_plan(ctx)
            answer = "План утверждён — можно выполнить следующий шаг"
        elif action == "replan":
            updated = replan(ctx, request.content)
            answer = "Требования сохранены — сформируйте новый план, готовые результаты доступны агенту"
        else:
            profile = None if self._profile_repository is None else profile_context(
                self._profile_repository.get(session.profile_id),
            )
            memory = self._memory.snapshot(session_id, session.profile_id)
            updated, answer = self._tasks.advance(
                ctx, orchestration=OrchestrationContext(profile=profile, invariants=policy.settings), memory=memory,
            )
        is_generation = action in {"generate_plan", "execute_step", "validate"}
        self._repository.update_task(
            session_id, updated,
            {"generate_plan": ctx.current, "execute_step": ctx.current, "validate": ctx.current,
             "approve": "Утвердить план", "pause": "Пауза",
             "resume": "Продолжить", "replan": "Пересмотреть план: " + request.content}[action],
            policy.apply(answer),
            expected_revision=None if is_generation else request.revision,
            expected_progress_revision=stored.progress_revision if is_generation else None,
            pause_only=action in {"pause", "resume"},
        )
        return self.get(session_id)
