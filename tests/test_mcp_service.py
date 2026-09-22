from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.agents.agent import Agent
from app.main import app, get_mcp_service
from app.services.chat_sessions import ChatSessionService
from app.services.mcp import McpService, McpTool
from app.storage.chat_sessions import SQLiteChatSessionRepository


def tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=f"Description for {name}",
        input_schema={"type": "object", "properties": {}},
    )


class FakeClient:
    def __init__(self) -> None:
        self.cursors: list[str | None] = []
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self, *, cursor: str | None = None) -> SimpleNamespace:
        self.cursors.append(cursor)
        if cursor is None:
            return SimpleNamespace(tools=[tool("first")], next_cursor="next")
        return SimpleNamespace(tools=[tool("second")], next_cursor=None)

    async def call_tool(self, name: str, arguments: dict) -> SimpleNamespace:
        self.calls.append((name, arguments))
        return SimpleNamespace(
            structured_content=None,
            content=[SimpleNamespace(text="Ответ DeepWiki")],
            is_error=False,
        )


class FakeClientContext:
    def __init__(self, client: FakeClient) -> None:
        self.client = client

    async def __aenter__(self) -> FakeClient:
        return self.client

    async def __aexit__(self, *args: object) -> None:
        return None


def mcp_with(client: FakeClient) -> McpService:
    return McpService(
        "https://example.test/mcp",
        client_factory=lambda _url: FakeClientContext(client),
    )


def test_mcp_service_lists_all_pages_and_calls_tool() -> None:
    client = FakeClient()
    service = mcp_with(client)

    tools = service.list_tools()
    result = service.call_tool("ask_question", {"repoName": "owner/repo"})

    assert [item.name for item in tools] == ["first", "second"]
    assert client.cursors == [None, "next"]
    assert result == "Ответ DeepWiki"
    assert client.calls == [("ask_question", {"repoName": "owner/repo"})]


def test_deepwiki_chat_command_uses_mcp_tool() -> None:
    client = FakeClient()
    answer = mcp_with(client).execute_chat_command(
        "/deepwiki modelcontextprotocol/python-sdk Как устроен Client?",
    )

    assert answer == (
        "MCP · ask_wiki_question · modelcontextprotocol/python-sdk\n\nОтвет DeepWiki"
    )
    assert client.calls == [(
        "ask_wiki_question",
        {
            "repoName": "modelcontextprotocol/python-sdk",
            "question": "Как устроен Client?",
        },
    )]


def test_mcp_command_validation_returns_usage_without_network() -> None:
    client = FakeClient()
    service = mcp_with(client)

    assert "owner/repo" in service.execute_chat_command("/deepwiki неверный вопрос")
    assert "валидным JSON" in service.execute_chat_command("/mcp-call ask_question не-json")
    assert client.calls == []


def test_model_selects_and_invokes_mcp_tool_from_natural_language() -> None:
    class RouterModel:
        def __init__(self) -> None:
            self.messages = []

        def generate_json(self, *, messages, max_tokens):
            self.messages = messages
            assert max_tokens == 800
            return '{"tool":"first","arguments":{}}'

    client = FakeClient()
    router = RouterModel()

    invocation = mcp_with(client).maybe_invoke(
        "Изучи owner/repo и расскажи, как он устроен",
        router,
    )

    assert invocation is not None
    assert invocation.tool_name == "first"
    assert invocation.result == "Ответ DeepWiki"
    assert client.calls == [("first", {})]
    assert "available_tools" in router.messages[1]["content"]


def test_model_can_decline_mcp_for_an_unrelated_question() -> None:
    class RouterModel:
        @staticmethod
        def generate_json(**_request):
            return '{"tool":null,"arguments":{}}'

    client = FakeClient()

    invocation = mcp_with(client).maybe_invoke("Сколько будет два плюс два?", RouterModel())

    assert invocation is None
    assert client.calls == []


def test_chat_routes_mcp_command_without_calling_language_model(tmp_path) -> None:
    class Model:
        def __init__(self) -> None:
            self.calls = []

        def generate(self, **request):
            self.calls.append(request)
            return "Обычный ответ"

    class ChatMcp:
        @staticmethod
        def handles(content: str) -> bool:
            return content.startswith("/mcp")

        @staticmethod
        def execute_chat_command(content: str) -> str:
            return "MCP · Инструменты (1)\nask_question"

    model = Model()
    service = ChatSessionService(
        SQLiteChatSessionRepository(tmp_path / "chat.sqlite3"),
        Agent(model),
        mcp_service=ChatMcp(),  # type: ignore[arg-type]
    )
    session = service.create()

    response = service.send(session.id, "/mcp-tools")
    service.send(session.id, "Обычный вопрос")

    assert response.assistant_message.content.startswith("MCP ·")
    assert [message.kind for message in service.get(session.id).messages[:2]] == [
        "command", "command",
    ]
    assert len(model.calls) == 1
    sent_messages = model.calls[0]["messages"]
    assert all("mcp-tools" not in message["content"] for message in sent_messages)


def test_chat_uses_automatically_selected_mcp_result(tmp_path) -> None:
    class Model:
        def __init__(self) -> None:
            self.calls = []

        def generate(self, **request):
            self.calls.append(request)
            return "Источник: MCP · first\n\nОтвет по репозиторию"

    class RouterModel:
        @staticmethod
        def generate_json(**_request):
            return '{"tool":"first","arguments":{}}'

    client = FakeClient()
    model = Model()
    service = ChatSessionService(
        SQLiteChatSessionRepository(tmp_path / "chat.sqlite3"),
        Agent(model),
        mcp_service=mcp_with(client),
        mcp_model=RouterModel(),
    )
    session = service.create()

    response = service.send(
        session.id,
        "Изучи owner/repo и объясни архитектуру естественным языком",
    )

    assert response.assistant_message.content.startswith("Источник: MCP · first")
    assert client.calls == [("first", {})]
    assert len(model.calls) == 1
    messages = model.calls[0]["messages"]
    assert "HOST_MCP_RESULT" in messages[0]["content"]
    assert "Ответ DeepWiki" in messages[0]["content"]
    assert messages[-1]["content"] == (
        "Изучи owner/repo и объясни архитектуру естественным языком"
    )


def test_mcp_http_endpoints() -> None:
    class HttpMcp:
        server_url = "https://example.test/mcp"

        @staticmethod
        def list_tools() -> list[McpTool]:
            return [McpTool("ask_question", "Ask", {"type": "object"})]

    app.dependency_overrides[get_mcp_service] = lambda: HttpMcp()
    try:
        with TestClient(app) as client:
            status = client.get("/api/mcp/status")
            tools = client.get("/api/mcp/tools")
        assert status.status_code == 200
        assert status.json() == {
            "connected": True,
            "endpoint": "https://example.test/mcp",
            "tool_count": 1,
        }
        assert tools.status_code == 200
        assert tools.json()[0]["name"] == "ask_question"
    finally:
        app.dependency_overrides.clear()
