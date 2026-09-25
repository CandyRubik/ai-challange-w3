from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from mcp import Client, StdioServerParameters

from app.agents.agent import Agent
from app.mcp.checklist_server import SQLiteChecklistRepository, create_checklist_server
from app.services.chat_sessions import ChatSessionService
from app.services.mcp import McpFlow, McpInvocation, McpService
from app.storage.chat_sessions import SQLiteChatSessionRepository


def test_checklist_server_keeps_state_between_tool_calls(tmp_path: Path) -> None:
    server = create_checklist_server(SQLiteChecklistRepository(tmp_path / "checklist.sqlite3"))

    async def exercise() -> tuple[object, object, object]:
        async with Client(server, raise_exceptions=True) as client:
            created = await client.call_tool("create_checklist", {"title": "Прогулка"})
            checklist_id = created.structured_content["id"]
            added = await client.call_tool(
                "add_checklist_item",
                {"checklist_id": checklist_id, "text": "Зонт"},
            )
            listed = await client.call_tool(
                "list_checklist_items",
                {"checklist_id": checklist_id},
            )
        return created, added, listed

    created, added, listed = asyncio.run(exercise())

    assert created.is_error is False
    assert added.structured_content["text"] == "Зонт"
    assert listed.structured_content["title"] == "Прогулка"
    assert [item["text"] for item in listed.structured_content["items"]] == ["Зонт"]


class MultiServerClient:
    def __init__(self, server: StdioServerParameters, calls: list[tuple[str, str, dict]]) -> None:
        self.server = "weather" if server.command == "weather" else "checklist"
        self.calls = calls

    async def list_tools(self, *, cursor: str | None = None) -> SimpleNamespace:
        assert cursor is None
        if self.server == "weather":
            tools = [SimpleNamespace(
                name="get_weather_forecast",
                title="Прогноз погоды",
                description="Получить прогноз",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "forecast_days": {"type": "integer"}},
                    "required": ["city"],
                },
            )]
        else:
            tools = [
                SimpleNamespace(
                    name="create_checklist",
                    title="Создать чек-лист",
                    description="Создать чек-лист",
                    input_schema={
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                    },
                ),
                SimpleNamespace(
                    name="add_checklist_item",
                    title="Добавить пункт",
                    description="Добавить пункт",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "checklist_id": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["checklist_id", "text"],
                    },
                ),
                SimpleNamespace(
                    name="list_checklist_items",
                    title="Показать пункты",
                    description="Показать пункты",
                    input_schema={
                        "type": "object",
                        "properties": {"checklist_id": {"type": "string"}},
                        "required": ["checklist_id"],
                    },
                ),
            ]
        return SimpleNamespace(tools=tools, next_cursor=None)

    async def call_tool(self, name: str, arguments: dict) -> SimpleNamespace:
        self.calls.append((self.server, name, arguments))
        if name == "get_weather_forecast":
            result = {
                "location": "Москва",
                "days": [{"relative_day": "завтра", "condition": "дождь", "precipitation_probability_max": 85}],
            }
        elif name == "create_checklist":
            result = {"id": "checklist-1", "title": arguments["title"], "items": []}
        elif name == "add_checklist_item":
            result = {"id": f"item-{len(self.calls)}", "text": arguments["text"]}
        else:
            result = {
                "id": arguments["checklist_id"],
                "title": "Прогулка",
                "items": [{"id": "item-1", "text": "Зонт"}, {"id": "item-2", "text": "Куртка"}],
            }
        return SimpleNamespace(structured_content=result, content=[], is_error=False)


class MultiServerContext:
    def __init__(self, client: MultiServerClient) -> None:
        self.client = client

    async def __aenter__(self) -> MultiServerClient:
        return self.client

    async def __aexit__(self, *_args: object) -> None:
        return None


def test_router_runs_ordered_flow_across_two_servers() -> None:
    calls: list[tuple[str, str, dict]] = []

    def factory(server: StdioServerParameters) -> MultiServerContext:
        return MultiServerContext(MultiServerClient(server, calls))

    class RouterModel:
        decisions = [
            {"tool": "weather.get_weather_forecast", "arguments": {"city": "Москва", "forecast_days": 2}},
            {"tool": "checklist.create_checklist", "arguments": {"title": "Прогулка по Москве"}},
            {"tool": "checklist.add_checklist_item", "arguments": {"checklist_id": "checklist-1", "text": "Зонт"}},
            {"tool": "checklist.add_checklist_item", "arguments": {"checklist_id": "checklist-1", "text": "Куртка"}},
            {"tool": "checklist.list_checklist_items", "arguments": {"checklist_id": "checklist-1"}},
            {"tool": None, "arguments": {}},
        ]

        def __init__(self) -> None:
            self.requests: list[dict] = []

        def generate_json(self, *, messages: list[dict[str, str]], max_tokens: int) -> str:
            assert max_tokens == 700
            request = json.loads(messages[1]["content"])
            self.requests.append(request)
            decision = self.decisions[len(self.requests) - 1]
            if decision["tool"] == "checklist.add_checklist_item":
                completed = request["completed_steps"]
                assert completed[1]["result"]
                assert json.loads(completed[1]["result"])["id"] == "checklist-1"
            return json.dumps(decision)

    model = RouterModel()
    service = McpService(
        servers={
            "weather": StdioServerParameters(command="weather"),
            "checklist": StdioServerParameters(command="checklist"),
        },
        client_factory=factory,
    )

    flow = service.run_flow("Проверь погоду и подготовь чек-лист прогулки", model)

    assert isinstance(flow, McpFlow)
    assert flow.stopped_reason == "completed"
    assert flow.server_names == ("weather", "checklist")
    assert [item.qualified_tool_name for item in flow.invocations] == [
        "weather.get_weather_forecast",
        "checklist.create_checklist",
        "checklist.add_checklist_item",
        "checklist.add_checklist_item",
        "checklist.list_checklist_items",
    ]
    assert calls == [
        ("weather", "get_weather_forecast", {"city": "Москва", "forecast_days": 2}),
        ("checklist", "create_checklist", {"title": "Прогулка по Москве"}),
        ("checklist", "add_checklist_item", {"checklist_id": "checklist-1", "text": "Зонт"}),
        ("checklist", "add_checklist_item", {"checklist_id": "checklist-1", "text": "Куртка"}),
        ("checklist", "list_checklist_items", {"checklist_id": "checklist-1"}),
    ]
    assert "Step 5" in flow.external_context()


def test_chat_passes_complete_multi_server_trace_to_agent(tmp_path: Path) -> None:
    class AnswerModel:
        def __init__(self) -> None:
            self.messages: list[dict] = []

        def generate(self, **request: object) -> str:
            self.messages = request["messages"]  # type: ignore[assignment]
            return "Источник: MCP · multi-server flow\n\nГотовый список: зонт и куртка."

    class FlowService:
        @staticmethod
        def run_flow(_content: str, _model: object) -> McpFlow:
            return McpFlow((
                McpInvocation(
                    "get_weather_forecast", {"city": "Москва"}, '{"condition":"дождь"}', "weather",
                ),
                McpInvocation(
                    "list_checklist_items", {"checklist_id": "checklist-1"}, '{"items":["Зонт"]}', "checklist",
                ),
            ))

    model = AnswerModel()
    service = ChatSessionService(
        SQLiteChatSessionRepository(tmp_path / "chat.sqlite3"),
        Agent(model),
        mcp_service=FlowService(),  # type: ignore[arg-type]
        mcp_model=object(),  # type: ignore[arg-type]
    )
    session = service.create()

    response = service.send(session.id, "Собери список для прогулки")

    assert response.assistant_message.content.startswith("Источник: MCP · multi-server flow")
    system_prompt = model.messages[0]["content"]
    assert "Step 1 · Server: weather" in system_prompt
    assert "Step 2 · Server: checklist" in system_prompt
