from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import httpx
from mcp import Client, StdioServerParameters

from app.agents.agent import Agent
from app.main import app, get_mcp_service
from app.mcp.weather_server import (
    FORECAST_URL,
    GEOCODING_URL,
    OpenMeteoWeatherApi,
    WeatherDay,
    WeatherForecast,
    create_weather_server,
)
from app.services.chat_sessions import ChatSessionService
from app.services.mcp import McpInvocation, McpService, McpTool
from app.storage.chat_sessions import SQLiteChatSessionRepository


class FakeWeatherApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def forecast(self, city: str, forecast_days: int) -> WeatherForecast:
        self.calls.append((city, forecast_days))
        return WeatherForecast(
            location="Москва",
            country="Россия",
            timezone="Europe/Moscow",
            latitude=55.75,
            longitude=37.62,
            days=[
                WeatherDay(
                    date="2026-09-22",
                    day_offset=0,
                    relative_day="сегодня",
                    weather_code=61,
                    condition="дождь",
                    temperature_min_c=8.5,
                    temperature_max_c=13.2,
                    precipitation_probability_max=80,
                ),
            ],
        )


def test_weather_server_registers_typed_tool_and_returns_structured_result() -> None:
    weather_api = FakeWeatherApi()
    server = create_weather_server(weather_api)

    async def exercise() -> tuple[object, object]:
        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()
            called = await client.call_tool(
                "get_weather_forecast",
                {"city": "Москва", "forecast_days": 1},
            )
        return listed, called

    listed, called = asyncio.run(exercise())
    tool = listed.tools[0]

    assert tool.name == "get_weather_forecast"
    assert tool.title == "Прогноз погоды"
    assert tool.input_schema["required"] == ["city"]
    assert tool.input_schema["properties"]["city"]["description"].startswith("Название")
    assert tool.input_schema["properties"]["forecast_days"]["maximum"] == 7
    assert called.is_error is False
    assert called.structured_content["location"] == "Москва"
    assert called.structured_content["days"][0]["precipitation_probability_max"] == 80
    assert called.structured_content["days"][0]["condition"] == "дождь"
    assert weather_api.calls == [("Москва", 1)]


def test_open_meteo_adapter_combines_geocoding_and_forecast() -> None:
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if str(request.url).startswith(GEOCODING_URL):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "Москва",
                            "country": "Россия",
                            "latitude": 55.75,
                            "longitude": 37.62,
                            "timezone": "Europe/Moscow",
                        },
                    ],
                },
            )
        assert str(request.url).startswith(FORECAST_URL)
        return httpx.Response(
            200,
            json={
                "timezone": "Europe/Moscow",
                "daily": {
                    "time": ["2026-09-22", "2026-09-23"],
                    "weather_code": [3, 61],
                    "temperature_2m_min": [8.0, 7.5],
                    "temperature_2m_max": [14.0, 12.0],
                    "precipitation_probability_max": [20, 85],
                },
            },
        )

    async def exercise() -> WeatherForecast:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await OpenMeteoWeatherApi(client).forecast("Москва", 2)

    result = asyncio.run(exercise())

    assert len(requested_urls) == 2
    assert result.location == "Москва"
    assert result.days[1].date == "2026-09-23"
    assert result.days[1].day_offset == 1
    assert result.days[1].relative_day == "завтра"
    assert result.days[1].condition == "дождь"
    assert result.days[1].precipitation_probability_max == 85


class FakeMcpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self, *, cursor: str | None = None) -> SimpleNamespace:
        assert cursor is None
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="get_weather_forecast",
                    title="Прогноз погоды",
                    description="Получить прогноз",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "forecast_days": {"type": "integer", "minimum": 1, "maximum": 7},
                        },
                        "required": ["city"],
                    },
                ),
            ],
            next_cursor=None,
        )

    async def call_tool(self, name: str, arguments: dict) -> SimpleNamespace:
        self.calls.append((name, arguments))
        return SimpleNamespace(
            structured_content={
                "location": "Москва",
                "days": [{"date": "2026-09-23", "precipitation_probability_max": 85}],
            },
            content=[],
            is_error=False,
        )


class FakeMcpContext:
    def __init__(self, client: FakeMcpClient) -> None:
        self.client = client

    async def __aenter__(self) -> FakeMcpClient:
        return self.client

    async def __aexit__(self, *_args: object) -> None:
        return None


def mcp_with(client: FakeMcpClient) -> McpService:
    server = StdioServerParameters(command="python", args=["-m", "weather"])
    return McpService(server, client_factory=lambda _server: FakeMcpContext(client))


def test_agent_router_selects_and_calls_weather_mcp() -> None:
    class RouterModel:
        @staticmethod
        def generate_json(**request: object) -> str:
            assert request["max_tokens"] == 500
            return '{"tool":"get_weather_forecast","arguments":{"city":"Москва","forecast_days":2}}'

    client = FakeMcpClient()
    invocation = mcp_with(client).maybe_invoke(
        "Нужен ли завтра зонт в Москве?",
        RouterModel(),
    )

    assert invocation is not None
    assert invocation.tool_name == "get_weather_forecast"
    assert client.calls == [
        ("get_weather_forecast", {"city": "Москва", "forecast_days": 2}),
    ]
    assert '"precipitation_probability_max": 85' in invocation.result


def test_chat_uses_mcp_result_in_the_final_agent_answer(tmp_path: Path) -> None:
    class Model:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def generate(self, **request: object) -> str:
            self.calls.append(request)
            return "Источник: MCP · get_weather_forecast\n\nЗонт стоит взять."

    class WeatherMcp:
        @staticmethod
        def maybe_invoke(_content: str, _model: object) -> McpInvocation:
            return McpInvocation(
                "get_weather_forecast",
                {"city": "Москва", "forecast_days": 2},
                '{"precipitation_probability_max":85}',
            )

    model = Model()
    service = ChatSessionService(
        SQLiteChatSessionRepository(tmp_path / "chat.sqlite3"),
        Agent(model),
        mcp_service=WeatherMcp(),  # type: ignore[arg-type]
        mcp_model=object(),  # type: ignore[arg-type]
    )
    session = service.create()

    response = service.send(session.id, "Нужен ли завтра зонт в Москве?")

    assert response.assistant_message.content.endswith("Зонт стоит взять.")
    system_prompt = model.calls[0]["messages"][0]["content"]
    assert "HOST_MCP_RESULT" in system_prompt
    assert "get_weather_forecast" in system_prompt
    assert "precipitation_probability_max" in system_prompt


def test_application_exposes_mcp_status_and_tool_schema() -> None:
    class HttpMcp:
        endpoint = "stdio: python -m app.mcp.weather_server"

        @staticmethod
        def list_tools() -> list[McpTool]:
            return [
                McpTool(
                    "get_weather_forecast",
                    "Прогноз погоды",
                    "Получить прогноз",
                    {"type": "object", "required": ["city"]},
                ),
            ]

    app.dependency_overrides[get_mcp_service] = lambda: HttpMcp()
    try:
        with TestClient(app) as client:
            status = client.get("/api/mcp/status")
            tools = client.get("/api/mcp/tools")
    finally:
        app.dependency_overrides.clear()

    assert status.json()["connected"] is True
    assert status.json()["tool_count"] == 1
    assert tools.json()[0]["name"] == "get_weather_forecast"
    assert tools.json()[0]["input_schema"]["required"] == ["city"]


def test_real_stdio_server_exposes_weather_tool() -> None:
    tools = McpService().list_tools()

    assert [tool.name for tool in tools] == ["get_weather_forecast"]
    assert tools[0].input_schema["required"] == ["city"]
    assert tools[0].output_schema is not None
    assert "days" in tools[0].output_schema["properties"]
