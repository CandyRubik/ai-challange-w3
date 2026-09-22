from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.mcp_client import inspect_server, list_all_tools


class FakeClient:
    server_info = SimpleNamespace(name="test-server", version="1.0")
    protocol_version = "2025-11-25"

    def __init__(self) -> None:
        self.cursors: list[str | None] = []

    async def list_tools(self, *, cursor: str | None = None) -> SimpleNamespace:
        self.cursors.append(cursor)
        if cursor is None:
            return SimpleNamespace(
                tools=[tool("first_tool")],
                next_cursor="page-2",
            )
        return SimpleNamespace(
            tools=[tool("second_tool")],
            next_cursor=None,
        )


class FakeClientContext:
    def __init__(self, client: FakeClient) -> None:
        self.client = client

    async def __aenter__(self) -> FakeClient:
        return self.client

    async def __aexit__(self, *args: object) -> None:
        return None


def tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=f"Description for {name}",
        input_schema={"type": "object", "properties": {}},
    )


def test_list_all_tools_follows_pagination() -> None:
    client = FakeClient()

    tools = asyncio.run(list_all_tools(client))

    assert [item.name for item in tools] == ["first_tool", "second_tool"]
    assert client.cursors == [None, "page-2"]


def test_inspect_server_connects_and_prints_tools(capsys) -> None:
    client = FakeClient()
    requested_urls: list[str] = []

    def client_factory(url: str) -> FakeClientContext:
        requested_urls.append(url)
        return FakeClientContext(client)

    tools = asyncio.run(inspect_server("https://example.test/mcp", client_factory))

    output = capsys.readouterr().out
    assert requested_urls == ["https://example.test/mcp"]
    assert [item.name for item in tools] == ["first_tool", "second_tool"]
    assert "Connected: test-server 1.0" in output
    assert "Protocol: 2025-11-25" in output
    assert "Tools (2):" in output
    assert "first_tool" in output
    assert '"type": "object"' in output
