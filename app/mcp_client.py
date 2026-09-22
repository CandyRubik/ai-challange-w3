from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from mcp import Client


DEFAULT_MCP_SERVER_URL = "https://mcp.deepwiki.com/mcp"


async def list_all_tools(client: Any) -> list[Any]:
    """Return every tool, following MCP pagination cursors when present."""
    tools: list[Any] = []
    cursor: str | None = None

    while True:
        page = await client.list_tools(cursor=cursor)
        tools.extend(page.tools)
        cursor = page.next_cursor
        if cursor is None:
            return tools


def print_connection(client: Any, server_url: str) -> None:
    server_info = client.server_info
    server_name = getattr(server_info, "name", None) or "unknown"
    server_version = getattr(server_info, "version", None) or "unknown"
    print(f"Connected: {server_name} {server_version}")
    print(f"Endpoint: {server_url}")
    print(f"Protocol: {client.protocol_version}")


def print_tools(tools: list[Any]) -> None:
    print(f"Tools ({len(tools)}):")
    for tool in tools:
        print(f"\n- {tool.name}")
        if tool.description:
            print(f"  {tool.description}")
        print("  Input schema:")
        schema = json.dumps(tool.input_schema, ensure_ascii=False, indent=2)
        print("\n".join(f"    {line}" for line in schema.splitlines()))


async def inspect_server(
    server_url: str,
    client_factory: Callable[[str], AbstractAsyncContextManager[Any]] = Client,
) -> list[Any]:
    """Connect to an MCP server, print connection data, and list its tools."""
    async with client_factory(server_url) as client:
        print_connection(client, server_url)
        tools = await list_all_tools(client)
        print_tools(tools)
        return tools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect to a remote MCP server and list its tools.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL),
        help="Streamable HTTP MCP endpoint (default: DeepWiki MCP).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(inspect_server(args.url))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("MCP request cancelled.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"MCP connection failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
