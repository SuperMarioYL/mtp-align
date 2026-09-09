"""Shared test harness: script a fake upstream SSE stream through the proxy app.

The proxy constructs its own ``httpx.AsyncClient`` inside the streaming
route, so the fake upstream is injected by swapping the module-level
``httpx`` reference in :mod:`mtp_align.proxy` with a shim that replays a
scripted list of SSE lines.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from mtp_align import proxy as proxy_mod
from mtp_align.config import MTPConfig


def sse_line(delta: dict[str, Any]) -> str:
    """One upstream SSE data frame carrying a chat-completion delta."""
    return "data: " + json.dumps({"choices": [{"delta": delta}]})


def tool_call_line(name: str = "search") -> str:
    """One SSE data frame carrying a tool-call delta."""
    return sse_line(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": name, "arguments": "{}"},
                }
            ]
        }
    )


class _StreamCtx:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.status_code = 200

    async def __aenter__(self) -> "_StreamCtx":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _ClientCtx:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self) -> "_ClientCtx":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def stream(self, method: str, url: str, json: Any = None) -> _StreamCtx:
        return _StreamCtx(self._lines)


class _HttpxShim:
    """Stands in for the ``httpx`` module inside ``mtp_align.proxy``."""

    def __init__(self, lines: list[str]) -> None:
        self.AsyncClient = lambda **kwargs: _ClientCtx(lines)  # noqa: E731


async def post_completion(
    app: Any, monkeypatch: pytest.MonkeyPatch, lines: list[str]
) -> httpx.Response:
    """POST a streaming completion through ``app`` with a scripted upstream."""
    monkeypatch.setattr(proxy_mod, "httpx", _HttpxShim(lines))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [], "stream": True},
        )


def make_config(**overrides: Any) -> MTPConfig:
    """A config for proxy tests, defaulting to the DeepSeek-V4 window of 4."""
    defaults: dict[str, Any] = {"window_size": 4, "flush_policy": "window_boundary"}
    return MTPConfig(**(defaults | overrides))
