"""The MTP-Align FastAPI proxy (m2 milestone).

Sits between an OpenAI-compatible agent framework and the llama.cpp server.
It transparently proxies ``/v1/chat/completions`` (streaming SSE) and, when
the model emits a tool-call chunk, defers it to the next MTP decode-window
boundary instead of letting the agent interrupt the stream mid-window — the
core thesis of MTP-Align.

This is the m2 milestone scaffold: the streaming passthrough, window
tracking, and tool-call flush-at-boundary are wired in so the proxy runs and
demonstrates the mechanism. Full validation against OpenCode/Hermes (m3)
and production hardening (backpressure, partial-chunk tool-call
reconstruction across deltas) are tracked as follow-ons.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from .config import MTPConfig
from .metrics import MetricsCollector
from .scheduler import FlushDecision, MTPScheduler


def create_app(config: MTPConfig | None = None) -> FastAPI:
    """Build the MTP-Align proxy FastAPI application."""
    cfg = config or MTPConfig()
    app = FastAPI(title="MTP-Align", version="0.1.0")
    app.state.config = cfg
    app.state.scheduler = MTPScheduler(
        window_size=cfg.window_size,
        max_batch=cfg.max_batch,
        flush_policy=cfg.flush_policy,
    )
    app.state.metrics = MetricsCollector(
        scheduler=app.state.scheduler,  # type: ignore[arg-type]
        interval_s=cfg.metrics_interval_s,
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        s: MTPScheduler = app.state.scheduler
        return {
            "status": "ok",
            "upstream": cfg.upstream,
            "window_size": cfg.window_size,
            "flush_policy": cfg.flush_policy,
            "tokens_seen": s.tokens_seen(),
            "windows_completed": s.windows_completed(),
            "continuity_pct": s.continuity_pct(),
        }

    @app.api_route(
        "/v1/chat/completions",
        methods=["POST"],
    )
    async def chat_completions(request: Request) -> StreamingResponse:
        """Proxy a streaming chat completion with MTP-aware tool-call flushing."""
        body = await request.json()
        upstream = cfg.upstream
        scheduler: MTPScheduler = app.state.scheduler
        metrics: MetricsCollector = app.state.metrics
        metrics.reset()

        async def stream_and_align() -> AsyncIterator[bytes]:
            tool_buffer: list[str] = []  # buffered tool-call SSE chunks (aligned)
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{upstream}/v1/chat/completions",
                    json=body,
                ) as resp:
                    if resp.status_code >= 400:
                        text = await resp.aread()
                        yield _sse_error(resp.status_code, text)
                        return
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        out, is_tool, decision = _inspect_line(line, scheduler)
                        if is_tool and cfg.flush_policy == "window_boundary":
                            # defer this tool-call chunk to the next boundary
                            tool_buffer.append(out)
                            if decision is FlushDecision.FLUSH:
                                for buffered in tool_buffer:
                                    yield buffered.encode()
                                tool_buffer.clear()
                            continue
                        # forward content/role chunks immediately; check for flush
                        if decision is FlushDecision.FLUSH and tool_buffer:
                            for buffered in tool_buffer:
                                yield buffered.encode()
                            tool_buffer.clear()
                        if out:
                            yield out.encode()
                    # drain anything still buffered at end-of-stream
                    for buffered in tool_buffer:
                        yield buffered.encode()

        return StreamingResponse(
            stream_and_align(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-MTP-Align": "0.1.0"},
        )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def passthrough(path: str, request: Request) -> JSONResponse:
        """Transparent passthrough for non-chat endpoints (e.g. /v1/models)."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(
                request.method,
                f"{cfg.upstream}/{path}",
                content=await request.body(),
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                params=request.query_params,
            )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)

    return app


def _inspect_line(line: str, scheduler: MTPScheduler) -> tuple[str, bool, FlushDecision]:
    """Inspect one upstream SSE line; advance the window and detect tool calls.

    Returns ``(outgoing_line, is_tool_call, flush_decision)``. Decoded tokens
    advance the MTP window; tool-call deltas are flagged for boundary flush.
    """
    out = line if line.endswith("\n") else line + "\n"
    if not line.startswith("data:"):
        return out, False, FlushDecision.WAIT
    data = line[5:].strip()
    if data == "[DONE]":
        return out, False, FlushDecision.WAIT
    is_tool = False
    try:
        payload = json.loads(data)
        choices = payload.get("choices") or []
        for choice in choices:
            delta = choice.get("delta") or {}
            if delta.get("tool_calls"):
                is_tool = True
            content = delta.get("content")
            if isinstance(content, str) and content:
                for _ in content:
                    # one token per non-empty content chunk (coarse; m2 refines)
                    decision = scheduler.on_token()
                    if decision is FlushDecision.FLUSH and is_tool:
                        return out, True, decision
                # if no content tokens but a tool call, still advance once
                if is_tool and not content:
                    decision = scheduler.on_token()
                    return out, True, decision
    except json.JSONDecodeError:
        pass
    return out, is_tool, FlushDecision.WAIT


def _sse_error(status: int, body: bytes) -> bytes:
    payload = {"error": {"upstream_status": status, "message": body.decode("utf-8", "replace")}}
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode()


__all__ = ["create_app"]
