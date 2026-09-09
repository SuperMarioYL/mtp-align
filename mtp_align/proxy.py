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

from . import __version__
from .config import MTPConfig
from .metrics import MetricsCollector
from .scheduler import FlushDecision, MTPScheduler, ToolCall


def create_app(config: MTPConfig | None = None) -> FastAPI:
    """Build the MTP-Align proxy FastAPI application."""
    cfg = config or MTPConfig()
    app = FastAPI(title="MTP-Align", version=__version__)
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
        # each request is an independent decode stream starting at offset 0
        scheduler.reset()
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
                        # end-of-stream is itself a flush point: drain before [DONE]
                        is_done = line.startswith("data:") and line[5:].strip() == "[DONE]"
                        out, is_tool, decision = _inspect_line(line, scheduler, metrics)
                        flush_now = decision is FlushDecision.FLUSH or is_done
                        if is_tool and cfg.flush_policy == "window_boundary":
                            # defer this tool-call chunk to the next boundary
                            tool_buffer.append(out)
                            if flush_now:
                                for chunk in _drain(tool_buffer, scheduler):
                                    yield chunk
                            continue
                        # forward content/role chunks immediately; flush buffered tool calls
                        if flush_now and tool_buffer:
                            for chunk in _drain(tool_buffer, scheduler):
                                yield chunk
                        if out:
                            yield out.encode()
                    # drain anything still buffered if the stream ended without [DONE]
                    for chunk in _drain(tool_buffer, scheduler):
                        yield chunk

        return StreamingResponse(
            stream_and_align(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-MTP-Align": __version__},
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


def _frame(line: str) -> str:
    """Re-frame one upstream SSE line for the downstream stream.

    ``aiter_lines`` strips terminators and blank separator lines, so the frame
    must be rebuilt: a ``data:`` line is a complete SSE event and needs the
    trailing blank line strict parsers (e.g. the OpenAI SDK) dispatch on;
    other lines (``event:``, comments) keep a single newline so they stay
    attached to the data line that ends their event.
    """
    if line.startswith("data:"):
        return line + "\n\n"
    return line + "\n"


def _drain(tool_buffer: list[str], scheduler: MTPScheduler) -> list[bytes]:
    """Release queued tool-call chunks and clear the scheduler batch."""
    scheduler.flush()
    chunks = [chunk.encode() for chunk in tool_buffer]
    tool_buffer.clear()
    return chunks


def _inspect_line(
    line: str, scheduler: MTPScheduler, metrics: MetricsCollector
) -> tuple[str, bool, FlushDecision]:
    """Inspect one upstream SSE line; advance the window and detect tool calls.

    Returns ``(outgoing_line, is_tool_call, flush_decision)``. Decoded tokens
    advance the MTP window and the metrics collector; tool-call deltas are
    queued with the scheduler (under ``window_boundary`` policy) so the next
    window boundary triggers a flush. The flush decision is propagated from
    every line, including content-only lines that cross a boundary.
    """
    out = _frame(line)
    if not line.startswith("data:"):
        return out, False, FlushDecision.WAIT
    data = line[5:].strip()
    if data == "[DONE]":
        return out, False, FlushDecision.WAIT
    is_tool = False
    decision = FlushDecision.WAIT
    try:
        payload = json.loads(data)
        choices = payload.get("choices") or []
        for choice in choices:
            delta = choice.get("delta") or {}
            tool_calls = delta.get("tool_calls")
            if tool_calls:
                is_tool = True
                if scheduler.flush_policy == "window_boundary":
                    # register pendingness so the next boundary triggers a flush
                    for tc in tool_calls:
                        fn = tc.get("function") or {}
                        scheduler.queue_tool_call(
                            ToolCall(
                                id=int(tc.get("index") or 0),
                                name=str(fn.get("name", "")),
                            )
                        )
            content = delta.get("content")
            if isinstance(content, str) and content:
                for _ in content:
                    # one token per non-empty content chunk (coarse; m2 refines)
                    if scheduler.on_token() is FlushDecision.FLUSH:
                        decision = FlushDecision.FLUSH
                    metrics.on_token()
            elif is_tool:
                # a tool-call chunk without content still advances the window once
                if scheduler.on_token() is FlushDecision.FLUSH:
                    decision = FlushDecision.FLUSH
                metrics.on_token()
    except json.JSONDecodeError:
        pass
    return out, is_tool, decision


def _sse_error(status: int, body: bytes) -> bytes:
    payload = {"error": {"upstream_status": status, "message": body.decode("utf-8", "replace")}}
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode()


__all__ = ["create_app"]
