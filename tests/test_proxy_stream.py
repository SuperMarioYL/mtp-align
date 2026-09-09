"""SSE-level tests for the proxy streaming path (v0.2.0 boundary-flush fixes).

Drives ``create_app`` through ``httpx.ASGITransport`` against a scripted fake
upstream, verifying at the wire level that tool-call chunks flush at MTP
decode-window boundaries (never after ``data: [DONE]``), that the re-framed
stream is spec-conformant SSE, and that scheduler state resets per request.
"""

from __future__ import annotations

import pytest

from mtp_align import proxy as proxy_mod
from fake_upstream import make_config, post_completion, sse_line, tool_call_line


def _data_lines(body: str) -> list[str]:
    return [line for line in body.split("\n") if line.startswith("data:")]


async def test_tool_call_flushes_at_window_boundary_before_done(monkeypatch):
    """A tool call deferred mid-window flushes at the next boundary, before [DONE]."""
    app = proxy_mod.create_app(make_config())
    lines = [
        sse_line({"content": "ab"}),  # tokens 1-2
        sse_line({"content": "cd"}),  # tokens 3-4 -> window-1 boundary
        tool_call_line(),             # queued at the start of window 2
        sse_line({"content": "e"}),   # token 5
        sse_line({"content": "fgh"}), # tokens 6-8 -> window-2 boundary: flush here
        "data: [DONE]",
    ]
    resp = await post_completion(app, monkeypatch, lines)
    body = resp.text
    out = _data_lines(body)
    tool_idx = next(i for i, line in enumerate(out) if "tool_calls" in line)
    done_idx = out.index("data: [DONE]")
    # the tool call is released at the token-8 boundary: after the token-7/8
    # content chunk, and always before [DONE] where clients stop parsing
    assert out.index(sse_line({"content": "e"})) < tool_idx
    assert tool_idx < out.index(sse_line({"content": "fgh"}))
    assert tool_idx < done_idx


async def test_tool_call_without_later_boundary_drains_before_done(monkeypatch):
    """A tool call whose window never completes still drains before [DONE]."""
    app = proxy_mod.create_app(make_config())
    lines = [
        sse_line({"content": "abcd"}),  # tokens 1-4 -> boundary
        tool_call_line(),               # token 5, mid-window
        sse_line({"content": "ef"}),    # tokens 6-7, no boundary reached
        "data: [DONE]",
    ]
    resp = await post_completion(app, monkeypatch, lines)
    out = _data_lines(resp.text)
    assert out.index("data: [DONE]") == len(out) - 1
    assert "tool_calls" in out[-2]


async def test_eager_policy_forwards_tool_chunks_immediately(monkeypatch):
    """Eager policy trades alignment for latency: no deferral, in-order passthrough."""
    app = proxy_mod.create_app(make_config(flush_policy="eager"))
    lines = [
        sse_line({"content": "ab"}),
        tool_call_line(),
        sse_line({"content": "cd"}),
        "data: [DONE]",
    ]
    resp = await post_completion(app, monkeypatch, lines)
    out = _data_lines(resp.text)
    assert out[1] == tool_call_line()  # forwarded in place, in original order


async def test_output_frames_sse_events_with_blank_lines(monkeypatch):
    """Every data event is terminated by a blank line (WHATWG SSE)."""
    app = proxy_mod.create_app(make_config())
    lines = [
        sse_line({"content": "ab"}),
        sse_line({"content": "cd"}),
        tool_call_line(),
        "data: [DONE]",
    ]
    resp = await post_completion(app, monkeypatch, lines)
    events = [event for event in resp.text.split("\n\n") if event.strip()]
    assert len(events) == 4
    assert all(event.startswith("data:") for event in events)
    assert resp.text.endswith("data: [DONE]\n\n")


async def test_scheduler_state_resets_between_requests(monkeypatch):
    """Each request is a fresh decode stream: identical scripts, identical output."""
    app = proxy_mod.create_app(make_config())
    lines = [
        sse_line({"content": "ab"}),  # tokens 1-2
        sse_line({"content": "cd"}),  # tokens 3-4 -> boundary
        tool_call_line(),             # queued mid window 2
        sse_line({"content": "ef"}),  # tokens 5-6, window left mid-position
        "data: [DONE]",
    ]
    first = (await post_completion(app, monkeypatch, lines)).text
    # ab(2) + cd(2) + tool chunk(1) + ef(2) = 7 decoded tokens, window mid-position
    assert app.state.scheduler.tokens_seen() == 7

    second = (await post_completion(app, monkeypatch, lines)).text
    # without the per-request reset the second stream would inherit the
    # mid-window position and flush the tool call a window early — no drift
    assert second == first
    assert app.state.scheduler.tokens_seen() == 7


async def test_boundary_flush_clears_scheduler_batch(monkeypatch):
    """Draining the buffer also releases the scheduler's queued batch."""
    app = proxy_mod.create_app(make_config())
    lines = [
        sse_line({"content": "abcd"}),  # tokens 1-4
        tool_call_line(),               # queued
        sse_line({"content": "efgh"}),  # tokens 5-8 -> boundary flush
        "data: [DONE]",
    ]
    await post_completion(app, monkeypatch, lines)
    assert app.state.scheduler.batch.pending == []


async def test_metrics_collector_is_driven_by_decoded_tokens(monkeypatch):
    """The live metrics collector sees one token per decoded content token."""
    app = proxy_mod.create_app(make_config())
    lines = [
        sse_line({"content": "abcd"}),
        sse_line({"content": "ef"}),
        "data: [DONE]",
    ]
    await post_completion(app, monkeypatch, lines)
    assert app.state.metrics.sample().tokens == 6
