"""Tests for the MTP decode-window scheduler — the core primitive."""

from __future__ import annotations

import pytest

from mtp_align.scheduler import (
    FlushDecision,
    MTPScheduler,
    ToolCall,
)


def test_window_boundary_flushes_pending_batch():
    """A queued tool call flushes exactly at the window boundary."""
    sched = MTPScheduler(window_size=4)
    sched.queue_tool_call(ToolCall(id=1, name="search", args={"q": "x"}))

    decisions = [sched.on_token() for _ in range(4)]
    # tokens 1..3: mid-window, no flush; token 4: boundary → flush
    assert decisions[:3] == [FlushDecision.WAIT] * 3
    assert decisions[3] is FlushDecision.FLUSH
    flushed = sched.flush()
    assert len(flushed) == 1
    assert flushed[0].name == "search"
    assert sched.windows_completed() == 1


def test_max_batch_forces_early_flush_mid_window():
    """A full batch flushes early even mid-window so a flood can't stall."""
    sched = MTPScheduler(window_size=4, max_batch=2, flush_policy="window_boundary")
    sched.queue_tool_call(ToolCall(id=1, name="a", args={}))
    sched.queue_tool_call(ToolCall(id=2, name="b", args={}))
    # batch is full → first token after queueing triggers an early flush
    decision = sched.on_token()
    assert decision is FlushDecision.FLUSH
    assert sched.flush() == [ToolCall(id=1, name="a", args={}), ToolCall(id=2, name="b", args={})]


def test_eager_policy_flushes_past_first_token():
    """Eager mode flushes a non-empty batch once past the first token."""
    sched = MTPScheduler(window_size=4, flush_policy="eager")
    sched.queue_tool_call(ToolCall(id=1, name="a", args={}))
    sched.on_token()  # token 1: position 0, no flush yet
    decision = sched.on_token()  # token 2: position 1 → eager flush
    assert decision is FlushDecision.FLUSH


def test_force_interrupt_records_and_resets_window():
    """An agent cutting to a tool mid-window abandons the window."""
    sched = MTPScheduler(window_size=4)
    sched.on_token()  # 1
    sched.on_token()  # 2
    sched.queue_tool_call(ToolCall(id=1, name="t", args={}))
    calls = sched.force_interrupt(offset=2)
    assert len(calls) == 1
    assert len(scheduler_interrupts(sched)) == 1
    # window reset: a fresh window starts
    assert sched.window.remaining == sched.window.size


def test_continuity_aligned_is_100pct():
    """No interrupts → every window completes → 100% continuity."""
    sched = MTPScheduler(window_size=4)
    for _ in range(8):  # two full windows
        sched.on_token()
    assert sched.continuity_pct() == 100.0
    assert sched.windows_completed() == 2


def test_continuity_drops_with_interrupts():
    """A mid-window interrupt lowers continuity below 100%."""
    sched = MTPScheduler(window_size=4)
    for _ in range(2):
        sched.on_token()
    sched.force_interrupt(offset=2)
    for _ in range(4):  # finish a fresh window
        sched.on_token()
    assert sched.continuity_pct() < 100.0
    assert sched.continuity_pct() > 0.0


def test_invalid_window_size_rejected():
    with pytest.raises(ValueError):
        MTPScheduler(window_size=0)


def scheduler_interrupts(sched: MTPScheduler) -> list[int]:
    return sched._interrupts  # noqa: SLF001
