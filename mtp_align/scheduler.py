"""The MTP decode-window primitive and the tool-call batching scheduler.

This module is the heart of MTP-Align. DeepSeek-V4's multi-token-prediction
(MTP) emits ``window_size`` tokens per forward pass. If an agent loop
interrupts the stream mid-window to fire a tool call, the remaining tokens of
that window are wasted and must be regenerated. :class:`MTPScheduler` tracks
the window horizon and defers tool-call flushes to window boundaries so MTP
throughput survives the agent tool-call loop.

The primitive is DeepSeek-specific: GPT/Claude single-token streams have no
decode window to align against, so there is nothing for this scheduler to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FlushDecision(Enum):
    """Decision returned by :meth:`MTPScheduler.on_token`."""

    WAIT = "wait"
    FLUSH = "flush"


@dataclass
class Token:
    """A single decoded token with its absolute offset in the stream."""

    offset: int
    content: str = ""


@dataclass
class ToolCall:
    """An agent tool invocation queued for flush."""

    id: int
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class MTPWindow:
    """Scheduler-level model of one MTP forward pass.

    ``remaining`` counts down as tokens stream out; when it hits zero the
    window boundary is reached and a queued batch may flush.
    """

    size: int
    remaining: int
    horizon: int  # absolute token offset of the next boundary


@dataclass
class ToolCallBatch:
    """Tool calls deferred to a window boundary."""

    pending: list[ToolCall] = field(default_factory=list)
    flush_horizon: int = 0  # window boundary at which to flush


class MTPScheduler:
    """Tracks the MTP decode window and decides when to flush tool calls.

    The scheduler is a pure state machine over the token stream. It does not
    touch the network — that is the proxy's job. This separation lets the
    benchmark harness drive it with a synthetic workload and lets the proxy
    drive it with a real SSE stream.

    Parameters
    ----------
    window_size:
        Tokens per MTP forward pass (model-specific; DeepSeek-V4 = 4).
    max_batch:
        Hard cap on queued tool calls — an early flush is forced once reached
        even under ``window_boundary`` policy so a flood of tool calls cannot
        stall the agent indefinitely.
    flush_policy:
        ``"window_boundary"`` flushes only at window boundaries (maximises
        continuity); ``"eager"`` flushes as soon as the batch is non-empty
        and the window is past its first token (latency-favouring A/B mode).
    """

    def __init__(
        self,
        window_size: int = 4,
        max_batch: int = 8,
        flush_policy: str = "window_boundary",
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self.max_batch = max_batch
        self.flush_policy = flush_policy
        self.window = MTPWindow(
            size=window_size, remaining=window_size, horizon=window_size
        )
        self.batch = ToolCallBatch(pending=[], flush_horizon=window_size)
        self._tokens_seen = 0
        self._windows_completed = 0
        self._interrupts: list[int] = []  # offsets where a flush mid-window occurred
        self._boundary_flushes: int = 0

    # -- state inspection -------------------------------------------------

    def tokens_seen(self) -> int:
        return self._tokens_seen

    def windows_completed(self) -> int:
        return self._windows_completed

    def in_window_position(self) -> int:
        """Zero-based index of the last consumed token within the window.

        Returns 0 on the first token of a fresh window, 1 on the second, and
        so on; ``-1`` when no token has been consumed in the current window.
        """
        return self.window_size - self.window.remaining - 1

    def at_boundary(self) -> bool:
        """True when the next token to arrive opens a fresh window."""
        return self.window.remaining == self.window.size and self._tokens_seen > 0

    # -- mutations --------------------------------------------------------

    def queue_tool_call(self, call: ToolCall) -> None:
        """Buffer a tool call for a future flush."""
        self.batch.pending.append(call)

    def on_token(self, token: Token | None = None) -> FlushDecision:
        """Advance the window by one token and decide whether to flush.

        Decrements ``remaining``. On a window boundary (``remaining`` hits
        zero), the window resets and any pending batch flushes. Under the
        ``eager`` policy a non-empty batch flushes once past the first token
        of a window. A batch hitting ``max_batch`` flushes early regardless.

        The token offset, when provided, is used only for bookkeeping
        (interrupt offsets); passing ``None`` is fine for a pure count.
        """
        self._tokens_seen += 1
        self.window.remaining -= 1

        offset = token.offset if token is not None else self._tokens_seen

        # Window boundary reached: reset window, then maybe flush.
        if self.window.remaining <= 0:
            self._windows_completed += 1
            self.window.horizon += self.window_size
            self.window.remaining = self.window.size
            if self.batch.pending:
                self._boundary_flushes += 1
                return FlushDecision.FLUSH
            return FlushDecision.WAIT

        # Mid-window decisions.
        if self.batch.pending:
            if len(self.batch.pending) >= self.max_batch:
                self._interrupts.append(offset)
                return FlushDecision.FLUSH
            if self.flush_policy == "eager" and self.in_window_position() >= 1:
                self._interrupts.append(offset)
                return FlushDecision.FLUSH

        return FlushDecision.WAIT

    def force_interrupt(self, offset: int | None = None) -> list[ToolCall]:
        """Simulate an agent cutting to a tool call mid-window.

        Records the interrupt at ``offset`` (default: current token count),
        flushes the pending batch, and resets the window so a fresh decode
        window starts on the next token — mirroring the engine re-decoding
        the abandoned window tail on resume. Used by the naive benchmark arm
        and (eventually) by the proxy's eager-interrupt path.
        """
        if offset is None:
            offset = self._tokens_seen
        self._interrupts.append(offset)
        calls = self.flush()
        # abandon the in-flight window; a fresh one starts next token
        self.window.horizon = self._tokens_seen + self.window_size
        self.window.remaining = self.window.size
        return calls

    def flush(self) -> list[ToolCall]:
        """Release and return the queued tool calls, resetting the batch."""
        calls = self.batch.pending
        self.batch.pending = []
        self.batch.flush_horizon = self.window.horizon
        return calls

    # -- continuity metric ------------------------------------------------

    def continuity_pct(self) -> float:
        """Percentage of windows completed without a mid-window interrupt.

        100% means every window ran to its boundary (max MTP throughput).
        Under the naive model, where every tool call interrupts mid-window,
        this number drops toward ``1 - tool_calls / windows``.
        """
        total = self._windows_completed + len(self._interrupts)
        if total == 0:
            return 100.0
        return round(100.0 * self._windows_completed / total, 2)


__all__ = [
    "FlushDecision",
    "Token",
    "ToolCall",
    "MTPWindow",
    "ToolCallBatch",
    "MTPScheduler",
]
