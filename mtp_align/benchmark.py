"""The MTP-Align benchmark harness — the m1 kill-gate.

``mtp-align bench`` measures decode-tps in two modes against the same synthetic
agent workload:

* **naive** — every tool call interrupts the decode stream mid-window, wasting
  the in-flight MTP window (the agent abandons the tail of the window and the
  engine must re-decode it on resume).
* **aligned** — tool calls are queued to window boundaries via
  :class:`~mtp_align.scheduler.MTPScheduler`, so every window runs to
  completion and no decode work is wasted.

The output is a ``rich`` table: mode · total tokens · wall time · tps ·
window-continuity % · delta %. The kill-gate is a delta below 5%.

Two engines are provided:

* :class:`MockEngine` — a deterministic synthetic model of MTP decode windows.
  Runs in CI and in the demo with no GPU. The numbers it prints are a
  *model-based upper bound*, not a GPU measurement; they exist so the harness
  runs end-to-end and the methodology is exercised. The real kill-gate is the
  HTTP engine below against a live llama.cpp server.
* :class:`HttpEngine` — drives a real OpenAI-compatible upstream (llama.cpp
  with DeepSeek-V4 + MTP). The naive arm aborts the stream at tool-call
  offsets to simulate an agent cutting to a tool; the aligned arm reads each
  window to completion. This is the weekend-1 measurement.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .config import MTPConfig
from .scheduler import FlushDecision, MTPScheduler, Token, ToolCall


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass
class ModeResult:
    """One row of the benchmark table."""

    mode: str
    total_tokens: int
    wall_s: float
    tps: float
    continuity_pct: float
    windows: int
    interrupts: int
    delta_pct: float | None = None  # set on the aligned row only

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "mode": self.mode,
            "tokens": self.total_tokens,
            "wall_s": round(self.wall_s, 3),
            "tps": round(self.tps, 2),
            "continuity_pct": round(self.continuity_pct, 1),
        }
        if self.delta_pct is not None:
            row["delta_pct"] = round(self.delta_pct, 1)
        return row


@dataclass
class Workload:
    """A synthetic agent workload: a token stream with interleaved tool calls."""

    total_tokens: int
    tool_call_offsets: list[int]
    window_size: int


def build_workload(config: MTPConfig, *, seed: int = 42) -> Workload:
    """Evenly distribute ``benchmark_tool_calls`` tool calls across the stream.

    Each offset is forced to land *inside* a window (never on its first token)
    so the naive arm always interrupts mid-window — the worst case the scheduler
    exists to fix. Deterministic for a fixed seed so tests are reproducible.
    """
    rng = random.Random(seed)
    n, k, w = config.benchmark_tokens, config.benchmark_tool_calls, config.window_size
    if k <= 0:
        return Workload(total_tokens=n, tool_call_offsets=[], window_size=w)
    span = max(n - 2 * w, 1)
    offsets: list[int] = []
    for i in range(k):
        base = w + int(span * (i + 0.5) / k)
        window_start = (base // w) * w
        mid = window_start + w // 2
        jitter = rng.randint(-(w // 4), w // 4) if w >= 4 else 0
        off = max(window_start + 1, min(mid + jitter, n - 1))
        offsets.append(off)
    offsets = sorted(set(offsets))
    return Workload(total_tokens=n, tool_call_offsets=offsets, window_size=w)


# ---------------------------------------------------------------------------
# Engine protocol
# ---------------------------------------------------------------------------


class Engine(Protocol):
    """Something that can run one mode of the workload and return a result."""

    async def run(self, workload: Workload, mode: str) -> ModeResult: ...


# ---------------------------------------------------------------------------
# Mock engine — deterministic, GPU-free, for CI / demo / tests
# ---------------------------------------------------------------------------


class MockEngine:
    """Synthetic MTP decode model.

    Models one MTP forward pass as ``window_latency_s`` of wall time that
    produces ``window_size`` tokens. In the naive arm, a tool call interrupting
    mid-window abandons the in-flight window: the engine re-decodes the whole
    window on resume, so each naive interrupt costs a full extra forward pass.
    In the aligned arm the interrupt waits for the boundary, costing nothing.
    The tool-call round-trip itself costs ``tool_overhead_s`` in both arms.

    This is a *model-based upper bound* — it makes the harness runnable without
    a GPU and exercises the scheduler end-to-end. It is NOT a measurement; the
    real kill-gate is :class:`HttpEngine` against a live llama.cpp server.
    """

    def __init__(self, config: MTPConfig) -> None:
        self.config = config

    async def run(self, workload: Workload, mode: str) -> ModeResult:
        cfg = self.config
        w = workload.window_size
        scheduler = MTPScheduler(
            window_size=w, max_batch=cfg.max_batch, flush_policy="window_boundary"
        )
        tool_set = set(workload.tool_call_offsets)
        waste_windows = 0

        for offset in range(1, workload.total_tokens + 1):
            if offset in tool_set:
                call = ToolCall(id=offset, name=f"tool_{offset}", args={})
                if mode == "naive":
                    # agent cuts to the tool mid-window → window abandoned
                    scheduler.queue_tool_call(call)
                    scheduler.force_interrupt(offset)
                    waste_windows += 1
                else:
                    scheduler.queue_tool_call(call)
            decision = scheduler.on_token(Token(offset=offset, content=str(offset)))
            if decision is FlushDecision.FLUSH:
                scheduler.flush()

        # wall time: full windows + naive waste windows + tool-call round-trips
        full_windows = workload.total_tokens // w
        passes = full_windows + (waste_windows if mode == "naive" else 0)
        wall = (
            passes * cfg.window_latency_s
            + len(workload.tool_call_offsets) * cfg.tool_overhead_s
        )
        await asyncio.sleep(0)  # cooperative yield; keeps tests fast

        tps = workload.total_tokens / wall if wall > 0 else 0.0
        return ModeResult(
            mode=mode,
            total_tokens=workload.total_tokens,
            wall_s=wall,
            tps=tps,
            continuity_pct=scheduler.continuity_pct(),
            windows=scheduler.windows_completed(),
            interrupts=len(scheduler._interrupts),  # noqa: SLF001
        )


# ---------------------------------------------------------------------------
# HTTP engine — the real weekend-1 measurement against llama.cpp
# ---------------------------------------------------------------------------


class HttpEngine:
    """Drives a live OpenAI-compatible upstream.

    The naive arm streams a long completion and aborts the connection at each
    tool-call offset (simulating an agent cutting to a tool), then re-requests
    with the decoded prefix to continue — the abandoned window must be
    re-decoded. The aligned arm reads each window to completion before issuing
    the next tool call, mirroring what ``mtp-align serve`` does.

    Requires a reachable upstream; raises a clear error otherwise. The naive
    A/B is a best-effort real measurement — validate on your hardware (that
    validation *is* the m1 kill-gate).
    """

    def __init__(self, config: MTPConfig, *, timeout_s: float = 120.0) -> None:
        self.config = config
        self.timeout_s = timeout_s

    async def run(self, workload: Workload, mode: str) -> ModeResult:
        cfg = self.config
        w = workload.window_size
        scheduler = MTPScheduler(window_size=w, max_batch=cfg.max_batch)
        total = 0
        wall_start = time.monotonic()
        tool_offsets = sorted(workload.tool_call_offsets)
        call_idx = 0
        decoded_prefix: list[str] = []

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            await self._probe(client)
            while total < workload.total_tokens and call_idx <= len(tool_offsets):
                budget = workload.total_tokens - total
                abort_at = None
                if mode == "naive" and call_idx < len(tool_offsets):
                    abort_at = tool_offsets[call_idx] - total
                delivered, chunk = await self._stream_once(
                    client, budget, abort_at, prefix="".join(decoded_prefix)
                )
                if delivered == 0:
                    break
                for i in range(delivered):
                    scheduler.on_token(Token(offset=total + i + 1))
                total += delivered
                decoded_prefix.append(chunk)

                if mode == "naive" and call_idx < len(tool_offsets) and total >= tool_offsets[call_idx]:
                    scheduler.force_interrupt(total)
                    scheduler.flush()
                    call_idx += 1
                else:
                    while (
                        call_idx < len(tool_offsets)
                        and total >= tool_offsets[call_idx]
                        and total % w == 0
                    ):
                        scheduler.queue_tool_call(
                            ToolCall(id=call_idx, name=f"tool_{call_idx}", args={})
                        )
                        scheduler.flush()
                        call_idx += 1
                if mode == "naive" and call_idx >= len(tool_offsets) and total < workload.total_tokens:
                    # drain remaining tokens without further aborts
                    pass

        wall = time.monotonic() - wall_start
        return ModeResult(
            mode=mode,
            total_tokens=total,
            wall_s=wall,
            tps=total / wall if wall > 0 else 0.0,
            continuity_pct=scheduler.continuity_pct(),
            windows=scheduler.windows_completed(),
            interrupts=len(scheduler._interrupts),  # noqa: SLF001
        )

    async def _probe(self, client: httpx.AsyncClient) -> None:
        """Fail fast with a clear message if the upstream is unreachable."""
        try:
            resp = await client.get(f"{self.config.upstream}/v1/models")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"upstream {self.config.upstream} unreachable: {exc}. "
                "Start your llama.cpp server, or run `mtp-align bench` (mock)."
            ) from exc

    async def _stream_once(
        self,
        client: httpx.AsyncClient,
        budget: int,
        abort_at: int | None,
        *,
        prefix: str,
    ) -> tuple[int, str]:
        """Stream one completion; return (tokens delivered, decoded text)."""
        content = (
            "Output the integers from 1 to 400, one per line, no commentary. Begin: 1"
            + (f"\nContinue after: {prefix[-200:]}" if prefix else "")
        )
        body = {
            "model": "deepseek-v4",
            "messages": [{"role": "user", "content": content}],
            "stream": True,
            "max_tokens": max(budget, 1),
            "temperature": 0.0,
        }
        delivered = 0
        chunk = []
        try:
            async with client.stream(
                "POST", f"{self.config.upstream}/v1/chat/completions", json=body
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    # one SSE data frame ≈ one token chunk in the streaming protocol
                    delivered += 1
                    chunk.append(data)
                    if abort_at is not None and delivered >= abort_at:
                        break  # agent cuts to a tool → connection aborts
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"stream from {self.config.upstream} failed: {exc}. "
                "Check the model is loaded with MTP/DSpark enabled."
            ) from exc
        return delivered, "".join(chunk)[:200]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkHarness:
    """Runs both modes and returns the comparable result pair."""

    config: MTPConfig
    engine: Engine | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.engine is None:
            self.engine = MockEngine(self.config)

    async def run(self) -> list[ModeResult]:
        workload = build_workload(self.config, seed=self.seed)
        naive = await self.engine.run(workload, "naive")  # type: ignore[union-attr]
        aligned = await self.engine.run(workload, "aligned")
        if naive.tps > 0:
            aligned.delta_pct = (aligned.tps - naive.tps) / naive.tps * 100.0
        return [naive, aligned]

    @property
    def kill_gate_pct(self) -> float:
        """The m1 kill threshold: deltas below this kill the project."""
        return 5.0


__all__ = [
    "ModeResult",
    "Workload",
    "build_workload",
    "Engine",
    "MockEngine",
    "HttpEngine",
    "BenchmarkHarness",
]
