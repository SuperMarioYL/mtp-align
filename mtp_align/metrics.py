"""Real-time metrics collection for the MTP-Align proxy and benchmark.

The :class:`MetricsCollector` records decode tps (tokens / wall time) and
window-continuity % (windows that ran to their boundary vs. interrupted by a
tool call). It is driven by the scheduler's state and emits a one-line
summary to stderr every ``metrics_interval_s``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import TextIO

from .scheduler import MTPScheduler


@dataclass
class MetricsSample:
    """One point-in-time snapshot of proxy throughput."""

    tokens: int
    wall_s: float
    tps: float
    continuity_pct: float
    windows: int
    interrupts: int


@dataclass
class MetricsCollector:
    """Accumulates decode-side metrics and logs them to a stream.

    Parameters
    ----------
    scheduler:
        The scheduler whose window/continuity state is being observed.
    interval_s:
        Minimum wall seconds between two stderr log lines during a run.
    stream:
        Where to write the periodic summary (defaults to stderr).
    """

    scheduler: MTPScheduler
    interval_s: float = 0.5
    stream: TextIO = field(default_factory=lambda: sys.stderr)

    _start: float = field(default_factory=time.monotonic)
    _last_log: float = 0.0
    _tokens: int = 0

    def reset(self) -> None:
        self._start = time.monotonic()
        self._last_log = 0.0
        self._tokens = 0

    def on_token(self) -> None:
        self._tokens += 1
        now = time.monotonic()
        if now - self._last_log >= self.interval_s:
            self._emit(now)

    def sample(self) -> MetricsSample:
        wall = max(time.monotonic() - self._start, 1e-9)
        tps = self._tokens / wall
        return MetricsSample(
            tokens=self._tokens,
            wall_s=round(wall, 4),
            tps=round(tps, 2),
            continuity_pct=self.scheduler.continuity_pct(),
            windows=self.scheduler.windows_completed(),
            interrupts=len(self.scheduler._interrupts),  # noqa: SLF001
        )

    def final(self) -> MetricsSample:
        sample = self.sample()
        self._emit(time.monotonic(), force=True)
        return sample

    def _emit(self, now: float, *, force: bool = False) -> None:
        if not force and now - self._last_log < self.interval_s:
            return
        self._last_log = now
        s = self.sample()
        print(
            f"mtp-align  tokens={s.tokens}  wall={s.wall_s:.3f}s  "
            f"tps={s.tps:.1f}  continuity={s.continuity_pct:.1f}%  "
            f"windows={s.windows}  interrupts={s.interrupts}",
            file=self.stream,
            flush=True,
        )


__all__ = ["MetricsSample", "MetricsCollector"]
