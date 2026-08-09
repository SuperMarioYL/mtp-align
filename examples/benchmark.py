"""Minimal MTP-Align usage example.

Three lines: configure, run the kill-gate benchmark, read the delta.
"""

import asyncio

from mtp_align import __version__
from mtp_align.benchmark import BenchmarkHarness, MockEngine
from mtp_align.config import MTPConfig


def main() -> None:
    print(f"mtp-align {__version__}")
    cfg = MTPConfig(window_size=4, benchmark_tokens=512, benchmark_tool_calls=12)
    harness = BenchmarkHarness(config=cfg, engine=MockEngine(cfg))
    naive, aligned = asyncio.run(harness.run())
    print(f"naive   tps={naive.tps:.2f}  continuity={naive.continuity_pct:.1f}%")
    print(f"aligned tps={aligned.tps:.2f}  continuity={aligned.continuity_pct:.1f}%")
    print(f"delta   = {aligned.delta_pct:+.1f}%  (kill-gate >= {harness.kill_gate_pct:.0f}%)")


if __name__ == "__main__":
    main()
