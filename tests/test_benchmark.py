"""Tests for the benchmark harness — the m1 kill-gate.

The mock engine is deterministic and GPU-free, so these tests exercise the
full naive-vs-aligned A/B end-to-end and assert the methodology:
both modes produce results, continuity is ≤ 100% everywhere, the aligned
arm never interrupts, and the delta is computed and reported.
"""

from __future__ import annotations

import asyncio

import pytest

from mtp_align.benchmark import (
    BenchmarkHarness,
    MockEngine,
    build_workload,
)
from mtp_align.config import MTPConfig


@pytest.fixture()
def cfg() -> MTPConfig:
    # tiny, fast workload for CI; window_latency dominates so the model is stable
    return MTPConfig(
        window_size=4,
        benchmark_tokens=128,
        benchmark_tool_calls=8,
        benchmark_warmup=8,
        window_latency_s=0.02,
        tool_overhead_s=0.003,
    )


def test_build_workload_is_deterministic_and_inside_windows(cfg: MTPConfig):
    w1 = build_workload(cfg, seed=42)
    w2 = build_workload(cfg, seed=42)
    assert w1.tool_call_offsets == w2.tool_call_offsets
    assert w1.total_tokens == cfg.benchmark_tokens
    # every tool call lands strictly inside a window (not on a boundary token)
    for off in w1.tool_call_offsets:
        assert off % cfg.window_size != 0
        assert 0 < off < cfg.benchmark_tokens


def test_harness_runs_both_modes_and_computes_delta(cfg: MTPConfig):
    harness = BenchmarkHarness(config=cfg)
    naive, aligned = asyncio.run(harness.run())
    assert naive.mode == "naive"
    assert aligned.mode == "aligned"
    assert naive.total_tokens == aligned.total_tokens == cfg.benchmark_tokens
    # continuity: aligned never interrupts → 100%; naive has interrupts → < 100%
    assert aligned.continuity_pct == 100.0
    assert aligned.interrupts == 0
    assert naive.interrupts == cfg.benchmark_tool_calls
    assert naive.continuity_pct < 100.0
    # delta is computed on the aligned row
    assert aligned.delta_pct is not None
    assert aligned.delta_pct > 0


def test_kill_gate_threshold_is_five_pct():
    harness = BenchmarkHarness(config=MTPConfig())
    assert harness.kill_gate_pct == 5.0


def test_mock_engine_aligned_uses_no_waste_windows(cfg: MTPConfig):
    """Aligned wall == full windows only; naive wall > aligned wall."""
    engine = MockEngine(cfg)
    workload = build_workload(cfg)
    naive = asyncio.run(engine.run(workload, "naive"))
    aligned = asyncio.run(engine.run(workload, "aligned"))
    assert naive.wall_s > aligned.wall_s
    assert aligned.tps > naive.tps
    # delta must clear the 5% kill-gate under the synthetic upper-bound model
    delta = (aligned.tps - naive.tps) / naive.tps * 100.0
    assert delta >= 5.0


def test_zero_tool_calls_means_no_delta() -> None:
    """With no tool calls there is nothing to interrupt → no delta."""
    cfg = MTPConfig(
        window_size=4,
        benchmark_tokens=128,
        benchmark_tool_calls=0,
        benchmark_warmup=8,
    )
    harness = BenchmarkHarness(config=cfg)
    naive, aligned = asyncio.run(harness.run())
    assert naive.interrupts == 0
    assert aligned.interrupts == 0
    assert aligned.delta_pct is not None
    assert abs(aligned.delta_pct) < 1e-6
