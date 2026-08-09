"""Configuration for the MTP-Align proxy, scheduler, and benchmark harness.

All knobs live in one :class:`MTPConfig` dataclass so the CLI, proxy, scheduler,
and benchmark share a single source of truth. Defaults assume DeepSeek-V4's MTP
window of 4 tokens per forward pass, a local llama.cpp server on ``:8080``, and
the MTP-Align proxy listening on ``:8090``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MTPConfig:
    """Runtime configuration shared across the proxy, scheduler, and benchmark.

    Attributes
    ----------
    upstream:
        Base URL of the upstream OpenAI-compatible inference server
        (a llama.cpp server running DeepSeek-V4 with MTP/DSpark enabled).
    listen_host / listen_port:
        Bind address for the MTP-Align FastAPI proxy.
    window_size:
        Tokens emitted per MTP forward pass. DeepSeek-V4 uses 4; this is the
        single model-specific knob the scheduler depends on.
    flush_policy:
        ``"window_boundary"`` (default) flushes queued tool calls only at window
        boundaries; ``"eager"`` flushes as soon as the batch is full even mid
        window (trades continuity for latency — kept for A/B experiments).
    max_batch:
        Maximum tool calls buffered before an early flush regardless of policy.
    benchmark_*:
        Synthetic workload shape used by ``mtp-align bench``: total decode
        tokens, number of interleaved tool calls, and warmup tokens discarded
        before timing.
    window_latency_s:
        Modelled wall time of one MTP forward pass — used by the synthetic
        mock upstream so the kill-gate benchmark can run without a GPU.
    tool_overhead_s:
        Modelled round-trip cost of one tool call (interrupt + resume).
    metrics_interval_s:
        How often the live metrics collector logs tps + continuity to stderr.
    """

    upstream: str = "http://localhost:8080"
    listen_host: str = "0.0.0.0"
    listen_port: int = 8090

    # MTP decode-window primitive
    window_size: int = 4
    flush_policy: str = "window_boundary"
    max_batch: int = 8

    # Synthetic benchmark workload
    benchmark_tokens: int = 512
    benchmark_tool_calls: int = 12
    benchmark_warmup: int = 32

    # Modelled latencies for the mock upstream (real upstream ignores these)
    window_latency_s: float = 0.08
    tool_overhead_s: float = 0.012

    # Live metrics
    metrics_interval_s: float = 0.5

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.flush_policy not in {"window_boundary", "eager"}:
            raise ValueError(
                f"flush_policy must be 'window_boundary' or 'eager', got {self.flush_policy!r}"
            )
        if self.benchmark_tokens <= self.benchmark_warmup:
            raise ValueError("benchmark_tokens must exceed benchmark_warmup")


__all__ = ["MTPConfig"]
