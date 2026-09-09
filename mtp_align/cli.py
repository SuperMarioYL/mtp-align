"""The MTP-Align command-line interface (Typer).

Two commands ship the v0.1 value:

* ``mtp-align bench`` — runs the kill-gate benchmark and prints a before/after
  tps table. Defaults to the synthetic mock (no GPU needed); pass
  ``--upstream`` to measure against a live llama.cpp server.
* ``mtp-align serve`` — starts the MTP-aware proxy on ``:8090`` so an agent
  can point at it and get tool-call batching for free (m2 milestone).

Run ``mtp-align --help`` for the full surface.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .benchmark import BenchmarkHarness, HttpEngine, MockEngine
from .config import MTPConfig

app = typer.Typer(
    name="mtp-align",
    help="DeepSeek-V4 MTP-aware agent-loop scheduler.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console(stderr=False)
err_console = Console(stderr=True)


def _build_config(**kwargs: Any) -> MTPConfig:
    """Build an MTPConfig, turning validation errors into a clean CLI error."""
    try:
        return MTPConfig(**kwargs)
    except ValueError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mtp-align {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed version and exit.",
    ),
) -> None:
    """DeepSeek-V4 MTP-aware agent-loop scheduler."""


@app.command()
def bench(
    upstream: str = typer.Option(
        "",
        "--upstream",
        help="OpenAI-compatible upstream base URL (e.g. http://localhost:8080). "
        "Omit to run the synthetic mock benchmark (no GPU needed).",
    ),
    tokens: int = typer.Option(512, "--tokens", help="Total decode tokens in the workload."),
    tool_calls: int = typer.Option(12, "--tool-calls", help="Interleaved tool calls."),
    window: int = typer.Option(4, "--window", help="MTP tokens per forward pass (DSV4=4)."),
    warmup: int = typer.Option(32, "--warmup", help="Warmup tokens discarded before timing."),
) -> None:
    """Run the before/after tps benchmark and print a falsifiable delta table."""
    cfg = _build_config(
        upstream=upstream or "http://localhost:8080",
        window_size=window,
        benchmark_tokens=tokens,
        benchmark_tool_calls=tool_calls,
        benchmark_warmup=warmup,
    )
    engine = HttpEngine(cfg) if upstream else MockEngine(cfg)
    synthetic = not bool(upstream)
    harness = BenchmarkHarness(config=cfg, engine=engine)

    label = "SYNTHETIC (model-based upper bound — not a GPU measurement)" if synthetic else "REAL upstream"
    err_console.print(f"[bold]mtp-align bench[/bold]  ·  {label}")
    err_console.print(
        f"  workload: {tokens} tokens, {tool_calls} tool calls, window={window}\n"
    )

    try:
        results = asyncio.run(harness.run())
    except RuntimeError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)

    table = Table(title="MTP-Align benchmark — naive vs. MTP-aligned decode")
    table.add_column("mode", style="cyan", no_wrap=True)
    table.add_column("tokens", justify="right")
    table.add_column("wall (s)", justify="right")
    table.add_column("tps", justify="right", style="green")
    table.add_column("continuity %", justify="right")
    table.add_column("delta %", justify="right", style="bold magenta")

    for r in results:
        delta = "—" if r.delta_pct is None else f"{r.delta_pct:+.1f}"
        table.add_row(
            r.mode,
            str(r.total_tokens),
            f"{r.wall_s:.3f}",
            f"{r.tps:.2f}",
            f"{r.continuity_pct:.1f}",
            delta,
        )

    console.print(table)

    aligned = results[-1]
    delta = aligned.delta_pct or 0.0
    gate = harness.kill_gate_pct
    if delta >= gate:
        console.print(
            f"\n[green]✓ kill-gate PASSED:[/green] tps delta [bold]+{delta:.1f}%[/bold] "
            f"(gate ≥ {gate:.0f}%). MTP-aligned decode recovers throughput."
        )
    else:
        console.print(
            f"\n[red]✗ kill-gate FAILED:[/red] tps delta [bold]+{delta:.1f}%[/bold] "
            f"< gate {gate:.0f}%. Thesis not supported — stop here."
        )
        if synthetic:
            err_console.print(
                "  (synthetic result; the real gate is `mtp-align bench --upstream <url>`)"
            )


@app.command()
def serve(
    upstream: str = typer.Option(
        "http://localhost:8080",
        "--upstream",
        help="Base URL of the upstream llama.cpp server.",
    ),
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address."),
    port: int = typer.Option(8090, "--port", help="Listen port."),
    window: int = typer.Option(4, "--window", help="MTP tokens per forward pass (DSV4=4)."),
    flush: str = typer.Option(
        "window_boundary",
        "--flush",
        help="Flush policy: window_boundary (default) or eager.",
    ),
) -> None:
    """Start the MTP-aware proxy. Point your agent at http://<host>:<port>/v1."""
    cfg = _build_config(
        upstream=upstream,
        listen_host=host,
        listen_port=port,
        window_size=window,
        flush_policy=flush,
    )
    try:
        import uvicorn
        from .proxy import create_app
    except ImportError as exc:  # pragma: no cover - fastapi/uvicorn are key deps
        err_console.print(f"[red]missing dependency:[/red] {exc.name}. Run `pip install mtp-align`.")
        raise typer.Exit(code=1)

    err_console.print(f"[bold]mtp-align serve[/bold]  upstream={upstream}  window={window}")
    err_console.print(
        f"  listening on http://{host}:{port}/v1  "
        "(point your agent here, e.g. OPENAI_BASE_URL=http://localhost:8090/v1)\n"
    )
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    app()
