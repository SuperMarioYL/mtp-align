"""CLI error-handling contract: invalid arguments exit 2 with a clean message.

Before v0.2.0 these paths dumped a raw ValueError traceback from
MTPConfig.__post_init__ instead of behaving like a CLI.
"""

from __future__ import annotations

from typer.testing import CliRunner

from mtp_align.cli import app as cli_app

runner = CliRunner()


def _all_output(result) -> str:
    """Combine captured stdout and stderr across click versions."""
    output = result.output
    try:
        output += result.stderr or ""
    except (ValueError, AttributeError):
        pass
    return output


def test_bench_warmup_not_below_tokens_exits_2():
    result = runner.invoke(cli_app, ["bench", "--tokens", "128", "--warmup", "128"])
    assert result.exit_code == 2
    out = _all_output(result)
    assert "benchmark_tokens must exceed benchmark_warmup" in out
    assert "Traceback" not in out


def test_serve_invalid_flush_policy_exits_2():
    result = runner.invoke(cli_app, ["serve", "--flush", "bogus"])
    assert result.exit_code == 2
    out = _all_output(result)
    assert "flush_policy must be 'window_boundary' or 'eager'" in out
    assert "Traceback" not in out


def test_serve_zero_window_exits_2():
    result = runner.invoke(cli_app, ["serve", "--window", "0"])
    assert result.exit_code == 2
    out = _all_output(result)
    assert "window_size must be >= 1" in out
    assert "Traceback" not in out


def test_bench_mock_happy_path_still_succeeds():
    result = runner.invoke(
        cli_app, ["bench", "--tokens", "128", "--tool-calls", "8", "--warmup", "8"]
    )
    assert result.exit_code == 0
    assert "kill-gate" in _all_output(result)
