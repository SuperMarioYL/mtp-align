"""Version lockstep: every version surface reports the same version."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from typer.testing import CliRunner

from fake_upstream import make_config, post_completion, sse_line
from mtp_align import __version__, proxy as proxy_mod
from mtp_align.cli import app as cli_app

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "0.2.0"


def test_version_file_matches():
    assert (REPO_ROOT / "VERSION").read_text().strip() == EXPECTED


def test_pyproject_version_matches():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == EXPECTED


def test_dunder_version_matches():
    assert __version__ == EXPECTED


def test_cli_version_flag_matches():
    result = CliRunner().invoke(cli_app, ["--version"])
    assert result.exit_code == 0
    assert EXPECTED in result.output


def test_fastapi_app_version_matches():
    assert proxy_mod.create_app(make_config()).version == EXPECTED


async def test_proxy_header_matches(monkeypatch):
    app = proxy_mod.create_app(make_config())
    resp = await post_completion(app, monkeypatch, [sse_line({"content": "ab"}), "data: [DONE]"])
    assert resp.headers["X-MTP-Align"] == EXPECTED


def test_site_content_version_matches():
    site = json.loads((REPO_ROOT / "web" / "site.json").read_text())
    assert site["content_version"] == "v" + EXPECTED
