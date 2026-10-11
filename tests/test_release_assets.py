"""Guard: the release workflow must attach dist artifacts to the GitHub release.

v0.2.0 shipped an asset-less release because release.yml uploaded dist/* only
as a workflow artifact — nothing ever attached the wheel, sdist, or checksums
to the release itself. These assertions pin the attach step so a future
workflow edit cannot silently ship another asset-less release.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"


def _workflow_text() -> str:
    return RELEASE_WORKFLOW.read_text()


def test_release_workflow_attaches_dist_to_the_release():
    text = _workflow_text()
    assert "action-gh-release" in text, (
        "release.yml has no step attaching artifacts to the GitHub release"
    )
    assert re.search(r"files:\s*dist/", text), "attach step does not publish dist/"


def test_release_workflow_still_uploads_ci_artifact():
    text = _workflow_text()
    assert "actions/upload-artifact" in text, "dist CI artifact upload was dropped"
