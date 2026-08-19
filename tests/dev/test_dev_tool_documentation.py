"""Contract tests for the documented development-tool invocation surfaces."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_TOOLS_README = REPO_ROOT / "scripts/dev/README.md"


def test_support_tools_are_classified_and_linked() -> None:
    """The two opt-in tools remain discoverable without becoming implicit gates."""

    text = DEV_TOOLS_README.read_text(encoding="utf-8")

    assert "check_pr_closing_reference.py" in text
    assert "check_merge_queue_protection.py" in text
    assert "pr_ready_check.sh" in text
    assert "merge_queue_gate.py" in text
    assert "--self-test" in text
    assert "--check --repo ll7/robot_sf_ll7" in text
    assert "must not infer" in text
    assert "read-only" in text


def test_support_tools_expose_help_without_live_github_access() -> None:
    """Both documented support tools expose a usable CLI help surface."""

    for script in ("check_pr_closing_reference.py", "check_merge_queue_protection.py"):
        result = subprocess.run(
            [sys.executable, f"scripts/dev/{script}", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
