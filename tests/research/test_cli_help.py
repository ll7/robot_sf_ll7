"""CLI help output tests (Phase 7 T080)."""

from __future__ import annotations

import subprocess


def _run(cmd: list[str]) -> str:
    """Run.

    Args:
        cmd: Auto-generated placeholder description.

    Returns:
        str: Auto-generated placeholder description.
    """
    return subprocess.check_output(cmd, text=True)


def test_generate_report_cli_help():
    """Test generate report cli help.

    Returns:
        Any: Auto-generated placeholder description.
    """
    out = _run(["uv", "run", "python", "scripts/research/generate_report.py", "--help"])
    assert "--experiment-name" in out
    assert "--tracker-run" in out or "--baseline" in out  # fallback if interface evolved


def test_compare_ablations_cli_help():
    """Test compare ablations cli help.

    Returns:
        Any: Auto-generated placeholder description.
    """
    out = _run(["uv", "run", "python", "scripts/research/compare_ablations.py", "--help"])
    assert "--config" in out or "--ablation-config" in out
    assert "ablation" in out.lower()
