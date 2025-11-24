"""CLI help output tests (Phase 7 T080)."""

from __future__ import annotations

import subprocess


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True)


def test_generate_report_cli_help():
    out = _run(["uv", "run", "python", "scripts/research/generate_report.py", "--help"])
    assert "--experiment-name" in out
    assert "--tracker-run" in out or "--baseline" in out  # fallback if interface evolved


def test_compare_ablations_cli_help():
    out = _run(["uv", "run", "python", "scripts/research/compare_ablations.py", "--help"])
    assert "--config" in out or "--ablation-config" in out
    assert "ablation" in out.lower()
