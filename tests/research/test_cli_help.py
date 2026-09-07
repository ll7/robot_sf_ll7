"""CLI help output tests (Phase 7 T080)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _run(cmd: list[str]) -> str:
    """TODO docstring. Document this function.

    Args:
        cmd: TODO docstring.

    Returns:
        TODO docstring.
    """
    return subprocess.check_output(cmd, text=True)


def test_generate_report_cli_help():
    """TODO docstring. Document this function."""
    out = _run(["uv", "run", "python", "scripts/research/generate_report.py", "--help"])
    assert "--experiment-name" in out
    assert "--tracker-run" in out or "--baseline" in out  # fallback if interface evolved


def test_compare_ablations_cli_help():
    """TODO docstring. Document this function."""
    out = _run(["uv", "run", "python", "scripts/research/compare_ablations.py", "--help"])
    assert "--config" in out or "--ablation-config" in out
    assert "ablation" in out.lower()


@pytest.mark.parametrize(
    "script",
    [
        "scripts/research/check_forecast_heavy_model_inventory.py",
        "scripts/research/check_ped_model_assumption_inventory.py",
        "scripts/research/run_continual_adaptation_diagnostics.py",
    ],
)
def test_diagnostic_cli_help_does_not_require_optional_analytics(script: str):
    """Core CLI help should not import the optional analytics stack."""
    repo_root = Path(__file__).resolve().parents[2]
    runner = textwrap.dedent(
        """
        import builtins
        import runpy
        import sys

        script = sys.argv[1]
        blocked = ("matplotlib", "pandas", "scipy", "seaborn", "stable_baselines3", "torch")
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if any(name == package or name.startswith(package + ".") for package in blocked):
                raise ModuleNotFoundError(f"blocked optional dependency: {name}")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        sys.argv = [script, "--help"]
        runpy.run_path(script, run_name="__main__")
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [sys.executable, "-c", runner, script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
