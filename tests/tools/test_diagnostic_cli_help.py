"""Diagnostic CLI ``--help`` must expose the canonical ``uv run`` invocation.

Each developer-facing diagnostic CLI under ``scripts/tools/`` ends its
``--help`` with an ``Example:`` block showing the canonical
``uv run python scripts/tools/<name>.py <required-arg>`` command, so a new
contributor has a copy-pasteable invocation (issue #4908).

These checks run the real CLI ``--help`` (via ``python -m``) and assert the
full canonical example line is present, which also guards acceptance criterion
#2: the example must match the tool's actual required positional/flag.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: (module dotted path, exact canonical example line expected in ``--help``)
DIAGNOSTIC_CLIS: list[tuple[str, str]] = [
    (
        "scripts.tools.validate_scenario",
        "uv run python scripts/tools/validate_scenario.py <scenario_config.yaml>",
    ),
    (
        "scripts.tools.validate_socnav_map_batch",
        "uv run python scripts/tools/validate_socnav_map_batch.py [--batch-id <id>]",
    ),
    (
        "scripts.tools.validate_experiment_registry",
        "uv run python scripts/tools/validate_experiment_registry.py [experiments/registry.yaml]",
    ),
    (
        "scripts.tools.validate_report",
        "uv run python scripts/tools/validate_report.py --report-dir <dir>",
    ),
    (
        "scripts.tools.check_artifact_root",
        "uv run python scripts/tools/check_artifact_root.py",
    ),
    (
        "scripts.tools.preflight_scenario_perturbations",
        "uv run python scripts/tools/preflight_scenario_perturbations.py <manifest.yaml>",
    ),
    (
        "scripts.tools.preflight_adversarial_package_b",
        "uv run python scripts/tools/preflight_adversarial_package_b.py [--manifest <path>]",
    ),
]


def _run_help(module: str) -> subprocess.CompletedProcess[str]:
    """Run ``<python> -m <module> --help`` and capture stdout/stderr."""
    return subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("module, example_line", DIAGNOSTIC_CLIS)
def test_diagnostic_cli_help_exposes_canonical_invocation(module: str, example_line: str) -> None:
    """Each diagnostic CLI ``--help`` ends with the canonical ``uv run`` example."""
    result = _run_help(module)
    assert result.returncode == 0, f"{module} --help failed:\n{result.stderr}"
    assert "Example:" in result.stdout, f"{module} --help has no Example: block"
    assert example_line in result.stdout, (
        f"{module} --help missing canonical line {example_line!r}; got:\n{result.stdout}"
    )
