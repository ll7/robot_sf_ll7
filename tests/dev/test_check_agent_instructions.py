"""Tests for the combined agent-instruction check entry point.

The wrapper is the single convenience entry point for instruction-contract changes; these fixtures
guard its executability, strict shell mode, and the canonical commands it composes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "dev" / "check_agent_instructions.sh"

CANONICAL_COMMANDS = (
    "scripts/dev/check_instruction_references.py",
    "scripts/tools/sync_ai_config.py --check",
    "tests/dev/test_instruction_references.py",
    "tests/dev/test_instruction_precedence.py",
    "tests/dev/test_task_scope_manifest.py",
    "tests/dev/test_maintainer_values.py",
    "tests/dev/test_delivery_and_friction.py",
    "tests/dev/test_instruction_task_fixtures.py",
    "tests/dev/test_compact_boot_router.py",
)


def test_wrapper_is_executable_and_strict() -> None:
    """The wrapper exists, is executable, and fails fast on any step."""
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & 0o111
    text = SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text


def test_wrapper_composes_the_canonical_checks() -> None:
    """Every contract check the wrapper claims to run is present."""
    text = SCRIPT.read_text(encoding="utf-8")
    for command in CANONICAL_COMMANDS:
        assert command in text, command


def test_wrapper_forwards_json_flag() -> None:
    """The optional --json flag reaches the instruction-reference checker."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "checker_args+=(--json)" in text
    assert '"${checker_args[@]}"' in text


def test_wrapper_shell_syntax_is_valid() -> None:
    """The shell script parses without executing it."""
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
