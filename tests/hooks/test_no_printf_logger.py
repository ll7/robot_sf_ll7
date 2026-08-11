"""Contract tests for the no-printf-logger pre-commit hook (issue #6837)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HOOK_PATH = Path(__file__).resolve().parents[2] / "hooks" / "no_printf_logger.py"
_SPEC = importlib.util.spec_from_file_location("no_printf_logger", _HOOK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HOOK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HOOK)

find_violations = _HOOK.find_printf_logger_violations


def test_flags_printf_placeholder_with_escaped_literal_braces() -> None:
    """Escaped braces must not hide a printf placeholder from the AST guard."""
    source = "from loguru import logger\nlogger.warning('payload={{}} failed: %s', exc)\n"

    violations = find_violations(source, "<escaped-braces>")

    assert len(violations) == 1
    assert violations[0][1] == "warning"
    assert "%s" in violations[0][2]
