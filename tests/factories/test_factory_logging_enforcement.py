"""T037: Logging enforcement (no stray print) test.

Scans `environment_factory.py` for un-commented `print(` usage outside docstrings/comments.
Allows occurrences inside strings or comments. Fails on raw code-level print calls.
"""

from __future__ import annotations

from pathlib import Path

FACTORY_PATH = Path("robot_sf/gym_env/environment_factory.py")


def test_no_raw_print_statements():
    """TODO docstring. Document this function."""
    text = FACTORY_PATH.read_text(encoding="utf-8")
    # Naive scan: split lines and ignore those starting with comment or inside triple quotes state
    in_doc = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            # Toggle docstring block state (handles multi-line docstrings)
            in_doc = not in_doc
            continue
        if in_doc:
            continue
        if stripped.startswith("#"):
            continue
        # Detect raw print(
        if "print(" in stripped:
            raise AssertionError(f"Raw print() usage disallowed: {line}")
