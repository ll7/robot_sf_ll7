"""One-shot migration script: re-parent remaining ad-hoc exception families onto RobotSfError.

Run from repo root in the issue-4993 worktree:
    uv run python scripts/dev/migrate_exceptions_4993.py

Transformation rules:
- class Foo(ValueError)  -> class Foo(RobotSfError, ValueError)
- class Foo(RuntimeError) -> class Foo(RobotSfError, RuntimeError)
- class Foo(Exception)   -> class Foo(RobotSfError)
- class Foo(SomeCustomError) -> no change (covered transitively once parent migrated)

Already-migrated files are skipped automatically.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SKIP = {
    ROOT / "robot_sf/errors.py",
    ROOT / "robot_sf/benchmark/errors.py",
    ROOT / "robot_sf/data/external/recording_shape_contract.py",
    ROOT / "robot_sf/data/external/atc.py",
    ROOT / "robot_sf/data/external/eth_ucy.py",
    ROOT / "robot_sf/data/external/ind.py",
    ROOT / "robot_sf/data/external/socnavbench_eth.py",
    ROOT / "robot_sf/data/external/crowdbot.py",
    ROOT / "robot_sf/data/external/scand.py",
    ROOT / "robot_sf/research/exceptions.py",
}

_CLASS_RE = re.compile(
    r"^(class\s+\w+)\((ValueError|RuntimeError|Exception)\)\s*:",
    re.MULTILINE,
)

_IMPORT_LINE = "from robot_sf.errors import RobotSfError"


def _needs_import(src: str) -> bool:
    return _IMPORT_LINE not in src


def _has_target_class(src: str) -> bool:
    return bool(_CLASS_RE.search(src))


def _add_import(src: str) -> str:
    """Insert the RobotSfError import safely, after all complete import blocks.

    Handles multi-line imports by tracking open parentheses depth so we never
    insert into the middle of a 'from X import (\n  ...\n)' block.
    """
    lines = src.splitlines(keepends=True)
    insert_after = -1
    in_docstring = False
    docstring_char = None
    paren_depth = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track module-level triple-quoted docstrings
        if not in_docstring and paren_depth == 0 and (
            stripped.startswith('"""') or stripped.startswith("'''")
        ):
            docstring_char = stripped[:3]
            count = stripped.count(docstring_char)
            if count >= 2 and len(stripped) > 3:
                insert_after = i  # single-line docstring
            else:
                in_docstring = True
            continue
        if in_docstring:
            if docstring_char and docstring_char in line:
                in_docstring = False
                insert_after = i
            continue

        # Track parenthesis depth across multi-line imports
        paren_depth += line.count("(") - line.count(")")

        if paren_depth > 0:
            # Inside a multi-line statement — do not mark as a completed import
            continue

        # At top level (paren_depth == 0 after the line)
        if stripped.startswith("from __future__"):
            insert_after = i
        elif stripped.startswith("from ") or stripped.startswith("import "):
            insert_after = i
        elif stripped.startswith("class ") or stripped.startswith("def ") or (
            stripped and not stripped.startswith("#") and not stripped.startswith("@")
            and not stripped.startswith("__")
        ):
            # First non-import, non-decorator, non-dunder-assignment line
            if insert_after >= 0:
                break

    new_lines = list(lines)
    if insert_after >= 0:
        new_lines.insert(insert_after + 1, _IMPORT_LINE + "\n")
    else:
        new_lines.insert(0, _IMPORT_LINE + "\n")

    return "".join(new_lines)


def _rewrite_classes(src: str) -> str:
    _builtins = {
        "ValueError": "RobotSfError, ValueError",
        "RuntimeError": "RobotSfError, RuntimeError",
        "Exception": "RobotSfError",
    }

    def _replacement(m: re.Match) -> str:
        prefix = m.group(1)
        parent = m.group(2)
        new_parent = _builtins[parent]
        return f"{prefix}({new_parent}):"

    return _CLASS_RE.sub(_replacement, src)


def process_file(path: Path) -> bool:
    src = path.read_text()
    if not _has_target_class(src):
        return False

    new_src = _rewrite_classes(src)
    if _needs_import(new_src):
        new_src = _add_import(new_src)

    if new_src == src:
        return False

    path.write_text(new_src)
    return True


def main() -> None:
    target_files = sorted(
        p
        for p in (ROOT / "robot_sf").rglob("*.py")
        if "__pycache__" not in str(p) and p not in SKIP
    )

    changed: list[Path] = []
    for path in target_files:
        if process_file(path):
            changed.append(path)
            print(f"  migrated: {path.relative_to(ROOT)}")

    print(f"\nTotal files modified: {len(changed)}")
    if not changed:
        print("Nothing to do — all exceptions already migrated or no targets found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
