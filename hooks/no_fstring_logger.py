#!/usr/bin/env python3
"""Pre-commit guard that flags f-string interpolation inside loguru ``logger.*()`` calls.

Why this hook exists
--------------------
``ruff`` rule G004 (``logging-f-string``) only fires for the standard-library
``logging`` module. This repository uses ``loguru`` (``from loguru import
logger``), which ruff cannot see, so f-string logging regressions slip past the
default lint stack. This hook provides the loguru-equivalent guard by
AST-scanning ``logger.<method>(f"...")`` calls and rejecting f-strings as the
message argument. Structured style is the project idiom::

    logger.info("Loaded {n} SVG map(s) from {dir}", n=len(out), dir=str(dir_path))

See ``robot_sf/nav/svg_map_parser.py`` for the canonical pattern.

Ratchet / allowlist
-------------------
Issue #6468 migrated only the hot-path modules under ``robot_sf/sim/`` and
``robot_sf/gym_env/`` (7 call sites). The remaining ~228 f-string sites live in
the grandfathered files listed in ``hooks/no_fstring_logger_allowlist.txt``. The
guard therefore passes today and prevents NEW f-string regressions in any
non-allowlisted file (notably the migrated hot paths). Shrinking the allowlist
is explicit follow-up scope, not this hook's job: once a file is migrated,
delete its line from the allowlist and the guard will start enforcing it.

Usage
-----
pre-commit passes repository-root-relative filenames::

    uv run python hooks/no_fstring_logger.py <file> [<file> ...]

With no filename arguments the hook scans ``./robot_sf`` (useful for manual
full-repo runs and ``pre-commit run --all-files``-style checks from the repo
root).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import NamedTuple

# loguru logger methods plus stdlib-style aliases. ``log`` takes the level as its
# first positional argument, so the message is the second argument for it.
LOGGER_METHODS: frozenset[str] = frozenset(
    {
        "trace",
        "debug",
        "info",
        "success",
        "warning",
        "warn",
        "error",
        "critical",
        "exception",
        "log",
    }
)

ALLOWLIST_FILENAME = "no_fstring_logger_allowlist.txt"


class Violation(NamedTuple):
    """A single f-string logger call detected in a source file."""

    lineno: int
    method: str
    preview: str


class HookError(Exception):
    """Raised when a file cannot be parsed or the allowlist is unreadable."""


def _logger_chain_root_is_logger(func: ast.Attribute) -> bool:
    """Return True if ``func`` is a logger method reached through a chain rooted at ``logger``.

    Handles direct calls (``logger.info(...)``), attribute receivers
    (``self.logger.info(...)``), and loguru chain calls
    (``logger.opt(...).info(...)`` / ``logger.bind(...).warning(...)``) by
    unwrapping attribute/call nodes until a terminal ``logger`` name is reached.
    """

    if func.attr not in LOGGER_METHODS:
        return False
    node: ast.expr = func.value
    while True:
        if isinstance(node, ast.Name):
            return node.id == "logger"
        if isinstance(node, ast.Attribute):
            if node.attr == "logger":  # e.g. self.logger
                return True
            node = node.value
            continue
        if isinstance(node, ast.Call):
            node = node.func
            continue
        return False


def find_fstring_logger_violations(source: str, filename: str = "<unknown>") -> list[Violation]:
    """Return every ``logger.<method>(f"...")`` violation in ``source``.

    The message argument is ``args[0]`` for all logger methods except ``log``,
    which takes the level as its first positional argument (so the message is
    ``args[1]``). Only an f-string (``ast.JoinedStr``) message is a violation;
    literal strings, ``%``-style, and ``{}``-style messages are allowed.
    """

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:  # pragma: no cover - defensive parse failure
        raise HookError(f"{filename}: could not parse Python source: {exc}") from exc

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if not _logger_chain_root_is_logger(func):
            continue
        message_index = 1 if func.attr == "log" else 0
        if len(node.args) <= message_index:
            continue
        message_arg = node.args[message_index]
        if isinstance(message_arg, ast.JoinedStr):
            preview = ast.unparse(message_arg)
            violations.append(Violation(node.lineno, func.attr, preview))
    return violations


def _load_allowlist(path: Path) -> set[str]:
    """Load grandfathered file paths, returning both repo-relative and absolute posix keys."""
    if not path.is_file():
        raise HookError(f"allowlist not found: {path}")
    entries: set[str] = set()
    cwd = Path.cwd()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(Path(line).as_posix())
        try:
            entries.add((cwd / line).resolve().as_posix())
        except OSError:  # pragma: no cover - pathological filesystem
            entries.add((cwd / line).as_posix())
    return entries


def _file_keys(path: Path) -> set[str]:
    """Return the set of posix path strings that identify ``path`` for allowlist matching."""
    keys: set[str] = {Path(path).as_posix()}
    try:
        keys.add(path.resolve().as_posix())
    except OSError:  # pragma: no cover - pathological filesystem
        pass
    return keys


def _discover_robot_sf_files(root: Path) -> list[Path]:
    robot_sf = root / "robot_sf"
    if not robot_sf.is_dir():
        return []
    return sorted(robot_sf.rglob("*.py"))


def main(argv: list[str] | None = None) -> int:
    """Scan files for f-string logger calls; fail on any non-allowlisted violation."""
    default_allowlist = Path(__file__).resolve().parent / ALLOWLIST_FILENAME
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("filenames", nargs="*", help="files to check (defaults to ./robot_sf)")
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=default_allowlist,
        help=f"path to the grandfathered allowlist (default: {default_allowlist.name})",
    )
    args = parser.parse_args(argv)

    try:
        allowlist = _load_allowlist(args.allowlist)
    except HookError as exc:
        sys.stderr.write(f"no-fstring-logger hook: {exc}\n")
        return 1

    cwd = Path.cwd()
    files = [Path(f) for f in args.filenames] or _discover_robot_sf_files(cwd)

    failure = False
    for path in files:
        if not path.is_file():
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            sys.stderr.write(f"no-fstring-logger hook: cannot read {path}: {exc}\n")
            failure = True
            continue
        try:
            violations = find_fstring_logger_violations(source, str(path))
        except HookError as exc:
            sys.stderr.write(f"{exc}\n")
            failure = True
            continue
        if not violations:
            continue
        if _file_keys(path) & allowlist:
            continue  # grandfathered: follow-up migration scope
        for violation in violations:
            sys.stderr.write(
                f"{path}:{violation.lineno}: logger.{violation.method}({violation.preview}) "
                "uses an f-string; use structured {key} + kwargs style "
                "(see robot_sf/nav/svg_map_parser.py). ruff G004 cannot see loguru.\n"
            )
        failure = True

    if failure:
        sys.stderr.write(
            "\nMigrate with a named placeholder plus keyword argument, e.g.\n"
            '  logger.info(f"loaded {n}")  ->  logger.info("loaded {n}", n=n)\n'
            "Once a file is fully migrated, remove it from "
            "hooks/no_fstring_logger_allowlist.txt so the guard enforces it.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
