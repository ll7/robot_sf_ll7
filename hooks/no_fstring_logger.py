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
Issue #6468 migrated nine call sites in three hot-path modules. Remaining calls
are grandfathered as stable per-call identities in
``hooks/no_fstring_logger_allowlist.txt``. A whole-file exemption would allow
new violations in legacy files, so the guard compares each current violation
against that exact baseline and rejects additions. Removing old calls is always
allowed; regenerate the deterministic baseline after a reviewed migration.

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
from collections import Counter
from hashlib import sha256
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
LOGGER_CHAIN_METHODS: frozenset[str] = frozenset({"bind", "opt", "patch"})

ALLOWLIST_FILENAME = "no_fstring_logger_allowlist.txt"
FINGERPRINT_LENGTH = 16


class Violation(NamedTuple):
    """A single f-string logger call detected in a source file."""

    lineno: int
    method: str
    preview: str
    scope: str
    fingerprint: str


class AllowlistKey(NamedTuple):
    """Stable identity for one grandfathered violation."""

    path: str
    scope: str
    method: str
    fingerprint: str


class HookError(Exception):
    """Raised when a file cannot be parsed or the allowlist is unreadable."""


def _matching_alias_names(aliases: list[ast.alias], target: str) -> set[str]:
    """Return local names for imports whose source name equals ``target``."""
    return {alias.asname or alias.name for alias in aliases if alias.name == target}


def _collect_import_bindings(tree: ast.AST) -> tuple[set[str], set[str], set[str]]:
    """Collect logger, Loguru-module, and get_logger import aliases."""
    logger_names: set[str] = set()
    loguru_modules: set[str] = set()
    get_logger_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            loguru_modules.update(_matching_alias_names(node.names, "loguru"))
        elif isinstance(node, ast.ImportFrom) and node.module == "loguru":
            logger_names.update(_matching_alias_names(node.names, "logger"))
        elif isinstance(node, ast.ImportFrom) and node.module == "robot_sf.common.logging":
            get_logger_names.update(_matching_alias_names(node.names, "get_logger"))
    return logger_names, loguru_modules, get_logger_names


def _assignment_is_loguru_logger(
    value: ast.expr,
    *,
    loguru_modules: set[str],
    get_logger_names: set[str],
) -> bool:
    """Return whether an assignment value is a supported Loguru logger factory."""
    if (
        isinstance(value, ast.Attribute)
        and value.attr == "logger"
        and isinstance(value.value, ast.Name)
    ):
        return value.value.id in loguru_modules
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in get_logger_names
    )


def _find_logger_bindings(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Return supported Loguru logger names and imported Loguru module aliases."""
    logger_names, loguru_modules, get_logger_names = _collect_import_bindings(tree)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None or not _assignment_is_loguru_logger(
            value,
            loguru_modules=loguru_modules,
            get_logger_names=get_logger_names,
        ):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                logger_names.add(target.id)

    return logger_names, loguru_modules


def _is_logger_receiver(
    node: ast.expr,
    *,
    logger_names: set[str],
    loguru_modules: set[str],
) -> bool:
    """Return whether ``node`` is a supported Loguru logger receiver."""
    if isinstance(node, ast.Name):
        return node.id in logger_names
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "logger"
        and isinstance(node.value, ast.Name)
    ):
        return node.value.id in loguru_modules
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in LOGGER_CHAIN_METHODS
    ):
        return _is_logger_receiver(
            node.func.value,
            logger_names=logger_names,
            loguru_modules=loguru_modules,
        )
    return False


def _logger_chain_root_is_logger(
    func: ast.Attribute,
    *,
    logger_names: set[str],
    loguru_modules: set[str],
) -> bool:
    """Return True if ``func`` is a logger method reached through a chain rooted at ``logger``.

    Handles direct calls and Loguru ``opt``/``bind``/``patch`` chains. It
    deliberately rejects unrelated ``self.logger`` attributes and arbitrary
    call chains even when an attribute happens to be named ``logger``.
    """

    return func.attr in LOGGER_METHODS and _is_logger_receiver(
        func.value,
        logger_names=logger_names,
        loguru_modules=loguru_modules,
    )


def _message_argument(node: ast.Call, method: str) -> ast.expr | None:
    """Return the positional Loguru message expression for ``node``."""
    message_index = 1 if method == "log" else 0
    return node.args[message_index] if len(node.args) > message_index else None


def _serialize_fstring(node: ast.JoinedStr) -> str:
    """Return a Python-version-independent canonical serialization of an f-string.

    ``ast.dump`` and ``ast.unparse`` are not stable across interpreter versions
    for f-strings: Python 3.13 omits empty ``Call.keywords`` in ``ast.dump``
    and selects a different outer quote in ``ast.unparse`` when an interpolation
    contains a literal quote (e.g. ``f"x {'; '.join(a['b'])}"``). A fingerprint
    derived from either representation therefore drifted between the CI
    interpreter (3.12) and developer interpreters (3.13), breaking the ratchet
    baseline (issue #6575). This serializer is stable because it fingerprints
    the f-string's semantic parts: each literal segment's string *value* and
    each interpolation's expression text, conversion, and recursive format spec.
    """
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(f"L\x00{value.value}")
        elif isinstance(value, ast.FormattedValue):
            expression = ast.unparse(value.value)
            conversion = str(value.conversion)
            format_spec = (
                _serialize_fstring(value.format_spec)
                if isinstance(value.format_spec, ast.JoinedStr)
                else ""
            )
            parts.append(f"F\x00{expression}\x00{conversion}\x00{format_spec}")
        elif isinstance(value, ast.JoinedStr):
            # A nested f-string used directly as a literal segment (PEP 701).
            parts.append(f"L\x00{_serialize_fstring(value)}")
        else:  # pragma: no cover - defensive for unexpected JoinedStr children
            parts.append(f"X\x00{ast.unparse(value)}")
    return "\x00".join(parts)


def _message_fingerprint(message: ast.JoinedStr) -> str:
    """Return a location- and Python-version-independent fingerprint for one f-string."""
    normalized = _serialize_fstring(message)
    return sha256(normalized.encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


class _ViolationVisitor(ast.NodeVisitor):
    """Collect violations while preserving their class/function scope."""

    def __init__(self, logger_names: set[str], loguru_modules: set[str]) -> None:
        self.logger_names = logger_names
        self.loguru_modules = loguru_modules
        self.scope: list[str] = []
        self.violations: list[Violation] = []

    def _visit_scope(self, node: ast.AST, name: str) -> None:
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and _logger_chain_root_is_logger(
            func,
            logger_names=self.logger_names,
            loguru_modules=self.loguru_modules,
        ):
            message_arg = _message_argument(node, func.attr)
            if isinstance(message_arg, ast.JoinedStr):
                self.violations.append(
                    Violation(
                        lineno=node.lineno,
                        method=func.attr,
                        preview=ast.unparse(message_arg),
                        scope=".".join(self.scope) or "<module>",
                        fingerprint=_message_fingerprint(message_arg),
                    )
                )
        self.generic_visit(node)


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

    logger_names, loguru_modules = _find_logger_bindings(tree)
    visitor = _ViolationVisitor(logger_names, loguru_modules)
    visitor.visit(tree)
    return visitor.violations


def _load_allowlist(path: Path) -> Counter[AllowlistKey]:
    """Load stable per-call ratchet entries from a tab-separated baseline."""
    if not path.is_file():
        raise HookError(f"allowlist not found: {path}")
    entries: Counter[AllowlistKey] = Counter()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 5:
            raise HookError(
                f"{path}:{lineno}: expected path<TAB>scope<TAB>method<TAB>fingerprint<TAB>count"
            )
        file_path, scope, method, fingerprint, raw_count = fields
        if method not in LOGGER_METHODS:
            raise HookError(f"{path}:{lineno}: unsupported logger method {method!r}")
        if len(fingerprint) != FINGERPRINT_LENGTH or any(
            char not in "0123456789abcdef" for char in fingerprint
        ):
            raise HookError(f"{path}:{lineno}: invalid fingerprint {fingerprint!r}")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise HookError(f"{path}:{lineno}: invalid count {raw_count!r}") from exc
        if count < 1:
            raise HookError(f"{path}:{lineno}: count must be positive")
        key = AllowlistKey(Path(file_path).as_posix(), scope, method, fingerprint)
        if key in entries:
            raise HookError(f"{path}:{lineno}: duplicate baseline identity {key!r}")
        entries[key] = count
    return entries


def _repo_relative_path(path: Path, root: Path) -> str:
    """Return a stable repository-relative path when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except OSError:  # pragma: no cover - pathological filesystem
        return path.as_posix()
    except ValueError:
        return path.as_posix()


def _allowlist_key(path: Path, violation: Violation, root: Path) -> AllowlistKey:
    return AllowlistKey(
        _repo_relative_path(path, root),
        violation.scope,
        violation.method,
        violation.fingerprint,
    )


def _discover_robot_sf_files(root: Path) -> list[Path]:
    robot_sf = root / "robot_sf"
    if not robot_sf.is_dir():
        return []
    return sorted(robot_sf.rglob("*.py"))


def _scan_file(path: Path) -> list[Violation]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HookError(f"cannot read {path}: {exc}") from exc
    return find_fstring_logger_violations(source, str(path))


def _generate_allowlist(files: list[Path], root: Path) -> int:
    """Print deterministic baseline rows for ``files``."""
    generated: Counter[AllowlistKey] = Counter()
    try:
        for path in files:
            if path.is_file():
                generated.update(_allowlist_key(path, item, root) for item in _scan_file(path))
    except HookError as exc:
        sys.stderr.write(f"no-fstring-logger hook: {exc}\n")
        return 1
    for key in sorted(generated):
        sys.stdout.write("\t".join((*key, str(generated[key]))) + "\n")
    return 0


def _check_files(
    files: list[Path],
    *,
    root: Path,
    remaining_allowlist: Counter[AllowlistKey],
) -> bool:
    """Return whether any scanned file has an uncovered violation or read error."""
    failure = False
    for path in files:
        if not path.is_file():
            continue
        try:
            violations = _scan_file(path)
        except HookError as exc:
            sys.stderr.write(f"no-fstring-logger hook: {exc}\n")
            failure = True
            continue
        for violation in violations:
            key = _allowlist_key(path, violation, root)
            if remaining_allowlist[key] > 0:
                remaining_allowlist[key] -= 1
                continue
            sys.stderr.write(
                f"{path}:{violation.lineno}: logger.{violation.method}({violation.preview}) "
                "uses an f-string; use structured {key} + kwargs style "
                "(see robot_sf/nav/svg_map_parser.py). ruff G004 cannot see loguru.\n"
            )
            failure = True
    return failure


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
    parser.add_argument(
        "--generate-allowlist",
        action="store_true",
        help="print a deterministic baseline for the selected files and exit",
    )
    args = parser.parse_args(argv)

    cwd = Path.cwd()
    files = [Path(f) for f in args.filenames] or _discover_robot_sf_files(cwd)

    if args.generate_allowlist:
        return _generate_allowlist(files, cwd)

    try:
        remaining_allowlist = _load_allowlist(args.allowlist)
    except HookError as exc:
        sys.stderr.write(f"no-fstring-logger hook: {exc}\n")
        return 1

    failure = _check_files(files, root=cwd, remaining_allowlist=remaining_allowlist)

    if failure:
        sys.stderr.write(
            "\nMigrate with a named placeholder plus keyword argument, e.g.\n"
            '  logger.info(f"loaded {n}")  ->  logger.info("loaded {n}", n=n)\n'
            "After removing grandfathered calls, regenerate and review "
            "hooks/no_fstring_logger_allowlist.txt to shrink the ratchet.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
