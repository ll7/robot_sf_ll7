#!/usr/bin/env python3
"""Pre-commit guard that flags loguru ``logger.*()`` calls using printf ``%``-placeholders.

Why this hook exists
--------------------
Loguru interpolates with ``str.format`` brace style, never ``%``-style. When a call
passes positional arguments against a ``%``-placeholder template, e.g.::

    logger.warning("Failed to export overlay: %s", exc)

loguru finds no ``{}`` field, **silently discards every argument**, and emits the
template literally. The exception text and counters are lost while the call looks
correct and passes tests. This destroyed diagnostic payloads across the repo
(issue #6837). ``ruff`` rule G002 (``logging-statement-uses-percent-format``) only
fires for the standard-library ``logging`` module, which it cannot see for loguru,
and where ``%``-style is in fact correct. This hook provides the loguru-equivalent
guard.

It AST-scans ``logger.<method>(...)`` calls and rejects any whose message template
is a string constant that carries a printf-style placeholder (``%s``/``%d``/``%.2f``
...) **and** passes positional arguments. This also catches mixed templates and
escaped literal braces, which otherwise still discard the printf arguments. Stdlib
``logging`` modules are not affected because they never bind a loguru logger.

Usage
-----
pre-commit runs the hook with ``pass_filenames: false`` so it scans the whole
in-scope tree once, mirroring ``hooks/no_fstring_logger.py``::

    uv run python hooks/no_printf_logger.py            # scan robot_sf, scripts, examples
    uv run python hooks/no_printf_logger.py <file> ... # scan explicit files (manual use)
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

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

# A printf-style placeholder: % [(name)] [flags] [width] [.prec] conversion.
# The flag set deliberately excludes the space flag so common prose like
# "5% of episodes" / "100% complete" is not mistaken for a format spec; every
# real placeholder observed in the audit (``%s``/``%d``/``%r``/``%.2f``/``%02d``)
# still matches. The bare ``%%`` literal is intentionally excluded so it cannot
# masquerade as a placeholder.
PRINTF_PLACEHOLDER_RE = re.compile(r"%(\([^)]*\))?[#0\-+]*\d*(\.\d+)?[diouxXeEfFgGcrsa]")


class HookError(Exception):
    """Raised when a file cannot be parsed."""


def _matching_alias_names(aliases: list[ast.alias], target: str) -> set[str]:
    """Return local names for imports whose source name equals ``target``."""
    return {alias.asname or alias.name for alias in aliases if alias.name == target}


def _collect_import_bindings(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Collect ``from loguru import logger`` names and ``import loguru`` module aliases."""
    logger_names: set[str] = set()
    loguru_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            loguru_modules.update(_matching_alias_names(node.names, "loguru"))
        elif isinstance(node, ast.ImportFrom) and node.module == "loguru":
            logger_names.update(_matching_alias_names(node.names, "logger"))
    return logger_names, loguru_modules


def _import_module_call_returns_loguru(value: ast.expr) -> bool:
    """Return True for ``importlib.import_module('loguru')`` (single or double quoted)."""
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "import_module"
        and bool(value.args)
        and isinstance(value.args[0], ast.Constant)
        and value.args[0].value == "loguru"
    )


def _assignment_is_loguru_logger(value: ast.expr, loguru_modules: set[str]) -> bool:
    """Return whether an assignment value binds a supported Loguru logger.

    Covers the two binding kinds named in issue #6837:

    * ``logger = importlib.import_module('loguru').logger`` (dynamic binding)
    * ``logger = <loguru_module_alias>.logger`` (``import loguru`` then ``loguru.logger``)
    """
    if not (isinstance(value, ast.Attribute) and value.attr == "logger"):
        return False
    inner = value.value
    if _import_module_call_returns_loguru(inner):
        return True
    return isinstance(inner, ast.Name) and inner.id in loguru_modules


def _find_logger_bindings(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Return supported Loguru logger names and imported Loguru module aliases.

    Covers module-level ``from loguru import logger`` and dynamic
    ``logger = importlib.import_module('loguru').logger`` bindings, the two
    binding kinds the issue audit enumerates.
    """
    logger_names, loguru_modules = _collect_import_bindings(tree)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None or not _assignment_is_loguru_logger(value, loguru_modules):
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


def _is_loguru_logger_call(
    func: ast.Attribute,
    *,
    logger_names: set[str],
    loguru_modules: set[str],
) -> bool:
    """Return True if ``func`` is a logger method reached through a chain rooted at ``logger``."""
    return func.attr in LOGGER_METHODS and _is_logger_receiver(
        func.value,
        logger_names=logger_names,
        loguru_modules=loguru_modules,
    )


def _message_argument(node: ast.Call, method: str) -> ast.expr | None:
    """Return the positional Loguru message expression for ``node``."""
    message_index = 1 if method == "log" else 0
    return node.args[message_index] if len(node.args) > message_index else None


class _ViolationVisitor(ast.NodeVisitor):
    """Collect printf-placeholder violations in a single source file."""

    def __init__(self, logger_names: set[str], loguru_modules: set[str]) -> None:
        self.logger_names = logger_names
        self.loguru_modules = loguru_modules
        self.violations: list[tuple[int, str, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and _is_loguru_logger_call(
            func,
            logger_names=self.logger_names,
            loguru_modules=self.loguru_modules,
        ):
            self._check_call(node, func.attr)
        self.generic_visit(node)

    def _check_call(self, node: ast.Call, method: str) -> None:
        message_arg = _message_argument(node, method)
        if not (isinstance(message_arg, ast.Constant) and isinstance(message_arg.value, str)):
            return
        template = message_arg.value
        if not PRINTF_PLACEHOLDER_RE.search(template):
            return
        # ``log`` carries (level, message, *args); every other method uses
        # (message, *args). Only flag when positional interpolation args are
        # present -- otherwise the literal ``%``-string is harmless.
        first_arg_index = 1 if method == "log" else 0
        positional_args = node.args[first_arg_index + 1 :]
        if not positional_args:
            return
        preview = template.replace("\n", "\\n")
        self.violations.append((node.lineno, method, preview))


def find_printf_logger_violations(
    source: str, filename: str = "<unknown>"
) -> list[tuple[int, str, str]]:
    """Return every ``logger.<method>("%...", *args)`` violation in ``source``."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:  # pragma: no cover - defensive parse failure
        raise HookError(f"{filename}: could not parse Python source: {exc}") from exc
    logger_names, loguru_modules = _find_logger_bindings(tree)
    visitor = _ViolationVisitor(logger_names, loguru_modules)
    visitor.visit(tree)
    return visitor.violations


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HookError(f"cannot read {path}: {exc}") from exc
    return find_printf_logger_violations(source, str(path))


def _discover_in_scope_files(root: Path) -> list[Path]:
    """Return all ``.py`` files under the in-scope directories (robot_sf, scripts, examples)."""
    files: list[Path] = []
    for sub in ("robot_sf", "scripts", "examples"):
        base = root / sub
        if base.is_dir():
            files.extend(sorted(base.rglob("*.py")))
    return files


def main(argv: list[str] | None = None) -> int:
    """Scan files for loguru printf-placeholder calls; fail on any violation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "filenames",
        nargs="*",
        help="files to check (defaults to robot_sf, scripts, examples under cwd)",
    )
    args = parser.parse_args(argv)

    cwd = Path.cwd()
    files = [Path(f) for f in args.filenames] or _discover_in_scope_files(cwd)

    failure = False
    for path in files:
        if not path.is_file():
            continue
        try:
            violations = _scan_file(path)
        except HookError as exc:
            sys.stderr.write(f"no-printf-logger hook: {exc}\n")
            failure = True
            continue
        for lineno, method, preview in violations:
            sys.stderr.write(
                f"{path}:{lineno}: logger.{method}({preview!r}) uses a printf %-placeholder "
                "with positional args; loguru silently discards them. Use brace {} style, e.g.\n"
                '  logger.warning("Failed: {}", exc)\n'
                "Inside except handlers prefer logger.opt(exception=True) so the traceback "
                "survives (see robot_sf/baselines/ppo.py). ruff G002 cannot see loguru.\n"
            )
            failure = True
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
