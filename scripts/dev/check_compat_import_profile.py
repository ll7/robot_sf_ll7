"""Verify the compatibility lane's test closure fits its slim sync profile.

The ``compat-matrix`` CI lane syncs ``base + viz + maps`` extras instead of
``--all-extras`` (issue #9355: full-extra downloads blew the 1,200s setup budget
before tests ran). This probe fails closed when a compatibility test module
imports a third-party package outside that profile, so a missing dependency can
never silently shrink lane coverage: the lane fails here with the exact module
and the profile that must change.

Usage:
    uv run python scripts/dev/check_compat_import_profile.py [--root DIR]
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

COMPAT_TEST_DIRS = (
    "tests/common",
    "tests/contract",
    "tests/factories",
    "tests/gym_env",
    "tests/maps",
    "tests/nav",
    "tests/ped_npc",
    "tests/render",
    "tests/scenarios",
    "tests/sensor",
    "tests/sim",
    "tests/unit",
)

# Third-party top-level imports provably covered by the compat sync profile
# (base dependencies + viz + maps extras + dev group). Evidence: slim-venv
# collection of the full lane plus runtime runs of the viz/maps-relevant dirs.
PROFILE_IMPORTS = frozenset(
    {
        # Base project dependencies.
        "numpy",
        "gymnasium",
        "numba",
        "svgelements",
        "loguru",
        "rich",
        "psutil",
        "jsonschema",
        "rfc8785",
        "shapely",
        "networkx",
        "pyvisgraph",
        "python_motion_planning",
        "tqdm",
        "yaml",
        "orjson",
        # Workspace member, always installed editable.
        "pysocialforce",
        # Viz extra (tests/render, matplotlib/pillow users).
        "pygame",
        "matplotlib",
        "PIL",
        # Maps extra (tests/unit/test_geojson_map_builder.py).
        "geopandas",
        "osmnx",
        "pyproj",
        # Transitive base closure (gymnasium/networkx/robot-sf requirement
        # chains, not direct extras). The runtime importability check below
        # guards against these drifting out of the closure after bumps.
        "scipy",
        "pandas",
        # Dev group (pytest runs the lane).
        "pytest",
        "pytest_cov",
    }
)

# importorskip modules accepted as explicit, reasoned skips (never silent
# coverage loss): the skipping tests name their reason and their dependency
# stays out of the lane profile by design.
ACCEPTED_SKIPS = {
    "playwright": "browser rendering needs Chromium, unavailable in CI runners",
    "sklearn": "optional-by-design annotation check, skips with explicit reason",
}

# In-repo namespaces that resolve from the checkout, not from the environment.
LOCAL_NAMESPACES = frozenset({"robot_sf", "tests", "examples", "hooks", "scripts"})

_BUCKETS = ("top_hard", "top_in_repo", "func_heavy", "skip_strings")


class _ProfileScanError(RuntimeError):
    """Raised when a selected source file cannot be inspected safely."""


def _new_buckets() -> dict[str, set[str]]:
    """Return empty import-classification buckets."""
    return {key: set() for key in _BUCKETS}


def _handler_catches_import_error(try_node: ast.Try) -> bool:
    """Return whether a try statement tolerates failed imports."""
    for handler in try_node.handlers:
        if handler.type is None:
            return True
        for node in ast.walk(handler.type):
            if isinstance(node, ast.Name) and node.id in {
                "ImportError",
                "ModuleNotFoundError",
                "Exception",
            }:
                return True
    return False


def _is_type_checking(node: ast.If) -> bool:
    """Return whether an if statement guards typing-only imports."""
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return (
        isinstance(test, ast.Attribute)
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
        and test.attr == "TYPE_CHECKING"
    )


def _names(node: ast.Import | ast.ImportFrom) -> list[str]:
    """Return dotted module names for one import statement."""
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if node.level:
        return []
    return [node.module] if node.module else []


def _record(names: list[str], buckets: dict[str, set[str]], *, collection_time: bool) -> None:
    """File names into hard/in-repo buckets (optional guarded attempts are dropped)."""
    for name in names:
        if name == "robot_sf" or name.startswith("robot_sf."):
            if collection_time:
                buckets["top_in_repo"].add(name)
        elif collection_time:
            buckets["top_hard"].add(name.split(".")[0])
        else:
            buckets["func_heavy"].add(name.split(".")[0])


def _visit_value(node: ast.AST, buckets: dict[str, set[str]]) -> None:
    """Collect importorskip module names from expression positions."""
    if isinstance(node, ast.Call):
        func = node.func
        attr = func.attr if isinstance(func, ast.Attribute) else ""
        if attr == "importorskip" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                buckets["skip_strings"].add(first.value.split(".")[0])
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.stmt):
            _visit_nested(child, buckets)
        else:
            _visit_value(child, buckets)


def _visit_nested(node: ast.stmt, buckets: dict[str, set[str]]) -> None:
    """Collect deferred (function-body) imports for selected test checks."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        _record(_names(node), buckets, collection_time=False)
        return
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.stmt):
            _visit_nested(child, buckets)
        else:
            _visit_value(child, buckets)


def _visit_branch(
    statements: list[ast.stmt],
    buckets: dict[str, set[str]],
    *,
    guarded: bool,
    top_level: bool,
) -> None:
    """Visit one statement list while preserving import execution context."""
    for sub in statements:
        if isinstance(sub, (ast.Import, ast.ImportFrom)):
            if not guarded:
                _record(_names(sub), buckets, collection_time=top_level)
        else:
            _visit_statement(sub, buckets, top_level=top_level, guarded=guarded)


def _visit_statement(
    node: ast.stmt, buckets: dict[str, set[str]], *, top_level: bool, guarded: bool
) -> None:
    """Classify one statement, tracking collection-time execution."""
    if isinstance(node, ast.If) and _is_type_checking(node):
        return
    if isinstance(node, ast.Try) and _handler_catches_import_error(node):
        # The import attempt itself is an explicitly tolerated optional path.
        # Its handler and cleanup still execute when the attempt fails, so
        # inspect those statements instead of dropping the complete try node.
        _visit_branch(node.body, buckets, guarded=True, top_level=top_level)
        for handler in node.handlers:
            _visit_branch(handler.body, buckets, guarded=guarded, top_level=top_level)
        for sub in node.orelse:
            _visit_statement(sub, buckets, top_level=top_level, guarded=guarded)
        for sub in node.finalbody:
            _visit_statement(sub, buckets, top_level=top_level, guarded=guarded)
        return
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        if not guarded:
            _record(_names(node), buckets, collection_time=top_level)
        return
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.stmt):
            _visit_statement(child, buckets, top_level=False, guarded=guarded)
        else:
            _visit_value(child, buckets)


def _scan_file(path: Path) -> dict[str, set[str]]:
    """Scan one file, separating collection-time imports from deferred ones.

    Returns:
        Buckets with ``top_hard`` (collection-time third-party imports),
        ``top_in_repo`` (collection-time in-repo modules), ``func_heavy``
        (deferred third-party imports), and ``skip_strings``.
    """
    import sys as _sys

    buckets = _new_buckets()
    try:
        source = path.read_bytes()
    except OSError as error:
        raise _ProfileScanError(f"{path}: cannot read source ({type(error).__name__})") from error
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeError, ValueError, RecursionError) as error:
        detail = getattr(error, "msg", None) or type(error).__name__
        detail = str(detail).replace("\n", " ")[:160]
        raise _ProfileScanError(f"{path}: cannot parse source ({detail})") from error
    try:
        for statement in tree.body:
            _visit_statement(statement, buckets, top_level=True, guarded=False)
        stdlib = set(_sys.stdlib_module_names)
        drop = stdlib | LOCAL_NAMESPACES | {"__future__", "typing", "typing_extensions"}
        for key in ("top_hard", "func_heavy", "skip_strings"):
            buckets[key] = {m for m in buckets[key] if m not in drop}
    except RecursionError as error:
        raise _ProfileScanError(f"{path}: cannot inspect source (recursion limit)") from error
    return buckets


def _scan_path(path: Path, errors: list[str]) -> dict[str, set[str]] | None:
    """Scan one selected path, retaining a bounded diagnostic on failure."""
    try:
        return _scan_file(path)
    except _ProfileScanError as error:
        errors.append(str(error))
        return None


def _resolve_module(root: Path, dotted: str) -> Path | None:
    """Map an in-repo dotted path to its source file, if present."""
    relative = Path(*dotted.split("."))
    for candidate in (
        root / (str(relative) + ".py"),
        root / relative / "__init__.py",
    ):
        if candidate.is_file():
            return candidate
    return None


def _check_violation(display: str, module: str, context: str) -> str:
    """Format one profile violation with its remedy."""
    return (
        f"{display} imports {module!r} at {context}: "
        "outside the compat profile (base+viz+maps); "
        "extend the profile or move the module out of the lane"
    )


def _selected_files(root: Path, dirname: str, errors: list[str]) -> list[Path]:
    """Return the selected test files, failing closed for missing roots."""
    target = root / dirname
    if dirname.endswith(".py"):
        if not target.is_file():
            errors.append(f"{dirname}: required compat profile file is missing")
            return []
        return [target]
    if not target.is_dir():
        errors.append(f"{dirname}: required compat profile directory is missing")
        return []
    try:
        return sorted(
            path
            for path in target.rglob("*.py")
            if path.name == "__init__.py" or path.name.startswith("test_")
        )
    except OSError as error:
        errors.append(f"{dirname}: cannot enumerate compat profile files ({type(error).__name__})")
        return []


def _check_deferred_test_imports(
    display: str,
    modules: set[str],
    errors: list[str],
    report: dict[str, list[str]],
) -> None:
    """Require function-body imports in selected tests to be profile-covered."""
    for module in sorted(modules):
        report.setdefault(module, []).append(display)
        if module not in PROFILE_IMPORTS:
            errors.append(_check_violation(display, module, "deferred test execution"))


def _collect_roots(root: Path, errors: list[str], report: dict[str, list[str]]) -> list[str]:
    """Scan test roots; return in-repo modules for transitive closure."""
    pending: list[str] = []
    for dirname in COMPAT_TEST_DIRS + ("tests/conftest.py",):
        files = _selected_files(root, dirname, errors)
        for path in files:
            buckets = _scan_path(path, errors)
            if buckets is None:
                continue
            display = str(path.relative_to(root))
            for module in sorted(buckets["top_hard"]):
                report.setdefault(module, []).append(display)
                if module not in PROFILE_IMPORTS:
                    errors.append(_check_violation(display, module, "collection time (test file)"))
            _check_deferred_test_imports(display, buckets["func_heavy"], errors, report)
            for module in sorted(buckets["skip_strings"]):
                if module in ACCEPTED_SKIPS or module in LOCAL_NAMESPACES:
                    continue
                report.setdefault(module, []).append(display)
                if module not in PROFILE_IMPORTS:
                    errors.append(
                        f"{display} silently skips on missing {module!r}: "
                        "outside the compat profile (base+viz+maps); "
                        "a skip here would shrink lane coverage without failing"
                    )
            pending.extend(sorted(buckets["top_in_repo"]))
    return pending


def _collect_closure(
    root: Path, pending: list[str], errors: list[str], report: dict[str, list[str]]
) -> None:
    """Follow collection-time in-repo imports from the selected test roots.

    Source-owner function bodies can expose optional runtime adapters that are
    outside this slim compatibility lane. Deferred imports in the selected test
    files are checked by ``_collect_roots``; this closure only guarantees the
    imports executed while those tests are collected.
    """
    seen_files: set[Path] = set()
    while pending:
        dotted = pending.pop()
        location = _resolve_module(root, dotted)
        if location is None or location in seen_files:
            continue
        seen_files.add(location)
        try:
            display = str(location.relative_to(root))
        except ValueError:
            display = str(location)
        buckets = _scan_path(location, errors)
        if buckets is None:
            continue
        for module in sorted(buckets["top_hard"]):
            report.setdefault(module, []).append(display)
            if module not in PROFILE_IMPORTS:
                errors.append(
                    _check_violation(display, module, "collection time (reached from compat tests)")
                )
        pending.extend(sorted(buckets["top_in_repo"]))


def check_profile(root: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Scan the compat closure and verify every hard import resolves.

    Returns:
        Tuple of (errors, report mapping third-party module to providing files).
    """
    errors: list[str] = []
    report: dict[str, list[str]] = {}
    pending = _collect_roots(root, errors, report)
    _collect_closure(root, pending, errors, report)
    missing = sorted(
        module
        for module in report
        if module in PROFILE_IMPORTS and importlib.util.find_spec(module) is None
    )
    for module in missing:
        errors.append(
            f"{module!r} is in the compat profile but not importable: "
            "sync is incomplete, refusing silent coverage loss"
        )
    return errors, report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the compat import-profile check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root.")
    args = parser.parse_args(argv)
    errors, report = check_profile(args.root)
    print(f"compat import profile: {len(report)} third-party modules across the lane")
    for module in sorted(report):
        print(f"  {module}: {len(report[module])} files")
    print(f"accepted skips: {', '.join(sorted(ACCEPTED_SKIPS))}")
    if errors:
        print("COMPAT PROFILE VIOLATIONS:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("compat import profile: OK (no silent coverage loss)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
