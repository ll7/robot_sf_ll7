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
import os
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

_BUCKETS = (
    "top_hard",
    "top_in_repo",
    "func_heavy",
    "deferred_in_repo",
    "skip_strings",
)


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


def _module_context(root: Path, path: Path) -> tuple[str, str]:
    """Return the module and package names represented by ``path``."""
    relative = path.resolve().relative_to(root.resolve())
    parts = list(relative.parts)
    if not parts or parts[-1] != "__init__.py":
        module_parts = [*parts[:-1], Path(parts[-1]).stem]
    else:
        module_parts = parts[:-1]
    package_parts = parts[:-1]
    return ".".join(module_parts), ".".join(package_parts)


def _relative_base(package_name: str, level: int) -> str:
    """Resolve the package prefix for a relative import."""
    parts = package_name.split(".") if package_name else []
    parent_count = level - 1
    if parent_count > len(parts):
        return ""
    return ".".join(parts[: len(parts) - parent_count])


def _names(
    node: ast.Import | ast.ImportFrom,
    *,
    package_name: str,
) -> list[str]:
    """Return absolute dotted module names for one import statement."""
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]

    base = _relative_base(package_name, node.level) if node.level else ""
    target = ".".join(part for part in (base, node.module or "") if part)
    if not target:
        return []

    # ``from package import child`` imports the package and may load a child
    # submodule. Include both so package initializers and relative imports in
    # the child enter the transitive closure.
    names = [target]
    names.extend(f"{target}.{alias.name}" for alias in node.names if alias.name != "*")
    return names


def _is_local(name: str) -> bool:
    """Return whether a dotted name belongs to a checkout namespace."""
    return any(
        name == namespace or name.startswith(f"{namespace}.") for namespace in LOCAL_NAMESPACES
    )


def _local_closure_names(name: str) -> set[str]:
    """Return ``name`` and its package prefixes for import closure traversal."""
    parts = name.split(".")
    return {".".join(parts[:index]) for index in range(1, len(parts) + 1)}


def _record(
    names: list[str],
    buckets: dict[str, set[str]],
    *,
    collection_time: bool,
    optional: bool = False,
) -> None:
    """File names into hard/in-repo buckets, preserving optional guards."""
    for name in names:
        if _is_local(name):
            bucket = "top_in_repo" if collection_time else "deferred_in_repo"
            buckets[bucket].update(_local_closure_names(name))
        elif optional:
            continue
        elif collection_time:
            buckets["top_hard"].add(name.split(".")[0])
        else:
            buckets["func_heavy"].add(name.split(".")[0])


def _imports_in_body(statements: list[ast.stmt]) -> list[ast.Import | ast.ImportFrom]:
    """Return import statements nested in a guarded body."""
    return [
        node
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def _optional_import_ids(
    nodes: list[ast.Import | ast.ImportFrom], *, package_name: str
) -> set[int]:
    """Return guarded imports that share one optional top-level package."""
    if not nodes:
        return set()
    roots = {
        name.split(".")[0]
        for node in nodes
        for name in _names(node, package_name=package_name)
        if name
    }
    # A guarded block may import several submodules of one optional package
    # (for example ``torch`` and ``torch.nn``). Distinct roots are siblings,
    # so there is no safe exemption and every import remains checked.
    return {id(node) for node in nodes} if len(roots) == 1 else set()


class _ImportCollector(ast.NodeVisitor):
    """Collect imports while retaining execution and optional-guard context."""

    def __init__(self, *, package_name: str) -> None:
        self.buckets = _new_buckets()
        self.package_name = package_name
        self.collection_time = True
        self.optional_import_ids: set[int] = set()

    def _visit_statements(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.visit(statement)

    def _visit_with_context(
        self,
        statements: list[ast.stmt],
        *,
        collection_time: bool | None = None,
        optional_import_ids: set[int] | None = None,
    ) -> None:
        previous_collection_time = self.collection_time
        previous_optional_import_ids = self.optional_import_ids
        if collection_time is not None:
            self.collection_time = collection_time
        if optional_import_ids is not None:
            self.optional_import_ids = optional_import_ids
        try:
            self._visit_statements(statements)
        finally:
            self.collection_time = previous_collection_time
            self.optional_import_ids = previous_optional_import_ids

    def _visit_import(self, node: ast.Import | ast.ImportFrom) -> None:
        _record(
            _names(node, package_name=self.package_name),
            self.buckets,
            collection_time=self.collection_time,
            optional=id(node) in self.optional_import_ids,
        )

    def visit_Import(self, node: ast.Import) -> None:
        self._visit_import(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._visit_import(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Collect literal ``pytest.importorskip`` targets."""
        func = node.func
        attr = func.attr if isinstance(func, ast.Attribute) else ""
        if attr == "importorskip" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                self.buckets["skip_strings"].add(first.value.split(".")[0])
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        """Skip type-only imports while still inspecting the runtime else branch."""
        if _is_type_checking(node):
            self._visit_statements(node.orelse)
            return
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        """Inspect guarded cleanup and sibling imports without hiding hard deps."""
        if not _handler_catches_import_error(node):
            self.generic_visit(node)
            return

        body_imports = _imports_in_body(node.body)
        # Imports from one optional package retain the existing exemption (for
        # example ``torch`` and ``torch.nn``). Distinct roots are siblings, so
        # there is no safe static way to prove which one is optional and every
        # sibling is checked fail-closed.
        optional_ids = _optional_import_ids(body_imports, package_name=self.package_name)
        self._visit_with_context(node.body, optional_import_ids=optional_ids)
        for handler in node.handlers:
            self._visit_statements(handler.body)
        self._visit_statements(node.orelse)
        self._visit_statements(node.finalbody)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Visit function declarations now and their bodies when called."""
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        self._visit_with_context(node.body, collection_time=False)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)


def _scan_file(path: Path, *, root: Path) -> dict[str, set[str]]:
    """Scan one file, separating collection-time imports from deferred ones.

    Returns:
        Buckets with ``top_hard`` (collection-time third-party imports),
        ``top_in_repo`` (collection-time in-repo modules), ``func_heavy``
        (deferred third-party imports), ``deferred_in_repo`` (deferred local
        modules), and ``skip_strings``.
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
        _module_name, package_name = _module_context(root, path)
        collector = _ImportCollector(package_name=package_name)
        collector._visit_statements(tree.body)
        buckets = collector.buckets
        stdlib = set(_sys.stdlib_module_names)
        drop = stdlib | LOCAL_NAMESPACES | {"__future__", "typing", "typing_extensions"}
        for key in ("top_hard", "func_heavy", "skip_strings"):
            buckets[key] = {m for m in buckets[key] if m not in drop}
    except RecursionError as error:
        raise _ProfileScanError(f"{path}: cannot inspect source (recursion limit)") from error
    return buckets


def _scan_path(root: Path, path: Path, errors: list[str]) -> dict[str, set[str]] | None:
    """Scan one selected path, retaining a bounded diagnostic on failure."""
    try:
        return _scan_file(path, root=root)
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
    """Return selected test files, failing closed on traversal gaps or emptiness."""
    target = root / dirname
    if dirname.endswith(".py"):
        if not target.is_file():
            errors.append(f"{dirname}: required compat profile file is missing")
            return []
        return [target]
    if not target.is_dir():
        errors.append(f"{dirname}: required compat profile directory is missing")
        return []

    traversal_errors: list[OSError] = []

    def onerror(error: OSError) -> None:
        """Retain traversal errors instead of letting ``os.walk`` hide them."""
        traversal_errors.append(error)

    files: list[Path] = []
    try:
        for directory, _subdirectories, filenames in os.walk(target, onerror=onerror):
            directory_path = Path(directory)
            files.extend(
                directory_path / filename
                for filename in filenames
                if filename.endswith(".py")
                and (filename == "__init__.py" or filename.startswith("test_"))
            )
    except (OSError, TypeError, ValueError) as error:
        errors.append(f"{dirname}: cannot enumerate compat profile files ({type(error).__name__})")
        return sorted(files)

    for error in traversal_errors:
        errors.append(
            f"{dirname}: cannot enumerate compat profile files ({type(error).__name__}: {error})"
        )
    if not files:
        errors.append(
            f"{dirname}: required compat profile directory contains no selected Python files"
        )
    return sorted(files)


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
            buckets = _scan_path(root, path, errors)
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
            pending.extend(sorted(buckets["deferred_in_repo"]))
    return pending


def _collect_closure(
    root: Path, pending: list[str], errors: list[str], report: dict[str, list[str]]
) -> None:
    """Follow local imports from the selected test roots through their owners.

    Source-owner function bodies can expose optional runtime adapters that are
    outside this slim compatibility lane. Deferred third-party imports in the
    selected test files are checked by ``_collect_roots``; local imports from
    either phase still enter the closure so their owners cannot hide imports.
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
        buckets = _scan_path(root, location, errors)
        if buckets is None:
            continue
        for module in sorted(buckets["top_hard"]):
            report.setdefault(module, []).append(display)
            if module not in PROFILE_IMPORTS:
                errors.append(
                    _check_violation(display, module, "collection time (reached from compat tests)")
                )
        pending.extend(sorted(buckets["top_in_repo"]))
        pending.extend(sorted(buckets["deferred_in_repo"]))


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
