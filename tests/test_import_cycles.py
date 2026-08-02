"""Import-cycle regression guard for ``robot_sf`` (issue #6455).

The guard builds a static import graph of the ``robot_sf`` package from AST
analysis and fails when any group of modules forms a load-time import cycle.

Graph semantics
---------------
Only imports that execute at module load time create edges:

* module-scope imports (including class bodies and ``try/except`` fallbacks),

while these deliberate, sanctioned cycle breaks do **not** create edges:

* imports inside function bodies (lazy/deferred imports),
* imports guarded by ``if TYPE_CHECKING:``,
* imports guarded by ``if __name__ == "__main__":``.

Issue #6455 accepts lazy/deferred imports and ``TYPE_CHECKING`` references as
valid resolutions for a circular pair, so the guard enforces the invariant
that actually prevents fragile import-order-dependent load failures: no
module-scope import cycle anywhere in ``robot_sf``. A naive all-static-import
scan (counting lazy and ``TYPE_CHECKING`` edges) reports more candidate pairs;
those are intentionally broken at load time and are not guard violations.

To regenerate the load-time inventory manually::

    uv run pytest tests/test_import_cycles.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _module_name(path: Path, scan_root: Path, package: str) -> str:
    rel = path.relative_to(scan_root / Path(*package.split("."))).with_suffix("")
    parts = [part for part in rel.parts if part != "__init__"]
    return ".".join([package, *parts])


def _package_of(module: str, is_package_init: bool) -> str:
    if is_package_init:
        return module
    return module.rpartition(".")[0]


def _resolve_relative(level: int, module: str | None, package: str) -> str | None:
    if level == 0:
        return module
    parts = package.split(".") if package else []
    if level - 1 > len(parts):
        return None
    base = ".".join(parts[: len(parts) - (level - 1)])
    if module:
        return f"{base}.{module}" if base else module
    return base or None


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return (
            test.attr == "TYPE_CHECKING"
            and isinstance(test.value, ast.Name)
            and test.value.id
            in {
                "typing",
                "typing_extensions",
            }
        )
    return False


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if (
        not isinstance(test, ast.Compare)
        or len(test.ops) != 1
        or not isinstance(test.ops[0], ast.Eq)
    ):
        return False
    left, right = test.left, test.comparators[0]
    pairs = ((left, right), (right, left))
    return any(
        isinstance(one, ast.Name)
        and one.id == "__name__"
        and isinstance(other, ast.Constant)
        and other.value == "__main__"
        for one, other in pairs
    )


def _iter_load_time_imports(tree: ast.AST):
    """Yield import statements that execute at module load time."""
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue  # deferred imports are a sanctioned cycle break (#6455)
        if isinstance(child, ast.If) and (_is_type_checking_guard(child) or _is_main_guard(child)):
            # The guarded body is not executed during import, but its else
            # branch is. Keep scanning that branch for load-time imports.
            for else_child in child.orelse:
                if isinstance(else_child, (ast.Import, ast.ImportFrom)):
                    yield else_child
                yield from _iter_load_time_imports(else_child)
            continue
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            yield child
        yield from _iter_load_time_imports(child)


def build_load_time_import_graph(
    scan_root: Path,
    package: str,
) -> tuple[set[str], dict[str, set[str]]]:
    """Return ``(modules, edges)`` for the load-time import graph of ``package``."""
    package_root = scan_root / Path(*package.split("."))
    modules = {_module_name(path, scan_root, package) for path in package_root.rglob("*.py")}
    graph: dict[str, set[str]] = {module: set() for module in modules}

    for path in sorted(package_root.rglob("*.py")):
        module = _module_name(path, scan_root, package)
        package_of_module = _package_of(module, path.name == "__init__.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _iter_load_time_imports(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            else:
                resolved = _resolve_relative(node.level, node.module, package_of_module)
                if resolved is None:
                    continue
                targets.append(resolved)
                targets.extend(
                    f"{resolved}.{alias.name}" for alias in node.names if alias.name != "*"
                )
            for target in targets:
                if target in modules:
                    edge = target
                else:
                    parent = target.rpartition(".")[0]
                    edge = parent if parent in modules else ""
                if edge and edge != module:
                    graph[module].add(edge)

    return modules, graph


def _reachable_modules(start: str, graph: dict[str, set[str]]) -> set[str]:
    """Return every module reachable from ``start`` (excluding ``start`` itself)."""
    seen: set[str] = set()
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for neighbor in graph.get(node, ()):
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return seen


def find_load_time_cycles(scan_root: Path, package: str = "robot_sf") -> list[list[str]]:
    """Return every strongly connected component of size > 1 (sorted, stable).

    Two modules share an SCC exactly when each is reachable from the other,
    which is cheap to compute for a package-scale import graph.
    """
    modules, graph = build_load_time_import_graph(scan_root, package)
    reachability = {module: _reachable_modules(module, graph) for module in sorted(modules)}

    components: dict[frozenset[str], None] = {}
    for module in sorted(modules):
        partners = {other for other in reachability[module] if module in reachability[other]}
        if partners:
            components[frozenset(partners | {module})] = None

    return sorted(sorted(component) for component in components)


def _format_cycles(cycles: list[list[str]]) -> str:
    return "\n".join("  " + " -> ".join([*cycle, cycle[0]]) for cycle in cycles)


def test_robot_sf_has_no_load_time_import_cycles() -> None:
    """No ``robot_sf`` module group may form a module-scope import cycle (#6455)."""
    cycles = find_load_time_cycles(REPO_ROOT)

    assert not cycles, (
        "Load-time import cycles detected in robot_sf (issue #6455 regression guard).\n"
        "Break each cycle by extracting shared types into a dependency-free module, "
        "inverting the dependency, or converting one direction to a deferred "
        "(function-level) or TYPE_CHECKING import:\n" + _format_cycles(cycles)
    )


def _write_package(root: Path, sources: dict[str, str]) -> Path:
    package = root / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name, source in sources.items():
        (package / name).write_text(source, encoding="utf-8")
    return root


def test_guard_detects_direct_two_module_cycle(tmp_path: Path) -> None:
    """A top-level mutual import must be reported as a cycle."""
    root = _write_package(
        tmp_path,
        {
            "a.py": "from pkg import b\n",
            "b.py": "from pkg import a\n",
        },
    )

    assert find_load_time_cycles(root, package="pkg") == [["pkg.a", "pkg.b"]]


def test_guard_detects_three_module_cycle(tmp_path: Path) -> None:
    """Longer module-scope cycles must be reported, not only pairs."""
    root = _write_package(
        tmp_path,
        {
            "a.py": "import pkg.b\n",
            "b.py": "import pkg.c\n",
            "c.py": "import pkg.a\n",
        },
    )

    assert find_load_time_cycles(root, package="pkg") == [["pkg.a", "pkg.b", "pkg.c"]]


def test_guard_detects_relative_import_cycle(tmp_path: Path) -> None:
    """Relative imports resolve to package modules and must create edges."""
    root = _write_package(
        tmp_path,
        {
            "a.py": "from . import b\n",
            "b.py": "from . import a\n",
        },
    )

    assert find_load_time_cycles(root, package="pkg") == [["pkg.a", "pkg.b"]]


def test_guard_detects_class_body_import_cycle(tmp_path: Path) -> None:
    """Class bodies execute at import time, so their imports create edges."""
    root = _write_package(
        tmp_path,
        {
            "a.py": "from pkg import b\n",
            "b.py": "class Holder:\n    from pkg import a\n",
        },
    )

    assert find_load_time_cycles(root, package="pkg") == [["pkg.a", "pkg.b"]]


def test_guard_ignores_type_checking_back_edge(tmp_path: Path) -> None:
    """TYPE_CHECKING imports are a sanctioned cycle break and create no edge."""
    root = _write_package(
        tmp_path,
        {
            "a.py": "from pkg import b\n",
            "b.py": (
                "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from pkg import a\n"
            ),
        },
    )

    assert find_load_time_cycles(root, package="pkg") == []


def test_guard_counts_type_checking_else_branch(tmp_path: Path) -> None:
    """The executable else branch of a TYPE_CHECKING guard remains load-time code."""
    root = _write_package(
        tmp_path,
        {
            "a.py": "from pkg import b\n",
            "b.py": (
                "from typing import TYPE_CHECKING\n\n"
                "if TYPE_CHECKING:\n    pass\n"
                "else:\n    from pkg import a\n"
            ),
        },
    )

    assert find_load_time_cycles(root, package="pkg") == [["pkg.a", "pkg.b"]]


def test_guard_does_not_treat_arbitrary_attribute_as_type_checking(
    tmp_path: Path,
) -> None:
    """Only typing's TYPE_CHECKING sentinel suppresses a guarded import."""
    root = _write_package(
        tmp_path,
        {
            "a.py": "from pkg import b\n",
            "b.py": "if runtime.TYPE_CHECKING:\n    from pkg import a\n",
        },
    )

    assert find_load_time_cycles(root, package="pkg") == [["pkg.a", "pkg.b"]]


def test_guard_ignores_function_level_back_edge(tmp_path: Path) -> None:
    """Deferred function-level imports are a sanctioned cycle break."""
    root = _write_package(
        tmp_path,
        {
            "a.py": "from pkg import b\n",
            "b.py": "def use_a():\n    from pkg import a\n\n    return a\n",
        },
    )

    assert find_load_time_cycles(root, package="pkg") == []


def test_guard_ignores_main_guard_back_edge(tmp_path: Path) -> None:
    """``if __name__ == '__main__'`` imports never run during import."""
    root = _write_package(
        tmp_path,
        {
            "a.py": "from pkg import b\n",
            "b.py": 'if __name__ == "__main__":\n    from pkg import a\n',
        },
    )

    assert find_load_time_cycles(root, package="pkg") == []


def test_guard_counts_main_guard_else_branch(tmp_path: Path) -> None:
    """The executable else branch of a __main__ guard remains load-time code."""
    root = _write_package(
        tmp_path,
        {
            "a.py": "from pkg import b\n",
            "b.py": 'if __name__ == "__main__":\n    pass\nelse:\n    from pkg import a\n',
        },
    )

    assert find_load_time_cycles(root, package="pkg") == [["pkg.a", "pkg.b"]]


def test_guard_ignores_acyclic_shared_dependency(tmp_path: Path) -> None:
    """Modules sharing a dependency without mutual imports are not a cycle."""
    root = _write_package(
        tmp_path,
        {
            "a.py": "from pkg import shared\n",
            "b.py": "from pkg import shared\n",
            "shared.py": "VALUE = 1\n",
        },
    )

    assert find_load_time_cycles(root, package="pkg") == []
