"""Every deferred robot_sf import in resource_lifecycle must resolve (issue #4826).

The subprocess arm worker defers its imports into function bodies, so a phantom module
name is invisible to collection-time import errors and to any test that does not execute
the worker end to end. That is exactly how `from robot_sf.benchmark.scenario_matrix
import load_scenario_matrix` — a module that never existed — shipped inside the arm
isolation feature and made every subprocess arm fail at runtime (Slurm job 13364),
hidden behind the arm_params serialization crash (job 13344) until that was fixed.

This test walks the module's AST, extracts every `import`/`from ... import` whose target
is a robot_sf module — wherever it appears, including inside functions — and requires
importlib to resolve the module AND getattr to resolve each imported name.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = _REPO_ROOT / "robot_sf/benchmark/camera_ready/resource_lifecycle.py"


def _robot_sf_imports(tree: ast.AST) -> list[tuple[str, list[str]]]:
    found: list[tuple[str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("robot_sf"):
            if node.level == 0:
                found.append((node.module, [a.name for a in node.names]))
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("robot_sf"):
                    found.append((a.name, []))
    return found


def test_all_robot_sf_imports_resolve() -> None:
    """Every deferred first-party import resolves its module and requested symbol."""
    tree = ast.parse(TARGET.read_text())
    imports = _robot_sf_imports(tree)
    assert imports, "expected robot_sf imports in resource_lifecycle"
    problems: list[str] = []
    for module, names in imports:
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            problems.append(f"{module}: module does not resolve ({exc})")
            continue
        for name in names:
            if name != "*" and not hasattr(mod, name):
                problems.append(f"{module}.{name}: name missing from module")
    assert not problems, "phantom imports in resource_lifecycle:\n  " + "\n  ".join(problems)
