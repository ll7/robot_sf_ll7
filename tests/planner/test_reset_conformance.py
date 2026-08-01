"""Conformance tests for the #6504 local-planner ``reset()`` migration."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _planner_reset_methods() -> list[tuple[str, str, ast.FunctionDef]]:
    """Collect planner reset methods, excluding the non-planner vector-env shim."""
    repo_root = Path(__file__).resolve().parents[2]
    methods: list[tuple[str, str, ast.FunctionDef]] = []

    for path in sorted((repo_root / "robot_sf" / "planner").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name == "_DummyVecEnv":
                continue
            methods.extend(
                (str(path.relative_to(repo_root)), node.name, member)
                for member in node.body
                if isinstance(member, ast.FunctionDef) and member.name == "reset"
            )

    return methods


@pytest.mark.parametrize(
    "rel_path,class_name,method",
    _planner_reset_methods(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_local_planner_reset_uses_keyword_only_seed(
    rel_path: str,
    class_name: str,
    method: ast.FunctionDef,
) -> None:
    """Every local-planner reset has the exact ``reset(*, seed=None) -> None`` shape."""
    args = method.args
    return_annotation = ast.unparse(method.returns) if method.returns is not None else "<missing>"
    seed_annotation = (
        ast.unparse(args.kwonlyargs[0].annotation)
        if len(args.kwonlyargs) == 1 and args.kwonlyargs[0].annotation is not None
        else "<missing>"
    )
    signature = f"{class_name}.reset({ast.unparse(args)}) -> {return_annotation}"
    location = f"{rel_path}:{method.lineno}"

    assert [arg.arg for arg in args.args] == ["self"], f"{location}: {signature}"
    assert not args.posonlyargs, f"{location}: {signature}"
    assert args.vararg is None, f"{location}: {signature}"
    assert args.kwarg is None, f"{location}: {signature}"
    assert [arg.arg for arg in args.kwonlyargs] == ["seed"], f"{location}: {signature}"
    assert len(args.kw_defaults) == 1 and isinstance(args.kw_defaults[0], ast.Constant), (
        f"{location}: {signature}"
    )
    assert args.kw_defaults[0].value is None, f"{location}: {signature}"
    assert seed_annotation == "int | None", f"{location}: {signature}"
    assert isinstance(method.returns, ast.Constant) and method.returns.value is None, (
        f"{location}: {signature}"
    )
