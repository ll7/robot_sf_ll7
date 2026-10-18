"""Production callers must use the guarded preregistered reference facade."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import yaml

UNGUARDED = {
    "robot_sf.research.lane_formation_reference": {
        "run_native_reference",
        "run_reference_campaign",
    },
    "robot_sf.research.lane_formation_parameter_screen": {"run_parameter_screen"},
}
GUARDED_FACADES = {
    "robot_sf/research/lane_formation_reference_guarded.py",
    "robot_sf/research/lane_formation_parameter_screen_guarded.py",
}
HISTORICAL_INPUTS = (
    {
        "path": "robot_sf/research/lane_formation_reference.py",
        "packet": "configs/benchmarks/issue_6969_lane_formation_stage_b_preregistration.yaml",
        "reason": "Preserve the preregistered native reference implementation bytes.",
    },
    {
        "path": "robot_sf/research/lane_formation_parameter_screen.py",
        "packet": "configs/benchmarks/issue_6969_lane_formation_stage_b_preregistration.yaml",
        "reason": "Preserve the preregistered Stage A parameter-screen implementation bytes.",
    },
    {
        "path": "scripts/validation/run_issue_6969_parameter_screen.py",
        "packet": "configs/benchmarks/issue_6969_lane_formation_stage_b_preregistration.yaml",
        "reason": "Preserve the preregistered Stage A command bytes; current campaigns use its successor.",
    },
    {
        "path": "tests/research/test_lane_formation_parameter_screen.py",
        "packet": "configs/benchmarks/issue_6969_lane_formation_stage_b_preregistration.yaml",
        "reason": "Preserve the preregistered Stage A test bytes; new controls exercise the facade.",
    },
)


def _validate_historical_inputs(root: Path, entries) -> None:
    for entry in entries:
        path = entry["path"]
        packet = entry["packet"]
        document = yaml.safe_load((root / packet).read_text())
        matching_keys = [
            key for key, value in document["source_contracts"].items() if value == path
        ]
        if not matching_keys or not entry.get("reason"):
            raise ValueError(f"historical exemption {path} is not pinned by {packet}")
        digest = hashlib.sha256((root / path).read_bytes()).hexdigest()
        if not all(document["source_sha256"].get(key) == digest for key in matching_keys):
            raise ValueError(f"historical exemption {path} does not match its immutable pin")


def _unguarded_imports(source: str) -> list[int]:  # noqa: C901 - cover direct and aliased imports
    tree = ast.parse(source)
    aliases = {}
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module in UNGUARDED:
                if any(
                    alias.name in UNGUARDED[node.module] or alias.name == "*"
                    for alias in node.names
                ):
                    violations.append(node.lineno)
            elif node.module == "robot_sf.research":
                for alias in node.names:
                    module = f"robot_sf.research.{alias.name}"
                    if module in UNGUARDED:
                        aliases[alias.asname or alias.name] = module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in UNGUARDED:
                    aliases[alias.asname or alias.name] = alias.name
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            module = aliases.get(ast.unparse(node.value))
            if module is not None and node.attr in UNGUARDED[module]:
                violations.append(node.lineno)
    return sorted(set(violations))


def test_production_reference_callers_import_guarded_execution():
    reference = "robot_sf.research.lane_formation_reference"
    assert _unguarded_imports(f"from {reference} import run_native_reference as run") == [1]
    assert _unguarded_imports(f"import {reference} as ref\nref.run_reference_campaign()") == [2]
    assert _unguarded_imports(
        "from robot_sf.research import lane_formation_reference as ref\nref.run_native_reference()"
    ) == [2]
    assert _unguarded_imports(f"from {reference} import ReferenceProtocol") == []

    root = Path(__file__).resolve().parents[2]
    _validate_historical_inputs(root, HISTORICAL_INPUTS)
    historical_paths = {entry["path"] for entry in HISTORICAL_INPUTS}
    violations = []
    for directory in ("robot_sf", "scripts", "tests", "examples"):
        for path in sorted((root / directory).rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if relative in GUARDED_FACADES or relative in historical_paths:
                continue
            violations.extend(f"{relative}:{line}" for line in _unguarded_imports(path.read_text()))
    assert not violations, "unguarded reference execution imports: " + ", ".join(violations)
