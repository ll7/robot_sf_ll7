#!/usr/bin/env python3
"""Generate the environment-configuration reference from typed dataclasses.

Reads ``robot_sf/gym_env/unified_config.py`` and ``CrowdSimulationConfig`` in
``robot_sf/gym_env/crowd_sim_env.py`` with :mod:`ast` (no runtime imports, so no
heavy optional dependencies are needed) and renders the deterministic
``docs/environment_config_reference.md``. Same-module inheritance is flattened
with each inherited field marked; external mixin bases are named, not expanded.

``--check`` regenerates to memory and fails when the committed reference
drifts (exit 1). Output is byte-stable for unchanged sources.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_CONFIG = REPO_ROOT / "robot_sf" / "gym_env" / "unified_config.py"
CROWD_SIM_ENV = REPO_ROOT / "robot_sf" / "gym_env" / "crowd_sim_env.py"
TELEMETRY_CONFIG = REPO_ROOT / "robot_sf" / "gym_env" / "telemetry_config.py"
REFERENCE_MD = REPO_ROOT / "docs" / "environment_config_reference.md"

TARGET_MODULES = (UNIFIED_CONFIG, CROWD_SIM_ENV)
SOURCE_MODULES = (TELEMETRY_CONFIG, *TARGET_MODULES)


@dataclass(frozen=True)
class ConfigField:
    """One rendered configuration field."""

    name: str
    annotation: str
    default: str
    required: bool
    doc: str
    stability: str
    inherited_from: str

    def anchor(self, owner: str) -> str:
        """Return the explicit Markdown anchor for hand-written references."""
        return f"{owner}.{self.name}"


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node).strip()
    except (ValueError, AttributeError, RecursionError):  # pragma: no cover - exotic nodes
        return ""


def _is_dataclass(classdef: ast.ClassDef) -> bool:
    return any(
        (isinstance(d, ast.Name) and d.id == "dataclass")
        or (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass")
        for d in classdef.decorator_list
    )


def _is_field_call(value: ast.AST) -> bool:
    """Return whether *value* is a call to the dataclass ``field`` helper."""
    if not isinstance(value, ast.Call):
        return False
    return (isinstance(value.func, ast.Name) and value.func.id == "field") or (
        isinstance(value.func, ast.Attribute) and value.func.attr == "field"
    )


def _field_default(value: ast.AST) -> tuple[str, bool]:
    """Return (default_text, required) for one dataclass field value."""
    if _is_field_call(value):
        assert isinstance(value, ast.Call)
        for keyword in value.keywords:
            if keyword.arg == "default_factory" and keyword.value is not None:
                factory = _unparse(keyword.value)
                if isinstance(keyword.value, ast.Lambda):
                    return _unparse(keyword.value.body), False
                return f"{factory}()", False
        for keyword in value.keywords:
            if keyword.arg == "default" and keyword.value is not None:
                return _unparse(keyword.value), False
        return "", True
    return _unparse(value), False


def _field_doc(value: ast.AST) -> str:
    if _is_field_call(value):
        assert isinstance(value, ast.Call)
        for keyword in value.keywords:
            if keyword.arg == "metadata" and isinstance(keyword.value, ast.Dict):
                for key, item in zip(keyword.value.keys, keyword.value.values, strict=True):
                    if isinstance(key, ast.Constant) and key.value == "doc":
                        return item.value if isinstance(item, ast.Constant) else _unparse(item)
    return ""


def _stability(name: str, lineno: int, comments: dict[int, str]) -> str:
    text = f"{name} {comments.get(lineno - 1, '')} {comments.get(lineno, '')}".lower()
    if "deprecat" in text or "alias" in text:
        return "deprecated"
    return "public"


def _line_comments(tree: ast.Module, source_lines: list[str]) -> dict[int, str]:
    comments: dict[int, str] = {}
    for number, line in enumerate(source_lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            comments[number] = stripped.lstrip("#").strip()
    return comments


@dataclass
class _ClassInfo:
    name: str
    bases: list[str]
    docstring: str
    fields: list[ConfigField]


def parse_config_module(path: Path) -> dict[str, _ClassInfo]:
    """Parse one config module into dataclass info records."""
    source_lines = path.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(source_lines))
    comments = _line_comments(tree, source_lines)
    classes: dict[str, _ClassInfo] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
            continue
        if not _is_dataclass(node):
            continue
        docstring = ast.get_docstring(node) or ""
        bases = [_unparse(base) for base in node.bases]
        fields: list[ConfigField] = []
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            name = item.target.id
            if name.startswith("_"):
                continue
            default, required = _field_default(item.value) if item.value else ("", True)
            fields.append(
                ConfigField(
                    name=name,
                    annotation=_unparse(item.annotation),
                    default=default,
                    required=required,
                    doc=_field_doc(item.value) if item.value else "",
                    stability=_stability(name, item.lineno, comments),
                    inherited_from="",
                )
            )
        classes[node.name] = _ClassInfo(node.name, bases, docstring, fields)
    return classes


def parse_config_modules(paths: tuple[Path, ...] = SOURCE_MODULES) -> dict[str, _ClassInfo]:
    """Parse target modules and lightweight local bases into one class index."""
    classes: dict[str, _ClassInfo] = {}
    for path in paths:
        classes.update(parse_config_module(path))
    return classes


def _display_default(raw: str) -> str:
    """Render one field default in value semantics instead of source syntax."""
    text = raw.strip()
    factory = re.match(r"^field\(\s*default_factory\s*=\s*(.+?)\s*\)$", text, re.DOTALL)
    if factory:
        return f"{factory.group(1).strip()}()"
    default = re.match(
        r"^field\(\s*default\s*=\s*(.+?)\s*(?:,\s*metadata\s*=.*)?\)$", text, re.DOTALL
    )
    if default:
        return default.group(1).strip()
    return text


def _flatten(classes: dict[str, _ClassInfo], info: _ClassInfo) -> list[ConfigField]:
    """Inline same-module base fields first, marked with their owner."""
    ordered: list[ConfigField] = []
    positions: dict[str, int] = {}

    def _inherit(parent: _ClassInfo, owner: str) -> None:
        for base in parent.bases:
            base_name = base.split("[")[0].split(".")[-1]
            grandparent = classes.get(base_name)
            if grandparent is not None and grandparent.name != parent.name:
                _inherit(grandparent, grandparent.name)
        for parent_field in parent.fields:
            inherited = ConfigField(
                parent_field.name,
                parent_field.annotation,
                parent_field.default,
                parent_field.required,
                parent_field.doc,
                parent_field.stability,
                inherited_from=owner,
            )
            position = positions.get(parent_field.name)
            if position is None:
                positions[parent_field.name] = len(ordered)
                ordered.append(inherited)
            else:
                # A nearer same-module parent may override a grandparent field.
                ordered[position] = inherited

    for base in info.bases:
        base_name = base.split("[")[0].split(".")[-1]
        parent = classes.get(base_name)
        if parent is None or parent.name == info.name:
            continue
        _inherit(parent, parent.name)
    for own in info.fields:
        position = positions.get(own.name)
        if position is None:
            positions[own.name] = len(ordered)
            ordered.append(own)
        else:
            # Dataclasses keep an overridden field in its inherited position while
            # applying the subclass annotation/default, so preserve that order here.
            ordered[position] = own
    return ordered


def _external_bases(classes: dict[str, _ClassInfo], info: _ClassInfo) -> list[str]:
    names = []
    for base in info.bases:
        base_name = base.split("[")[0].split(".")[-1]
        if base_name not in classes and base_name not in {"object"}:
            names.append(base)
    return names


def render() -> str:
    """Render the complete reference Markdown deterministically."""
    modules: dict[str, dict[str, _ClassInfo]] = {
        str(path.relative_to(REPO_ROOT)): parse_config_module(path) for path in TARGET_MODULES
    }
    all_classes = parse_config_modules()
    lines = [
        "# Environment Configuration Reference",
        "",
        "> _This file is generated by `scripts/dev/generate_environment_config_reference.py`._",
        "> _Edit the typed dataclasses, never this file directly._",
        "",
        "Public environment configuration: field name, resolved annotation, default semantics,",
        "required status, and stability class, in declaration order. Units and ranges appear only",
        "when stated in field metadata or docstrings; otherwise they remain unspecified.",
        "",
    ]
    for module, classes in modules.items():
        lines += [f"## Module `{module}`", ""]
        for name in sorted(classes):
            info = classes[name]
            fields = _flatten(all_classes, info)
            lines += [f"### `{name}`", ""]
            if info.docstring:
                lines += [info.docstring.strip().splitlines()[0], ""]
            externals = _external_bases(all_classes, info)
            if externals:
                lines += [f"_Inherits unexpanded external bases: {', '.join(externals)}._", ""]
            if not fields:
                lines += ["_No public fields._", ""]
                continue
            lines += [
                "| Field | Type | Default | Required | Stability | Notes |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for entry in fields:
                anchor = entry.anchor(name)
                default = f"`{_display_default(entry.default)}`" if entry.default else "—"
                required = "yes" if entry.required else "no"
                inherited = (
                    f"inherited from `{entry.inherited_from}`; " if entry.inherited_from else ""
                )
                doc = entry.doc.replace("|", "\\|").replace("\n", " ") if entry.doc else ""
                notes = f"{inherited}{doc}".strip()
                lines.append(
                    f'| <a id="{anchor}"></a>`{entry.name}` | `{entry.annotation}` '
                    f"| {default} | {required} | {entry.stability} | {notes} |"
                )
            lines.append("")
    lines += [
        "## Hand-written examples",
        "",
        "These snippets reference generated anchors above instead of repeating field tables.",
        "",
        "Select the differential-drive robot with planning enabled "
        "([`robot_config`](#RobotSimulationConfig.robot_config)) and a clearance margin "
        "([`planner_clearance_margin`](#RobotSimulationConfig.planner_clearance_margin)):",
        "",
        "```python",
        "from robot_sf.gym_env.unified_config import RobotSimulationConfig",
        "from robot_sf.robot.differential_drive import DifferentialDriveSettings",
        "",
        "config = RobotSimulationConfig(",
        "    robot_config=DifferentialDriveSettings(),",
        "    use_planner=True,",
        "    planner_clearance_margin=0.3,",
        ")",
        "```",
        "",
        "Run crowd-only simulation without robot configuration "
        "([`map_id`](#CrowdSimulationConfig.map_id), "
        "[`recording_enabled`](#CrowdSimulationConfig.recording_enabled)):",
        "",
        "```python",
        "from robot_sf.gym_env.crowd_sim_env import CrowdSimulationConfig",
        "",
        "config = CrowdSimulationConfig(map_id=None, recording_enabled=False)",
        "```",
        "",
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: list[str] | None = None) -> int:
    """Generate the reference or fail when ``--check`` observes drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when the reference drifts")
    parser.add_argument("--output", type=Path, default=REFERENCE_MD)
    args = parser.parse_args(argv)
    rendered = render()
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.is_file() else None
        if current != rendered:
            print(f"drift: {args.output} differs from generated reference")
            return 1
        print(f"reference stable: {args.output}")
        return 0
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
