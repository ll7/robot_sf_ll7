"""Parity tests for the generated environment-config reference (issue #8729)."""

from __future__ import annotations

import ast
from pathlib import Path

from scripts.dev.generate_environment_config_reference import (
    TARGET_MODULES,
    _flatten,
    parse_config_module,
    parse_config_modules,
    render,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_MD = REPO_ROOT / "docs" / "environment_config_reference.md"

FORBIDDEN_IMPORTS = ("pygame", "torch", "stable_baselines3", "sb3", "carla")


def _public_dataclass_fields() -> set[str]:
    """Collect flattened ``Class.field`` names, mirroring generator semantics."""
    names: set[str] = set()
    all_classes = parse_config_modules()
    for module in TARGET_MODULES:
        classes = parse_config_module(module)
        for name, info in classes.items():
            for entry in _flatten(all_classes, info):
                names.add(f"{name}.{entry.name}")
    return names


def test_reference_is_byte_stable() -> None:
    """Regeneration must reproduce the committed reference exactly."""
    assert render() == REFERENCE_MD.read_text(encoding="utf-8")


def test_every_public_field_appears_exactly_once() -> None:
    """No public config field may be missing or duplicated in the reference."""
    import re

    text = REFERENCE_MD.read_text(encoding="utf-8")
    anchors = re.findall(r'<a id="([^"]+)"></a>', text)
    assert sorted(anchors) == sorted(_public_dataclass_fields())
    assert len(anchors) == len(set(anchors))


def test_subclass_overrides_use_subclass_defaults() -> None:
    """Inherited fields must reflect a subclass's replacement annotation/default."""
    classes = parse_config_module(TARGET_MODULES[0])
    fields = {field.name: field for field in _flatten(classes, classes["ImageRobotConfig"])}
    assert fields["use_image_obs"].default == "True"
    assert not fields["use_image_obs"].inherited_from
    assert "`use_image_obs` | `bool` | `True`" in render()

    transitive_fields = {
        field.name: field for field in _flatten(classes, classes["RobotEnvSettings"])
    }
    assert transitive_fields["use_image_obs"].default == "True"
    assert transitive_fields["use_image_obs"].inherited_from == "ImageRobotConfig"


def test_local_mixin_fields_are_documented() -> None:
    """Public fields from lightweight local mixins must not be silently omitted."""
    classes = parse_config_modules()
    fields = {field.name: field for field in _flatten(classes, classes["BaseSimulationConfig"])}
    assert fields["telemetry_metrics"].default == "list(DEFAULT_TELEMETRY_METRICS)"
    assert fields["telemetry_metrics"].inherited_from == "TelemetryConfigMixin"
    reference = render()
    assert "list(DEFAULT_TELEMETRY_METRICS)" in reference
    assert "lambda: list(DEFAULT_TELEMETRY_METRICS)()" not in reference
    assert "unexpanded external bases: TelemetryConfigMixin" not in reference


def test_generator_avoids_heavy_imports() -> None:
    """The generator must stay AST-based without heavy third-party imports."""
    tree = ast.parse(
        (REPO_ROOT / "scripts" / "dev" / "generate_environment_config_reference.py").read_text(
            encoding="utf-8"
        )
    )
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & set(FORBIDDEN_IMPORTS))
    assert "robot_sf" not in imported


def test_hand_written_examples_reference_generated_anchors() -> None:
    """Examples must link anchors that the reference actually defines."""
    import re

    text = REFERENCE_MD.read_text(encoding="utf-8")
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', text))
    example_links = set(re.findall(r"\]\(([^)]+)\)", text.split("## Hand-written examples")[1]))
    for link in example_links:
        if link.startswith("#") and "." in link:
            assert link[1:] in anchors, link
