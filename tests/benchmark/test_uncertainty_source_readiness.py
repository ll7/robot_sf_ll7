"""Tests benchmark uncertainty-source readiness helpers."""

from __future__ import annotations

import ast
import sys
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING

from robot_sf.benchmark import uncertainty_source_readiness as readiness

if TYPE_CHECKING:
    from pathlib import Path


def test_default_inventory_payload_and_legacy_inventory_contract(monkeypatch) -> None:
    """Default and legacy inventory helpers expose deterministic readiness payloads."""

    report = readiness.inspect_uncertainty_source_readiness()
    payload = report.as_dict()

    assert payload["schema_version"] == readiness.SCHEMA_VERSION
    assert payload["issue"] == readiness.ISSUE
    assert payload["ready"] is False
    assert readiness.EXISTENCE_DEGRADATION in payload["ready_sources"]
    assert readiness.VISIBILITY_OCCLUSION in payload["blocked_sources"]
    assert "does not claim" in payload["claim_boundary"]
    assert payload["sources"][0]["condition_builder"]["present"] is True

    monkeypatch.setattr(readiness, "_discover_condition_builders", lambda: frozenset({"builder"}))
    monkeypatch.setattr(readiness, "_discover_scenario_hooks", lambda: frozenset({"hook"}))
    legacy = readiness.build_uncertainty_source_readiness_inventory(
        (
            readiness.UncertaintySourceRunSpec(
                source="legacy_ready",
                condition_builder="builder",
                scenario_hook="hook",
            ),
        )
    )

    assert legacy["schema_version"] == readiness.UNCERTAINTY_SOURCE_READINESS_SCHEMA
    assert legacy["ready_sources"] == ["legacy_ready"]
    assert legacy["blocked_sources"] == []


def test_static_ast_helpers_find_supported_top_level_symbols() -> None:
    """Static helper recognizes functions, classes, annotated names, and modules."""

    tree = ast.parse(
        "def function_owner():\n    pass\nclass ClassOwner:\n    pass\nANNOTATED_OWNER: int = 1\n"
    )

    assert readiness._module_ast_defines(tree, "function_owner")
    assert readiness._module_ast_defines(tree, "ClassOwner")
    assert not readiness._module_ast_defines(tree, "ANNOTATED_OWNER")
    assert (
        readiness._module_source_path("robot_sf.benchmark.uncertainty_source_readiness").name
        == "uncertainty_source_readiness.py"
    )


def test_source_field_discovery_and_surrogate_missing_fields(monkeypatch) -> None:
    """Surrogate inspection covers static source fields and missing-field failures."""

    field_names = readiness._class_field_names_from_source(
        "robot_sf.representation.uncertainty_source_generalization:SourceContrast"
    )
    assert field_names is not None
    assert set(readiness.EXPECTED_SURROGATE_OUTPUTS) <= field_names

    module = types.ModuleType("_issue_3557_incomplete_dynamic_owner")

    class IncompleteContrast:
        source: str

    module.IncompleteContrast = IncompleteContrast
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(readiness, "_module_source_path", lambda _module_name: None)
    monkeypatch.setattr(readiness, "_class_field_names_from_source", lambda _owner: None)

    present, evidence = readiness._source_contrast_has_expected_fields(
        f"{module.__name__}:IncompleteContrast"
    )

    assert not present
    assert "missing fields" in evidence


def test_dynamic_surrogate_output_fallback_accepts_dataclass(monkeypatch) -> None:
    """Dynamic fallback supports dataclasses when static source inspection is unavailable."""

    module = types.ModuleType("_issue_3557_dataclass_dynamic_owner")

    @dataclass(frozen=True)
    class DataclassContrast:
        source: str
        retained_unsafe_commit_rate: float
        dropped_unsafe_commit_rate: float
        min_separation_delta_m: float
        n_episodes: int

    module.DataclassContrast = DataclassContrast
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(readiness, "_module_source_path", lambda _module_name: None)
    monkeypatch.setattr(readiness, "_class_field_names_from_source", lambda _owner: None)

    present, evidence = readiness._source_contrast_has_expected_fields(
        f"{module.__name__}:DataclassContrast"
    )

    assert present
    assert "exposes" in evidence


def test_static_owner_discovery_handles_destructuring_and_parse_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Static owner checks handle common source shapes and fail closed on bad source."""

    tree = ast.parse("first, second = object(), object()\n")
    assert not readiness._module_ast_defines(tree, "second")
    assert not readiness._module_ast_defines(tree, "third")

    bad_source = tmp_path / "bad_owner.py"
    bad_source.write_text("def broken(:\n", encoding="utf-8")
    monkeypatch.setattr(readiness, "_module_source_path", lambda _module_name: bad_source)

    present, evidence = readiness._resolve_owner("bad_owner:anything")

    assert not present
    assert "cannot read" in evidence


def test_dynamic_surrogate_output_fallback_accepts_annotated_non_dataclass(monkeypatch) -> None:
    """Dynamic fallback can inspect plain annotated classes without crashing."""

    module = types.ModuleType("_issue_3557_dynamic_owner")

    class PlainContrast:
        source: str
        retained_unsafe_commit_rate: float
        dropped_unsafe_commit_rate: float
        min_separation_delta_m: float
        n_episodes: int

    module.PlainContrast = PlainContrast
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(readiness, "_module_source_path", lambda _module_name: None)
    monkeypatch.setattr(readiness, "_class_field_names_from_source", lambda _owner: None)

    present, evidence = readiness._source_contrast_has_expected_fields(
        f"{module.__name__}:PlainContrast"
    )

    assert present
    assert "exposes" in evidence


def test_readiness_status_reports_first_missing_component() -> None:
    """Source readiness status distinguishes each missing prerequisite class."""

    present = readiness.ReadinessComponent(present=True, owner="owner:symbol", evidence="ok")
    missing = readiness.ReadinessComponent(present=False, owner=None, evidence="missing")

    assert (
        readiness.UncertaintySourceReadiness(
            source="missing_builder",
            condition_builder=missing,
            scenario_hook=present,
            expected_surrogate_outputs=present,
        ).status
        == readiness.MISSING_CONDITION_BUILDER
    )
    assert (
        readiness.UncertaintySourceReadiness(
            source="missing_hook",
            condition_builder=present,
            scenario_hook=missing,
            expected_surrogate_outputs=present,
        ).status
        == readiness.MISSING_SCENARIO_HOOK
    )
    assert (
        readiness.UncertaintySourceReadiness(
            source="missing_output",
            condition_builder=present,
            scenario_hook=present,
            expected_surrogate_outputs=missing,
        ).status
        == readiness.MISSING_SURROGATE_OUTPUT
    )


def test_owner_resolution_defensive_branches(monkeypatch) -> None:
    """Owner resolution fails closed for malformed, missing, and dynamic owners."""

    module = types.ModuleType("_issue_3557_dynamic_resolution")
    module.available = object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(readiness, "_module_source_path", lambda _module_name: None)

    assert readiness._resolve_owner("not-a-module-owner")[0] is False
    assert readiness._resolve_owner("_issue_3557_missing_module:Thing")[0] is False
    assert readiness._resolve_owner(f"{module.__name__}:missing")[0] is False

    present, evidence = readiness._resolve_owner(f"{module.__name__}:available")

    assert present
    assert "resolved" in evidence


def test_static_path_and_target_helpers_cover_edge_cases() -> None:
    """Static helpers support packages and ignore non-binding targets."""

    package_path = readiness._module_source_path("robot_sf.benchmark")
    assert package_path is not None
    assert package_path.name == "__init__.py"
    assert readiness._module_source_path("_issue_3557_no_such_module") is None

    assert not readiness._module_ast_defines(ast.parse("holder.value = 1\n"), "value")


def test_class_field_source_and_surrogate_fail_closed(monkeypatch, tmp_path: Path) -> None:
    """Source field discovery and surrogate inspection fail closed."""

    no_class_source = tmp_path / "no_class.py"
    no_class_source.write_text("OTHER = object()\n", encoding="utf-8")
    monkeypatch.setattr(readiness, "_module_source_path", lambda _module_name: no_class_source)
    assert readiness._class_field_names_from_source("no_class:Missing") is None

    assert readiness._source_contrast_has_expected_fields(None)[0] is False
