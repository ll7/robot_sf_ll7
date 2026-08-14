"""Tests for trace dossier cell-binding metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError

from robot_sf.benchmark.trace_dossier_cell_binding import (
    TRACE_DOSSIER_CELL_BINDING_SCHEMA_VERSION,
    TraceDossierCellBindingError,
    TraceDossierCellIdentity,
    TraceDossierSelectedTrace,
    build_trace_dossier_cell_binding,
)

_TRACE_SHA256 = "a" * 64
_SCHEMA = (
    Path(__file__).parents[2] / "robot_sf/benchmark/schemas/trace_dossier_cell_binding.v1.json"
)


def _cell() -> dict[str, object]:
    """Return a minimal campaign-cell identity."""

    return {
        "campaign_id": "release-0.0.3",
        "cell_id": "classic-crossing::orca::nominal",
        "scenario_id": "classic_group_crossing_medium",
        "scenario_family": "classic_group_crossing",
        "planner_id": "orca",
        "release_arm_id": "nominal",
    }


def _selected_trace() -> dict[str, object]:
    """Return a selected trace identity."""

    return {
        "cell_id": "classic-crossing::orca::nominal",
        "episode_id": "episode-007",
        "seed": 7,
        "trace_artifact_uri": "traces/episode-007.json",
        "trace_sha256": _TRACE_SHA256,
        "terminal_verdict": "success",
    }


def test_build_trace_dossier_cell_binding_records_counts_and_selected_trace() -> None:
    """The metadata block names the cell, selected trace, and verdict denominator."""

    binding = build_trace_dossier_cell_binding(
        cell=_cell(),
        selected_trace=_selected_trace(),
        terminal_verdict_counts={"timeout": 2, "success": 28},
    )

    assert binding.schema_version == TRACE_DOSSIER_CELL_BINDING_SCHEMA_VERSION
    assert binding.cell.cell_id == "classic-crossing::orca::nominal"
    assert binding.selected_trace.episode_id == "episode-007"
    assert binding.cell_episode_count == 30
    assert binding.selected_verdict_count == 28
    assert binding.terminal_verdict_counts == {"success": 28, "timeout": 2}
    assert binding.evidence_boundary.startswith("metadata_only")


def test_binding_to_dict_is_deterministic_and_json_shaped() -> None:
    """The manifest block is sorted and contains no clock-bearing fields."""

    binding = build_trace_dossier_cell_binding(
        cell=_cell(),
        selected_trace=_selected_trace(),
        terminal_verdict_counts={"timeout": 2, "success": 28},
    )

    payload = binding.to_dict()

    assert list(payload["terminal_verdict_counts"]) == ["success", "timeout"]
    assert payload["cell"]["campaign_id"] == "release-0.0.3"
    assert payload["selected_trace"]["trace_sha256"] == _TRACE_SHA256
    assert "generated_at" not in payload


def test_dataclass_inputs_are_accepted_without_mutating_contract() -> None:
    """Validated dataclass inputs can be passed directly by future renderers."""

    binding = build_trace_dossier_cell_binding(
        cell=TraceDossierCellIdentity(
            campaign_id="campaign-a",
            cell_id="cell-a",
            scenario_id="scenario-a",
            planner_id="planner-a",
        ),
        selected_trace=TraceDossierSelectedTrace(
            cell_id="cell-a",
            episode_id="episode-a",
            seed=1,
            trace_artifact_uri="trace-a.json",
            trace_sha256="b" * 64,
            terminal_verdict="collision",
        ),
        terminal_verdict_counts={"collision": 1},
    )

    assert binding.cell.release_arm_id is None
    assert binding.cell_episode_count == 1
    assert binding.selected_verdict_count == 1


def test_binding_schema_accepts_serialized_contract() -> None:
    """The tracked JSON schema accepts the builder's deterministic shape."""

    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    binding = build_trace_dossier_cell_binding(
        cell=_cell(),
        selected_trace=_selected_trace(),
        terminal_verdict_counts={"timeout": 2, "success": 28},
    )

    Draft202012Validator(schema).validate(binding.to_dict())


def test_text_and_seed_inputs_are_normalized() -> None:
    """Whitespace is trimmed and integral scalar seeds become plain integers."""

    cell = _cell()
    cell["cell_id"] = "  classic-crossing::orca::nominal  "
    cell["release_arm_id"] = " nominal "
    selected_trace = _selected_trace()
    selected_trace["cell_id"] = "classic-crossing::orca::nominal"
    selected_trace["seed"] = np.int64(7)
    selected_trace["terminal_verdict"] = " success "

    binding = build_trace_dossier_cell_binding(
        cell=cell,
        selected_trace=selected_trace,
        terminal_verdict_counts={" success ": 30},
    )

    assert binding.cell.cell_id == "classic-crossing::orca::nominal"
    assert binding.cell.release_arm_id == "nominal"
    assert binding.selected_trace.seed == 7
    assert type(binding.selected_trace.seed) is int
    assert binding.selected_trace.terminal_verdict == "success"
    assert binding.terminal_verdict_counts == {"success": 30}


def test_normalized_duplicate_verdict_labels_fail_closed() -> None:
    """Distinct raw labels cannot collapse into one normalized verdict silently."""

    with pytest.raises(TraceDossierCellBindingError, match="duplicate label"):
        build_trace_dossier_cell_binding(
            cell=_cell(),
            selected_trace=_selected_trace(),
            terminal_verdict_counts={"success": 15, " success ": 15},
        )


def test_binding_schema_rejects_blank_verdict_label() -> None:
    """The JSON schema rejects blank and whitespace-only verdict keys."""

    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(
            {
                "schema_version": TRACE_DOSSIER_CELL_BINDING_SCHEMA_VERSION,
                "cell": {
                    "campaign_id": "campaign",
                    "cell_id": "cell",
                    "scenario_id": "scenario",
                    "planner_id": "planner",
                },
                "selected_trace": {
                    "cell_id": "cell",
                    "episode_id": "episode",
                    "seed": 1,
                    "trace_artifact_uri": "trace.json",
                    "trace_sha256": _TRACE_SHA256,
                    "terminal_verdict": "success",
                },
                "terminal_verdict_counts": {" ": 1},
                "cell_episode_count": 1,
                "selected_verdict_count": 1,
                "evidence_boundary": (
                    "metadata_only_not_benchmark_evidence_until_trace_export_and_renderer_provenance_pass"
                ),
            }
        )


def test_selected_verdict_must_be_present_in_cell_counts() -> None:
    """A dossier cannot bind a selected trace to a verdict absent from its cell."""

    with pytest.raises(TraceDossierCellBindingError, match="terminal_verdict"):
        build_trace_dossier_cell_binding(
            cell=_cell(),
            selected_trace=_selected_trace(),
            terminal_verdict_counts={"timeout": 30},
        )


def test_selected_trace_must_match_declared_cell() -> None:
    """A trace from another cell cannot be rebound by changing only the manifest cell."""

    selected_trace = _selected_trace()
    selected_trace["cell_id"] = "other-cell"

    with pytest.raises(TraceDossierCellBindingError, match="must match cell.cell_id"):
        build_trace_dossier_cell_binding(
            cell=_cell(),
            selected_trace=selected_trace,
            terminal_verdict_counts={"success": 30},
        )


def test_selected_verdict_count_must_be_positive() -> None:
    """A selected trace cannot claim a verdict with a zero cell count."""

    with pytest.raises(TraceDossierCellBindingError, match="positive count"):
        build_trace_dossier_cell_binding(
            cell=_cell(),
            selected_trace=_selected_trace(),
            terminal_verdict_counts={"success": 0, "timeout": 30},
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("campaign_id", "", "cell.campaign_id"),
        ("cell_id", " ", "cell.cell_id"),
        ("scenario_id", None, "cell.scenario_id"),
        ("planner_id", 7, "cell.planner_id"),
        ("release_arm_id", "", "cell.release_arm_id"),
    ],
)
def test_invalid_cell_identity_fields_fail_closed(field: str, value: object, match: str) -> None:
    """Blank or non-text campaign-cell identities are rejected."""

    cell = _cell()
    cell[field] = value

    with pytest.raises(TraceDossierCellBindingError, match=match):
        build_trace_dossier_cell_binding(
            cell=cell,
            selected_trace=_selected_trace(),
            terminal_verdict_counts={"success": 30},
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("cell_id", "other-cell", "must match cell.cell_id"),
        ("episode_id", "", "selected_trace.episode_id"),
        ("seed", True, "selected_trace.seed"),
        ("seed", -1, "non-negative integer"),
        ("trace_artifact_uri", "", "selected_trace.trace_artifact_uri"),
        ("trace_sha256", "A" * 64, "lowercase SHA-256"),
        ("trace_sha256", "abc", "lowercase SHA-256"),
        ("terminal_verdict", "", "selected_trace.terminal_verdict"),
    ],
)
def test_invalid_selected_trace_fields_fail_closed(field: str, value: object, match: str) -> None:
    """The selected trace identity must be explicit and checksum-bound."""

    selected_trace = _selected_trace()
    selected_trace[field] = value

    with pytest.raises(TraceDossierCellBindingError, match=match):
        build_trace_dossier_cell_binding(
            cell=_cell(),
            selected_trace=selected_trace,
            terminal_verdict_counts={"success": 30},
        )


@pytest.mark.parametrize(
    ("counts", "match"),
    [
        ({}, "non-empty"),
        ({"success": 0}, "total"),
        ({"success": -1}, "non-negative"),
        ({"success": True}, "non-negative"),
        ({"": 1}, "label"),
    ],
)
def test_invalid_terminal_verdict_counts_fail_closed(
    counts: dict[object, object], match: str
) -> None:
    """Missing or malformed cell counts cannot enter a dossier manifest."""

    with pytest.raises(TraceDossierCellBindingError, match=match):
        build_trace_dossier_cell_binding(
            cell=_cell(),
            selected_trace=_selected_trace(),
            terminal_verdict_counts=counts,
        )


def test_unknown_mapping_fields_fail_closed() -> None:
    """Versioned metadata must not silently drop caller-provided identity fields."""

    cell = _cell()
    cell["unexpected"] = "value"

    with pytest.raises(TraceDossierCellBindingError, match="unknown fields"):
        build_trace_dossier_cell_binding(
            cell=cell,
            selected_trace=_selected_trace(),
            terminal_verdict_counts={"success": 30},
        )
