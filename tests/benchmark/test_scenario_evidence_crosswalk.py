"""Tests for the scenario-evidence crosswalk (issue #5602)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.scenario_evidence_crosswalk import (
    KNOWN_PREDICATE_SCHEMAS,
    PREDICATE_EXPORT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ScenarioEvidenceCrosswalkError,
    build_scenario_evidence_crosswalk,
    crosswalk_markdown,
    validate_scenario_evidence_crosswalk,
    write_scenario_evidence_crosswalk,
)
from scripts.tools.export_scenario_evidence_crosswalk import load_scenario_matrix


def _scenario(
    name: str,
    *,
    scenario_family: str = "doorway",
    geometry_group: str = "doorway",
    interaction_class: str | None = None,
    hazard_tags: list[str] | None = None,
    seeds: list[int] | None = None,
    map_id: str = "classic_doorway",
) -> dict[str, object]:
    """Return a compact scenario fixture for crosswalk tests."""
    metadata: dict[str, object] = {
        "scenario_family": scenario_family,
        "archetype": geometry_group,
    }
    if interaction_class is not None:
        metadata["interaction_class"] = interaction_class
    if hazard_tags is not None:
        metadata["expected_failure_modes"] = hazard_tags
    return {
        "name": name,
        "map_id": map_id,
        "metadata": metadata,
        "seeds": seeds if seeds is not None else [1, 2, 3],
    }


def test_crosswalk_schema_version_and_one_row_per_scenario() -> None:
    """Exactly one row per canonical scenario, with stable schema version."""
    scenarios = [
        _scenario("doorway_low", scenario_family="doorway", geometry_group="doorway"),
        _scenario("bottleneck_low", scenario_family="bottleneck", geometry_group="bottleneck"),
    ]

    crosswalk = build_scenario_evidence_crosswalk(scenarios, source="fixture.yaml")

    assert crosswalk["schema_version"] == SCHEMA_VERSION
    assert crosswalk["summary"]["scenario_count"] == 2
    assert [r["scenario_id"] for r in crosswalk["rows"]] == [
        "bottleneck_low",
        "doorway_low",
    ]
    errors = validate_scenario_evidence_crosswalk(crosswalk)
    assert errors == []


def test_predicate_fields_unavailable_when_export_absent() -> None:
    """Without #5593 export, predicate fields are explicit unavailable, not inferred."""
    scenarios = [_scenario("doorway_low")]

    crosswalk = build_scenario_evidence_crosswalk(scenarios, source="fixture.yaml")
    pred = crosswalk["rows"][0]["predicate_availability"]

    assert pred["export_status"] == "unavailable"
    assert pred["export_status_reason"] == "predicate_export_unavailable"
    assert pred["exported_predicates"] == []
    motivated = {m["predicate"] for m in pred["motivated_not_exported_predicates"]}
    assert motivated == set(KNOWN_PREDICATE_SCHEMAS)


def test_predicate_export_is_consumed_when_present() -> None:
    """A supplied #5593 export populates exported predicates and drops motivated-only."""
    scenarios = [_scenario("doorway_low")]
    export = {
        "schema_version": PREDICATE_EXPORT_SCHEMA_VERSION,
        "rows": [
            {
                "scenario_id": "doorway_low",
                "predicates": [
                    {
                        "predicate": "oscillatory_control",
                        "schema_version": KNOWN_PREDICATE_SCHEMAS["oscillatory_control"],
                        "status": "ok",
                    }
                ],
            }
        ],
    }

    crosswalk = build_scenario_evidence_crosswalk(
        scenarios, source="fixture.yaml", predicate_export=export
    )
    row = crosswalk["rows"][0]
    assert row["predicate_availability"]["export_status"] == "available"
    assert {p["predicate"] for p in row["predicate_availability"]["exported_predicates"]} == {
        "oscillatory_control"
    }
    motivated = {
        m["predicate"] for m in row["predicate_availability"]["motivated_not_exported_predicates"]
    }
    assert "oscillatory_control" not in motivated


def test_geometry_group_and_validated_mechanism_are_separate_fields() -> None:
    """Geometry group is a descriptive label; validated_mechanism stays null in crosswalk."""
    scenarios = [_scenario("doorway_low", geometry_group="doorway_geometry")]
    evidence_catalog = {
        "doorway_low": {
            "eligibility": "eligible",
            "trace_ids": ["trace_1"],
            "replay_ids": ["replay_1"],
        }
    }

    crosswalk = build_scenario_evidence_crosswalk(
        scenarios, source="fixture.yaml", evidence_catalog=evidence_catalog
    )
    row = crosswalk["rows"][0]
    assert row["geometry_group"] == "doorway_geometry"
    assert row["geometry_group_provenance"] == "config_metadata"
    assert row["evidence"]["validated_mechanism"] is None
    assert row["evidence"]["validated_mechanism_provenance"] is None
    assert row["evidence"]["trace_ids"] == ["trace_1"]
    assert row["evidence"]["eligibility"] == "eligible"


def test_non_eligible_evidence_requires_exclusion_reason() -> None:
    """Fail closed: a non-eligible evidence entry without a reason is rejected."""
    scenarios = [_scenario("doorway_low")]
    evidence_catalog = {"doorway_low": {"eligibility": "excluded", "trace_ids": []}}

    with pytest.raises(ScenarioEvidenceCrosswalkError, match="exclusion_reason"):
        build_scenario_evidence_crosswalk(
            scenarios, source="fixture.yaml", evidence_catalog=evidence_catalog
        )


def test_validated_mechanism_must_not_be_supplied_by_crosswalk() -> None:
    """Crosswalk must not encode a mechanism; such input is rejected."""
    scenarios = [_scenario("doorway_low")]
    evidence_catalog = {
        "doorway_low": {
            "eligibility": "eligible",
            "validated_mechanism": "static_deadlock_or_local_minimum",
        }
    }

    with pytest.raises(ScenarioEvidenceCrosswalkError, match="validated_mechanism"):
        build_scenario_evidence_crosswalk(
            scenarios, source="fixture.yaml", evidence_catalog=evidence_catalog
        )


def test_broken_artifact_reference_fails_closed() -> None:
    """Referenced evidence artifacts that are missing raise rather than pass."""
    scenarios = [_scenario("doorway_low")]
    evidence_catalog = {
        "doorway_low": {
            "eligibility": "eligible",
            "trace_ids": ["does_not_exist.jsonl"],
        }
    }

    with pytest.raises(ScenarioEvidenceCrosswalkError, match="broken artifact reference"):
        build_scenario_evidence_crosswalk(
            scenarios,
            source="fixture.yaml",
            evidence_catalog=evidence_catalog,
            artifact_root=Path(__file__).resolve().parents[2] / "does-not-exist",
        )


def test_unsafe_artifact_reference_is_structured_and_regular_file_only(tmp_path: Path) -> None:
    """Traversal, symlinks, and directories cannot enter the trusted artifact root."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "episode.json").write_text("{}", encoding="utf-8")
    (artifact_root / "subdir").mkdir()
    (artifact_root / "subdir" / "episode.json").write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    symlink = artifact_root / "symlink.json"
    symlink.symlink_to(outside)

    for reference in ("../outside.json", str(outside), "symlink.json", "subdir"):
        evidence_catalog = {
            "doorway_low": {"eligibility": "eligible", "trace_ids": [reference]}
        }
        with pytest.raises(
            ScenarioEvidenceCrosswalkError,
            match="unreadable_episode_artifact",
        ):
            build_scenario_evidence_crosswalk(
                [_scenario("doorway_low")],
                source="fixture.yaml",
                evidence_catalog=evidence_catalog,
                artifact_root=artifact_root,
            )


def test_seed_coercion_accepts_only_strict_integers() -> None:
    """Boolean, float, and string seed values are not silently coerced."""
    crosswalk = build_scenario_evidence_crosswalk(
        [_scenario("doorway_low", seeds=[1, True, 2.0, "3", 4])], source="fixture.yaml"
    )
    assert crosswalk["rows"][0]["seeds"] == [1, 4]


def test_duplicate_scenario_ids_are_rejected() -> None:
    """Duplicate scenario ids must fail rather than produce ambiguous rows."""
    scenarios = [_scenario("same"), _scenario("same")]

    with pytest.raises(ScenarioEvidenceCrosswalkError, match="duplicate scenario id"):
        build_scenario_evidence_crosswalk(scenarios, source="dupes.yaml")


def test_unknown_predicate_schema_version_fails_validation() -> None:
    """An export carrying an unrecognized predicate schema version is rejected."""
    scenarios = [_scenario("doorway_low")]
    export = {
        "schema_version": PREDICATE_EXPORT_SCHEMA_VERSION,
        "rows": [
            {
                "scenario_id": "doorway_low",
                "predicates": [
                    {
                        "predicate": "oscillatory_control",
                        "schema_version": "safety_predicate.oscillatory_control.v9",
                    }
                ],
            }
        ],
    }

    with pytest.raises(ScenarioEvidenceCrosswalkError, match="unknown predicate schema version"):
        build_scenario_evidence_crosswalk(scenarios, source="fixture.yaml", predicate_export=export)


def test_output_is_deterministic_checksum_identical() -> None:
    """Same inputs produce checksum-identical content JSON."""
    scenarios = [_scenario("doorway_low"), _scenario("bottleneck_low")]

    first = build_scenario_evidence_crosswalk(scenarios, source="fixture.yaml")
    second = build_scenario_evidence_crosswalk(scenarios, source="fixture.yaml")
    assert first["content_sha256"] == second["content_sha256"]


def test_writers_emit_json_markdown_csv(tmp_path: Path) -> None:
    """JSON, Markdown, and CSV artifacts are written and pass validation."""
    scenarios = [_scenario("doorway_low", hazard_tags=["near_miss", "collision"])]
    crosswalk = build_scenario_evidence_crosswalk(scenarios, source="fixture.yaml")

    json_path = tmp_path / "cw.json"
    md_path = tmp_path / "cw.md"
    csv_path = tmp_path / "cw.csv"
    write_scenario_evidence_crosswalk(
        crosswalk, json_path=json_path, markdown_path=md_path, csv_path=csv_path
    )

    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == SCHEMA_VERSION
    assert "Scenario-Evidence Crosswalk" in md_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "doorway_low" in csv_text
    assert "collision;near_miss" in csv_text


def test_csv_preserves_generated_and_case_capsule_ids(tmp_path: Path) -> None:
    """CSV output must retain all evidence-id fields in the versioned schema."""
    crosswalk = build_scenario_evidence_crosswalk(
        [_scenario("doorway_low")],
        source="fixture.yaml",
        evidence_catalog={
            "doorway_low": {
                "eligibility": "eligible",
                "trace_ids": [],
                "generated_candidate_ids": ["candidate_1"],
                "case_capsule_ids": ["capsule_1"],
            }
        },
    )
    csv_path = tmp_path / "cw.csv"
    write_scenario_evidence_crosswalk(crosswalk, csv_path=csv_path)
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["generated_candidate_ids"] == "candidate_1"
    assert rows[0]["case_capsule_ids"] == "capsule_1"


def test_exporter_rejects_scalar_single_document(tmp_path: Path) -> None:
    """The exporter must reject a scalar YAML document before scenario loading."""
    matrix = tmp_path / "scalar.yaml"
    matrix.write_text("just-a-scalar\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping or list"):
        load_scenario_matrix(matrix)


def test_markdown_labels_geometry_as_descriptive_not_causal() -> None:
    """The report must separate geometry (descriptive) from evidence eligibility."""
    scenarios = [_scenario("doorway_low")]
    crosswalk = build_scenario_evidence_crosswalk(scenarios, source="fixture.yaml")
    md = crosswalk_markdown(crosswalk)

    assert "Geometry (descriptive)" in md
    assert "Evidence" in md
    assert "doorway_low" in md
