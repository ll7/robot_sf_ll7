"""Tests for the scenario-evidence crosswalk (issue #5602)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.identity.hash_utils import read_jsonl as _read_jsonl
from robot_sf.benchmark.scenario_evidence_crosswalk import (
    KNOWN_PREDICATE_SCHEMAS,
    LEGACY_LATE_EVASIVE_PREDICATE_SCHEMA,
    PREDICATE_EXPORT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SUPPORTED_PREDICATE_SCHEMAS,
    ScenarioEvidenceCrosswalkError,
    build_scenario_evidence_crosswalk,
    crosswalk_markdown,
    validate_scenario_evidence_crosswalk,
    write_scenario_evidence_crosswalk,
)
from robot_sf.benchmark.trace_predicate_export import (
    EXPORT_STATUS_DEGRADED,
    EXPORT_STATUS_MISSING,
    build_trace_predicate_export,
)
from scripts.tools.export_scenario_evidence_crosswalk import load_scenario_matrix
from scripts.tools.export_scenario_evidence_crosswalk import main as export_main

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        evidence_catalog = {"doorway_low": {"eligibility": "eligible", "trace_ids": [reference]}}
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


def test_legacy_late_evasive_v1_flows_through_with_exact_provenance() -> None:
    """Legacy late_evasive.v1 (issue #5935) is accepted and retained as v1, not normalized.

    The real issue4206 trace-capable bundle carries v1 late-evasive records (0 v2 records).
    The producer accepts both; the crosswalk must also accept v1 and preserve the exact source
    schema version as provenance rather than silently rewriting it to v2.
    """
    scenarios = [_scenario("doorway_low")]
    export = {
        "schema_version": PREDICATE_EXPORT_SCHEMA_VERSION,
        "rows": [
            {
                "scenario_id": "doorway_low",
                "predicates": [
                    {
                        "predicate": "late_evasive",
                        "schema_version": LEGACY_LATE_EVASIVE_PREDICATE_SCHEMA,
                        "status": "ok",
                    }
                ],
            }
        ],
    }

    crosswalk = build_scenario_evidence_crosswalk(
        scenarios, source="fixture.yaml", predicate_export=export
    )
    exported = crosswalk["rows"][0]["predicate_availability"]["exported_predicates"]
    assert len(exported) == 1
    assert exported[0]["predicate"] == "late_evasive"
    # Exact source provenance preserved (NOT normalized to v2).
    assert exported[0]["schema_version"] == LEGACY_LATE_EVASIVE_PREDICATE_SCHEMA
    # The v1 record passes full schema + semantic validation.
    assert validate_scenario_evidence_crosswalk(crosswalk) == []


def test_late_evasive_v1_and_v2_coexist_with_distinct_provenance() -> None:
    """A bundle carrying v1 in one scenario and v2 in another preserves each version distinctly."""
    scenarios = [_scenario("legacy_scenario"), _scenario("current_scenario")]
    export = {
        "schema_version": PREDICATE_EXPORT_SCHEMA_VERSION,
        "rows": [
            {
                "scenario_id": "legacy_scenario",
                "predicates": [
                    {
                        "predicate": "late_evasive",
                        "schema_version": LEGACY_LATE_EVASIVE_PREDICATE_SCHEMA,
                        "status": "ok",
                    }
                ],
            },
            {
                "scenario_id": "current_scenario",
                "predicates": [
                    {
                        "predicate": "late_evasive",
                        "schema_version": KNOWN_PREDICATE_SCHEMAS["late_evasive"],
                        "status": "ok",
                    }
                ],
            },
        ],
    }

    crosswalk = build_scenario_evidence_crosswalk(
        scenarios, source="fixture.yaml", predicate_export=export
    )
    by_id = {r["scenario_id"]: r for r in crosswalk["rows"]}
    legacy = by_id["legacy_scenario"]["predicate_availability"]["exported_predicates"][0]
    current = by_id["current_scenario"]["predicate_availability"]["exported_predicates"][0]
    assert legacy["schema_version"] == LEGACY_LATE_EVASIVE_PREDICATE_SCHEMA
    assert current["schema_version"] == KNOWN_PREDICATE_SCHEMAS["late_evasive"]
    assert validate_scenario_evidence_crosswalk(crosswalk) == []


def test_supported_predicate_schemas_is_superset_of_known_motivated() -> None:
    """The accepted provenance set is a superset of the motivated/current versions."""
    motivated = set(KNOWN_PREDICATE_SCHEMAS.values())
    accepted = set().union(*SUPPORTED_PREDICATE_SCHEMAS.values())
    assert motivated.issubset(accepted)
    # v1 is accepted for late_evasive specifically (issue #5935).
    assert LEGACY_LATE_EVASIVE_PREDICATE_SCHEMA in SUPPORTED_PREDICATE_SCHEMAS["late_evasive"]
    assert KNOWN_PREDICATE_SCHEMAS["late_evasive"] in SUPPORTED_PREDICATE_SCHEMAS["late_evasive"]


def test_motivated_not_exported_still_references_current_v2() -> None:
    """Accepting v1 provenance does not change the motivated/current schema reference (v2)."""
    scenarios = [_scenario("doorway_low")]
    export = {
        "schema_version": PREDICATE_EXPORT_SCHEMA_VERSION,
        "rows": [
            {
                "scenario_id": "doorway_low",
                "predicates": [
                    {
                        "predicate": "late_evasive",
                        "schema_version": LEGACY_LATE_EVASIVE_PREDICATE_SCHEMA,
                        "status": "ok",
                    }
                ],
            }
        ],
    }

    crosswalk = build_scenario_evidence_crosswalk(
        scenarios, source="fixture.yaml", predicate_export=export
    )
    motivated = {
        m["predicate"]: m["schema_version"]
        for m in crosswalk["rows"][0]["predicate_availability"]["motivated_not_exported_predicates"]
    }
    # late_evasive is exported, so it is NOT in motivated_not_exported; the other two remain v1.
    assert "late_evasive" not in motivated
    assert motivated["oscillatory_control"] == KNOWN_PREDICATE_SCHEMAS["oscillatory_control"]
    assert motivated["occlusion_near_miss"] == KNOWN_PREDICATE_SCHEMAS["occlusion_near_miss"]


def test_unknown_late_evasive_version_still_fails_closed() -> None:
    """Accepting v1 must not widen to an arbitrary late_evasive version (fail-closed boundary)."""
    scenarios = [_scenario("doorway_low")]
    export = {
        "schema_version": PREDICATE_EXPORT_SCHEMA_VERSION,
        "rows": [
            {
                "scenario_id": "doorway_low",
                "predicates": [
                    {
                        "predicate": "late_evasive",
                        "schema_version": "safety_predicate.late_evasive.v9",
                        "status": "ok",
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


def test_exporter_reports_input_errors_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI input failures use the clean error path and nonzero status."""
    matrix = tmp_path / "scalar.yaml"
    matrix.write_text("just-a-scalar\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["export_scenario_evidence_crosswalk.py", str(matrix)])

    assert export_main() == 1
    assert "Crosswalk export failed:" in capsys.readouterr().err


def test_markdown_labels_geometry_as_descriptive_not_causal() -> None:
    """The report must separate geometry (descriptive) from evidence eligibility."""
    scenarios = [_scenario("doorway_low")]
    crosswalk = build_scenario_evidence_crosswalk(scenarios, source="fixture.yaml")
    md = crosswalk_markdown(crosswalk)

    assert "Geometry (descriptive)" in md
    assert "Evidence" in md
    assert "doorway_low" in md


class TestRealLegacyLateEvasiveSmoke:
    """Real-campaign smoke exercising the legacy late_evasive.v1 boundary (issue #5935).

    The real issue4206 trace-capable bundle carries v1 late-evasive records (6291 v1,
    0 v2 records). This feeds the actual producer export through the crosswalk's
    consumer contract, asserting the legacy v1 provenance flows end-to-end with exact
    provenance and passes validation. ``output/`` is gitignored, so the test skips
    cleanly in fresh worktrees that lack the bundle.
    """

    RELATIVE_BUNDLE = Path(
        "output/benchmarks/camera_ready/issue4206_trace_capable_h600_rerun_20260704/"
        "runs/goal__differential_drive/episodes.jsonl"
    )

    @classmethod
    def _bundle_path(cls) -> Path | None:
        """Resolve the real campaign bundle, checking the current repo and parent checkout."""
        for root in (REPO_ROOT,):
            bundle = root / cls.RELATIVE_BUNDLE
            if bundle.is_file():
                return bundle
        # Sibling main checkout. The worktree container here is
        # ``.../robot_sf_ll7.worktrees/<branch>/``; the main checkout is ``.../robot_sf_ll7``.
        container = REPO_ROOT.parent
        if container.name.endswith(".worktrees"):
            main_root = container.with_name(container.name[: -len(".worktrees")])
            alt = main_root / cls.RELATIVE_BUNDLE
            if alt.is_file():
                return alt
        return None

    def test_real_legacy_v1_export_flows_through_crosswalk(self) -> None:
        """The real v1 issue4206 bundle reaches the crosswalk and validates with v1 provenance."""
        bundle = self._bundle_path()
        if bundle is None:
            pytest.skip("real campaign bundle not present")
        episodes = _read_jsonl(bundle)
        assert episodes, "bundle must contain episode records"

        # Produce the per-episode export (the producer accepts both v1 and v2 and
        # preserves the exact source schema_version, including legacy v1).
        export_rows = build_trace_predicate_export(episodes, run_id="goal__differential_drive")
        assert export_rows, "producer must emit export rows"

        # The bundle is legacy v1; at least one late_evasive block must carry v1.
        v1_seen = any(
            row["predicates"]["late_evasive"]["schema_version"]
            == LEGACY_LATE_EVASIVE_PREDICATE_SCHEMA
            for row in export_rows
            if row["predicates"]["late_evasive"]["export_status"] != EXPORT_STATUS_MISSING
        )
        assert v1_seen, "real bundle must carry legacy v1 late_evasive records"

        # Build the crosswalk-consumable envelope (one row per scenario, predicates as a
        # list of {predicate, schema_version, status} records) from the real producer rows.
        # Status is derived from the producer's export_status so a missing/degraded predicate
        # is never misrepresented as a clean measurement (fail-closed surfacing).
        per_scenario: dict[str, list[dict[str, str]]] = {}
        for row in export_rows:
            sid = row["scenario_id"]
            records = per_scenario.setdefault(sid, [])
            for predicate, block in row["predicates"].items():
                if block["export_status"] == EXPORT_STATUS_MISSING:
                    status = "missing"
                elif block["export_status"] == EXPORT_STATUS_DEGRADED:
                    status = "degraded"
                else:
                    status = "ok"
                records.append(
                    {
                        "predicate": predicate,
                        "schema_version": block["schema_version"],
                        "status": status,
                    }
                )
        predicate_export = {
            "schema_version": PREDICATE_EXPORT_SCHEMA_VERSION,
            "rows": [
                {"scenario_id": sid, "predicates": preds}
                for sid, preds in sorted(per_scenario.items())
            ],
        }

        scenarios = [
            {"name": sid, "map_id": "m", "metadata": {"scenario_family": "f", "archetype": "g"}}
            for sid in sorted(per_scenario)
        ]
        crosswalk = build_scenario_evidence_crosswalk(
            scenarios, source="issue4206_real_smoke", predicate_export=predicate_export
        )

        # Exact v1 source provenance is preserved end-to-end, not normalized away.
        exported_versions = {
            rec["schema_version"]
            for row in crosswalk["rows"]
            for rec in row["predicate_availability"]["exported_predicates"]
            if rec["predicate"] == "late_evasive"
        }
        assert LEGACY_LATE_EVASIVE_PREDICATE_SCHEMA in exported_versions
        # No real v2 records exist in this bundle; none should be fabricated.
        assert KNOWN_PREDICATE_SCHEMAS["late_evasive"] not in exported_versions
        assert validate_scenario_evidence_crosswalk(crosswalk) == []
