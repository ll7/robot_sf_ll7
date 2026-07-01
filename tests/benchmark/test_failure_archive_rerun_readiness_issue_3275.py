"""Tests for issue #3275 failure-archive rerun readiness/leakage checks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.failure_archive_rerun_readiness import (
    BLOCKED,
    DIAGNOSTIC_ONLY,
    READY,
    classify_failure_archive_rerun_readiness,
)
from scripts.validation.check_failure_archive_rerun_readiness import main as readiness_cli_main

if TYPE_CHECKING:
    from pathlib import Path


def _entry(
    archive_id: str,
    *,
    family: str,
    seed: int,
    certified: bool = True,
) -> dict:
    """Build a minimal failure-archive entry fixture."""

    entry = {
        "archive_id": archive_id,
        "cluster_key": {
            "policy": "goal",
            "scenario_template": family,
            "primary_failure": "collision",
            "termination_reason": "collision",
        },
        "candidate": {"scenario_seed": seed},
        "failure_attribution": {
            "primary_failure": "collision",
            "details": {"termination_reason": "collision"},
        },
    }
    if certified:
        entry["certification_metadata"] = {
            "status": "passed",
            "source": "unit-test scenario_cert.v1 fixture",
        }
    return entry


def _archive(
    path: Path,
    entries: list[dict],
    *,
    source_manifests: list[str] | None = None,
) -> Path:
    """Write an adversarial failure archive fixture."""

    payload = {
        "schema_version": "adversarial_failure_archive.v1",
        "config": {
            "source_manifests": source_manifests
            if source_manifests is not None
            else [f"search-manifests/{path.stem}.json"],
        },
        "entries": entries,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _null_test_prerequisites() -> dict:
    """Return complete null-test prerequisite metadata."""

    return {
        "null_tests_reject_null": True,
        "shuffled_outcome_null_test": {"status": "complete", "p_value": 0.01},
        "ranking_permutation_test": {"status": "complete", "p_value": 0.02},
    }


def test_disjoint_certified_archives_are_ready(tmp_path: Path) -> None:
    """Disjoint archive IDs and certified rerun rows pass the metadata gate."""

    source = _archive(
        tmp_path / "source.json",
        [
            _entry("source_0000", family="family_a", seed=1),
            _entry("source_0001", family="family_a", seed=2),
        ],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [
            _entry("rerun_0000", family="family_b", seed=101),
            _entry("rerun_0001", family="family_b", seed=102),
        ],
    )

    readiness = classify_failure_archive_rerun_readiness(
        source,
        rerun,
        null_test_prerequisites=_null_test_prerequisites(),
    )

    assert readiness.status == READY
    assert readiness.ready is True
    assert readiness.archive_id_overlap == []
    assert readiness.missing_certification_archive_ids == []
    assert readiness.to_payload()["claim_boundary"].startswith("readiness/leakage check only")


def test_missing_archive_source_lineage_blocks_readiness(tmp_path: Path) -> None:
    """Archive-level source manifests are required before disjointness is trusted."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
        source_manifests=[],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )

    readiness = classify_failure_archive_rerun_readiness(
        source,
        rerun,
        null_test_prerequisites=_null_test_prerequisites(),
    )

    assert readiness.status == BLOCKED
    assert readiness.missing_archive_lineage == ["source:config.source_manifests"]
    assert "missing_archive_lineage:1" in readiness.blockers


def test_shared_archive_source_lineage_blocks_readiness(tmp_path: Path) -> None:
    """Different entries still fail closed when archives come from one manifest."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
        source_manifests=["search-manifests/shared.json"],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
        source_manifests=["search-manifests/shared.json"],
    )

    readiness = classify_failure_archive_rerun_readiness(
        source,
        rerun,
        null_test_prerequisites=_null_test_prerequisites(),
    )

    assert readiness.status == BLOCKED
    assert readiness.source_manifest_overlap == ["search-manifests/shared.json"]
    assert "source_manifest_overlap:1" in readiness.blockers


def test_missing_null_test_prerequisite_input_blocks_readiness(tmp_path: Path) -> None:
    """Absent null-test prerequisite metadata fails closed."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )

    readiness = classify_failure_archive_rerun_readiness(source, rerun)

    assert readiness.status == BLOCKED
    assert readiness.ready is False
    assert readiness.null_test_prerequisite_status == BLOCKED
    assert readiness.missing_null_test_prerequisites == ["null_test_prerequisites"]
    assert "missing_null_test_prerequisites:1" in readiness.blockers


def test_overlapping_archive_ids_block_leakage(tmp_path: Path) -> None:
    """Archive-ID overlap between source and rerun archives blocks readiness."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("shared_failure", family="family_a", seed=1)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("shared_failure", family="family_b", seed=101)],
    )

    readiness = classify_failure_archive_rerun_readiness(source, rerun)

    assert readiness.status == BLOCKED
    assert readiness.archive_id_overlap == ["shared_failure"]
    assert "archive_id_overlap:1" in readiness.blockers


def test_scenario_family_and_seed_overlap_is_reported_as_blockers(tmp_path: Path) -> None:
    """Family and seed overlap metadata must be treated as leakage blockers."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=42)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_a", seed=42)],
    )

    readiness = classify_failure_archive_rerun_readiness(source, rerun)

    assert readiness.status == BLOCKED
    assert "scenario_family_overlap:1" in readiness.blockers
    assert "seed_overlap:1" in readiness.blockers
    assert len(readiness.overlap_provenance["scenario_family_overlap"]) == 1
    assert (
        json.loads(readiness.overlap_provenance["scenario_family_overlap"][0])
        == _entry("_", family="family_a", seed=0)["cluster_key"]
    )
    assert readiness.overlap_provenance["seed_overlap"] == [42]


def test_missing_overlap_metadata_blocks_readiness(tmp_path: Path) -> None:
    """Missing archive-id, scenario-family, or seed metadata must fail closed."""

    source_entry = _entry("source_0000", family="family_a", seed=1)
    source_entry.pop("archive_id")
    source_entry.pop("cluster_key")
    source_entry.pop("failure_attribution")
    rerun_entry = _entry("rerun_0000", family="family_b", seed=101)
    rerun_entry["candidate"].pop("scenario_seed")
    source = _archive(tmp_path / "source.json", [source_entry])
    rerun = _archive(tmp_path / "rerun.json", [rerun_entry])

    readiness = classify_failure_archive_rerun_readiness(source, rerun)
    payload = readiness.to_payload()

    assert readiness.status == BLOCKED
    assert readiness.missing_overlap_metadata_archive_ids == [
        "source:source:<entry:0>:archive_id,scenario_family",
        "rerun:rerun_0000:scenario_seed",
    ]
    assert "missing_overlap_metadata:2" in readiness.blockers
    assert payload["missing_overlap_metadata_archive_ids"] == (
        readiness.missing_overlap_metadata_archive_ids
    )


def test_blank_overlap_metadata_blocks_readiness(tmp_path: Path) -> None:
    """Blank overlap metadata must fail closed like absent metadata."""

    source_entry = _entry("source_0000", family="family_a", seed=1)
    source_entry["archive_id"] = "   "
    source_entry["cluster_key"] = {
        "policy": "",
        "scenario_template": " ",
        "primary_failure": "",
        "termination_reason": "",
    }
    source_entry.pop("failure_attribution")
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )
    source = _archive(tmp_path / "source.json", [source_entry])

    readiness = classify_failure_archive_rerun_readiness(
        source,
        rerun,
        null_test_prerequisites=_null_test_prerequisites(),
    )

    assert readiness.status == BLOCKED
    assert readiness.missing_overlap_metadata_archive_ids == [
        "source:source:<entry:0>:archive_id,scenario_family",
    ]
    assert "missing_overlap_metadata:1" in readiness.blockers


def test_user_archive_id_matching_fallback_prefix_is_not_missing(tmp_path: Path) -> None:
    """User-provided archive IDs may look like synthetic fallback IDs."""

    source_entry = _entry("source:<entry:0>:custom", family="family_a", seed=1)
    source_entry.pop("cluster_key")
    source_entry.pop("failure_attribution")
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )
    source = _archive(tmp_path / "source.json", [source_entry])

    readiness = classify_failure_archive_rerun_readiness(
        source,
        rerun,
        null_test_prerequisites=_null_test_prerequisites(),
    )

    assert readiness.status == BLOCKED
    assert readiness.missing_overlap_metadata_archive_ids == [
        "source:source:<entry:0>:custom:scenario_family",
    ]
    assert "missing_overlap_metadata:1" in readiness.blockers


def test_non_string_cluster_key_value_is_not_placeholder_only(tmp_path: Path) -> None:
    """Non-null JSON cluster-key values can identify a scenario family."""

    source_entry = _entry("source_0000", family="family_a", seed=1)
    source_entry["cluster_key"] = {
        "policy": None,
        "scenario_template": 1,
        "primary_failure": "",
        "termination_reason": " ",
    }
    source_entry.pop("failure_attribution")
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )
    source = _archive(tmp_path / "source.json", [source_entry])

    readiness = classify_failure_archive_rerun_readiness(
        source,
        rerun,
        null_test_prerequisites=_null_test_prerequisites(),
    )

    assert readiness.status == READY
    assert readiness.missing_overlap_metadata_archive_ids == []
    assert not any(blocker.startswith("missing_overlap_metadata") for blocker in readiness.blockers)


def test_missing_archive_payload_blocks_ready_path(tmp_path: Path) -> None:
    """Missing archive file paths must fail closed for both source and rerun."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    missing = tmp_path / "missing.json"

    readiness = classify_failure_archive_rerun_readiness(source, missing)

    assert readiness.status == BLOCKED
    assert any(
        blocker.startswith("rerun_archive_blocked:path_missing") for blocker in readiness.blockers
    )
    assert readiness.missing_archive_lineage == []
    assert not any(blocker.startswith("source_archive_blocked") for blocker in readiness.blockers)


def test_malformed_archive_payload_fails_closed(tmp_path: Path) -> None:
    """Malformed or non-object archives are blockers, never treated as ready."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")

    readiness = classify_failure_archive_rerun_readiness(source, malformed)

    assert readiness.status == BLOCKED
    assert any(
        blocker.startswith("rerun_archive_blocked:unreadable") for blocker in readiness.blockers
    )
    assert readiness.missing_archive_lineage == []


def test_non_object_archive_entries_fail_closed(tmp_path: Path) -> None:
    """Archive entries must be objects before readiness metadata can be trusted."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    rerun_payload = {
        "schema_version": "adversarial_failure_archive.v1",
        "entries": ["not-an-entry"],
    }
    rerun = tmp_path / "rerun.json"
    rerun.write_text(json.dumps(rerun_payload), encoding="utf-8")

    readiness = classify_failure_archive_rerun_readiness(
        source,
        rerun,
        null_test_prerequisites=_null_test_prerequisites(),
    )

    assert readiness.status == BLOCKED
    assert readiness.rerun_entry_count == 0
    assert "rerun_archive_blocked:archive_entries_not_objects:0" in readiness.blockers


def test_mixed_archive_entry_shapes_fail_closed_with_index(tmp_path: Path) -> None:
    """Malformed source entry indexes are reported instead of silently dropped."""

    source_payload = {
        "schema_version": "adversarial_failure_archive.v1",
        "entries": [
            _entry("source_0000", family="family_a", seed=1),
            ["not-an-entry"],
        ],
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )

    readiness = classify_failure_archive_rerun_readiness(
        source,
        rerun,
        null_test_prerequisites=_null_test_prerequisites(),
    )

    assert readiness.status == BLOCKED
    assert readiness.source_entry_count == 1
    assert "source_archive_blocked:archive_entries_not_objects:1" in readiness.blockers


def test_missing_certification_metadata_blocks_rerun_archive(tmp_path: Path) -> None:
    """Every rerun archive entry must carry certification metadata."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101, certified=False)],
    )

    readiness = classify_failure_archive_rerun_readiness(source, rerun)

    assert readiness.status == BLOCKED
    assert readiness.missing_certification_archive_ids == ["rerun_0000"]
    assert "missing_certification_metadata:1" in readiness.blockers


def test_missing_source_certification_metadata_blocks_readiness(tmp_path: Path) -> None:
    """The source archive must also carry certification metadata."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1, certified=False)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )

    readiness = classify_failure_archive_rerun_readiness(source, rerun)

    assert readiness.status == BLOCKED
    assert readiness.source_missing_certification_archive_ids == ["source_0000"]
    assert "source_missing_certification_metadata:1" in readiness.blockers
    payload = readiness.to_payload()
    assert payload["source_missing_certification_archive_ids"] == ["source_0000"]


def test_failed_or_falsy_certification_status_is_invalid(tmp_path: Path) -> None:
    """Explicitly failed or falsy certification status must fail closed, not pass."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    cases = [
        {"status": "failed", "source": "unit-test"},
        {"status": False},
        {"status": None},
        {"status": ""},
        "",
        {},
    ]
    for index, certification in enumerate(cases):
        entry = _entry(f"rerun_{index:04d}", family="family_b", seed=200 + index)
        entry["certification_metadata"] = certification
        rerun = _archive(tmp_path / f"rerun_{index}.json", [entry])

        readiness = classify_failure_archive_rerun_readiness(source, rerun)

        assert readiness.status == BLOCKED, certification
        assert readiness.invalid_certification_archive_ids == [f"rerun_{index:04d}"], certification
        assert readiness.missing_certification_archive_ids == [], certification
        assert "invalid_certification_status:1" in readiness.blockers, certification


def test_failed_source_certification_status_is_invalid(tmp_path: Path) -> None:
    """Explicit source certification failures must fail closed, not pass."""

    source_entry = _entry("source_0000", family="family_a", seed=1)
    source_entry["certification_metadata"] = {"status": "failed", "source": "unit-test"}
    source = _archive(tmp_path / "source.json", [source_entry])
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )

    readiness = classify_failure_archive_rerun_readiness(source, rerun)

    assert readiness.status == BLOCKED
    assert readiness.source_invalid_certification_archive_ids == ["source_0000"]
    assert "source_invalid_certification_status:1" in readiness.blockers
    payload = readiness.to_payload()
    assert payload["source_invalid_certification_archive_ids"] == ["source_0000"]


def test_diagnostic_only_output_caps_otherwise_ready_inputs(tmp_path: Path) -> None:
    """Diagnostic-only rerun outputs cannot be promoted to benchmark evidence."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )
    rerun_output = tmp_path / "rerun_output.json"
    rerun_output.write_text(
        json.dumps(
            {
                "schema_version": "proposal_model_rerun.v1",
                "result_classification": "diagnostic_only",
                "benchmark_evidence": False,
            }
        ),
        encoding="utf-8",
    )

    readiness = classify_failure_archive_rerun_readiness(
        source,
        rerun,
        rerun_output=rerun_output,
        null_test_prerequisites=_null_test_prerequisites(),
    )

    assert readiness.status == DIAGNOSTIC_ONLY
    assert readiness.ready is False
    assert readiness.blockers == []
    assert readiness.diagnostic_only_outputs == ["result_classification:diagnostic_only"]


def test_null_test_prerequisites_can_be_checked_from_report(tmp_path: Path) -> None:
    """Complete null-test prerequisite metadata keeps otherwise ready inputs ready."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(
        json.dumps(
            {
                "schema_version": "adversarial_proposal_comparison.v1",
                "null_tests": {
                    "null_tests_reject_null": True,
                    "shuffled_outcome_null_test": {"status": "complete", "p_value": 0.01},
                    "ranking_permutation_test": {"status": "complete", "p_value": 0.02},
                },
            }
        ),
        encoding="utf-8",
    )

    readiness = classify_failure_archive_rerun_readiness(
        source,
        rerun,
        null_test_prerequisites=prerequisites,
    )

    assert readiness.status == READY
    assert readiness.null_test_prerequisite_source == str(prerequisites)
    assert readiness.null_test_prerequisite_status == READY
    assert readiness.missing_null_test_prerequisites == []
    assert readiness.invalid_null_test_prerequisites == []


def test_missing_null_test_prerequisites_block_readiness(tmp_path: Path) -> None:
    """Incomplete null-test prerequisite metadata fails closed."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(
        json.dumps(
            {
                "null_tests": {
                    "null_tests_reject_null": False,
                    "shuffled_outcome_null_test": {"status": "complete", "p_value": 0.5},
                }
            }
        ),
        encoding="utf-8",
    )

    readiness = classify_failure_archive_rerun_readiness(
        source,
        rerun,
        null_test_prerequisites=prerequisites,
    )

    assert readiness.status == BLOCKED
    assert readiness.null_test_prerequisite_status == BLOCKED
    assert readiness.missing_null_test_prerequisites == ["ranking_permutation_test"]
    assert "null_tests_reject_null_not_true" in readiness.invalid_null_test_prerequisites
    assert "missing_null_test_prerequisites:1" in readiness.blockers
    assert "invalid_null_test_prerequisites:1" in readiness.blockers


def test_invalid_null_test_p_values_block_readiness(tmp_path: Path) -> None:
    """Malformed null-test p-values block readiness."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )
    prerequisites = {
        "null_tests_reject_null": True,
        "shuffled_outcome_null_test": {"status": "complete", "p_value": "0.01"},
        "ranking_permutation_test": {"status": "complete", "p_value": 1.5},
    }

    readiness = classify_failure_archive_rerun_readiness(
        source,
        rerun,
        null_test_prerequisites=prerequisites,
    )

    assert readiness.status == BLOCKED
    assert readiness.null_test_prerequisite_status == BLOCKED
    assert readiness.invalid_null_test_prerequisites == [
        "shuffled_outcome_null_test_invalid_p_value",
        "ranking_permutation_test_invalid_p_value",
    ]
    assert "invalid_null_test_prerequisites:2" in readiness.blockers


def test_incomplete_null_test_status_does_not_validate_p_value(tmp_path: Path) -> None:
    """Incomplete null tests report status without duplicate p-value blockers."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )
    prerequisites = {
        "null_tests_reject_null": True,
        "shuffled_outcome_null_test": {"status": "failed"},
        "ranking_permutation_test": {"status": "complete", "p_value": 0.05},
    }

    readiness = classify_failure_archive_rerun_readiness(
        source,
        rerun,
        null_test_prerequisites=prerequisites,
    )

    assert readiness.status == BLOCKED
    assert readiness.invalid_null_test_prerequisites == [
        "shuffled_outcome_null_test_status_not_complete",
    ]


def test_null_test_p_value_probability_bounds_are_allowed(tmp_path: Path) -> None:
    """Boundary p-values are valid probabilities."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )
    prerequisites = {
        "null_tests_reject_null": True,
        "shuffled_outcome_null_test": {"status": "complete", "p_value": 0.0},
        "ranking_permutation_test": {"status": "complete", "p_value": 1.0},
    }

    readiness = classify_failure_archive_rerun_readiness(
        source,
        rerun,
        null_test_prerequisites=prerequisites,
    )

    assert readiness.status == READY
    assert readiness.invalid_null_test_prerequisites == []


def test_cli_exit_codes_and_writes_report(tmp_path: Path, capsys) -> None:
    """CLI returns 0 for ready inputs and writes the JSON payload."""

    source = _archive(
        tmp_path / "source.json",
        [_entry("source_0000", family="family_a", seed=1)],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [_entry("rerun_0000", family="family_b", seed=101)],
    )
    output = tmp_path / "readiness.json"
    prerequisites = tmp_path / "prerequisites.json"
    prerequisites.write_text(
        json.dumps(
            {
                "null_tests_reject_null": True,
                "shuffled_outcome_null_test": {"status": "complete", "p_value": 0.01},
                "ranking_permutation_test": {"status": "complete", "p_value": 0.02},
            }
        ),
        encoding="utf-8",
    )

    exit_code = readiness_cli_main(
        [
            "--source-archive",
            str(source),
            "--rerun-archive",
            str(rerun),
            "--null-test-prerequisites",
            str(prerequisites),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    output_payload = json.loads(output.read_text(encoding="utf-8"))
    stdout_payload = json.loads(capsys.readouterr().out)
    assert output_payload["status"] == READY
    assert output_payload["null_test_prerequisite_status"] == READY
    assert stdout_payload["status"] == READY
