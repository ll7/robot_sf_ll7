"""Tests for issue #4932's review-pending generated scenario catalog contract."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from robot_sf.benchmark.scenario_generation import (
    GeneratedScenarioCatalogValidationError,
    extract_critical_segment,
    validate_catalog_entry,
)
from robot_sf.benchmark.scenario_generation.catalog_schema import load_catalog_entry_schema
from robot_sf.benchmark.scenario_generation.replay_adapter import materialize_generated_scenario
from robot_sf.benchmark.scenario_generation.review_manifest import validate_review_manifest


def _episode() -> dict:
    """Return a synthetic randomized episode trace with a known closest approach."""

    return {
        "episode_id": "random-campus-0007",
        "seed": 4932,
        "source_map": "maps/svg_maps/classic_crossing.svg",
        "steps": [
            {
                "time_s": 0.0,
                "robot": {"position": [0.0, 0.0]},
                "pedestrians": [{"position": [4.0, 0.0]}],
            },
            {
                "time_s": 1.0,
                "robot": {"position": [1.0, 0.0]},
                "pedestrians": [{"position": [1.3, 0.0]}],
            },
            {
                "time_s": 2.0,
                "robot": {"position": [2.0, 0.0]},
                "pedestrians": [{"position": [5.0, 0.0]}],
            },
        ],
    }


def test_distiller_extracts_known_minimum_window_with_provenance() -> None:
    """A synthetic minimum-clearance frame yields a valid, bounded catalog entry."""

    entry = extract_critical_segment(_episode(), pre_margin_s=1.0, post_margin_s=1.0)

    assert entry["criticality"] == {
        "signal": "min_clearance",
        "observed_at_s": 1.0,
        "source_metrics": {"min_clearance_m": pytest.approx(0.3)},
    }
    assert entry["segment"]["window_start_s"] == 0.0
    assert entry["segment"]["window_end_s"] == 2.0
    assert entry["segment"]["initial_robot_state"] == {"position": [0.0, 0.0]}
    assert entry["metadata"]["required_manual_review"] is True
    assert entry["metadata"]["benchmark_evidence"] is False
    assert entry["source_episode"] == {
        "episode_id": "random-campus-0007",
        "source_seed": 4932,
        "source_map": "maps/svg_maps/classic_crossing.svg",
    }
    assert entry["replay"]["status"] == "not_representable_yet"
    assert entry["replay"]["warnings"][0].startswith("replay_gap:")
    assert entry["provenance"]["reviewed"] is False
    validate_catalog_entry(entry)


def test_review_manifest_requires_complete_certified_provenance(tmp_path: Path) -> None:
    """A certified generated replay remains non-benchmark and review-covered."""

    entry = extract_critical_segment(_episode())
    entry["replay"] = {
        **entry["replay"],
        "status": "replay_validated",
        "warnings": [],
    }
    reason = "Pinned geometry, route, density, critical frame, and dedup checks passed."
    entry["provenance"] = {
        **entry["provenance"],
        "reviewed": True,
        "review": {"verdict": "certified", "reason": reason},
    }
    catalog_path = tmp_path / "catalog.yaml"
    candidate_path = tmp_path / "candidate.yaml"
    review_path = tmp_path / "review.yaml"
    candidate = materialize_generated_scenario(entry)
    candidate_document = candidate.scenario_document
    assert candidate_document is not None
    candidate_document["scenarios"][0]["metadata"]["generated_replay"]["replay_status"] = (
        "replay_validated"
    )
    candidate_path.write_text(yaml.safe_dump(candidate_document, sort_keys=True), encoding="utf-8")
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "generated-scenario-catalog.v1",
                "metadata": {"benchmark_evidence": False},
                "entries": [entry],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    review_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "generated-scenario-review-manifest.v1",
                "claim_boundary": "generated scenario hypotheses only",
                "deduplication_distance_threshold": 1.0,
                "entries": [
                    {
                        "scenario_id": entry["scenario_id"],
                        "materialized_scenario": "candidate.yaml",
                        "verdict": "certified",
                        "reason": reason,
                        "checklist": {
                            "geometry_valid": True,
                            "route_feasible": True,
                            "pedestrian_density_plausible": True,
                            "critical_window_present": True,
                            "dedup_correct": True,
                        },
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assert validate_review_manifest(catalog_path, review_path)["reviewed_count"] == 1

    review_packet = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    review_packet["entries"] = []
    review_path.write_text(yaml.safe_dump(review_packet, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="cover exactly"):
        validate_review_manifest(catalog_path, review_path)

    review_packet["entries"] = [
        {
            "scenario_id": entry["scenario_id"],
            "materialized_scenario": "candidate.yaml",
            "verdict": "certified",
            "reason": reason,
            "checklist": {
                "geometry_valid": True,
                "route_feasible": True,
                "pedestrian_density_plausible": True,
                "critical_window_present": True,
                "dedup_correct": True,
            },
        }
    ]
    review_packet["entries"][0]["checklist"].pop("dedup_correct")
    review_path.write_text(yaml.safe_dump(review_packet, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="checklist is incomplete"):
        validate_review_manifest(catalog_path, review_path)

    review_packet["entries"][0]["checklist"]["dedup_correct"] = True
    review_path.write_text(yaml.safe_dump(review_packet, sort_keys=True), encoding="utf-8")
    catalog_packet = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    catalog_packet["entries"][0]["provenance"]["reviewed"] = False
    catalog_path.write_text(yaml.safe_dump(catalog_packet, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="not marked reviewed"):
        validate_review_manifest(catalog_path, review_path)

    catalog_packet["entries"][0]["provenance"]["reviewed"] = True
    catalog_packet["entries"][0]["criticality"]["observed_at_s"] = 1.5
    catalog_path.write_text(yaml.safe_dump(catalog_packet, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="no trace frame matches observed_at_s"):
        validate_review_manifest(catalog_path, review_path)


def test_distiller_is_deterministic_for_one_trace() -> None:
    """A catalog identity is stable for the same episode and extraction bounds."""

    assert extract_critical_segment(_episode()) == extract_critical_segment(_episode())


def test_distiller_preserves_explicit_zero_margins() -> None:
    """An explicit zero margin selects only the critical frame, not the default window."""

    entry = extract_critical_segment(_episode(), pre_margin_s=0.0, post_margin_s=0.0)

    assert entry["segment"]["window_start_s"] == 1.0
    assert entry["segment"]["window_end_s"] == 1.0


def test_catalog_entry_schema_is_a_valid_json_schema() -> None:
    """The stored contract is itself valid before an entry relies on it."""

    Draft202012Validator.check_schema(load_catalog_entry_schema())


def test_validator_rejects_benchmark_evidence_marker() -> None:
    """Generated entries fail closed if they assert benchmark evidence."""

    entry = extract_critical_segment(_episode())
    entry["metadata"]["benchmark_evidence"] = True

    with pytest.raises(GeneratedScenarioCatalogValidationError, match="False was expected"):
        validate_catalog_entry(entry)


def test_validator_rejects_criticality_outside_segment_bounds() -> None:
    """A stored event timestamp cannot fall outside its extracted trace window."""

    entry = extract_critical_segment(_episode())
    entry["criticality"]["observed_at_s"] = 3.0

    with pytest.raises(GeneratedScenarioCatalogValidationError, match="must lie inside"):
        validate_catalog_entry(entry)


def test_validator_requires_a_pinned_replay_seed_and_replay_gap() -> None:
    """A replay-pending entry cannot silently lose its source replay contract."""

    entry = extract_critical_segment(_episode())
    entry["replay"]["source_seed"] = 1
    with pytest.raises(GeneratedScenarioCatalogValidationError, match="must equal"):
        validate_catalog_entry(entry)

    entry = extract_critical_segment(_episode())
    entry["replay"]["warnings"] = []
    with pytest.raises(GeneratedScenarioCatalogValidationError, match="replay_gap"):
        validate_catalog_entry(entry)


def test_validator_rejects_an_incomplete_review_verdict() -> None:
    """A recorded review cannot omit the one-line reviewer reason."""

    entry = extract_critical_segment(_episode())
    entry["provenance"] = {
        **entry["provenance"],
        "reviewed": True,
        "review": {"verdict": "certified"},
    }

    with pytest.raises(GeneratedScenarioCatalogValidationError, match="reason"):
        validate_catalog_entry(entry)


def test_distiller_requires_pedestrian_trace_for_minimum_clearance() -> None:
    """The min-clearance extractor rejects traces without an observable counterpart."""

    episode = deepcopy(_episode())
    for step in episode["steps"]:
        step["pedestrians"] = []

    with pytest.raises(ValueError, match="no pedestrian positions"):
        extract_critical_segment(episode)


def test_cli_writes_validated_catalog_entry(tmp_path: Path) -> None:
    """The narrow CLI turns a JSON trace into one review-pending entry."""

    input_path = tmp_path / "episode.json"
    output_path = tmp_path / "catalog-entry.json"
    input_path.write_text(json.dumps(_episode()), encoding="utf-8")
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/benchmark/distill_generated_scenario_segment.py"
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--episode-json",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    entry = json.loads(output_path.read_text(encoding="utf-8"))
    validate_catalog_entry(entry)
