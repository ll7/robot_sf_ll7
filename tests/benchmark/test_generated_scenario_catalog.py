"""Tests for issue #4932's review-pending generated scenario catalog contract."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from robot_sf.benchmark.scenario_generation import (
    GeneratedScenarioCatalogValidationError,
    extract_critical_segment,
    validate_catalog_entry,
)
from robot_sf.benchmark.scenario_generation.catalog_schema import load_catalog_entry_schema


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
    validate_catalog_entry(entry)


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
