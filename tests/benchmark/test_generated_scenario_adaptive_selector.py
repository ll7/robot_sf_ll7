"""Tests for #4932 adaptive proposal selection over generated hypotheses."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from robot_sf.benchmark.scenario_generation.adaptive_selector import (
    AdaptiveSelectionSpec,
    GeneratedScenarioAdaptiveSelectionError,
    run_adaptive_selection,
    select_generated_proposals,
)
from robot_sf.benchmark.scenario_generation.segment_extraction import extract_critical_segment

if TYPE_CHECKING:
    from pathlib import Path


def _entry(episode_id: str, clearance_m: float, observed_at_s: float = 1.0) -> dict[str, Any]:
    entry = extract_critical_segment(
        {
            "episode_id": episode_id,
            "seed": 4932,
            "source_map": "maps/svg_maps/classic_crossing.svg",
            "steps": [
                {
                    "time_s": 0.0,
                    "robot": {"position": [0.0, 0.0]},
                    "pedestrians": [{"position": [3.0, 0.0]}],
                },
                {
                    "time_s": 1.0,
                    "robot": {"position": [1.0, 0.0]},
                    "pedestrians": [{"position": [1.0 + clearance_m, 0.0]}],
                },
                {
                    "time_s": 2.0,
                    "robot": {"position": [2.0, 0.0]},
                    "pedestrians": [{"position": [5.0, 0.0]}],
                },
            ],
        }
    )
    entry["criticality"]["observed_at_s"] = observed_at_s
    return entry


def _spec(*, proposal_count: int = 2) -> AdaptiveSelectionSpec:
    return AdaptiveSelectionSpec.from_payload(
        {
            "schema_version": "generated-scenario-adaptive-proposal-selection.v1",
            "proposal_count": proposal_count,
            "selector": {
                "type": "adaptive_min_max_rank.v1",
                "criteria": [
                    {
                        "field": "criticality.source_metrics.min_clearance_m",
                        "direction": "lower_is_better",
                        "weight": 3.0,
                    },
                    {
                        "field": "criticality.observed_at_s",
                        "direction": "higher_is_better",
                        "weight": 1.0,
                    },
                ],
            },
            "claim_boundary": "generated scenario hypotheses only",
        }
    )


def _write_config_and_archive(tmp_path: Path) -> tuple[Path, Path, Path]:
    archive_path = tmp_path / "archive.yaml"
    archive_payload = {
        "schema_version": "generated-scenario-catalog.v1",
        "metadata": {
            "source": "auto_generated",
            "required_manual_review": True,
            "benchmark_evidence": False,
        },
        "entries": [
            _entry("critical", 0.1, 0.5),
            _entry("balanced", 0.5, 1.0),
            _entry("late", 1.0, 1.5),
        ],
    }
    archive_path.write_text(yaml.safe_dump(archive_payload, sort_keys=True), encoding="utf-8")
    output_path = tmp_path / "selection.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "generated-scenario-adaptive-proposal-selection.v1",
                "source_archive": archive_path.as_posix(),
                "proposal_count": 2,
                "selector": {
                    "type": "adaptive_min_max_rank.v1",
                    "criteria": [
                        {
                            "field": "criticality.source_metrics.min_clearance_m",
                            "direction": "lower_is_better",
                            "weight": 1.0,
                        }
                    ],
                },
                "output_path": output_path.as_posix(),
                "claim_boundary": "generated scenario hypotheses only",
            }
        ),
        encoding="utf-8",
    )
    return config_path, archive_path, output_path


def test_selection_is_deterministic_order_independent_and_configurable() -> None:
    """Weighted criteria rank identically regardless of archive record ordering."""

    entries = [
        _entry("critical", 0.1, 0.5),
        _entry("balanced", 0.5, 1.0),
        _entry("late", 1.0, 1.5),
    ]
    first = select_generated_proposals(entries, spec=_spec())
    second = select_generated_proposals(list(reversed(entries)), spec=_spec())

    assert first == second
    assert [row["entry"]["source_episode"]["episode_id"] for row in first["selected"]] == [
        "critical",
        "balanced",
    ]
    assert first["normalization_ranges"]["criticality.observed_at_s"] == {
        "min": 0.5,
        "max": 1.5,
    }
    clearance_range = first["normalization_ranges"]["criticality.source_metrics.min_clearance_m"]
    assert clearance_range["min"] == pytest.approx(0.1)
    assert clearance_range["max"] == pytest.approx(1.0)
    assert all(row["entry"]["metadata"]["benchmark_evidence"] is False for row in first["selected"])


def test_score_ties_use_stable_scenario_ids() -> None:
    """Equal archive-adaptive scores use stable ids rather than input ordering."""

    entries = [_entry("first", 0.5), _entry("second", 0.5)]
    selected = select_generated_proposals(entries, spec=_spec(proposal_count=2))["selected"]

    assert [row["scenario_id"] for row in selected] == sorted(
        entry["scenario_id"] for entry in entries
    )
    assert all(row["score"] == 1.0 for row in selected)


def test_missing_configured_field_fails_closed() -> None:
    """A scoring typo cannot silently drop a hypothesis or criterion."""

    payload = {
        "schema_version": "generated-scenario-adaptive-proposal-selection.v1",
        "proposal_count": 1,
        "selector": {
            "type": "adaptive_min_max_rank.v1",
            "criteria": [
                {
                    "field": "criticality.source_metrics.missing_metric",
                    "direction": "higher_is_better",
                    "weight": 1.0,
                }
            ],
        },
        "claim_boundary": "generated scenario hypotheses only",
    }

    with pytest.raises(GeneratedScenarioAdaptiveSelectionError, match="missing configured"):
        select_generated_proposals(
            [_entry("candidate", 0.2)], spec=AdaptiveSelectionSpec.from_payload(payload)
        )


def test_malformed_entries_and_invalid_scoring_specs_fail_closed() -> None:
    """Governance and scoring controls cannot degrade to permissive defaults."""

    malformed = deepcopy(_entry("candidate", 0.2))
    malformed["metadata"]["benchmark_evidence"] = True
    with pytest.raises(GeneratedScenarioAdaptiveSelectionError, match="entry 0 is invalid"):
        select_generated_proposals([malformed], spec=_spec(proposal_count=1))

    payload = {
        "schema_version": "generated-scenario-adaptive-proposal-selection.v1",
        "proposal_count": 1,
        "selector": {
            "type": "adaptive_min_max_rank.v1",
            "criteria": [
                {
                    "field": "criticality.observed_at_s",
                    "direction": "higher_is_better",
                    "weight": float("nan"),
                }
            ],
        },
        "claim_boundary": "generated scenario hypotheses only",
    }
    with pytest.raises(GeneratedScenarioAdaptiveSelectionError, match="finite and > 0"):
        AdaptiveSelectionSpec.from_payload(payload)


def test_run_persists_scores_source_provenance_and_review_only_governance(tmp_path: Path) -> None:
    """Standalone selection records inputs, scoring decisions, and claim boundaries."""

    config_path, archive_path, output_path = _write_config_and_archive(tmp_path)
    result = run_adaptive_selection(config_path)
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted == result
    assert result["schema_version"] == "generated-scenario-adaptive-proposal-selection-result.v1"
    assert (
        result["source_archive"]["sha256"] == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    )
    assert result["selector"]["normalization"] == "current_archive_min_max.v1"
    assert len(result["scored_candidates"]) == 3
    assert all(row["score_components"] for row in result["scored_candidates"])
    assert result["governance"] == {
        "required_manual_review": True,
        "benchmark_evidence": False,
        "scenario_certification": False,
        "automatic_promotion": False,
    }

    with pytest.raises(FileExistsError, match="already exists"):
        run_adaptive_selection(config_path)


def test_run_rejects_non_generated_archive_metadata(tmp_path: Path) -> None:
    """Hand-authored or evidence-bearing archives cannot enter the selector."""

    config_path, archive_path, _output_path = _write_config_and_archive(tmp_path)
    archive = yaml.safe_load(archive_path.read_text(encoding="utf-8"))
    archive["metadata"]["source"] = "hand_authored"
    archive_path.write_text(yaml.safe_dump(archive), encoding="utf-8")

    with pytest.raises(GeneratedScenarioAdaptiveSelectionError, match="metadata.source"):
        run_adaptive_selection(config_path)
