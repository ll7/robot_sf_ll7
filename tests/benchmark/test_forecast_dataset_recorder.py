"""Tests for ForecastDataset.v1 recording and split manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.simulation_trace_export import (
    load_simulation_trace_export,
)
from robot_sf.benchmark.forecast_dataset_recorder import (
    FORECAST_DATASET_SCHEMA_VERSION,
    record_forecast_dataset_from_trace_exports,
    validate_forecast_dataset_manifest,
)

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
)
TRACE_FIXTURES = [
    FIXTURE_ROOT / "minimal_trace.json",
]


def _feature_schema() -> dict[str, object]:
    return {
        "name": "forecast_dataset_recorder_test_v1",
        "features": ["position_m", "velocity_mps"],
    }


def _write_trace_copy(tmp_path: Path, index: int) -> Path:
    """Write a valid trace fixture copy with distinct split metadata."""
    payload = load_simulation_trace_export(TRACE_FIXTURES[0]).to_dict()
    payload["trace_id"] = f"fixture_trace_{index:03d}"
    payload["source"]["scenario_id"] = f"fixture_scenario_{index:03d}"
    payload["source"]["seed"] = index
    payload["source"]["episode_id"] = f"fixture_episode_{index:03d}"
    path = tmp_path / f"fixture_trace_{index:03d}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_record_forecast_dataset_writes_jsonl_and_manifest(tmp_path: Path) -> None:
    """A tiny durable trace set should produce rows plus a validated manifest."""
    result = record_forecast_dataset_from_trace_exports(
        TRACE_FIXTURES[:1],
        tmp_path,
        feature_schema=_feature_schema(),
        horizons_s=[0.1],
        dataset_id="forecast_dataset_test",
    )

    assert result.dataset_path.exists()
    assert result.manifest_path.exists()
    assert result.manifest["schema_version"] == FORECAST_DATASET_SCHEMA_VERSION
    assert result.manifest["example_count"] > 0
    rows = [
        json.loads(line) for line in result.dataset_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["observation_tier"] == "oracle_full_state"
    assert rows[0]["oracle_state"] is True
    assert rows[0]["map_id"] == rows[0]["scenario_id"]
    assert rows[0]["map_id_source"] == "simulation_trace_export.source.scenario_id"
    assert rows[0]["feature_schema"]["name"] == "forecast_dataset_recorder_test_v1"
    assert rows[0]["label"]["source"] == "simulation_trace_export.pedestrians"


def test_record_forecast_dataset_manifest_has_train_validation_test_splits(
    tmp_path: Path,
) -> None:
    """Split metadata should expose all three split names without leakage."""
    result = record_forecast_dataset_from_trace_exports(
        [_write_trace_copy(tmp_path, index) for index in range(3)],
        tmp_path,
        feature_schema=_feature_schema(),
        horizons_s=[0.1],
        dataset_id="forecast_dataset_split_test",
    )

    validate_forecast_dataset_manifest(result.manifest)
    assert set(result.manifest["splits"]) == {"train", "validation", "test"}
    split_trace_counts = {
        split: payload["trace_count"] for split, payload in result.manifest["splits"].items()
    }
    assert split_trace_counts["train"] >= 1
    assert split_trace_counts["validation"] == 1
    assert split_trace_counts["test"] == 1


def test_validate_manifest_rejects_scenario_leakage() -> None:
    """A scenario cannot appear in more than one split."""
    manifest = {
        "schema_version": FORECAST_DATASET_SCHEMA_VERSION,
        "dataset_id": "leaky",
        "example_count": 2,
        "examples_path": "leaky.jsonl",
        "feature_schema": _feature_schema(),
        "splits": {
            "train": {
                "scenario_ids": ["same"],
                "scenario_seed_keys": ["same:1"],
            },
            "validation": {
                "scenario_ids": ["same"],
                "scenario_seed_keys": ["same:2"],
            },
            "test": {
                "scenario_ids": [],
                "scenario_seed_keys": [],
            },
        },
    }

    with pytest.raises(ValueError, match="scenario_ids leakage"):
        validate_forecast_dataset_manifest(manifest)


def test_validate_manifest_rejects_scenario_seed_leakage() -> None:
    """A scenario/seed key cannot appear in more than one split."""
    manifest = {
        "schema_version": FORECAST_DATASET_SCHEMA_VERSION,
        "dataset_id": "leaky-seed",
        "example_count": 2,
        "examples_path": "leaky.jsonl",
        "feature_schema": _feature_schema(),
        "splits": {
            "train": {
                "scenario_ids": ["a"],
                "scenario_seed_keys": ["a:1"],
            },
            "validation": {
                "scenario_ids": ["b"],
                "scenario_seed_keys": ["a:1"],
            },
            "test": {
                "scenario_ids": [],
                "scenario_seed_keys": [],
            },
        },
    }

    with pytest.raises(ValueError, match="scenario_seed_keys leakage"):
        validate_forecast_dataset_manifest(manifest)


def test_recorder_rejects_missing_feature_schema(tmp_path: Path) -> None:
    """Dataset rows must not be emitted without feature-schema provenance."""
    with pytest.raises(ValueError, match="feature_schema is required"):
        record_forecast_dataset_from_trace_exports(
            TRACE_FIXTURES[:1],
            tmp_path,
            feature_schema={},
            horizons_s=[0.1],
        )


def test_recorder_rows_use_trace_source_metadata(tmp_path: Path) -> None:
    """Rows should preserve scenario, seed, planner, and episode metadata."""
    result = record_forecast_dataset_from_trace_exports(
        [TRACE_FIXTURES[0]],
        tmp_path,
        feature_schema=_feature_schema(),
        horizons_s=[0.1],
    )
    trace = load_simulation_trace_export(TRACE_FIXTURES[0])
    first_row = json.loads(result.dataset_path.read_text(encoding="utf-8").splitlines()[0])

    assert first_row["scenario_id"] == trace.source.scenario_id
    assert first_row["map_id"] == trace.source.scenario_id
    assert first_row["seed"] == trace.source.seed
    assert first_row["planner_id"] == trace.source.planner_id
    assert first_row["episode_id"] == trace.source.episode_id
