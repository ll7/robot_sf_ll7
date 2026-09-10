"""Loader and semantic validator for the shared SNQI example fixture."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from robot_sf.benchmark.aggregate import compute_aggregates, filter_evidence_eligible_records
from robot_sf.benchmark.snqi import compute_snqi, validate_weights_mapping

FIXTURE_PATH = Path(__file__).with_name("fixture.json")
CLAIM_BOUNDARY = "diagnostic-only; synthetic fixture output is not benchmark or scientific evidence"


def load_fixture() -> dict[str, Any]:
    """Load the immutable JSON fixture and verify its semantic contract."""
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if fixture["fixture_version"] != "snqi-example-fixture.v1":
        raise ValueError("unsupported SNQI fixture version")
    if fixture["provenance"]["source_kind"] != "synthetic_fixture":
        raise ValueError("fixture provenance must identify synthetic_fixture")
    boundary = fixture["provenance"]["claim_boundary"].lower()
    if (
        fixture["provenance"]["evidence_status"] != "diagnostic-only"
        or "not benchmark" not in boundary
    ):
        raise ValueError("fixture claim boundary must remain diagnostic-only")
    validate_weights_mapping(fixture["weights"])
    records = fixture["episodes"]
    scores = [
        compute_snqi(row["metrics"], fixture["weights"], fixture["baseline"]) for row in records
    ]
    expected = fixture["expected"]
    components = expected["components"]
    first = records[0]["metrics"]
    actual_components = {
        "success": float(first["success"]),
        "time_penalty": fixture["weights"]["w_time"] * first["time_to_goal_norm"],
        "collision_penalty": fixture["weights"]["w_collisions"] * first["collisions"],
        "near_miss_penalty": fixture["weights"]["w_near"] * first["near_misses"],
        "comfort_penalty": fixture["weights"]["w_comfort"] * first["comfort_exposure"],
        "force_penalty": fixture["weights"]["w_force_exceed"] * first["force_exceed_events"],
        "jerk_penalty": fixture["weights"]["w_jerk"] * first["jerk_mean"],
    }
    if actual_components != components or not math.isclose(scores[0], expected["tie_score"]):
        raise ValueError("fixture SNQI component or score contract drifted")
    if scores[0] != scores[1]:
        raise ValueError("fixture exact tie was not preserved")
    eligible, excluded = filter_evidence_eligible_records(records)
    denominators = fixture["declared_denominators"]
    if (len(records), len(eligible), excluded) != (
        denominators["input_records"],
        denominators["eligible_records"],
        denominators["excluded_records"],
    ):
        raise ValueError("fixture denominator contract drifted")
    excluded_ids = [r["episode_id"] for r in records if r not in eligible]
    if excluded_ids != expected["excluded_episode_ids"]:
        raise ValueError("fixture exclusion contract drifted")
    aggregate = compute_aggregates(
        records,
        snqi_weights=fixture["weights"],
        snqi_baseline=fixture["baseline"],
        recompute_snqi=True,
    )
    if aggregate["_meta"]["evidence_eligibility"]["excluded_record_count"] != excluded:
        raise ValueError("canonical aggregate exclusion count drifted")
    return fixture


def materialize_inputs(out_dir: Path) -> tuple[Path, Path, Path]:
    """Write fixture inputs into a caller-owned directory for CLI consumers."""
    fixture = load_fixture()
    inputs = out_dir / "fixture_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    episodes = inputs / "episodes.jsonl"
    episodes.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in fixture["episodes"]
        ),
        encoding="utf-8",
    )
    weights = inputs / "weights.json"
    weights.write_text(
        json.dumps(fixture["weights"], sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    baseline = inputs / "baseline.json"
    baseline.write_text(
        json.dumps(fixture["baseline"], sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return episodes, weights, baseline


def write_summary(out_dir: Path, *, example: str, output_files: list[str]) -> Path:
    """Write a bounded, byte-stable semantic summary for a fixture run."""
    fixture = load_fixture()
    summary = {
        "schema_version": "snqi-example-summary.v1",
        "example": example,
        "fixture_version": fixture["fixture_version"],
        "provenance": fixture["provenance"],
        "declared_denominators": fixture["declared_denominators"],
        "expected_components": fixture["expected"]["components"],
        "tie_score": fixture["expected"]["tie_score"],
        "excluded_episode_ids": fixture["expected"]["excluded_episode_ids"],
        "outputs": sorted(output_files),
    }
    path = out_dir / "snqi_fixture_summary.json"
    path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path
