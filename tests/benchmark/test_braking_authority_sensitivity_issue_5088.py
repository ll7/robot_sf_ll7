"""Contract tests for issue #5088's reproducible braking-authority smoke."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from robot_sf.benchmark.braking_authority_sensitivity import (
    analyze_smoke_results,
    load_smoke_config,
    materialize_arm_scenario,
    validate_smoke_config,
    write_report,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmarks/issue_5088_braking_authority_sensitivity_smoke.yaml"


def test_config_commits_fixed_scenario_seeds_planner_and_two_authorities() -> None:
    """The smoke must freeze every comparator input except braking authority."""
    config = load_smoke_config(CONFIG)

    assert config["scenario"] == {
        "path": "configs/scenarios/classic_interactions.yaml",
        "name": "classic_cross_trap_high",
    }
    assert config["seeds"] == [101, 102, 103]
    assert config["planner"] == {
        "algo": "social_force",
        "benchmark_profile": "baseline-safe",
    }
    assert [arm["max_linear_decel_m_s2"] for arm in config["arms"]] == [0.25, 2.0]
    assert config["run"]["workers"] == 1
    assert config["run"]["record_simulation_step_trace"] is True
    assert config["signal"]["minimum_valid_seeds"] == 2


def test_materialized_arms_only_change_braking_authority_and_arm_metadata() -> None:
    """Arm materialization protects the controlled-comparison contract."""
    source = {
        "name": "crossing",
        "map_id": "classic_cross_trap",
        "simulation_config": {"max_episode_steps": 100, "ped_density": 0.08},
        "robot_config": {"max_linear_accel": 1.0},
        "metadata": {"archetype": "crossing"},
        "seeds": [999],
    }
    weak = materialize_arm_scenario(
        source,
        arm={"key": "weak", "max_linear_decel_m_s2": 0.25},
        seeds=[101, 102],
    )
    strong = materialize_arm_scenario(
        source,
        arm={"key": "strong", "max_linear_decel_m_s2": 2.0},
        seeds=[101, 102],
    )

    weak_compare = deepcopy(weak)
    strong_compare = deepcopy(strong)
    weak_compare["robot_config"].pop("max_linear_decel")
    strong_compare["robot_config"].pop("max_linear_decel")
    weak_compare["metadata"].pop("braking_authority_sensitivity_arm")
    strong_compare["metadata"].pop("braking_authority_sensitivity_arm")
    assert weak_compare == strong_compare
    assert weak["robot_config"]["max_linear_decel"] == pytest.approx(0.25)
    assert strong["robot_config"]["max_linear_decel"] == pytest.approx(2.0)
    assert weak["seeds"] == strong["seeds"] == [101, 102]


def test_config_validation_fails_closed_on_invalid_control_contracts() -> None:
    """Malformed comparison controls must fail before any episode executes."""
    valid = load_smoke_config(CONFIG)
    cases = [
        (("schema_version",), "wrong", "schema_version"),
        (("issue",), 1, "issue must be 5088"),
        (("claim_boundary",), "benchmark success", "claim_boundary"),
        (("scenario", "name"), "", "scenario.path and scenario.name"),
        (("seeds",), [101, 101], "seeds must be unique"),
        (("planner", "benchmark_profile"), "unsafe", "benchmark_profile"),
        (("run", "workers"), 2, "workers must be 1"),
        (("signal", "metrics"), ["unsupported"], "signal.metrics"),
    ]

    for path, value, match in cases:
        invalid = deepcopy(valid)
        target = invalid
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match=match):
            validate_smoke_config(invalid)


def test_report_detects_metric_signal_and_preserves_runtime_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A changed metric activates the smoke without upgrading its evidence tier."""
    config = load_smoke_config(CONFIG)
    config["seeds"] = [101]
    config["signal"]["minimum_valid_seeds"] = 1
    records = {
        "weak_braking": [_record(101, near_misses=2, min_clearance=0.1)],
        "strong_braking": [_record(101, near_misses=1, min_clearance=0.5)],
    }
    summaries = {
        "weak_braking": _summary(distance=8.0),
        "strong_braking": _summary(distance=1.0),
    }
    monkeypatch.setattr(
        "robot_sf.benchmark.braking_authority_sensitivity._ttc_from_record",
        lambda record: (float(record["diagnostic_ttc"]), "available"),
    )

    report = analyze_smoke_results(
        config,
        arm_records=records,
        arm_summaries=summaries,
        config_path=CONFIG.relative_to(ROOT).as_posix(),
        run_commit="abc123",
        raw_artifact_root="output/issue_5088",
        raw_artifacts={
            "weak_braking": {"path": "weak.jsonl", "sha256": "a", "row_count": 1},
            "strong_braking": {"path": "strong.jsonl", "sha256": "b", "row_count": 1},
        },
        reproduction_command="uv run python scripts/tools/run_braking_authority_sensitivity_smoke.py",
    )

    assert report["status"] == "signal_activated"
    assert report["evidence_tier"] == "targeted-smoke"
    assert report["comparison"]["signal_activated"] is True
    assert set(report["comparison"]["activated_metrics"]) == {
        "near_misses",
        "min_clearance",
        "time_to_collision_min",
    }
    assert report["comparison"]["changed_seed_counts"] == {
        "near_misses": 1,
        "min_clearance": 1,
        "time_to_collision_min": 1,
    }
    assert report["raw_artifact_classification"] == "local-scratch; not durable evidence"

    write_report(report, tmp_path)
    markdown = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Metric-sensitivity signal activated: `True`" in markdown
    assert "Changed-seed counts:" in markdown
    assert "## Reproduce" in markdown
    assert "Replace `<fresh-artifact-dir>` with an empty local scratch directory." in markdown
    assert (tmp_path / "report.json").is_file()
    checksum_lines = (tmp_path / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert [line.split(maxsplit=1)[1] for line in checksum_lines] == [
        "README.md",
        "report.json",
    ]


def test_report_fails_closed_for_degraded_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback or degraded arms can never become successful smoke evidence."""
    config = load_smoke_config(CONFIG)
    config["seeds"] = [101]
    config["signal"]["minimum_valid_seeds"] = 1
    records = {
        "weak_braking": [_record(101, near_misses=2, min_clearance=0.1)],
        "strong_braking": [_record(101, near_misses=1, min_clearance=0.5)],
    }
    summaries = {
        "weak_braking": _summary(distance=8.0),
        "strong_braking": _summary(distance=1.0, readiness="degraded", success=False),
    }
    monkeypatch.setattr(
        "robot_sf.benchmark.braking_authority_sensitivity._ttc_from_record",
        lambda record: (float(record["diagnostic_ttc"]), "available"),
    )

    with pytest.raises(ValueError, match="non-evidence"):
        analyze_smoke_results(
            config,
            arm_records=records,
            arm_summaries=summaries,
            config_path=CONFIG.relative_to(ROOT).as_posix(),
            run_commit="abc123",
            raw_artifact_root="output/issue_5088",
            raw_artifacts={
                "weak_braking": {"path": "weak.jsonl", "sha256": "a", "row_count": 1},
                "strong_braking": {"path": "strong.jsonl", "sha256": "b", "row_count": 1},
            },
            reproduction_command=(
                "uv run python scripts/tools/run_braking_authority_sensitivity_smoke.py"
            ),
        )


def _record(seed: int, *, near_misses: int, min_clearance: float) -> dict[str, object]:
    return {
        "seed": seed,
        "status": "failure",
        "metrics": {"near_misses": near_misses, "min_clearance": min_clearance},
        "algorithm_metadata": {},
        "diagnostic_ttc": 0.5 if near_misses > 1 else 1.5,
    }


def _summary(
    *, distance: float, readiness: str = "adapter", success: bool = True
) -> dict[str, object]:
    return {
        "benchmark_availability": {
            "execution_mode": "adapter",
            "readiness_status": readiness,
            "availability_status": "available" if success else "not_available",
            "benchmark_success": success,
        },
        "provenance": {
            "config_identity": {
                "metric_affecting_config": {
                    "actuation_envelope": {
                        "max_braking_decel_m_s2": 0.25 if distance == 8.0 else 2.0,
                        "stopping_distance_envelope_m": distance,
                    }
                }
            }
        },
    }
