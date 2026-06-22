"""Tests for compact adversarial manifest quality summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_sf.adversarial.config import CandidateSpec, Pose2D
from robot_sf.adversarial.manifest_quality import (
    MANIFEST_QUALITY_SCHEMA_VERSION,
    _collect_paths,
    _control_vector,
    _load_records,
    _read_text_jsonl,
    _safe_float,
    _safe_int,
    _safe_rate,
    load_adversarial_manifest_quality_records,
    summarize_adversarial_manifest_quality,
    summarize_adversarial_manifest_quality_records,
)
from robot_sf.adversarial.manifest_quality import main as quality_cli_main
from robot_sf.adversarial.scenario_manifest import compute_control_hash


def _candidate_controls(
    *,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    scenario_seed: int = 7,
) -> dict:
    return {
        "start": {"x": float(start_x), "y": float(start_y)},
        "goal": {"x": float(goal_x), "y": float(goal_y)},
        "spawn_time_s": 0.0,
        "pedestrian_speed_mps": 1.0,
        "pedestrian_delay_s": 0.0,
        "scenario_seed": scenario_seed,
    }


def _naturalistic_prior_payload(*, passed: bool, flags: list[str] | None = None) -> dict:
    return {
        "schema_version": "naturalistic_vru_prior.v1",
        "profile": "urban_vru_default_v1",
        "constraints": [
            {
                "field": "pedestrian_speed_mps",
                "min": 0.4,
                "max": 2.2,
                "observed": 1.0 if passed else 3.5,
                "passed": passed,
                "description": "bounded walking-to-running VRU speed for plausible hard cases",
            }
        ],
        "passed": passed,
        "violation_flags": flags or [],
    }


def _manifest_payload(
    controls: dict,
    status: str,
    *,
    naturalistic_prior: dict | None = None,
) -> dict:
    candidate = CandidateSpec(
        start=Pose2D(controls["start"]["x"], controls["start"]["y"]),
        goal=Pose2D(controls["goal"]["x"], controls["goal"]["y"]),
        spawn_time_s=float(controls["spawn_time_s"]),
        pedestrian_speed_mps=float(controls["pedestrian_speed_mps"]),
        pedestrian_delay_s=float(controls["pedestrian_delay_s"]),
        scenario_seed=int(controls["scenario_seed"]),
    )
    payload = {
        "schema_version": "adversarial_scenario_manifest.v1",
        "candidate_controls": controls,
        "validation": {
            "status": status,
            "errors": [],
            "warnings": [],
            "normalized_control_hash": compute_control_hash(candidate),
        },
    }
    if naturalistic_prior is not None:
        payload["naturalistic_prior"] = naturalistic_prior
    return payload


def _write_manifest(
    path: Path,
    controls: dict,
    status: str,
    *,
    naturalistic_prior: dict | None = None,
) -> None:
    path.write_text(
        yaml.safe_dump(
            _manifest_payload(controls, status, naturalistic_prior=naturalistic_prior),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _write_manifest_without_schema(path: Path, controls: dict, status: str) -> None:
    payload = _manifest_payload(controls, status)
    del payload["schema_version"]
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def test_safe_coercion_helpers_reject_invalid_values() -> None:
    """Safe coercion helpers normalize finite numeric inputs and reject ambiguous values."""
    assert _safe_float("1.25") == 1.25
    assert _safe_float(None) is None
    assert _safe_float("not-a-number") is None
    assert _safe_float(float("inf")) is None
    assert _safe_float(float("nan")) is None

    assert _safe_int(3) == 3
    assert _safe_int("4.0") == 4
    assert _safe_int(True) is None
    assert _safe_int(None) is None
    assert _safe_int("4.2") is None
    assert _safe_int(float("nan")) is None

    assert _safe_rate(1, 0) == 0.0
    assert _safe_rate(1, 3) == 0.333333


def test_jsonl_reader_ignores_missing_malformed_and_non_object_rows(tmp_path: Path) -> None:
    """Planner smoke JSONL loading must tolerate partial or malformed diagnostic logs."""
    assert _read_text_jsonl(tmp_path / "missing.jsonl") == []

    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                "",
                '{"status": "success"}',
                "not-json",
                '["not", "an", "object"]',
                '{"metrics": {"near_misses": 1}}',
            ]
        ),
        encoding="utf-8",
    )

    assert _read_text_jsonl(jsonl_path) == [
        {"status": "success"},
        {"metrics": {"near_misses": 1}},
    ]


def test_collect_paths_reports_bad_inputs_and_discovers_yaml(tmp_path: Path) -> None:
    """Path collection should fail clearly for bad inputs and recurse over YAML manifests."""
    with pytest.raises(FileNotFoundError, match="Manifest input missing"):
        _collect_paths([tmp_path / "missing"])

    text_file = tmp_path / "not_manifest.txt"
    text_file.write_text("not yaml", encoding="utf-8")
    with pytest.raises(ValueError, match=r"must use \.yml or \.yaml"):
        _collect_paths([text_file])

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No manifest files discovered"):
        _collect_paths([empty_dir])

    nested = tmp_path / "nested"
    nested.mkdir()
    yaml_path = nested / "a.yaml"
    yml_path = nested / "b.yml"
    yaml_path.write_text("{}", encoding="utf-8")
    yml_path.write_text("{}", encoding="utf-8")

    assert _collect_paths([tmp_path]) == [yaml_path, yml_path]


def test_control_vector_rejects_missing_or_non_finite_controls() -> None:
    """Perturbation vectors should be omitted when controls are incomplete or non-finite."""
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)

    assert _control_vector("not-a-mapping") is None
    assert _control_vector({**controls, "spawn_time_s": "nan"}) is None

    incomplete = dict(controls)
    del incomplete["goal"]
    assert _control_vector(incomplete) is None


def test_load_records_captures_parse_error_and_hash_fallback(tmp_path: Path) -> None:
    """Record loading should keep bad manifests visible and compute hashes when omitted."""
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    fallback_hash_path = tmp_path / "fallback_hash.yaml"
    payload = _manifest_payload(controls, "valid")
    del payload["validation"]["normalized_control_hash"]
    fallback_hash_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(
        yaml.safe_dump({"candidate_controls": [], "validation": {"status": "valid"}}),
        encoding="utf-8",
    )

    records, parse_errors = _load_records(
        [fallback_hash_path, bad_path],
        reference_vector=None,
    )

    assert len(records) == 2
    assert records[0].status == "valid"
    assert records[0].normalized_control_hash is not None
    assert records[1].status == "invalid"
    assert records[1].parse_error is not None
    assert parse_errors == [f"{bad_path.as_posix()}: candidate_controls missing or not a mapping"]


def test_summarize_manifest_rates_and_novelty(tmp_path: Path) -> None:
    controls_a = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    controls_b = _candidate_controls(start_x=1.5, start_y=2.0, goal_x=5.0, goal_y=2.0)
    controls_c = _candidate_controls(start_x=2.0, start_y=2.0, goal_x=6.0, goal_y=2.0)

    _write_manifest(tmp_path / "a.yaml", controls_a, "valid")
    _write_manifest(tmp_path / "b.yaml", controls_b, "invalid")
    _write_manifest(tmp_path / "c.yaml", controls_c, "degenerate")
    _write_manifest(tmp_path / "d.yaml", controls_a, "valid")

    result = summarize_adversarial_manifest_quality([tmp_path])

    assert result.manifest_count == 4
    assert result.status_counts["valid"] == 2
    assert result.status_counts["invalid"] == 1
    assert result.status_counts["degenerate"] == 1
    assert result.validity_rate == 0.5
    assert result.invalid_rate == 0.25
    assert result.degenerate_rate == 0.25
    assert result.hashable_count == 4
    assert result.duplicate_hash_count == 1
    assert result.unique_hash_count == 3
    assert result.novelty_rate == 0.75
    assert result.duplicate_rate == 0.25


def test_summarize_naturalistic_prior_rates_and_filters(tmp_path: Path) -> None:
    controls_a = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    controls_b = _candidate_controls(start_x=1.5, start_y=2.0, goal_x=5.0, goal_y=2.0)
    controls_c = _candidate_controls(start_x=2.0, start_y=2.0, goal_x=6.0, goal_y=2.0)

    violation_flag = "pedestrian_speed_mps_outside_urban_vru_default_v1"
    _write_manifest(
        tmp_path / "passed.yaml",
        controls_a,
        "valid",
        naturalistic_prior=_naturalistic_prior_payload(passed=True),
    )
    _write_manifest(
        tmp_path / "violated.yaml",
        controls_b,
        "valid",
        naturalistic_prior=_naturalistic_prior_payload(passed=False, flags=[violation_flag]),
    )
    _write_manifest(tmp_path / "legacy.yaml", controls_c, "valid")

    result = summarize_adversarial_manifest_quality([tmp_path])

    assert result.naturalistic_prior_available_count == 2
    assert result.naturalistic_prior_pass_count == 1
    assert result.naturalistic_prior_fail_count == 1
    assert result.naturalistic_prior_unavailable_count == 1
    assert result.naturalistic_prior_pass_rate == 0.5
    assert result.naturalistic_prior_fail_rate == 0.5
    assert result.naturalistic_prior_violation_counts == {violation_flag: 1}

    passed = summarize_adversarial_manifest_quality([tmp_path], naturalistic_status="passed")
    violated = summarize_adversarial_manifest_quality([tmp_path], naturalistic_status="violated")
    missing = summarize_adversarial_manifest_quality([tmp_path], naturalistic_status="missing")

    assert passed.manifest_count == 1
    assert passed.naturalistic_prior_pass_count == 1
    assert violated.manifest_count == 1
    assert violated.naturalistic_prior_fail_count == 1
    assert missing.manifest_count == 1
    assert missing.naturalistic_prior_unavailable_count == 1


def test_naturalistic_status_filter_rejects_unknown_value(tmp_path: Path) -> None:
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    _write_manifest(tmp_path / "a.yaml", controls, "valid")

    with pytest.raises(ValueError, match="Unsupported naturalistic status"):
        summarize_adversarial_manifest_quality([tmp_path], naturalistic_status="other")


def test_public_record_loader_feeds_summary_without_path_reload(tmp_path: Path) -> None:
    controls_a = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    controls_b = _candidate_controls(start_x=1.5, start_y=2.0, goal_x=5.0, goal_y=2.0)
    _write_manifest(tmp_path / "a.yaml", controls_a, "valid")
    _write_manifest(tmp_path / "b.yaml", controls_b, "invalid")

    records = load_adversarial_manifest_quality_records([tmp_path])
    result = summarize_adversarial_manifest_quality_records(records)

    assert result.manifest_count == 2
    assert result.status_counts == {"valid": 1, "invalid": 1}
    assert result.validity_rate == 0.5


def test_missing_manifest_schema_version_stays_none(tmp_path: Path) -> None:
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    manifest_path = tmp_path / "a.yaml"
    _write_manifest_without_schema(manifest_path, controls, "valid")

    records, parse_errors = _load_records([manifest_path], reference_vector=None)

    assert parse_errors == []
    assert records[0].schema_version is None


def test_summarize_perturbation_distance(tmp_path: Path) -> None:
    base = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    moved = _candidate_controls(start_x=2.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    far = _candidate_controls(
        start_x=1.0,
        start_y=4.0,
        goal_x=5.0,
        goal_y=4.0,
        scenario_seed=999,
    )

    ref_path = tmp_path / "reference.yaml"
    _write_manifest(ref_path, base, "valid")

    _write_manifest(tmp_path / "moved.yaml", moved, "valid")
    _write_manifest(tmp_path / "far.yaml", far, "valid")

    result = summarize_adversarial_manifest_quality(
        [tmp_path],
        reference_manifest=ref_path,
    )

    assert result.perturbation_reference == ref_path.as_posix()
    assert result.perturbation_count == 2
    assert result.perturbation_min == 1.0
    assert result.perturbation_max == 2.828427
    assert result.perturbation_mean == 1.914214


def test_reference_manifest_exclusion_normalizes_equivalent_paths(tmp_path: Path) -> None:
    base = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    moved = _candidate_controls(start_x=2.0, start_y=2.0, goal_x=5.0, goal_y=2.0)

    ref_path = tmp_path / "reference.yaml"
    _write_manifest(ref_path, base, "valid")
    _write_manifest(tmp_path / "moved.yaml", moved, "valid")
    (tmp_path / "subdir").mkdir()

    result = summarize_adversarial_manifest_quality(
        [tmp_path],
        reference_manifest=tmp_path / "subdir" / ".." / "reference.yaml",
    )

    assert result.manifest_count == 1
    assert result.perturbation_count == 1
    assert result.perturbation_min == 1.0


def test_summarize_planner_yields_from_smoke_summary(tmp_path: Path) -> None:
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    _write_manifest(tmp_path / "a.yaml", controls, "valid")

    episodes = tmp_path / "episodes_goal.jsonl"
    episodes.write_text(
        "\n".join(
            [
                '{"status": "success", "termination_reason": "success", "metrics": {"near_misses": 0}}',
                '{"status": "collision", "termination_reason": "collision", "metrics": {"near_misses": 2}}',
                '{"status": "truncated", "termination_reason": "truncated", "metrics": {"near_misses": 1}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    smoke_summary = tmp_path / "smoke_summary.json"
    smoke_summary.write_text(
        '{"planner_runs": [{"planner": "goal", "out_path": "' + episodes.as_posix() + '"}]}',
        encoding="utf-8",
    )

    result = summarize_adversarial_manifest_quality(
        [tmp_path],
        smoke_summary_json=smoke_summary,
    )

    assert result.planner_outcomes is not None
    planners = result.planner_outcomes.planners
    assert len(planners) == 1
    assert planners[0].failure_count == 2
    assert planners[0].near_miss_count == 2
    assert planners[0].failure_yield == pytest.approx(2 / 3)
    assert planners[0].near_miss_yield == pytest.approx(2 / 3)


def test_summarize_planner_yields_from_aggregate_smoke_summary(tmp_path: Path) -> None:
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    _write_manifest(tmp_path / "a.yaml", controls, "valid")

    smoke_summary = tmp_path / "smoke_summary.json"
    smoke_summary.write_text(
        json.dumps(
            {
                "planner_runs": [
                    {
                        "planner": "social_force",
                        "written": 2,
                        "total_jobs": 2,
                        "metrics": {
                            "episodes": 2,
                            "success": {"sum": 0.0},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = summarize_adversarial_manifest_quality(
        [tmp_path],
        smoke_summary_json=smoke_summary,
    )

    assert result.planner_outcomes is not None
    planners = result.planner_outcomes.planners
    assert result.planner_outcomes.available is True
    assert len(planners) == 1
    assert planners[0].source == "aggregate_metrics"
    assert planners[0].episodes == 2
    assert planners[0].failure_count == 2
    assert planners[0].failure_yield == 1.0
    assert planners[0].near_miss_count is None
    assert planners[0].near_miss_yield is None


def test_aggregate_success_yield_requires_count_like_sum(tmp_path: Path) -> None:
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    _write_manifest(tmp_path / "a.yaml", controls, "valid")

    smoke_summary = tmp_path / "smoke_summary.json"
    smoke_summary.write_text(
        json.dumps(
            {
                "planner_runs": [
                    {
                        "planner": "ambiguous",
                        "written": 2,
                        "metrics": {
                            "episodes": 2,
                            "success": {"sum": 1.5},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = summarize_adversarial_manifest_quality(
        [tmp_path],
        smoke_summary_json=smoke_summary,
    )

    assert result.planner_outcomes is not None
    assert result.planner_outcomes.available is True
    planner = result.planner_outcomes.planners[0]
    assert planner.source == "aggregate_metrics"
    assert planner.episodes == 2
    assert planner.failure_count is None
    assert planner.failure_yield is None


def test_aggregate_episode_count_preserves_written_zero(tmp_path: Path) -> None:
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    _write_manifest(tmp_path / "a.yaml", controls, "valid")

    smoke_summary = tmp_path / "smoke_summary.json"
    smoke_summary.write_text(
        json.dumps(
            {
                "planner_runs": [
                    {
                        "planner": "empty",
                        "written": 0,
                        "total_jobs": 2,
                        "metrics": {
                            "success": {"sum": 0.0},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = summarize_adversarial_manifest_quality(
        [tmp_path],
        smoke_summary_json=smoke_summary,
    )

    assert result.planner_outcomes is not None
    planner = result.planner_outcomes.planners[0]
    assert planner.episodes == 0
    assert planner.failure_count == 0
    assert planner.failure_yield == 0.0


def test_planner_summary_handles_missing_runs_empty_rows_and_missing_metrics(
    tmp_path: Path,
) -> None:
    """Planner diagnostics should expose unavailable and partial smoke-summary inputs."""
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    _write_manifest(tmp_path / "a.yaml", controls, "valid")

    no_runs_summary = tmp_path / "no_runs.json"
    no_runs_summary.write_text("{}", encoding="utf-8")
    no_runs_result = summarize_adversarial_manifest_quality(
        [tmp_path / "a.yaml"],
        smoke_summary_json=no_runs_summary,
    )
    assert no_runs_result.planner_outcomes is not None
    assert no_runs_result.planner_outcomes.available is False
    assert no_runs_result.planner_outcomes.planners == []

    empty_rows = tmp_path / "empty_rows.jsonl"
    empty_rows.write_text("", encoding="utf-8")
    missing_rows_summary = tmp_path / "missing_rows.json"
    missing_rows_summary.write_text(
        json.dumps(
            {
                "planner_runs": [
                    {"planner": "empty", "out_path": empty_rows.name, "total_jobs": 3},
                    {"planner": "no_metrics", "total_jobs": 4},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = summarize_adversarial_manifest_quality(
        [tmp_path / "a.yaml"],
        smoke_summary_json=missing_rows_summary,
    )

    assert result.planner_outcomes is not None
    assert result.planner_outcomes.available is False
    empty_planner, missing_metrics_planner = result.planner_outcomes.planners
    assert empty_planner.source == "no_rows"
    assert empty_planner.episodes == 3
    assert empty_planner.failure_yield == 0.0
    assert missing_metrics_planner.source == "planner_summary_missing_rows"
    assert missing_metrics_planner.episodes == 4
    assert missing_metrics_planner.failure_yield is None


def test_manifest_quality_cli_writes_output_json(tmp_path: Path) -> None:
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    _write_manifest(tmp_path / "a.yaml", controls, "valid")

    output_json = tmp_path / "quality_summary.json"

    exit_code = quality_cli_main([str(tmp_path), "--output-json", str(output_json)])

    assert exit_code == 0
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == MANIFEST_QUALITY_SCHEMA_VERSION
    assert loaded["manifest_count"] == 1
    assert loaded["rates"]["validity_rate"] == 1.0
    assert loaded["naturalistic_prior"]["unavailable_count"] == 1


def test_manifest_quality_cli_filters_naturalistic_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    controls_a = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    controls_b = _candidate_controls(start_x=1.5, start_y=2.0, goal_x=5.0, goal_y=2.0)
    _write_manifest(
        tmp_path / "passed.yaml",
        controls_a,
        "valid",
        naturalistic_prior=_naturalistic_prior_payload(passed=True),
    )
    _write_manifest(
        tmp_path / "legacy.yaml",
        controls_b,
        "valid",
    )

    assert quality_cli_main([str(tmp_path), "--naturalistic-status", "passed"]) == 0
    stdout = capsys.readouterr().out
    assert json.loads(stdout)["manifest_count"] == 1


def test_manifest_quality_cli_prints_json_and_reports_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI mode should print summaries to stdout and convert input errors to exit code 1."""
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    _write_manifest(tmp_path / "a.yaml", controls, "valid")

    assert quality_cli_main([str(tmp_path / "a.yaml")]) == 0
    stdout = capsys.readouterr().out
    assert json.loads(stdout)["manifest_count"] == 1

    assert quality_cli_main([str(tmp_path / "missing.yaml")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Manifest input missing" in captured.err
