"""Focused tests for the issue #6095 discriminability report."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark.build_issue_6095_discriminability_report import (
    EXPECTED_KINEMATICS,
    EpisodeRow,
    RegimeData,
    ReportContractError,
    _checkpoint_receipt,
    _episode_row,
    _provenance_interpretation_status,
    _provenance_limitation_lines,
    _validate_campaign_receipts,
    _validate_episode_row,
    _validate_summary_rows,
    bootstrap_mean_ci,
    classify_stress_floor,
    main,
)


def _episode(
    planner_key: str,
    scenario_id: str,
    seed: int,
    *,
    success: float = 0.0,
    collision: float = 0.0,
    near_misses: float = 0.0,
) -> EpisodeRow:
    """Build a minimal validated episode row for pure classification tests."""
    return EpisodeRow(
        planner_key=planner_key,
        scenario_id=scenario_id,
        seed=seed,
        success=success,
        collision=collision,
        near_misses=near_misses,
        near_miss_any=float(near_misses > 0.0),
        execution_mode="native",
        observation_level="tracked_agents_no_noise",
        model_id=None,
        horizon=100,
        dt=0.1,
    )


def _regime() -> RegimeData:
    """Build a four-scenario stress fixture covering every floor class."""
    scenarios = ("both_some", "one_some", "collision_only", "near_miss_only")
    seeds = (111, 112, 113)
    rows: dict[tuple[str, str, int], EpisodeRow] = {}
    for planner in ("orca", "ppo"):
        for scenario in scenarios:
            for seed in seeds:
                kwargs: dict[str, float] = {}
                if scenario == "both_some" and seed == 111:
                    kwargs["success"] = 1.0
                if scenario == "one_some" and planner == "ppo" and seed == 111:
                    kwargs["success"] = 1.0
                if scenario == "collision_only" and planner == "orca" and seed == 111:
                    kwargs["collision"] = 1.0
                if scenario == "near_miss_only" and planner == "ppo" and seed == 111:
                    kwargs["near_misses"] = 2.0
                rows[(planner, scenario, seed)] = _episode(planner, scenario, seed, **kwargs)
    return RegimeData(
        name="stress",
        root=Path("."),
        campaign_id="fixture",
        scenario_matrix="fixture",
        scenario_matrix_hash="fixture",
        git_commit="fixture",
        scenario_ids=scenarios,
        seeds=seeds,
        rows=rows,
        blockers=[],
        warnings=[],
        checkpoint={},
        metadata={"kinematics": EXPECTED_KINEMATICS},
    )


def _write_json(path: Path, payload: object) -> None:
    """Write one compact JSON fixture artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_checkpoint_fixture(
    root: Path,
    *,
    field: str | None = None,
    value: object = None,
    load_status: str = "not_run",
) -> None:
    """Write a complete checkpoint receipt, optionally corrupting one boolean field."""
    payload: dict[str, object] = {
        "mode": "enforced_staged",
        "stage": True,
        "submit_safe": True,
        "arms": [
            {
                "planner_key": "ppo",
                "model_id": "model",
                "checkpoint_sha256": "sha",
                "status": "staged",
                "hash_source": "computed_file",
                "resolved_path": "/cache/model",
                "load_status": load_status,
                "load_succeeded": None,
                "fallback_triggered": None,
            }
        ],
    }
    if field in {"stage", "submit_safe"}:
        payload[field] = value
    elif field is not None:
        payload["arms"][0][field] = value  # type: ignore[index]
    _write_json(root / "preflight" / "checkpoint_staging.json", payload)


def _write_campaign_fixture(root: Path, *, regime: str, row_case: str) -> None:
    """Write a small complete campaign with one deliberately invalid row case."""
    scenario_ids = tuple(f"{regime}_{index}" for index in range(4))
    matrix = (
        "configs/scenarios/nominal_v1.yaml"
        if regime == "nominal"
        else "configs/scenarios/classic_interactions_francis2023.yaml"
    )
    matrix_hash = f"{regime}-hash"
    campaign_id = f"fixture-{regime}"
    _write_json(
        root / "campaign_manifest.json",
        {
            "campaign_id": campaign_id,
            "scenario_matrix": matrix,
            "scenario_matrix_hash": matrix_hash,
            "seed_policy": {"resolved_seeds": [111]},
            "git": {"commit": "fixture"},
        },
    )
    _write_json(
        root / "preflight" / "preview_scenarios.json",
        {"scenarios": [{"name": scenario, "seeds": [111]} for scenario in scenario_ids]},
    )
    _write_checkpoint_fixture(root)
    _write_json(root / "reports" / "matrix_summary.json", {"rows": []})
    _write_json(
        root / "reports" / "campaign_integrity.json",
        {"status": "valid", "benchmark_success_allowed": True, "blockers": []},
    )
    _write_json(
        root / "reports" / "campaign_summary.json",
        {
            "campaign": {
                "campaign_id": campaign_id,
                "scenario_matrix": matrix,
                "scenario_matrix_hash": matrix_hash,
                "git_hash": "fixture",
                "benchmark_success": True,
                "evidence_status": "valid",
                "campaign_execution_status": "completed",
                "total_episodes": 8,
                "row_status_summary": {
                    "successful_evidence_rows": 8,
                    "accepted_unavailable_rows": 0,
                    "fallback_or_degraded_rows": 0,
                    "unexpected_failed_rows": 0,
                },
            },
            "planner_rows": [
                {
                    "planner_key": "orca",
                    "episodes": 4,
                    "status": "ok",
                    "benchmark_success": True,
                    "availability_status": "available",
                    "readiness_status": "adapter",
                    "execution_mode": "adapter",
                },
                {
                    "planner_key": "ppo",
                    "episodes": 4,
                    "status": "ok",
                    "benchmark_success": True,
                    "availability_status": "available",
                    "readiness_status": "native",
                    "execution_mode": "native",
                },
            ],
        },
    )

    for planner_key, execution_mode in (("orca", "adapter"), ("ppo", "native")):
        run_dir = root / "runs" / f"{planner_key}__differential_drive"
        records: list[dict[str, object]] = []
        for scenario_id in scenario_ids:
            if row_case == "missing" and planner_key == "ppo" and scenario_id == scenario_ids[0]:
                continue
            record: dict[str, object] = {
                "scenario_id": scenario_id,
                "seed": 111,
                "termination_reason": "success",
                "status": "success",
                "git_hash": "fixture",
                "horizon": 100,
                "metrics": {"success": 1.0, "collisions": 0.0, "near_misses": 0.0},
                "outcome": {
                    "route_complete": True,
                    "collision_event": False,
                    "timeout_event": False,
                },
                "result_provenance": {"simulator_settings": {"horizon": 100, "dt": 0.1}},
                "scenario_params": {"robot_config": {"type": "differential_drive"}},
                "algorithm_metadata": {
                    "planner_kinematics": {
                        "execution_mode": execution_mode,
                        "robot_kinematics": "differential_drive",
                    },
                    "learned_checkpoint_observation_contract": {
                        "observation_level": "tracked_agents_no_noise"
                    },
                },
            }
            if planner_key == "ppo":
                record["algorithm_metadata"]["config"] = {"model_id": "model"}  # type: ignore[index]
            if row_case == "malformed" and planner_key == "orca" and scenario_id == scenario_ids[0]:
                record.pop("outcome")
            if row_case == "fallback" and planner_key == "ppo" and scenario_id == scenario_ids[0]:
                record["algorithm_metadata"]["status"] = "fallback"  # type: ignore[index]
            records.append(record)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "episodes.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )


def _run_report_cli(
    nominal_root: Path,
    stress_root: Path,
    output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    """Run the real report CLI against the compact fixture roots."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_issue_6095_discriminability_report.py",
            "--nominal-root",
            str(nominal_root),
            "--stress-root",
            str(stress_root),
            "--output-dir",
            str(output_dir),
            "--expected-commit",
            "fixture",
            "--expected-model-id",
            "model",
            "--expected-model-sha256",
            "sha",
            "--expected-seed",
            "111",
            "--s3-seed",
            "111",
            "--bootstrap-samples",
            "100",
        ],
    )
    return main()


def test_bootstrap_mean_ci_is_deterministic_and_seed_scenario_aware() -> None:
    """The declared bootstrap returns repeatable finite bounds for a 2-D matrix."""
    matrix = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    first = bootstrap_mean_ci(matrix, bootstrap_seed=6095, bootstrap_samples=200)
    second = bootstrap_mean_ci(matrix, bootstrap_seed=6095, bootstrap_samples=200)

    assert first == second
    assert first[0] == 0.5
    assert 0.0 <= first[1] <= first[2] <= 1.0


def test_classify_stress_floor_counts_both_zero_metric_discriminability() -> None:
    """Both-zero scenarios are separated from one-planner and shared successes."""
    result = classify_stress_floor(_regime(), seeds=(111, 112, 113))

    assert result["class_counts"] == {
        "both_planners_some_success": 1,
        "both_planners_zero_success": 2,
        "exactly_one_planner_some_success": 1,
    }
    assert result["both_zero_count"] == 2
    assert result["both_zero_distinguished_count"] == 2
    assert result["both_zero_distinguished_by_collision_count"] == 1
    assert result["both_zero_distinguished_by_near_miss_count"] == 1


def test_provenance_markdown_tracks_staged_receipts() -> None:
    """Human-readable provenance caveats must reflect staged receipt status."""
    receipt = {
        "status": "staged_receipt",
        "identity_matches_expected": True,
        "hash_source": "computed_file",
        "submit_safe": True,
        "load_status": "not_run",
    }

    lines = _provenance_limitation_lines({"nominal": receipt, "stress": receipt})

    rendered = "\n".join(lines)
    assert "staged" in rendered
    assert "metadata-only" not in rendered
    assert "nominal=not_run, stress=not_run" in rendered


def test_checkpoint_receipt_requires_computed_file_hash_source(tmp_path: Path) -> None:
    """A staged identity receipt without a computed-file hash remains unresolved."""
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    (preflight / "checkpoint_staging.json").write_text(
        '{"mode":"staged","stage":true,"submit_safe":true,'
        '"arms":[{"planner_key":"ppo","model_id":"model",'
        '"checkpoint_sha256":"sha","status":"staged",'
        '"hash_source":"declared","load_status":"not_run"}]}\n',
        encoding="utf-8",
    )

    receipt = _checkpoint_receipt(
        tmp_path,
        expected_model_id="model",
        expected_sha256="sha",
    )

    assert receipt["status"] == "unresolved"


def test_checkpoint_receipt_rejects_failed_staged_runtime_receipt(tmp_path: Path) -> None:
    """A failed or pathless staged receipt must not support interpretation."""
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    (preflight / "checkpoint_staging.json").write_text(
        '{"mode":"enforced_staged","stage":true,"submit_safe":true,'
        '"arms":[{"planner_key":"ppo","model_id":"model",'
        '"checkpoint_sha256":"sha","status":"staged",'
        '"hash_source":"computed_file","load_status":"failed",'
        '"load_succeeded":false,"fallback_triggered":true}]}' + "\n",
        encoding="utf-8",
    )

    receipt = _checkpoint_receipt(
        tmp_path,
        expected_model_id="model",
        expected_sha256="sha",
    )

    assert receipt["status"] == "unresolved"


def test_checkpoint_receipt_accepts_complete_enforced_staged_receipt(tmp_path: Path) -> None:
    """A complete enforced-staged preflight receipt remains admissible."""
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    (preflight / "checkpoint_staging.json").write_text(
        '{"mode":"enforced_staged","stage":true,"submit_safe":true,'
        '"arms":[{"planner_key":"ppo","model_id":"model",'
        '"checkpoint_sha256":"sha","status":"staged",'
        '"hash_source":"computed_file","resolved_path":"/cache/model",'
        '"load_status":"not_run","load_succeeded":null,"fallback_triggered":null}]}\n',
        encoding="utf-8",
    )

    receipt = _checkpoint_receipt(
        tmp_path,
        expected_model_id="model",
        expected_sha256="sha",
    )

    assert receipt["status"] == "staged_receipt"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage", "true"),
        ("stage", "false"),
        ("submit_safe", "true"),
        ("submit_safe", "false"),
        ("load_succeeded", "true"),
        ("load_succeeded", "false"),
        ("fallback_triggered", "true"),
        ("fallback_triggered", "false"),
    ],
)
def test_checkpoint_receipt_rejects_string_boolean_flags(
    tmp_path: Path, field: str, value: str
) -> None:
    """String booleans must not pass the staged checkpoint receipt gate."""
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    _write_checkpoint_fixture(tmp_path, field=field, value=value)

    receipt = _checkpoint_receipt(
        tmp_path,
        expected_model_id="model",
        expected_sha256="sha",
    )

    assert receipt["status"] == "unresolved"
    assert receipt["boolean_fields_valid"] is False
    assert receipt[field] == value


def test_checkpoint_receipt_requires_boolean_load_success_when_loaded(tmp_path: Path) -> None:
    """A loaded receipt with a string load result remains unresolved."""
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    _write_checkpoint_fixture(
        tmp_path,
        field="load_succeeded",
        value="true",
        load_status="loaded",
    )

    receipt = _checkpoint_receipt(
        tmp_path,
        expected_model_id="model",
        expected_sha256="sha",
    )

    assert receipt["status"] == "unresolved"
    assert receipt["boolean_fields_valid"] is False


def test_checkpoint_interpretation_status_blocks_non_staged_receipts() -> None:
    """Invalid checkpoint receipts must not render as declared and receipted."""
    assert (
        _provenance_interpretation_status(({"status": "staged_receipt"}, {"status": "unresolved"}))
        == "blocked"
    )


@pytest.mark.parametrize("readiness_status", ["fallback", "degraded", "unknown", ""])
def test_summary_row_rejects_non_native_or_adapter_readiness(
    readiness_status: str,
) -> None:
    """Summary readiness must not bypass the native/adapter evidence contract."""
    summary_rows = {
        "orca": {
            "episodes": 1,
            "status": "ok",
            "benchmark_success": True,
            "availability_status": "available",
            "readiness_status": readiness_status,
            "execution_mode": "adapter",
        },
        "ppo": {
            "episodes": 1,
            "status": "ok",
            "benchmark_success": True,
            "availability_status": "available",
            "readiness_status": "native",
            "execution_mode": "native",
        },
    }

    blockers, _warnings = _validate_summary_rows(
        name="nominal",
        summary_rows=summary_rows,
        expected_episode_count=1,
    )

    assert any("readiness status" in blocker for blocker in blockers)


@pytest.mark.parametrize(
    "algorithm_metadata",
    [
        {"status": "policy_step_timeout_fallback"},
        {"status": "ok", "policy_step_timeout": {"fallback_actions": 1}},
    ],
)
def test_episode_row_rejects_runner_degraded_fallback_markers(
    algorithm_metadata: dict[str, object],
) -> None:
    """Runner-emitted fallback markers must not become benchmark evidence."""
    row = _episode("orca", "scenario", 111)
    blockers = _validate_episode_row(
        name="nominal",
        key=("orca", "scenario", 111),
        record={"algorithm_metadata": algorithm_metadata},
        row=row,
        scenario_ids=("scenario",),
        expected_seeds=(111,),
        expected_commit="fixture",
        expected_model_id="model",
    )

    assert any("fallback/degradation" in blocker for blocker in blockers)


def test_episode_row_rejects_missing_or_unknown_termination_reason(tmp_path: Path) -> None:
    """Malformed terminal metadata must not silently become a zero outcome."""
    record = {
        "scenario_id": "scenario",
        "seed": 111,
        "metrics": {"near_misses": 0.0},
    }

    with pytest.raises(ReportContractError, match="termination_reason"):
        _episode_row(record, planner_key="orca", source=tmp_path / "episodes.jsonl")

    record["termination_reason"] = "unknown"
    with pytest.raises(ReportContractError, match="termination_reason"):
        _episode_row(record, planner_key="orca", source=tmp_path / "episodes.jsonl")


def test_episode_row_rejects_missing_canonical_outcome_payload(tmp_path: Path) -> None:
    """Schema-required outcome flags must not be inferred from termination metadata."""
    record = {
        "scenario_id": "scenario",
        "seed": 111,
        "termination_reason": "success",
        "metrics": {"near_misses": 0.0},
    }

    with pytest.raises(ReportContractError, match="canonical outcome payload"):
        _episode_row(record, planner_key="orca", source=tmp_path / "episodes.jsonl")


def test_episode_row_rejects_outcome_metric_contradiction(tmp_path: Path) -> None:
    """Contradictory collision metrics must not be normalized into a zero collision."""
    record = {
        "scenario_id": "scenario",
        "seed": 111,
        "termination_reason": "success",
        "metrics": {"collisions": 1.0, "near_misses": 0.0},
        "outcome": {
            "route_complete": True,
            "collision_event": False,
            "timeout_event": False,
        },
    }

    with pytest.raises(ReportContractError, match="integrity contradiction"):
        _episode_row(record, planner_key="orca", source=tmp_path / "episodes.jsonl")


def test_episode_row_validates_planner_execution_and_observation_contract() -> None:
    """The report must reject rows that do not match the frozen runtime contract."""
    row = _episode(
        "orca",
        "scenario",
        111,
        success=0.0,
    )
    blockers = _validate_episode_row(
        name="nominal",
        key=("orca", "scenario", 111),
        record={},
        row=replace(row, execution_mode="native", observation_level="oracle_full_state"),
        scenario_ids=("scenario",),
        expected_seeds=(111,),
        expected_commit="fixture",
        expected_model_id="model",
    )

    assert any("execution mode" in blocker for blocker in blockers)
    assert any("observation level" in blocker for blocker in blockers)


def test_episode_row_rejects_failed_or_degraded_statuses() -> None:
    """Raw failed or degraded rows must not become diagnostic metric values."""
    row = _episode("orca", "scenario", 111)
    record = {
        "termination_reason": "error",
        "status": "failure",
        "algorithm_metadata": {"status": "fallback"},
    }

    blockers = _validate_episode_row(
        name="nominal",
        key=("orca", "scenario", 111),
        record=record,
        row=row,
        scenario_ids=("scenario",),
        expected_seeds=(111,),
        expected_commit="fixture",
        expected_model_id="model",
    )

    assert any("failed episode termination" in blocker for blocker in blockers)
    assert any("algorithm status 'fallback'" in blocker for blocker in blockers)


def test_campaign_receipts_require_complete_zero_failure_row_summary() -> None:
    """Missing or non-zero row-status receipts must fail closed."""
    summary = {
        "campaign": {
            "scenario_matrix": "matrix",
            "git_hash": "fixture",
            "benchmark_success": True,
            "evidence_status": "valid",
            "campaign_execution_status": "completed",
            "row_status_summary": {
                "successful_evidence_rows": 1,
                "accepted_unavailable_rows": 1,
                "fallback_or_degraded_rows": 0,
                "unexpected_failed_rows": 0,
            },
        }
    }
    manifest = {
        "scenario_matrix": "matrix",
        "git": {"commit": "fixture"},
        "seed_policy": {"resolved_seeds": [111]},
    }
    integrity = {"status": "valid", "benchmark_success_allowed": True}

    _campaign, blockers = _validate_campaign_receipts(
        name="nominal",
        summary=summary,
        manifest=manifest,
        integrity=integrity,
        expected_matrix="matrix",
        expected_seeds=(111,),
        expected_commit="fixture",
    )

    assert any("accepted unavailable rows are present" in blocker for blocker in blockers)

    summary["campaign"].pop("row_status_summary")
    _campaign, blockers = _validate_campaign_receipts(
        name="nominal",
        summary=summary,
        manifest=manifest,
        integrity=integrity,
        expected_matrix="matrix",
        expected_seeds=(111,),
        expected_commit="fixture",
    )

    assert any("row_status_summary is missing or invalid" in blocker for blocker in blockers)


@pytest.mark.parametrize("row_case", ["malformed", "missing", "fallback"])
def test_cli_writes_blocked_no_numeric_report_for_invalid_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row_case: str
) -> None:
    """Malformed, incomplete, and fallback rows write blocked artifacts without metrics."""
    nominal_root = tmp_path / "nominal"
    stress_root = tmp_path / "stress"
    output_dir = tmp_path / "report"
    _write_campaign_fixture(nominal_root, regime="nominal", row_case=row_case)
    _write_campaign_fixture(stress_root, regime="stress", row_case=row_case)

    assert _run_report_cli(nominal_root, stress_root, output_dir, monkeypatch) == 2

    report = json.loads(
        (output_dir / "issue6095_discriminability_report.json").read_text(encoding="utf-8")
    )
    markdown = (output_dir / "issue6095_discriminability_report.md").read_text(encoding="utf-8")

    assert report["status"] == "blocked_validation"
    assert report["benchmark_success_allowed"] is False
    assert report["interpretation_allowed"] is False
    assert report["numeric_data_available"] is False
    assert report["numeric_data"] == {
        "status": "unavailable",
        "reason": "validation_blockers",
        "rows": [],
    }
    assert report["nominal_vs_stress"]["s10"] is None
    assert report["stress_floor"]["scenarios"] == []
    assert "Numeric data: **unavailable**" in markdown
    assert "No aggregate metric rows" in markdown
    expected_blocker_text = {
        "malformed": "invalid episode row",
        "missing": "raw identity set",
        "fallback": "fallback",
    }[row_case]
    assert any(expected_blocker_text in blocker.lower() for blocker in report["blockers"])


def test_cli_keeps_numeric_report_for_complete_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete fixture still takes the existing numeric report path."""
    nominal_root = tmp_path / "nominal"
    stress_root = tmp_path / "stress"
    output_dir = tmp_path / "report"
    _write_campaign_fixture(nominal_root, regime="nominal", row_case="complete")
    _write_campaign_fixture(stress_root, regime="stress", row_case="complete")

    assert _run_report_cli(nominal_root, stress_root, output_dir, monkeypatch) == 0

    report = json.loads(
        (output_dir / "issue6095_discriminability_report.json").read_text(encoding="utf-8")
    )
    markdown = (output_dir / "issue6095_discriminability_report.md").read_text(encoding="utf-8")

    assert report["numeric_data_available"] is True
    assert report["regimes"]["nominal"]["s10"] is not None
    assert "Success and collision estimates" in markdown
