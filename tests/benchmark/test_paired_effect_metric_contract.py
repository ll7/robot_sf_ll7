"""Tests for the issue #6970 paired-effect retained-row contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from robot_sf.benchmark import runner
from robot_sf.benchmark.paired_effect_metric_contract import (
    REQUIRED_METRIC_NAMES,
    PairedEffectMetricContractError,
    load_paired_effect_metric_contract,
    validate_paired_effect_metric_contract,
    validate_paired_effect_metric_record,
    validate_paired_effect_metric_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "configs/benchmarks/paired_effect_metric_contract_v1.yaml"
SCHEMA_PATH = REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
CHECK_SCRIPT = REPO_ROOT / "scripts/benchmark/check_paired_effect_metric_contract.py"


def _contract() -> dict[str, object]:
    return load_paired_effect_metric_contract(CONTRACT_PATH)


def _valid_metric_values() -> dict[str, float]:
    return {
        "exact_collision_probability": 0.0,
        "near_miss_probability": 1.0,
        "min_predicted_separation_m": 0.42,
        "completion_probability": 1.0,
        "progress_at_timeout": 0.75,
        "false_positive_stop_rate": 0.0,
        "stop_yield_latency_s": 0.8,
        "wrapper_intervention_rate": 0.25,
    }


def _map_scenario() -> dict[str, object]:
    return {
        "name": "issue-6970-contract-smoke",
        "map_file": "maps/svg_maps/francis2023/francis2023_blind_corner.svg",
        "simulation_config": {"max_episode_steps": 1},
        "robot_config": {"kinematics": "differential_drive"},
    }


def test_contract_declares_exact_report_builder_roster() -> None:
    """The versioned contract is aligned with the #4598 report-builder outcome roster."""
    contract = _contract()
    assert contract["schema_version"] == "paired_effect_metric_contract.v1"
    assert contract["report_builder_issue"] == 4598
    assert tuple(contract["required_metric_names"]) == REQUIRED_METRIC_NAMES  # type: ignore[arg-type]
    assert [field["path"] for field in contract["fields"]] == [  # type: ignore[index]
        f"metric_values.{name}" for name in REQUIRED_METRIC_NAMES
    ]


def test_metric_values_producer_derives_available_sources() -> None:
    """The map-runner producer emits the five fields with available sources."""
    from robot_sf.benchmark.map_runner import map_runner_episode

    values = map_runner_episode._paired_effect_metric_values(
        metrics_raw={
            "success": 1.0,
            "collisions": 1.0,
            "near_misses": 2.0,
            "min_distance": 0.42,
        },
        metrics={"wrapper_intervention_rate": 0.25},
    )
    assert values["exact_collision_probability"] == 1.0
    assert values["near_miss_probability"] == 1.0
    assert values["min_predicted_separation_m"] == 0.42
    assert values["completion_probability"] == 1.0
    assert values["wrapper_intervention_rate"] == 0.25


def test_metric_values_producer_omits_unavailable_sources() -> None:
    """Fields whose sources are missing are omitted, never fabricated as zero."""
    from robot_sf.benchmark.map_runner import map_runner_episode

    values = map_runner_episode._paired_effect_metric_values(
        metrics_raw={},
        metrics={},
    )
    assert values == {}
    assert "exact_collision_probability" not in values


def test_metric_values_producer_finalize_attaches_mapping() -> None:
    """_finalize_episode_metrics attaches the metric_values mapping."""
    from robot_sf.benchmark.map_runner import map_runner_episode

    metrics = map_runner_episode._finalize_episode_metrics(
        {"success": 0.0, "collisions": 1.0, "near_misses": 0.0, "min_distance": 0.5},
        algo_meta={},
        actuation_controller=None,
        actuation_summary={},
        tracking_precision_summary={
            "min_separation_corrupted_m": 0.5,
            "contract_honored": True,
            "contract_honored_rate": 1.0,
        },
        tracking_precision_spec={"target_motp_m": 0.3},
        safety_wrapper_summary=None,
        cbf_filter_summary=None,
        snqi_weights=None,
        snqi_baseline=None,
    )
    assert "metric_values" in metrics
    assert metrics["metric_values"]["exact_collision_probability"] == 1.0
    assert metrics["metric_values"]["completion_probability"] == 0.0


def test_valid_retained_row_passes() -> None:
    """A row with all exact finite fields passes without using legacy aliases."""
    report = validate_paired_effect_metric_record(
        {"metric_values": _valid_metric_values()},
        _contract(),
        row_index=3,
    )
    assert report["status"] == "ok"
    assert report["missing_fields"] == []
    assert report["invalid_fields"] == []


def test_issue_13775_shape_is_blocked_without_alias_substitution() -> None:
    """The historical metrics-only row cannot satisfy the retained paired-effect contract."""
    report = validate_paired_effect_metric_rows(
        [{"metrics": {"clearing_distance_min": 0.42, "success": 1.0}}],
        _contract(),
    )
    assert report["status"] == "blocked"
    assert report["complete"] is False
    assert set(report["missing_field_counts"]) == set(REQUIRED_METRIC_NAMES)


def test_row_validation_uses_bounded_diagnostics_by_default() -> None:
    """Runner-facing reports stay compact while diagnostics retain every row report."""

    rows = [{"metrics": {"clearing_distance_min": float(index)}} for index in range(12)]

    compact = validate_paired_effect_metric_rows(rows, _contract())
    assert "row_reports" not in compact
    assert len(compact["invalid_row_samples"]) == 10
    assert [report["row_index"] for report in compact["invalid_row_samples"]] == list(range(10))

    diagnostic = validate_paired_effect_metric_rows(rows, _contract(), include_row_reports=True)
    assert len(diagnostic["row_reports"]) == len(rows)


@pytest.mark.parametrize(
    "field_name, value, reason",
    [
        ("wrapper_intervention_rate", True, "boolean_is_not_scalar"),
        ("wrapper_intervention_rate", float("nan"), "non_finite"),
        ("wrapper_intervention_rate", 1.1, "out_of_bounds"),
        ("progress_at_timeout", 1.1, "out_of_bounds"),
    ],
)
def test_invalid_retained_values_are_blocked(field_name: str, value: object, reason: str) -> None:
    """Boolean, non-finite, and out-of-range values fail closed."""
    metric_values = _valid_metric_values()
    metric_values[field_name] = value  # type: ignore[assignment]
    report = validate_paired_effect_metric_record(
        {"metric_values": metric_values},
        _contract(),
    )
    assert report["status"] == "blocked"
    assert report["invalid_fields"][0]["reason"] == reason


def test_contract_rejects_wrong_retained_path() -> None:
    """The contract cannot silently move the report-builder fields to a legacy mapping."""
    payload = _contract()
    fields = [dict(field) for field in payload["fields"]]  # type: ignore[index]
    fields[0]["path"] = "metrics.exact_collision_probability"
    payload["fields"] = fields
    with pytest.raises(PairedEffectMetricContractError, match="path"):
        validate_paired_effect_metric_contract(payload)


def test_runner_fails_closed_on_metrics_only_map_output(tmp_path: Path) -> None:
    """The run-batch post-run gate rejects a successful-looking but unidentifiable row."""
    out_path = tmp_path / "episodes.jsonl"

    def fake_run_map_batch(*args: object, **_kwargs: object) -> dict[str, object]:
        Path(args[1]).write_text(
            json.dumps({"metrics": {"clearing_distance_min": 0.42}}) + "\n",
            encoding="utf-8",
        )
        return {"status": "ok", "total_jobs": 1, "written": 1, "failures": []}

    with patch.object(runner, "run_map_batch", side_effect=fake_run_map_batch):
        with pytest.raises(PairedEffectMetricContractError, match="retained-row contract failed"):
            runner.run_batch(
                [_map_scenario()],
                out_path=out_path,
                schema_path=SCHEMA_PATH,
                horizon=1,
                dt=0.1,
                retained_metric_contract_path=CONTRACT_PATH,
            )


def test_runner_contract_error_uses_compact_validation_details(tmp_path: Path) -> None:
    """A failed runner gate does not serialize the full per-row diagnostic collection."""
    out_path = tmp_path / "episodes.jsonl"

    def fake_run_map_batch(*args: object, **_kwargs: object) -> dict[str, object]:
        Path(args[1]).write_text(
            json.dumps({"metrics": {"clearing_distance_min": 0.42}}) + "\n",
            encoding="utf-8",
        )
        return {"status": "ok", "total_jobs": 1, "written": 1, "failures": []}

    with patch.object(runner, "run_map_batch", side_effect=fake_run_map_batch):
        with pytest.raises(
            PairedEffectMetricContractError,
            match="retained-row contract failed",
        ) as exc:
            runner.run_batch(
                [_map_scenario()],
                out_path=out_path,
                schema_path=SCHEMA_PATH,
                horizon=1,
                dt=0.1,
                retained_metric_contract_path=CONTRACT_PATH,
            )
    assert "row_reports" not in str(exc.value)


def test_runner_accepts_complete_retained_metric_row(tmp_path: Path) -> None:
    """The post-run gate returns a structured success report for a complete retained row."""
    out_path = tmp_path / "episodes.jsonl"

    def fake_run_map_batch(*args: object, **_kwargs: object) -> dict[str, object]:
        Path(args[1]).write_text(
            json.dumps({"metric_values": _valid_metric_values()}) + "\n",
            encoding="utf-8",
        )
        return {"status": "ok", "total_jobs": 1, "written": 1, "failures": []}

    with patch.object(runner, "run_map_batch", side_effect=fake_run_map_batch):
        summary = runner.run_batch(
            [_map_scenario()],
            out_path=out_path,
            schema_path=SCHEMA_PATH,
            horizon=1,
            dt=0.1,
            retained_metric_contract_path=CONTRACT_PATH,
        )
    assert summary["retained_metric_contract"]["status"] == "ok"
    assert summary["retained_metric_contract"]["complete"] is True


def test_runner_loads_contract_before_dispatch(tmp_path: Path) -> None:
    """An invalid contract reference blocks dispatch before any arm can run."""
    called = False

    def fake_run_map_batch(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    with patch.object(runner, "run_map_batch", side_effect=fake_run_map_batch):
        with pytest.raises(PairedEffectMetricContractError, match="cannot be read"):
            runner.run_batch(
                [_map_scenario()],
                out_path=tmp_path / "episodes.jsonl",
                schema_path=SCHEMA_PATH,
                retained_metric_contract_path=tmp_path / "missing.yaml",
            )
    assert called is False


def test_contract_audit_cli_reports_contract_complete_exposure_without_running_campaign() -> None:
    """The audit command remains diagnostic-only while reporting complete config coverage."""
    completed = subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--contract",
            str(CONTRACT_PATH),
            "--audit-configs",
            "--json",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["exposure"]["status"] == "ok"
    assert payload["exposure"]["config_count"] >= 2
    assert payload["exposure"]["claim_boundary"].startswith("Exposure audit only.")
    assert payload["exposure"]["counts"]["missing_reference"] == 0
    assert payload["exposure"]["counts"]["invalid_reference"] == 0
    assert payload["exposure"]["counts"]["invalid_contract"] == 0


def test_contract_audit_marks_malformed_referenced_contract(tmp_path: Path) -> None:
    """A readable but schema-invalid reference is not reported as covered."""
    from scripts.benchmark.check_paired_effect_metric_contract import _audit_configs

    config_dir = tmp_path / "configs" / "research"
    config_dir.mkdir(parents=True)
    (config_dir / "invalid_reference.yaml").write_text(
        "report_contract:\n  paired_report_builder_issue: 4598\n"
        "retained_metric_contract: invalid_contract.yaml\n",
        encoding="utf-8",
    )
    (config_dir / "invalid_contract.yaml").write_text(
        "schema_version: paired_effect_metric_contract.v1\n",
        encoding="utf-8",
    )

    audit = _audit_configs(tmp_path)

    assert audit["status"] == "findings"
    assert audit["counts"]["invalid_contract"] == 1
    assert audit["configs"][0]["status"] == "invalid_contract"
