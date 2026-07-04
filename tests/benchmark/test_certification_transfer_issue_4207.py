"""Tests for issue #4207 certification-transfer probe helpers."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.certification_transfer import (
    AGGREGATABLE_METRICS,
    CLAIM_BOUNDARY,
    PREFLIGHT_BLOCKED_NO_EVALUABLE_GATE_FAMILY,
    PREFLIGHT_OK,
    _repo_relative_path,
    build_certification_transfer_report,
    classify_interaction_status,
    load_yaml_mapping,
    preflight_gate_evaluability,
    validate_gate_spec,
    validate_probe_config,
    write_certification_transfer_evidence,
)
from robot_sf.benchmark.certification_transfer import (
    REPO_ROOT as CERT_REPO_ROOT,
)
from robot_sf.sim.pedestrian_model_variants import HSFM_TOTAL_FORCE_V1, SOCIAL_FORCE_DEFAULT

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERACTING_SMOKE_CONFIG = REPO_ROOT / "configs/benchmarks/issue_4207_interacting_smoke_probe.yaml"
INTERACTING_SMOKE_GATES = (
    REPO_ROOT / "configs/benchmarks/release_gates/issue_4207_interacting_smoke_gates.yaml"
)
INTERACTING_SMOKE_FIXTURE = (
    REPO_ROOT
    / "tests/benchmark/fixtures/issue_4207_interacting_smoke/interacting_smoke_episodes.jsonl"
)
INTERACTING_PHYSICS_CONFIG = (
    REPO_ROOT / "configs/benchmarks/issue_4207_interacting_physics_probe.yaml"
)
INTERACTING_PHYSICS_PACKET = (
    REPO_ROOT / "docs/context/evidence/issue_4207_interacting_physics_2026-07"
)

_RUNNER_MODULE_PATH = (
    REPO_ROOT / "scripts" / "benchmark" / "run_certification_transfer_issue_4207.py"
)
_spec = importlib.util.spec_from_file_location("run_cert_transfer_4207", _RUNNER_MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_run_cert_transfer = importlib.util.module_from_spec(_spec)
sys.modules["run_cert_transfer_4207"] = _run_cert_transfer
_spec.loader.exec_module(_run_cert_transfer)


def test_transfer_matrix_detects_pass_fail_flips(tmp_path: Path) -> None:
    """A 2x2 fixture emits all transfer statuses deterministically."""

    config_path, gate_path, config, gates = _write_config_pair(tmp_path)
    records = [
        _record("stable", SOCIAL_FORCE_DEFAULT, collision_rate=0.0),
        _record("stable", HSFM_TOTAL_FORCE_V1, collision_rate=0.0),
        _record("fragile", SOCIAL_FORCE_DEFAULT, collision_rate=0.0),
        _record("fragile", HSFM_TOTAL_FORCE_V1, collision_rate=1.0),
        _record("conservative", SOCIAL_FORCE_DEFAULT, collision_rate=1.0),
        _record("conservative", HSFM_TOTAL_FORCE_V1, collision_rate=0.0),
        _record("blocked", SOCIAL_FORCE_DEFAULT, collision_rate=0.0, include_required=False),
        _record("blocked", HSFM_TOTAL_FORCE_V1, collision_rate=0.0),
    ]

    report = build_certification_transfer_report(
        records,
        probe_config=config,
        gate_spec=gates,
        config_path=config_path,
        gate_spec_path=gate_path,
        generated_at_utc="2026-07-03T00:00:00+00:00",
    )

    rows = report["certification_transfer_matrix"]
    assert len(rows) == 16
    statuses = {row["transfer_status"] for row in rows}
    assert {
        "stable_pass",
        "fragile_pass_to_fail",
        "conservative_fail_to_pass",
        "stable_fail",
        "not_evaluable",
    }.issubset(statuses)
    assert report["flip_cases"]
    assert report["claim_boundary"] == CLAIM_BOUNDARY


def test_missing_gate_metrics_are_not_evaluable_not_pass(tmp_path: Path) -> None:
    """Required missing gate metrics fail closed as not_evaluable."""

    config_path, gate_path, config, gates = _write_config_pair(tmp_path)
    report = build_certification_transfer_report(
        [_record("stable", SOCIAL_FORCE_DEFAULT, include_required=False)],
        probe_config=config,
        gate_spec=gates,
        config_path=config_path,
        gate_spec_path=gate_path,
    )

    cell = next(
        row
        for row in report["gate_cells"]
        if row["planner_key"] == "stable" and row["evaluation_model"] == SOCIAL_FORCE_DEFAULT
    )
    assert cell["gate_status"] == "not_evaluable"
    assert "near_miss_rate_limit" in cell["not_evaluable_gate_ids"]


def test_unsupported_pedestrian_model_fails_closed(tmp_path: Path) -> None:
    """Probe config rejects unsupported or undeclared pedestrian-model labels."""

    _config_path, _gate_path, config, _gates = _write_config_pair(tmp_path)
    config["pedestrian_models"] = [SOCIAL_FORCE_DEFAULT, "bogus_model"]
    with pytest.raises(ValueError, match="Unsupported pedestrian_model"):
        validate_probe_config(config, base_dir=tmp_path)


def test_arm_algo_configs_must_resolve(tmp_path: Path) -> None:
    """Planner arm config paths must exist before execution."""

    _config_path, _gate_path, config, _gates = _write_config_pair(tmp_path)
    config["arms"][0]["algo_config"] = "missing.yaml"
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        validate_probe_config(config, base_dir=tmp_path)


def test_learned_predictive_missing_checkpoint_excluded_from_trained_planner_claims(
    tmp_path: Path,
) -> None:
    """Fallback or missing-checkpoint trained arms cannot support comparison claims."""

    config_path, gate_path, config, gates = _write_config_pair(tmp_path)
    config["arms"][1]["structural_class"] = "learned_policy"
    config["arms"][1]["algo"] = "ppo"
    config["arms"][2]["structural_class"] = "predictive"
    config["arms"][2]["algo"] = "prediction_planner"
    records = [
        _record("stable", SOCIAL_FORCE_DEFAULT, collision_rate=0.0),
        _record("fragile", SOCIAL_FORCE_DEFAULT, collision_rate=0.0),
        _record("conservative", SOCIAL_FORCE_DEFAULT, collision_rate=0.0),
    ]

    report = build_certification_transfer_report(
        records,
        probe_config=config,
        gate_spec=gates,
        config_path=config_path,
        gate_spec_path=gate_path,
    )

    arms = {arm["key"]: arm for arm in report["arms"]}
    assert arms["stable"]["trained_planner_claim_status"] == "not_a_trained_planner"
    assert (
        arms["fragile"]["trained_planner_claim_status"] == "excluded_missing_checkpoint_or_config"
    )
    assert (
        arms["conservative"]["trained_planner_claim_status"]
        == "excluded_missing_checkpoint_or_config"
    )
    rows = {
        row["planner_key"]: row
        for row in report["certification_transfer_matrix"]
        if row["evaluation_model"] == SOCIAL_FORCE_DEFAULT
    }
    assert rows["fragile"]["trained_planner_claim_exclusion"] == "missing_checkpoint_or_config"
    assert rows["conservative"]["trained_planner_claim_exclusion"] == "missing_checkpoint_or_config"
    assert report["trained_planner_claim_status_counts"]["excluded_missing_checkpoint_or_config"]
    readiness = report["trained_planner_readiness"]
    assert readiness["all_trained_planner_arms_ready"] is False
    assert readiness["blocker_count"] == 2
    blocked = {
        row["planner_key"]: row
        for row in readiness["rows"]
        if row["readiness_status"] == "blocked_missing_artifact_provenance"
    }
    assert blocked["fragile"]["missing_fields"] == [
        "algo_config",
        "checkpoint",
        "training_manifest",
    ]
    assert blocked["conservative"]["missing_fields"] == [
        "algo_config",
        "checkpoint",
        "training_manifest",
    ]


def test_trained_planner_readiness_turns_ready_with_artifact_provenance(
    tmp_path: Path,
) -> None:
    """Checkpoint-backed trained arm records readiness for a fresh probe."""

    config_path, gate_path, config, gates = _write_config_pair(tmp_path)
    checkpoint_path = tmp_path / "policy.zip"
    manifest_path = tmp_path / "training_manifest.json"
    checkpoint_path.write_bytes(b"fixture checkpoint")
    manifest_path.write_text('{"schema_version": "fixture"}\n', encoding="utf-8")
    config["arms"][1].update(
        {
            "structural_class": "learned_policy",
            "algo": "ppo",
            "algo_config": str(tmp_path / "algo.yaml"),
            "checkpoint": str(checkpoint_path),
            "training_manifest": str(manifest_path),
        }
    )
    records = [
        _record("stable", SOCIAL_FORCE_DEFAULT, collision_rate=0.0),
        _record("fragile", SOCIAL_FORCE_DEFAULT, collision_rate=0.0),
    ]

    report = build_certification_transfer_report(
        records,
        probe_config=config,
        gate_spec=gates,
        config_path=config_path,
        gate_spec_path=gate_path,
    )

    readiness = report["trained_planner_readiness"]
    row = next(row for row in readiness["rows"] if row["planner_key"] == "fragile")
    assert row["eligible_for_trained_planner_claim"] is True
    assert row["readiness_status"] == "ready_for_fresh_probe"
    assert row["missing_fields"] == []


def test_declared_fallback_execution_excluded_from_trained_planner_claims(tmp_path: Path) -> None:
    """Declared fallback execution is distinct from checkpoint-backed eligibility."""

    config_path, _gate_path, config, _gates = _write_config_pair(tmp_path)
    config["arms"][0]["structural_class"] = "learned_policy"
    config["arms"][0]["algo"] = "ppo"
    config["arms"][0]["fallback_execution"] = True
    normalized = validate_probe_config(config, base_dir=config_path.parent)
    arm = normalized["arms"][0]
    assert arm["trained_planner_claim_status"] == "excluded_fallback_execution"
    assert arm["trained_planner_claim_exclusion"] == "fallback_execution"


def test_provenance_separates_certification_evaluation_and_development(tmp_path: Path) -> None:
    """Certification model is separate from declared policy development provenance."""

    config_path, gate_path, config, gates = _write_config_pair(tmp_path)
    config["arms"][0]["development_pedestrian_model"] = "training_manifest_declared_sfm"
    report = build_certification_transfer_report(
        [_record("stable", HSFM_TOTAL_FORCE_V1, collision_rate=0.0)],
        probe_config=config,
        gate_spec=gates,
        config_path=config_path,
        gate_spec_path=gate_path,
    )

    cell = next(
        row
        for row in report["gate_cells"]
        if row["planner_key"] == "stable" and row["evaluation_model"] == HSFM_TOTAL_FORCE_V1
    )
    assert cell["certification_pedestrian_model"] == HSFM_TOTAL_FORCE_V1
    assert cell["evaluation_model"] == HSFM_TOTAL_FORCE_V1
    assert cell["development_pedestrian_model"] == "training_manifest_declared_sfm"


def test_evidence_writer_emits_checksums_without_raw_artifacts(tmp_path: Path) -> None:
    """Evidence writer emits compact report files and excludes raw logs/videos/JSONL."""

    config_path, gate_path, config, gates = _write_config_pair(tmp_path)
    report = build_certification_transfer_report(
        [_record("stable", SOCIAL_FORCE_DEFAULT, collision_rate=0.0)],
        probe_config=config,
        gate_spec=gates,
        config_path=config_path,
        gate_spec_path=gate_path,
    )
    paths = write_certification_transfer_evidence(report, tmp_path / "evidence")

    assert Path(paths["sha256sums"]).exists()
    written_names = {path.name for path in (tmp_path / "evidence").iterdir()}
    assert "summary.json" in written_names
    assert "certification_transfer_matrix.csv" in written_names
    assert not any(
        path.suffix in {".jsonl", ".log", ".mp4"} for path in (tmp_path / "evidence").iterdir()
    )
    loaded = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    assert loaded["issue"] == 4207
    metadata = json.loads(Path(paths["metadata_json"]).read_text(encoding="utf-8"))
    assert metadata["trained_planner_readiness"]["schema_version"] == "trained-planner-readiness.v1"
    readme = Path(paths["readme"]).read_text(encoding="utf-8")
    assert "Trained-planner readiness" in readme
    with Path(paths["certification_transfer_matrix_csv"]).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows


def test_provenance_paths_are_not_absolute(tmp_path: Path) -> None:
    """Config/gate provenance paths must never leak an absolute home-dir path (#4324).

    The committed packet previously baked an author worktree path
    (``/home/<user>/git/robot_sf_ll7.worktrees/...``) into ``config.path`` /
    ``gate_spec.path``, which is non-reproducible across machines/CI.
    """
    config_path, gate_path, config, gates = _write_config_pair(tmp_path)
    report = build_certification_transfer_report(
        [_record("stable", SOCIAL_FORCE_DEFAULT, collision_rate=0.0)],
        probe_config=config,
        gate_spec=gates,
        config_path=config_path,
        gate_spec_path=gate_path,
    )
    for field in ("config", "gate_spec"):
        recorded = report[field]["path"]
        assert not Path(recorded).is_absolute(), f"{field}.path is absolute: {recorded}"
        for leak in ("/home/", "/Users/", "/root/", ".worktrees"):
            assert leak not in recorded, f"{field}.path leaks {leak!r}: {recorded}"


def test_repo_relative_path_normalizes_in_repo_and_falls_back() -> None:
    """`_repo_relative_path` emits repo-relative POSIX in-repo, basename otherwise."""
    in_repo = CERT_REPO_ROOT / "configs/benchmarks/issue_4207_interacting_smoke_probe.yaml"
    assert (
        _repo_relative_path(in_repo) == "configs/benchmarks/issue_4207_interacting_smoke_probe.yaml"
    )
    # A path outside this checkout (e.g. a sibling worktree) must not leak an
    # absolute home path; it collapses to the bare file name.
    outside = "/home/someone/git/robot_sf_ll7.worktrees/other/configs/x/probe.yaml"
    result = _repo_relative_path(outside)
    assert result == "probe.yaml"
    assert "/home/" not in result


def test_interaction_status_classifier_distinguishes_near_field_contact() -> None:
    """Proximity metrics decide interacting vs non_interacting vs unknown."""

    assert classify_interaction_status({"robot_ped_within_5m_frac": 0.4}) == "interacting"
    assert classify_interaction_status({"min_clearance_m": 1.2}) == "interacting"
    # No near-field contact: robot never within 5 m and min clearance far outside the band.
    assert (
        classify_interaction_status({"robot_ped_within_5m_frac": 0.0, "min_clearance_m": 20.0})
        == "non_interacting"
    )
    # No proximity metric at all (e.g. an empty / not_evaluable cell).
    assert classify_interaction_status({"collision_rate": 0.0}) == "unknown"
    assert classify_interaction_status({}) == "unknown"


def test_non_interacting_stable_transfer_is_flagged_not_model_robust(tmp_path: Path) -> None:
    """A stable_pass built from non_interacting cells must not read as model robustness."""

    config_path, gate_path, config, gates = _write_config_pair(tmp_path)
    # The "stable" arm passes the gates but never enters the 5 m pedestrian band (mirroring the
    # committed 2026-07 packet: robot_ped_within_5m_frac=0, min_clearance ~20 m); the "fragile"
    # arm here does enter the near field. Both use the 4-arm fixture config keys.
    records = [
        _record(
            "stable",
            SOCIAL_FORCE_DEFAULT,
            collision_rate=0.0,
            within_5m_frac=0.0,
            min_clearance_m=20.0,
        ),
        _record(
            "stable",
            HSFM_TOTAL_FORCE_V1,
            collision_rate=0.0,
            within_5m_frac=0.0,
            min_clearance_m=20.0,
        ),
        _record(
            "fragile",
            SOCIAL_FORCE_DEFAULT,
            collision_rate=0.0,
            within_5m_frac=0.3,
            min_clearance_m=1.5,
        ),
        _record(
            "fragile",
            HSFM_TOTAL_FORCE_V1,
            collision_rate=0.0,
            within_5m_frac=0.3,
            min_clearance_m=1.5,
        ),
    ]

    report = build_certification_transfer_report(
        records,
        probe_config=config,
        gate_spec=gates,
        config_path=config_path,
        gate_spec_path=gate_path,
    )

    stable_rows = [
        r for r in report["certification_transfer_matrix"] if r["planner_key"] == "stable"
    ]
    assert stable_rows
    assert all(r["transfer_status"] == "stable_pass" for r in stable_rows)
    # The stable status is vacuous: no cell entered the near field, so it is not exercised.
    assert all(r["interaction_status"] == "non_interacting" for r in stable_rows)
    assert all(r["interaction_exercised"] is False for r in stable_rows)

    contact_rows = [
        r for r in report["certification_transfer_matrix"] if r["planner_key"] == "fragile"
    ]
    assert all(r["interaction_status"] == "interacting" for r in contact_rows)
    assert all(r["interaction_exercised"] is True for r in contact_rows)

    # Model sensitivity is exercised overall because the fragile arm entered the near field.
    assert report["model_sensitivity_exercised"] is True
    assert report["interaction_status_counts"].get("non_interacting", 0) >= 1


def test_all_non_interacting_report_marks_sensitivity_unexercised(tmp_path: Path) -> None:
    """When every cell stays outside the near field, model sensitivity is not exercised."""

    config_path, gate_path, config, gates = _write_config_pair(tmp_path)
    records = [
        _record(arm, model, collision_rate=0.0, within_5m_frac=0.0, min_clearance_m=18.0)
        for arm in ("stable", "fragile", "conservative", "blocked")
        for model in (SOCIAL_FORCE_DEFAULT, HSFM_TOTAL_FORCE_V1)
    ]

    report = build_certification_transfer_report(
        records,
        probe_config=config,
        gate_spec=gates,
        config_path=config_path,
        gate_spec_path=gate_path,
    )

    assert report["model_sensitivity_exercised"] is False
    assert set(report["interaction_status_counts"]) <= {"non_interacting"}


def test_committed_interacting_smoke_family_exercises_model_sensitivity() -> None:
    """The committed interacting smoke family drives the guard to model_sensitivity_exercised=true.

    This is the end-to-end positive path over the real checked-in config, gate spec, and episode
    fixture (not synthetic in-test records): the companion francis2023_blind_corner run was all
    non_interacting, so this family is what flips the guard for issue #4207.
    """

    config = load_yaml_mapping(INTERACTING_SMOKE_CONFIG)
    gates = load_yaml_mapping(INTERACTING_SMOKE_GATES)
    records = [
        json.loads(line)
        for line in INTERACTING_SMOKE_FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    report = build_certification_transfer_report(
        records,
        probe_config=config,
        gate_spec=gates,
        config_path=INTERACTING_SMOKE_CONFIG,
        gate_spec_path=INTERACTING_SMOKE_GATES,
        generated_at_utc="2026-07-03T00:00:00+00:00",
    )

    # Every transfer cell entered the near field, so the guard reports exercised model sensitivity.
    assert report["model_sensitivity_exercised"] is True
    matrix = report["certification_transfer_matrix"]
    assert matrix
    assert all(row["interaction_status"] == "interacting" for row in matrix)
    assert all(row["interaction_exercised"] is True for row in matrix)
    assert set(report["interaction_status_counts"]) == {"interacting"}

    # A genuine interacting flip exists (model-assumption fragility, not a vacuous stable status).
    flips = report["flip_cases"]
    assert flips
    assert all(flip["interaction_exercised"] is True for flip in flips)
    assert {flip["planner_key"] for flip in flips} == {"ppo"}
    fragile = next(
        row
        for row in matrix
        if row["planner_key"] == "ppo"
        and row["certification_model"] == SOCIAL_FORCE_DEFAULT
        and row["evaluation_model"] == HSFM_TOTAL_FORCE_V1
    )
    assert fragile["transfer_status"] == "fragile_pass_to_fail"


def test_committed_interacting_smoke_config_and_gates_validate() -> None:
    """The committed interacting smoke config and gate spec pass schema validation."""

    config = load_yaml_mapping(INTERACTING_SMOKE_CONFIG)
    normalized = validate_probe_config(config, base_dir=INTERACTING_SMOKE_CONFIG.parent)
    assert normalized["scenario_family"] == "issue4207_interacting_smoke"
    assert normalized["pedestrian_models"] == [SOCIAL_FORCE_DEFAULT, HSFM_TOTAL_FORCE_V1]
    assert [arm["key"] for arm in normalized["arms"]] == [
        "goal",
        "ppo",
        "prediction_planner",
        "guarded_ppo",
    ]


def test_committed_physics_config_validates_and_uses_traversal_horizon() -> None:
    """The committed physics probe config validates and uses a horizon long enough for contact.

    The smoke config's horizon 60 (6 s) is too short for the robot to traverse the blind-corner
    L-route, which is why the real smoke-horizon run stayed non_interacting; the physics config
    raises the horizon to the scenario's own max_episode_steps (400) so the robot can reach the
    corridor the pedestrian walks.
    """

    config = load_yaml_mapping(INTERACTING_PHYSICS_CONFIG)
    normalized = validate_probe_config(config, base_dir=INTERACTING_PHYSICS_CONFIG.parent)
    assert normalized["scenario_family"] == "issue4207_interacting_smoke"
    assert normalized["pedestrian_models"] == [SOCIAL_FORCE_DEFAULT, HSFM_TOTAL_FORCE_V1]
    assert [arm["key"] for arm in normalized["arms"]] == [
        "goal",
        "ppo",
        "prediction_planner",
        "guarded_ppo",
    ]
    assert normalized["horizon"] >= 400
    assert normalized["paper_facing"] is False


def test_committed_physics_packet_records_real_near_field_contact() -> None:
    """The committed physics evidence packet proves nonzero near-field contact from simulation.

    Unlike the synthetic smoke packet (fixture design choices), this packet is generated from an
    actual CPU physics run. It must show at least one interacting cell with
    ``robot_ped_within_5m_frac > 0`` and portable, repo-relative provenance paths (no absolute
    home-dir leak), satisfying the issue #4207 / #4327 acceptance criterion.
    """

    summary = json.loads((INTERACTING_PHYSICS_PACKET / "summary.json").read_text(encoding="utf-8"))

    assert summary["issue"] == 4207
    assert summary["paper_facing"] is False
    assert summary["claim_boundary"] == CLAIM_BOUNDARY
    assert summary["trained_planner_claim_policy"]["excluded_statuses"] == [
        "excluded_missing_checkpoint_or_config",
        "excluded_fallback_execution",
    ]
    assert summary["trained_planner_claim_status_counts"]["excluded_missing_checkpoint_or_config"]
    assert summary["model_sensitivity_exercised"] is True
    metric_summary = summary["interaction_metric_summary"]
    assert metric_summary["physics_near_field_confirmed"] is True
    assert metric_summary["interacting_cell_count"] > 0
    assert metric_summary["max_robot_ped_within_5m_frac"] > 0.0
    assert metric_summary["min_clearance_m"] < 5.0

    interacting_cells = [
        cell for cell in summary["gate_cells"] if cell["interaction_status"] == "interacting"
    ]
    assert interacting_cells, "physics packet must contain at least one interacting cell"
    assert any(
        cell["metrics"].get("robot_ped_within_5m_frac", 0.0) > 0.0 for cell in interacting_cells
    ), "at least one interacting cell must record robot_ped_within_5m_frac > 0"
    # Clearance metrics are recorded for the interacting cells (issue acceptance criterion).
    assert all("min_clearance_m" in cell["metrics"] for cell in interacting_cells)

    # Provenance must be portable: committed evidence never bakes in absolute home-dir paths.
    for key in ("config", "gate_spec"):
        recorded = summary[key]["path"]
        assert not recorded.startswith(("/home/", "/Users/", "/root/")), recorded
        assert recorded.startswith("configs/"), recorded


def test_preflight_passes_when_required_gate_metrics_are_aggregatable(tmp_path: Path) -> None:
    """Preflight reports ok when every required gate metric is aggregatable."""

    _config_path, _gate_path, config, gates = _write_config_pair(tmp_path)
    normalized_config = validate_probe_config(config, base_dir=tmp_path)
    normalized_gates = validate_gate_spec(
        gates, scenario_family=normalized_config["scenario_family"]
    )

    result = preflight_gate_evaluability(normalized_config, normalized_gates)

    assert result["status"] == PREFLIGHT_OK
    assert result["not_evaluable_gate_ids"] == []
    assert result["not_evaluable_gate_metrics"] == []
    assert set(result["aggregatable_metrics"]) == set(AGGREGATABLE_METRICS)
    assert set(result["required_gate_metrics"]) <= set(AGGREGATABLE_METRICS)


def test_preflight_blocks_when_required_gate_metric_is_not_aggregatable(
    tmp_path: Path,
) -> None:
    """A required gate on a non-aggregatable metric fails closed as blocked_no_evaluable_gate_family."""

    _config_path, _gate_path, config, gates = _write_config_pair(tmp_path)
    gates["gates"][0]["metric"] = "bogus_metric"
    normalized_config = validate_probe_config(config, base_dir=tmp_path)
    normalized_gates = validate_gate_spec(
        gates, scenario_family=normalized_config["scenario_family"]
    )

    result = preflight_gate_evaluability(normalized_config, normalized_gates)

    assert result["status"] == PREFLIGHT_BLOCKED_NO_EVALUABLE_GATE_FAMILY
    assert result["not_evaluable_gate_ids"] == ["collision_rate_zero"]
    assert result["not_evaluable_gate_metrics"] == ["bogus_metric"]
    assert result["scenario_family"] == normalized_config["scenario_family"]


def test_preflight_optional_gate_with_non_aggregatable_metric_does_not_block(
    tmp_path: Path,
) -> None:
    """An optional gate on a non-aggregatable metric surfaces as not_evaluable but does not block."""

    _config_path, _gate_path, config, gates = _write_config_pair(tmp_path)
    gates["gates"].append(
        {
            "id": "optional_unknown_metric",
            "metric": "bogus_metric",
            "threshold": 0.0,
            "direction": "max",
            "category": "diagnostic",
            "provenance": "fixture",
            "required": False,
            "scope": {"scenario_family": "fixture_family"},
        }
    )
    normalized_config = validate_probe_config(config, base_dir=tmp_path)
    normalized_gates = validate_gate_spec(
        gates, scenario_family=normalized_config["scenario_family"]
    )

    result = preflight_gate_evaluability(normalized_config, normalized_gates)

    assert result["status"] == PREFLIGHT_OK
    assert result["not_evaluable_gate_ids"] == []


def test_runner_validate_only_passes_preflight_on_committed_smoke_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--validate-only runs preflight on the committed smoke config without simulation."""

    exit_code = _run_cert_transfer.main(
        [
            "--config",
            str(INTERACTING_SMOKE_CONFIG),
            "--gate-spec",
            str(INTERACTING_SMOKE_GATES),
            "--validate-only",
        ]
    )

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["preflight"]["status"] == PREFLIGHT_OK
    assert out["preflight"]["scenario_family"] == "issue4207_interacting_smoke"
    assert out["preflight"]["not_evaluable_gate_ids"] == []


def test_runner_validate_only_fails_closed_on_non_evaluable_gate_family(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--validate-only exits 2 with blocked_no_evaluable_gate_family for a non-aggregatable required gate."""

    config_path, gate_path = _write_runner_yaml_pair(tmp_path, gate_metric="bogus_metric")

    exit_code = _run_cert_transfer.main(
        [
            "--config",
            str(config_path),
            "--gate-spec",
            str(gate_path),
            "--validate-only",
        ]
    )

    assert exit_code == 2
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == PREFLIGHT_BLOCKED_NO_EVALUABLE_GATE_FAMILY
    assert out["not_evaluable_gate_metrics"] == ["bogus_metric"]
    assert "gate_one" in out["not_evaluable_gate_ids"]
    assert "aggregatable_metrics" in out


def test_runner_validate_only_does_not_require_output_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--validate-only succeeds without --output-dir and writes no evidence artifacts."""

    config_path, gate_path = _write_runner_yaml_pair(tmp_path)
    evidence_dir = tmp_path / "evidence"

    exit_code = _run_cert_transfer.main(
        [
            "--config",
            str(config_path),
            "--gate-spec",
            str(gate_path),
            "--validate-only",
        ]
    )

    assert exit_code == 0
    assert not evidence_dir.exists()


def _write_runner_yaml_pair(
    tmp_path: Path, *, gate_metric: str = "collision_rate", required: bool = True
) -> tuple[Path, Path]:
    """Write a minimal valid probe config and gate spec as real YAML for runner-level tests.

    Unlike ``_write_config_pair`` (which writes placeholder file bodies and returns in-memory
    mappings for ``build_certification_transfer_report`` tests), this helper writes YAML that the
    runner's ``load_yaml_mapping`` can actually parse, so ``main`` can be exercised end-to-end up
    to (but not including) simulation.
    """

    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("scenarios: []\n", encoding="utf-8")
    algo_path = tmp_path / "algo.yaml"
    algo_path.write_text("{}\n", encoding="utf-8")
    config = {
        "name": "issue_4207_certification_transfer_probe",
        "schema_version": "certification-transfer-probe.v1",
        "issue": 4207,
        "paper_facing": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "pedestrian_models": [SOCIAL_FORCE_DEFAULT, HSFM_TOTAL_FORCE_V1],
        "scenario_family": "fixture_family",
        "scenario_matrix": str(scenario_path),
        "seed_policy": {"mode": "fixed-list", "seeds": [111, 112, 113]},
        "arms": [
            {
                "key": "goal_a",
                "structural_class": "baseline_reactive",
                "algo": "goal",
                "algo_config": str(algo_path),
                "development_pedestrian_model": "unknown",
            },
            {
                "key": "goal_b",
                "structural_class": "baseline_reactive",
                "algo": "goal",
                "development_pedestrian_model": "unknown",
            },
            {
                "key": "goal_c",
                "structural_class": "baseline_reactive",
                "algo": "goal",
                "development_pedestrian_model": "unknown",
            },
        ],
        "horizon": 60,
        "dt": 0.1,
        "workers": 1,
    }
    gates = {
        "schema_version": "benchmark_release_gate_spec.v1",
        "gates": [
            {
                "id": "gate_one",
                "metric": gate_metric,
                "threshold": 0.0,
                "direction": "max",
                "category": "safety",
                "provenance": "fixture",
                "required": required,
                "scope": {"scenario_family": "fixture_family"},
            },
        ],
    }
    config_path = tmp_path / "config.yaml"
    gate_path = tmp_path / "gates.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    gate_path.write_text(yaml.safe_dump(gates), encoding="utf-8")
    return config_path, gate_path


def _write_config_pair(tmp_path: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("scenarios: []\n", encoding="utf-8")
    algo_path = tmp_path / "algo.yaml"
    algo_path.write_text("{}\n", encoding="utf-8")
    config: dict[str, object] = {
        "name": "issue_4207_certification_transfer_probe",
        "schema_version": "certification-transfer-probe.v1",
        "issue": 4207,
        "paper_facing": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "pedestrian_models": [SOCIAL_FORCE_DEFAULT, HSFM_TOTAL_FORCE_V1],
        "scenario_family": "fixture_family",
        "scenario_matrix": str(scenario_path),
        "seed_policy": {"mode": "fixed-list", "seeds": [111, 112, 113]},
        "arms": [
            {
                "key": "stable",
                "structural_class": "fixture",
                "algo": "goal",
                "algo_config": str(algo_path),
                "development_pedestrian_model": "unknown",
            },
            {"key": "fragile", "structural_class": "fixture", "algo": "goal"},
            {"key": "conservative", "structural_class": "fixture", "algo": "goal"},
            {"key": "blocked", "structural_class": "fixture", "algo": "goal"},
        ],
    }
    gates: dict[str, object] = {
        "schema_version": "benchmark_release_gate_spec.v1",
        "gates": [
            {
                "id": "collision_rate_zero",
                "metric": "collision_rate",
                "threshold": 0.0,
                "direction": "max",
                "category": "safety",
                "provenance": "fixture",
                "required": True,
                "scope": {"scenario_family": "fixture_family"},
            },
            {
                "id": "near_miss_rate_limit",
                "metric": "near_miss_rate",
                "threshold": 0.05,
                "direction": "max",
                "category": "safety",
                "provenance": "fixture",
                "required": True,
                "scope": {"scenario_family": "fixture_family"},
            },
        ],
    }
    config_path = tmp_path / "config.yaml"
    gate_path = tmp_path / "gates.yaml"
    config_path.write_text("fixture\n", encoding="utf-8")
    gate_path.write_text("fixture\n", encoding="utf-8")
    return config_path, gate_path, config, gates


def _record(
    planner_key: str,
    evaluation_model: str,
    *,
    collision_rate: float = 0.0,
    include_required: bool = True,
    within_5m_frac: float | None = None,
    min_clearance_m: float | None = None,
) -> dict[str, object]:
    metrics: dict[str, float] = {"collision_rate": collision_rate}
    if include_required:
        metrics["near_miss_rate"] = 0.0
    if within_5m_frac is not None:
        metrics["robot_ped_within_5m_frac"] = within_5m_frac
    if min_clearance_m is not None:
        metrics["min_clearance_m"] = min_clearance_m
    return {
        "planner_key": planner_key,
        "scenario_family": "fixture_family",
        "evaluation_pedestrian_model": evaluation_model,
        "certification_pedestrian_model": evaluation_model,
        "development_pedestrian_model": "unknown",
        "metrics": metrics,
    }
