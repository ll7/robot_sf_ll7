"""Fixture-only characterization of the camera-ready campaign orchestrator (#7327).

These tests lock observable coordination behavior before a later production refactor:

* the six coordinator phases and their injected values;
* fail-closed exception propagation and partial-output boundaries;
* stable return-path/status vocabulary;
* resume context, partial episode counts, and deterministic plan projections.

The tests never invoke a campaign worker, planner, benchmark, publication bundle, or external
process. Existing subprocess and GPU-cleanup suites remain the executable contract for those
lower-level boundaries.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

import robot_sf.benchmark.camera_ready.campaign as campaign_module
from robot_sf.benchmark.camera_ready._resume_plan import ResumeMismatchError

if TYPE_CHECKING:
    from pathlib import Path


def _fixture_paths(tmp_path: Path) -> SimpleNamespace:
    """Return the minimum preflight path surface consumed by the coordinator."""
    campaign_root = tmp_path / "fixture_campaign"
    reports_dir = campaign_root / "reports"
    return SimpleNamespace(
        campaign_id="fixture-campaign",
        campaign_root=campaign_root,
        reports_dir=reports_dir,
        manifest_payload={"campaign_id": "fixture-campaign"},
        scenarios=[{"name": "fixture-scenario"}],
        resolved_seeds=[111],
        config_hash="fixture-config-hash",
        matrix_summary_json_path=reports_dir / "matrix_summary.json",
        matrix_summary_csv_path=reports_dir / "matrix_summary.csv",
    )


def _fixture_artifacts(tmp_path: Path) -> SimpleNamespace:
    """Return a path-complete report artifact stub for return-contract assertions."""
    reports_dir = tmp_path / "fixture_campaign" / "reports"
    outcome = SimpleNamespace(
        campaign_outcome=SimpleNamespace(
            non_success_runs=0,
            accepted_unavailable_runs=0,
            unexpected_failed_runs=0,
        ),
        successful_runs=1,
        campaign_status_axes=SimpleNamespace(campaign_execution_status="completed"),
        row_status_summary={"successful_evidence_rows": 1},
        success_counters={
            "benchmark_success_basis": "core",
            "core_successful_runs": 1,
            "core_total_runs": 1,
        },
        campaign_evidence_status="diagnostic-only",
        campaign_status="completed",
        campaign_status_reason="fixture completion",
        campaign_exit_code=0,
        benchmark_success=False,
        total_episodes=1,
        runtime_sec=0.1,
    )
    snqi = SimpleNamespace(
        snqi_diagnostics_json_path=reports_dir / "snqi_diagnostics.json",
        snqi_diagnostics_md_path=reports_dir / "snqi_diagnostics.md",
        snqi_sensitivity_csv_path=reports_dir / "snqi_sensitivity.csv",
        soft_contract_warning=False,
    )
    path_values = {
        "summary_json_path": reports_dir / "campaign_summary.json",
        "report_md_path": reports_dir / "campaign_report.md",
        "credibility_scorecard_json_path": reports_dir / "campaign_credibility_scorecard.json",
        "csv_path": reports_dir / "campaign_table.csv",
        "md_table_path": reports_dir / "campaign_table.md",
        "core_csv_path": reports_dir / "campaign_table_core.csv",
        "core_md_path": reports_dir / "campaign_table_core.md",
        "experimental_csv_path": reports_dir / "campaign_table_experimental.csv",
        "experimental_md_path": reports_dir / "campaign_table_experimental.md",
        "scenario_csv_path": reports_dir / "scenario_breakdown.csv",
        "scenario_md_path": reports_dir / "scenario_breakdown.md",
        "family_csv_path": reports_dir / "scenario_family_breakdown.csv",
        "family_md_path": reports_dir / "scenario_family_breakdown.md",
        "parity_csv_path": reports_dir / "kinematics_parity_table.csv",
        "parity_md_path": reports_dir / "kinematics_parity_table.md",
        "skipped_csv_path": reports_dir / "kinematics_skipped_combinations.csv",
        "skipped_md_path": reports_dir / "kinematics_skipped_combinations.md",
        "seed_variability_json_path": reports_dir / "seed_variability_by_scenario.json",
        "seed_variability_csv_path": reports_dir / "seed_variability_by_scenario.csv",
        "seed_episode_rows_csv_path": reports_dir / "seed_episode_rows.csv",
        "statistical_sufficiency_json_path": reports_dir / "statistical_sufficiency.json",
        "actuation_envelope_json_path": None,
        "actuation_envelope_md_path": None,
    }
    return SimpleNamespace(outcome=outcome, snqi=snqi, **path_values)


def test_orchestrator_phase_ledger_is_stable_and_injected_values_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coordinator preserves phase order and forwards each collaborator's values."""
    paths = _fixture_paths(tmp_path)
    artifacts = _fixture_artifacts(tmp_path)
    dependencies = SimpleNamespace(name="fixture-dependencies")
    phase_ledger: list[str] = []
    cfg = SimpleNamespace(name="fixture-config")

    def prepare(
        received_cfg, received_dependencies, output_root, label, campaign_id, invoked_command
    ):
        assert received_cfg is cfg
        assert received_dependencies is dependencies
        assert (output_root, label, campaign_id, invoked_command) == (
            tmp_path / "output",
            "fixture",
            "campaign-id",
            "fixture command",
        )
        phase_ledger.append("preflight")
        return paths, {"weights": "fixture"}, {"baseline": "fixture"}, 12.5

    def matrix(
        received_cfg, received_paths, weights, baseline, received_dependencies, arm_isolation
    ):
        assert (received_cfg, received_paths, received_dependencies) == (cfg, paths, dependencies)
        assert (weights, baseline, arm_isolation) == (
            {"weights": "fixture"},
            {"baseline": "fixture"},
            "in_process",
        )
        phase_ledger.append("planner_matrix")
        return ["run-entry"], ["planner-row"], ["warning"], ["seed-record"], ("differential_drive",)

    def integrity(received_cfg, **kwargs):
        assert received_cfg is cfg
        assert kwargs["manifest_payload"] is paths.manifest_payload
        assert kwargs["run_entries"] == ["run-entry"]
        phase_ledger.append("integrity")
        return {"status": "valid"}, ["arm-rollup"], {"status": "not_applicable"}

    def report(received_cfg, **kwargs):
        assert received_cfg is cfg
        assert kwargs["paths"] is paths
        assert kwargs["start"] == 12.5
        phase_ledger.append("report")
        return artifacts

    def finalize(received_cfg, **kwargs):
        assert received_cfg is cfg
        assert kwargs["paths"] is paths
        assert kwargs["artifacts"] is artifacts
        assert kwargs["skip_publication_bundle"] is True
        phase_ledger.append("finalize")
        return {"status": "skipped", "reason": "fixture"}

    def build_return(**kwargs):
        assert kwargs["paths"] is paths
        assert kwargs["artifacts"] is artifacts
        assert kwargs["run_entries"] == ["run-entry"]
        assert kwargs["campaign_integrity"] == {"status": "valid"}
        phase_ledger.append("return")
        return {"status": "fixture", "publication_bundle": kwargs["publication_payload"]}

    monkeypatch.setattr(campaign_module, "_prepare_campaign_execution", prepare)
    monkeypatch.setattr(campaign_module, "_execute_planner_matrix_phase", matrix)
    monkeypatch.setattr(campaign_module, "_post_run_integrity_and_fairness", integrity)
    monkeypatch.setattr(campaign_module, "_write_campaign_report_artifacts", report)
    monkeypatch.setattr(campaign_module, "_finalize_campaign_outputs", finalize)
    monkeypatch.setattr(campaign_module, "_build_orchestrator_return", build_return)

    result = campaign_module._run_campaign_orchestrator(
        cfg,
        output_root=tmp_path / "output",
        label="fixture",
        campaign_id="campaign-id",
        skip_publication_bundle=True,
        invoked_command="fixture command",
        dependencies=dependencies,
        arm_isolation="in_process",
    )

    assert phase_ledger == [
        "preflight",
        "planner_matrix",
        "integrity",
        "report",
        "finalize",
        "return",
    ]
    ledger_bytes = json.dumps(phase_ledger, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(ledger_bytes).hexdigest() == (
        "26008da2515f471faba6b1da3435619174569a33f24eac2780d526293a2e230f"
    )
    assert result == {
        "status": "fixture",
        "publication_bundle": {"status": "skipped", "reason": "fixture"},
    }


@pytest.mark.parametrize(
    ("failed_phase", "expected_phases"),
    [
        ("preflight", ["preflight"]),
        ("planner_matrix", ["preflight", "planner_matrix"]),
        ("integrity", ["preflight", "planner_matrix", "integrity"]),
        ("report", ["preflight", "planner_matrix", "integrity", "report"]),
        ("finalize", ["preflight", "planner_matrix", "integrity", "report", "finalize"]),
    ],
)
def test_orchestrator_phase_failures_are_not_swallowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_phase: str,
    expected_phases: list[str],
) -> None:
    """Preflight, worker, report, and finalization failures preserve their original exception."""
    paths = _fixture_paths(tmp_path)
    artifacts = _fixture_artifacts(tmp_path)
    dependencies = SimpleNamespace(name="fixture-dependencies")
    phase_ledger: list[str] = []
    cfg = SimpleNamespace(name="fixture-config")

    def maybe_fail(phase: str) -> None:
        phase_ledger.append(phase)
        if phase == failed_phase:
            raise RuntimeError(f"fixture-{phase}-failure")

    def prepare(*_args, **_kwargs):
        maybe_fail("preflight")
        return paths, None, None, 12.5

    def matrix(*_args, **_kwargs):
        maybe_fail("planner_matrix")
        return [], [], [], [], ("differential_drive",)

    def integrity(*_args, **_kwargs):
        maybe_fail("integrity")
        return {}, [], {}

    def report(*_args, **_kwargs):
        maybe_fail("report")
        return artifacts

    def finalize(*_args, **_kwargs):
        maybe_fail("finalize")

    monkeypatch.setattr(campaign_module, "_prepare_campaign_execution", prepare)
    monkeypatch.setattr(campaign_module, "_execute_planner_matrix_phase", matrix)
    monkeypatch.setattr(campaign_module, "_post_run_integrity_and_fairness", integrity)
    monkeypatch.setattr(campaign_module, "_write_campaign_report_artifacts", report)
    monkeypatch.setattr(campaign_module, "_finalize_campaign_outputs", finalize)
    monkeypatch.setattr(
        campaign_module,
        "_build_orchestrator_return",
        lambda **_kwargs: pytest.fail("return construction must not follow a failed phase"),
    )

    with pytest.raises(RuntimeError, match=f"fixture-{failed_phase}-failure"):
        campaign_module._run_campaign_orchestrator(
            cfg,
            dependencies=dependencies,
            skip_publication_bundle=True,
        )

    assert phase_ledger == expected_phases


def test_orchestrator_return_contract_preserves_artifact_paths_and_status_axes(
    tmp_path: Path,
) -> None:
    """The final return maps stable artifact names and status axes without claiming success."""
    paths = _fixture_paths(tmp_path)
    artifacts = _fixture_artifacts(tmp_path)
    result = campaign_module._build_orchestrator_return(
        paths=paths,
        artifacts=artifacts,
        run_entries=[{"status": "ok"}],
        campaign_integrity={"status": "valid"},
        warnings=["fixture warning"],
        publication_payload=None,
    )

    assert result["campaign_id"] == "fixture-campaign"
    assert result["campaign_root"] == str(paths.campaign_root)
    assert result["summary_json"].endswith("reports/campaign_summary.json")
    assert result["table_csv"].endswith("reports/campaign_table.csv")
    assert result["report_md"].endswith("reports/campaign_report.md")
    assert result["matrix_summary_json"].endswith("reports/matrix_summary.json")
    assert result["matrix_summary_csv"].endswith("reports/matrix_summary.csv")
    assert result["campaign_execution_status"] == "completed"
    assert result["evidence_status"] == "diagnostic-only"
    assert result["benchmark_success"] is False
    assert result["status"] == "completed"
    assert result["status_reason"] == "fixture completion"
    assert result["publication_bundle"] is None
    assert result["campaign_integrity"] == {"status": "valid"}
    assert result["warnings"] == ["fixture warning"]


def test_finalize_registers_crosswalk_before_final_artifact_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finalization writes the crosswalk reference before the final export stage observes it."""
    paths = _fixture_paths(tmp_path)
    artifacts = _fixture_artifacts(tmp_path)
    dependencies = SimpleNamespace(name="fixture-dependencies")
    events: list[str] = []
    campaign_summary = {"artifacts": {}}

    def build_summary(*_args, **_kwargs):
        events.append("summary")
        return campaign_summary, {}

    def write_crosswalk(reports_dir, **kwargs):
        assert reports_dir is paths.reports_dir
        assert kwargs["campaign_id"] == paths.campaign_id
        events.append("crosswalk")
        return reports_dir / "report_crosswalk.json"

    def export_final(*args, **_kwargs):
        events.append("export")
        observed_summary = args[3]
        assert observed_summary["artifacts"]["report_crosswalk_json"].endswith(
            "report_crosswalk.json"
        )
        return {"status": "skipped"}

    monkeypatch.setattr(campaign_module, "_build_summary_and_write_run_files", build_summary)
    monkeypatch.setattr(campaign_module, "write_crosswalk_sidecar", write_crosswalk)
    monkeypatch.setattr(campaign_module, "_export_and_write_final_artifacts", export_final)

    result = campaign_module._finalize_campaign_outputs(
        SimpleNamespace(name="fixture-config"),
        paths=paths,
        artifacts=artifacts,
        run_entries=[],
        planner_rows=[],
        campaign_integrity={"status": "valid"},
        arm_rollup=[],
        fairness_report={},
        warnings=[],
        kinematics_matrix=("differential_drive",),
        invoked_command=None,
        skip_publication_bundle=True,
        dependencies=dependencies,
    )

    assert events == ["summary", "crosswalk", "export"]
    assert result == {"status": "skipped"}


def test_resume_partial_plan_is_deterministic_and_context_mismatch_is_fail_closed(
    tmp_path: Path,
) -> None:
    """Partial episodes produce a stable projection, while stale campaign context stops resume."""
    campaign_root = tmp_path / "campaign"
    runs_dir = campaign_root / "runs"
    arm_dir = runs_dir / "goal__differential_drive"
    arm_dir.mkdir(parents=True)
    (campaign_root / "campaign_manifest.json").write_text(
        json.dumps({"campaign_id": "campaign", "config_hash": "config"}) + "\n",
        encoding="utf-8",
    )
    (arm_dir / "episodes.jsonl").write_text('{"episode_id": "fixture-1"}\n', encoding="utf-8")
    cfg = SimpleNamespace(
        resume=True,
        planners=(SimpleNamespace(key="goal", enabled=True),),
        kinematics_matrix=("differential_drive",),
    )
    kwargs = {
        "cfg": cfg,
        "campaign_id": "campaign",
        "config_hash": "config",
        "campaign_root": campaign_root,
        "runs_dir": runs_dir,
        "scenarios": [{"name": "fixture", "repeats": 2}],
    }

    verdicts = campaign_module._emit_resume_plan_preflight(**kwargs)
    assert len(verdicts) == 1
    assert verdicts[0].verdict == "continue-from-1"
    assert verdicts[0].episodes_remaining == 1

    plan_path = campaign_root / "resume_plan.json"
    first = json.loads(plan_path.read_text(encoding="utf-8"))
    assert first["schema_version"] == "benchmark-resume-plan.v1"
    assert first["context_check"] == {"config_hash_match": True, "campaign_id_match": True}
    assert first["episodes_banked"] == 1
    assert first["episodes_to_run"] == 1
    first_projection = dict(first)
    first_projection.pop("generated_at_utc")

    campaign_module._emit_resume_plan_preflight(**kwargs)
    second_projection = json.loads(plan_path.read_text(encoding="utf-8"))
    second_projection.pop("generated_at_utc")
    assert second_projection == first_projection
    projection_bytes = json.dumps(first_projection, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    assert hashlib.sha256(projection_bytes).hexdigest() == (
        "cb498b5a51593e7342bf9ee19c8503401e04416b23fdab53f9a76f7970e59443"
    )

    (campaign_root / "campaign_manifest.json").write_text(
        json.dumps({"campaign_id": "campaign", "config_hash": "stale-config"}) + "\n",
        encoding="utf-8",
    )
    prior_plan_bytes = plan_path.read_bytes()
    with pytest.raises(ResumeMismatchError, match="config-hash mismatch"):
        campaign_module._emit_resume_plan_preflight(**kwargs)
    assert plan_path.read_bytes() == prior_plan_bytes


def test_partial_duplicate_episode_ledger_is_not_accepted_as_valid_campaign_state(
    tmp_path: Path,
) -> None:
    """Duplicate logical coverage remains a structured blocker, not a successful resume receipt."""
    episodes_path = tmp_path / "runs" / "goal__differential_drive" / "episodes.jsonl"
    episodes_path.parent.mkdir(parents=True)
    row = {
        "scenario_id": "fixture",
        "seed": 111,
        "config_hash": "config",
        "git_hash": "commit",
        "result_provenance": {
            "scenario_id": "fixture",
            "seed": 111,
            "config_hash": "config",
            "repo_commit": "commit",
        },
    }
    episodes_path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )

    verdict = campaign_module.validate_campaign_integrity(
        [
            {
                "status": "ok",
                "planner": {"key": "goal", "kinematics": "differential_drive"},
                "episodes_path": str(episodes_path),
                "summary": {"episodes_total": 2},
            }
        ],
        scenarios=[{"id": "fixture", "seeds": [111]}],
        resolved_seeds=[111],
        campaign_root=tmp_path,
        campaign_manifest={"git": {"commit": "commit"}},
    )

    assert verdict["status"] == "invalid"
    assert any(
        blocker["invariant"] == "duplicate_logical_coverage" for blocker in verdict["blockers"]
    )
