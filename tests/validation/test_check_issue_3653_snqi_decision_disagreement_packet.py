"""Tests for the issue #3653 SNQI decision-disagreement packet checker."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/benchmarks/issue_3653_snqi_decision_disagreement_packet.yaml"
SCRIPT = REPO_ROOT / "scripts/validation/check_issue_3653_snqi_decision_disagreement_packet.py"

_SPEC = importlib.util.spec_from_file_location("_issue_3653_snqi_packet_check", SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _load_packet() -> dict:
    payload = yaml.safe_load(PACKET.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _point_packet_at_episodes(packet: dict, episodes: Path) -> None:
    episodes_rel = os.path.relpath(episodes, REPO_ROOT)
    packet["inputs"]["episodes_jsonl"] = episodes_rel
    packet["raw_episode_artifact"]["expected_path"] = episodes_rel
    command_template = packet["export"]["command_template"]
    command_template[command_template.index("--episodes") + 1] = episodes_rel


def _write_ready_episode_fixture(
    path: Path,
    *,
    missing_metric: str | None = None,
    malformed_metric: tuple[str, float] | None = None,
) -> None:
    planners = [
        "goal",
        "hybrid_rule_v3_fast_progress_static_escape",
        "orca",
        "ppo",
        "prediction_planner",
        "sacadrl",
        "scenario_adaptive_hybrid_orca_v1",
        "social_force",
        "socnav_sampling",
    ]
    rows = []
    for planner_index, planner in enumerate(planners):
        for episode_index in range(960):
            rows.append(
                {
                    "scenario_id": "crossing",
                    "horizon": 500,
                    "planner_key": planner,
                    "metrics": {
                        "success": 1,
                        "time_to_goal_norm": min(0.95, 0.05 + planner_index * 0.08),
                        "collisions": 1 if planner_index == 0 else 0,
                        "near_misses": planner_index % 3,
                        "comfort_exposure": 0.01 * planner_index,
                        "force_exceed_events": float(planner_index + 1),
                        "jerk_mean": 0.08 + planner_index * 0.01,
                    },
                    "episode_index": episode_index,
                }
            )
    if missing_metric is not None:
        rows[0]["metrics"].pop(missing_metric)
    if malformed_metric is not None:
        metric, value = malformed_metric
        if metric not in rows[0]["metrics"]:
            raise KeyError(f"fixture metric not found: {metric}")
        rows[0]["metrics"][metric] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_issue_3653_packet_passes_fail_closed_contract() -> None:
    """The checked packet is a no-submit, artifact-blocked application contract."""

    summary = _MODULE.validate_packet(_load_packet())

    assert summary["status"] == "ok"
    assert summary["issue"] == 3653
    assert summary["current_status"] == "diagnostic_application_exported"
    assert summary["target_host"] == "imech036"
    assert summary["source_job_id"] == 13175
    assert summary["expected_episode_count"] == 8640
    assert summary["artifact_count"] == 5
    assert summary["evidence_artifact_count"] == 6
    assert summary["decision_disagreement_rate"] == 0.1388888888888889
    assert (
        summary["evidence_artifact_root"]
        == "docs/context/evidence/issue_3653_snqi_decision_disagreement_job_13175"
    )
    assert (
        summary["raw_episode_artifact_status"]
        == "hydrated_from_submit_host_recorded_job_13175"
    )
    assert summary["raw_episode_artifact_sha256"] == "fd15480d6892dd634e374fb9f79e1e3600d24c88604d9ff05f33d8227b4e6460"
    assert (
        summary["evidence_packet"]
        == "docs/context/evidence/issue_3798_post_13175_s20_s30_evidence_gap_packet.json"
    )
    assert (
        summary["episodes_jsonl"]
        == "output/issue1554-s20-h500-l40s-mem180/13175/reports/episodes.jsonl"
    )


def test_issue_3653_packet_rejects_untracked_evidence_packet() -> None:
    """The packet must point at the tracked 13175 diagnostic metadata."""

    packet = _load_packet()
    packet["source_campaign"]["evidence_packet"] = (
        "docs/context/evidence/issue_3798_post_13175_s20_s30_evidence_gap_packet.md"
    )

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "evidence packet hash mismatch" in str(exc)
    else:
        raise AssertionError("packet should reject untracked or mismatched evidence metadata")


def test_issue_3653_packet_rejects_claim_boundary_without_paper_guard() -> None:
    """The packet must not be broadened into paper-facing claims."""

    packet = _load_packet()
    packet["claim_boundary"] = (
        "Diagnostic packet only. This does not run a full benchmark campaign, "
        "does not submit Slurm/GPU work, and does not establish SNQI as a "
        "primary safety ranking."
    )

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "paper/dissertation" in str(exc)
    else:
        raise AssertionError("packet should reject missing paper claim guard")


def test_issue_3653_packet_rejects_compute_submission() -> None:
    """Compute submission remains outside this issue slice."""

    packet = _load_packet()
    packet["execution_boundary"]["compute_submit_authorized"] = True

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "compute_submit_authorized must be false" in str(exc)
    else:
        raise AssertionError("packet should reject compute submission authorization")


def test_issue_3653_packet_rejects_synthetic_raw_episode_source() -> None:
    """Raw campaign episodes must come from a durable job source, not a fixture shortcut."""

    packet = _load_packet()
    artifact = packet["raw_episode_artifact"]
    artifact["allowed_sources"].append("synthesized_fixture")
    artifact["forbidden_sources"].remove("synthesized_fixture")

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "allowed_sources mismatch" in str(exc)
    else:
        raise AssertionError("packet should reject synthetic raw episode acquisition")


def test_issue_3653_packet_rejects_obsolete_blocked_raw_episode_status() -> None:
    """The exported packet must not regress to the pre-hydration blocked status."""

    packet = _load_packet()
    packet["raw_episode_artifact"]["current_status"] = "blocked_until_durable_episode_jsonl_promoted_or_hydratable"

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "raw_episode_artifact.current_status mismatch" in str(exc)
    else:
        raise AssertionError("packet should reject obsolete raw episode blocked status")


def test_issue_3653_packet_rejects_missing_decision_disagreement_artifact() -> None:
    """The empirical application must export the decision-disagreement CSV."""

    packet = _load_packet()
    packet["export"]["required_artifacts"].remove(
        "snqi_scalarization_sensitivity_decision_disagreement.csv"
    )

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "required_artifacts mismatch" in str(exc)
    else:
        raise AssertionError("packet should reject missing decision-disagreement export")


def test_issue_3653_packet_rejects_malformed_evidence_file_entry() -> None:
    """Evidence artifact file entries fail closed when malformed."""

    packet = _load_packet()
    packet["evidence_artifacts"]["files"].append("not-a-file-entry")

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "each evidence file entry must be a mapping" in str(exc)
    else:
        raise AssertionError("packet should reject malformed evidence file entries")


def test_issue_3653_check_cli_json() -> None:
    """The checker CLI returns a machine-readable success summary."""

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--packet", str(PACKET), "--json"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["issue"] == 3653
    assert payload["current_status"] == "diagnostic_application_exported"


def test_issue_3653_export_if_ready_blocks_missing_campaign_input(tmp_path: Path) -> None:
    """The executable handoff refuses export when job 13175 JSONL is absent."""

    packet = _load_packet()
    _point_packet_at_episodes(packet, tmp_path / "missing" / "episodes.jsonl")

    try:
        _MODULE.export_if_ready(packet)
    except _MODULE.PacketError as exc:
        assert "episodes_jsonl missing" in str(exc)
        assert "blocked_missing_valid_campaign_episodes" in str(exc)
    else:
        raise AssertionError("export_if_ready should block missing raw campaign episodes")

def test_issue_3653_export_report_contract_rejects_missing_required_field() -> None:
    """Application export success requires configured decision/Pareto report sections."""

    packet = _load_packet()
    report = {
        field: {}
        for field in packet["success_criteria"]["required_report_fields"]
        if field != "decision_disagreement"
    }

    try:
        _MODULE._require_export_report_fields(packet, report)
    except _MODULE.PacketError as exc:
        assert "export report missing required fields" in str(exc)
        assert "decision_disagreement" in str(exc)
    else:
        raise AssertionError("missing report fields should fail closed")


def test_issue_3653_export_report_contract_rejects_empty_pareto_points() -> None:
    """Application export success requires populated Pareto-front evidence."""

    packet = _load_packet()
    report = {
        "decision_disagreement": {
            "snqi_winner": "planner_a",
            "constraints_first_winner": "planner_b",
            "winner_disagreement": True,
            "pairwise_disagreement_rate": 0.5,
            "pairwise_reversal_count": 1,
        },
        "pareto_front": {"x": "constraints_first_score", "y": "snqi_mean", "points": []},
        "planner_rows": [{"planner": "planner_a"}],
        "weight_zero_ablation": {"w_success": {}},
        "weight_sweep": {"w_success": []},
        "term_dominance": [{"component": "success"}],
    }

    try:
        _MODULE._require_export_report_fields(packet, report)
    except _MODULE.PacketError as exc:
        assert "pareto_front.points" in str(exc)
    else:
        raise AssertionError("empty Pareto-front export should fail closed")


def test_issue_3653_export_report_contract_rejects_invalid_decision_rate() -> None:
    """Application export success requires finite decision-disagreement values."""

    packet = _load_packet()
    report = {
        "decision_disagreement": {
            "snqi_winner": "planner_a",
            "constraints_first_winner": "planner_b",
            "winner_disagreement": True,
            "pairwise_disagreement_rate": 1.5,
            "pairwise_reversal_count": 1,
        },
        "pareto_front": {
            "x": "constraints_first_score",
            "y": "snqi_mean",
            "points": [
                {
                    "planner": "planner_a",
                    "constraints_first_score": 0.4,
                    "snqi_mean": 0.2,
                }
            ],
        },
        "planner_rows": [{"planner": "planner_a"}],
        "weight_zero_ablation": {"w_success": {}},
        "weight_sweep": {"w_success": []},
        "term_dominance": [{"component": "success"}],
    }

    try:
        _MODULE._require_export_report_fields(packet, report)
    except _MODULE.PacketError as exc:
        assert "decision_disagreement rate" in str(exc)
    else:
        raise AssertionError("invalid decision-disagreement rate should fail closed")


def test_issue_3653_export_report_contract_rejects_fractional_reversal_count() -> None:
    """Decision-disagreement reversal counts must be whole counts."""

    packet = _load_packet()
    report = {
        "decision_disagreement": {
            "snqi_winner": "planner_a",
            "constraints_first_winner": "planner_b",
            "winner_disagreement": True,
            "pairwise_disagreement_rate": 0.5,
            "pairwise_reversal_count": 1.5,
        },
        "pareto_front": {
            "x": "constraints_first_score",
            "y": "snqi_mean",
            "points": [
                {
                    "planner": "planner_a",
                    "constraints_first_score": 0.4,
                    "snqi_mean": 0.2,
                }
            ],
        },
        "planner_rows": [{"planner": "planner_a"}],
        "weight_zero_ablation": {"w_success": {}},
        "weight_sweep": {"w_success": []},
        "term_dominance": [{"component": "success"}],
    }

    try:
        _MODULE._require_export_report_fields(packet, report)
    except _MODULE.PacketError as exc:
        assert "reversal count must be finite integer" in str(exc)
    else:
        raise AssertionError("fractional reversal count should fail closed")


def test_issue_3653_export_if_ready_blocks_missing_normalized_metric(tmp_path: Path) -> None:
    """Application export refuses campaign-shaped rows missing normalized SNQI terms."""

    packet = _load_packet()
    episodes = tmp_path / "episodes.jsonl"
    output_dir = tmp_path / "export"
    _write_ready_episode_fixture(episodes, missing_metric="time_to_goal_norm")
    _point_packet_at_episodes(packet, episodes)

    try:
        _MODULE.export_if_ready(packet, output_dir=output_dir)
    except _MODULE.PacketError as exc:
        assert "SNQI scalarization-sensitivity preflight not ready: blocked" in str(exc)
    else:
        raise AssertionError("missing normalized metric should fail closed before export")

    assert not output_dir.exists()


def test_issue_3653_export_if_ready_blocks_malformed_normalized_metric(tmp_path: Path) -> None:
    """Application export refuses out-of-range bounded normalized SNQI terms."""

    packet = _load_packet()
    episodes = tmp_path / "episodes.jsonl"
    output_dir = tmp_path / "export"
    _write_ready_episode_fixture(
        episodes,
        malformed_metric=("time_to_goal_norm", 1.8),
    )
    _point_packet_at_episodes(packet, episodes)

    try:
        _MODULE.export_if_ready(packet, output_dir=output_dir)
    except _MODULE.PacketError as exc:
        assert "SNQI scalarization-sensitivity preflight not ready: malformed" in str(exc)
    else:
        raise AssertionError("malformed normalized metric should fail closed before export")

    assert not output_dir.exists()


def test_issue_3653_export_if_ready_writes_required_artifacts(tmp_path: Path) -> None:
    """Ready campaign-shaped inputs run through preflight and emit the packet artifact set."""

    packet = _load_packet()
    episodes = tmp_path / "episodes.jsonl"
    output_dir = tmp_path / "export"
    _write_ready_episode_fixture(episodes)
    _point_packet_at_episodes(packet, episodes)

    summary = _MODULE.export_if_ready(packet, output_dir=output_dir)

    assert summary["status"] == "exported"
    assert summary["preflight_status"] == "ready"
    assert summary["decision_disagreement_rate"] >= 0.0
    assert (output_dir / "preflight.json").is_file()
    for artifact in packet["export"]["required_artifacts"]:
        assert (output_dir / artifact).is_file()
