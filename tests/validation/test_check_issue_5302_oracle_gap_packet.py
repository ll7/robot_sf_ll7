"""Tests for the issue #5302 oracle-gap analysis packet checker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.benchmark.campaign_arm_admission import (
    ArmAdmissionFinding,
    CampaignArmAdmissionError,
    check_campaign_arm_admission,
)
from scripts.validation import check_issue_5302_oracle_gap_packet as checker

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET_PATH = REPO_ROOT / "configs/analysis/issue_5302_oracle_gap_packet.yaml"


def _packet() -> dict[str, Any]:
    payload = yaml.safe_load(PACKET_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _required(packet: dict[str, Any]) -> list[dict[str, Any]]:
    roster = packet["planner_roster"]
    assert isinstance(roster, dict)
    required = roster["required"]
    assert isinstance(required, list)
    return required


def _arm(packet: dict[str, Any], planner_id: str) -> dict[str, Any]:
    for arm in _required(packet):
        if arm.get("planner_id") == planner_id:
            return arm
    raise AssertionError(f"arm {planner_id!r} not in roster")


def test_shipped_packet_structural_contract_is_ready() -> None:
    """The structural contract still validates against tracked sources (shape checks)."""
    result = checker.validate_packet(_packet(), repo_root=REPO_ROOT, check_admission=False)
    assert result["status"] == "ok"
    assert result["planner_count"] == 5
    assert result["ceiling_count"] == 4
    assert result["campaign_execution_allowed"] is False


def test_shipped_packet_fails_admission_until_amended() -> None:
    """The shipped packet declares arms that cannot instantiate as declared (issue #5961).

    This is the acceptance criterion: the CURRENT packet fails the instantiability admission
    check until its PPO readiness/fallback and hybrid-leader claims are amended. The checker
    must surface every defect, not pass on shape alone.
    """
    with pytest.raises(checker.PacketError, match="cannot instantiate as declared") as exc_info:
        checker.validate_packet(_packet(), repo_root=REPO_ROOT)
    message = str(exc_info.value)
    # The ppo arm is declared artifact_qualified_only but the resolved model is only a
    # benchmark_candidate, and the paper-grade gate is not satisfied.
    assert "artifact_qualified_only" in message
    assert "benchmark_candidate" in message
    assert "paper-grade gate" in message
    # The ppo fallback flag contradicts the packet's own execution_boundary.
    assert "fallback_to_goal" in message
    assert "fallback_or_degraded_success_allowed" in message
    # The hybrid leader role asserts superiority without citing registered evidence.
    assert "scenario_adaptive_hybrid_leader" in message


def test_shipped_packet_admission_summary_reports_four_defects() -> None:
    """The admission summary resolves exactly the four S30-class defects in the shipped packet."""
    summary = check_campaign_arm_admission(_packet(), repo_root=REPO_ROOT)
    assert summary.admissible is False
    assert summary.arm_count == 5
    # Four distinct defects: ppo readiness (model not promoted), ppo readiness (paper gate),
    # ppo fallback contradiction, hybrid leader without evidence.
    assert summary.finding_count == 4
    contracts = {finding.contract for finding in summary.findings}
    assert contracts == {"readiness", "fallback", "evidence"}
    failing_arms = {finding.planner_id for finding in summary.findings}
    assert failing_arms == {"ppo", "scenario_adaptive_hybrid_orca_v1"}


def test_missing_packet_raises_file_not_found(tmp_path: Path) -> None:
    """A missing packet is a file error, not an apparently invalid packet."""
    with pytest.raises(FileNotFoundError, match="packet not found"):
        checker.load_packet(tmp_path / "missing.yaml")


def test_rejects_roster_expansion() -> None:
    """Adding a registered planner must not silently change the pre-registered roster."""
    packet = _packet()
    roster = packet["planner_roster"]
    assert isinstance(roster, dict)
    required = roster["required"]
    assert isinstance(required, list)
    required.append({"planner_id": "goal", "role": "extra"})
    with pytest.raises(checker.PacketError, match="planner roster"):
        checker.validate_packet(packet, repo_root=REPO_ROOT)


def test_missing_planner_config_raises_file_not_found() -> None:
    """A required planner config cannot be treated as a generic packet error."""
    packet = _packet()
    roster = packet["planner_roster"]
    assert isinstance(roster, dict)
    required = roster["required"]
    assert isinstance(required, list)
    ppo = required[1]
    assert isinstance(ppo, dict)
    ppo["config_path"] = "configs/algos/missing.yaml"
    with pytest.raises(FileNotFoundError, match="planner config missing"):
        checker.validate_packet(packet, repo_root=REPO_ROOT)


def test_rejects_split_leakage() -> None:
    """Family holdouts are mandatory rather than an optional reporting detail."""
    packet = _packet()
    inputs = packet["input_contract"]
    assert isinstance(inputs, dict)
    split = inputs["split_contract"]
    assert isinstance(split, dict)
    split["selection_and_evaluation_family_sets_must_be_disjoint"] = False
    with pytest.raises(
        checker.PacketError, match="selection_and_evaluation_family_sets_must_be_disjoint"
    ):
        checker.validate_packet(packet, repo_root=REPO_ROOT)


def test_rejects_transient_routing_state() -> None:
    """Host and queue routing belong to private ops/state, never this packet."""
    packet = _packet()
    packet["execution_boundary"]["target_host"] = "imech036"  # type: ignore[index]
    with pytest.raises(checker.PacketError, match="transient routing state"):
        checker.validate_packet(packet, repo_root=REPO_ROOT)


def test_rejects_non_native_success_policy() -> None:
    """Fallback/degraded rows cannot be promoted by the analysis packet."""
    packet = _packet()
    policy = packet["row_status_policy"]
    assert isinstance(policy, dict)
    policy["eligible_execution_modes"] = ["native", "adapter"]
    with pytest.raises(checker.PacketError, match="only native"):
        checker.validate_packet(packet, repo_root=REPO_ROOT)


def test_rejects_output_sibling_path() -> None:
    """A sibling of output/ must not satisfy the disposable-root contract."""
    packet = _packet()
    outputs = packet["outputs"]
    assert isinstance(outputs, dict)
    outputs["local_root"] = "output_sibling/benchmarks/issue_5302_oracle_gap"
    with pytest.raises(checker.PacketError, match="outputs.local_root"):
        checker.validate_packet(packet, repo_root=REPO_ROOT)


def test_rejects_durable_evidence_sibling_path() -> None:
    """A sibling of docs/context/evidence/ must not satisfy the durable-root contract."""
    packet = _packet()
    outputs = packet["outputs"]
    assert isinstance(outputs, dict)
    durable = outputs["durable_evidence"]
    assert isinstance(durable, dict)
    durable["path"] = "docs/context/evidence_sibling/issue_5302_oracle_gap"
    with pytest.raises(checker.PacketError, match="durable evidence"):
        checker.validate_packet(packet, repo_root=REPO_ROOT)


def test_rejects_missing_provenance_path() -> None:
    """A missing canonical schema owner blocks readiness."""
    packet = _packet()
    inputs = packet["input_contract"]
    assert isinstance(inputs, dict)
    provenance = inputs["provenance"]
    assert isinstance(provenance, dict)
    provenance["required_paths"].append("missing/canonical_owner.py")
    with pytest.raises(FileNotFoundError, match="provenance path missing"):
        checker.validate_packet(packet, repo_root=REPO_ROOT)


def test_rejects_directory_as_provenance_path() -> None:
    """A directory cannot stand in for a required provenance file."""
    packet = _packet()
    inputs = packet["input_contract"]
    assert isinstance(inputs, dict)
    provenance = inputs["provenance"]
    assert isinstance(provenance, dict)
    provenance["required_paths"].append("configs")
    with pytest.raises(FileNotFoundError, match="provenance path missing"):
        checker.validate_packet(packet, repo_root=REPO_ROOT)


def test_invalid_inference_contract_is_packet_error() -> None:
    """Inference-contract failures stay inside the packet checker's error contract."""
    packet = _packet()
    contract = packet["inference_contract"]
    assert isinstance(contract, dict)
    decision_rule = contract["decision_rule"]
    assert isinstance(decision_rule, dict)
    decision_rule["threshold"] = ""
    with pytest.raises(checker.PacketError, match="decision_rule.threshold"):
        checker.validate_packet(packet, repo_root=REPO_ROOT)


def test_main_reports_invalid_inference_contract_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI emits structured JSON instead of a traceback for a bad contract."""
    packet = _packet()
    contract = packet["inference_contract"]
    assert isinstance(contract, dict)
    decision_rule = contract["decision_rule"]
    assert isinstance(decision_rule, dict)
    decision_rule["threshold"] = ""
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(packet), encoding="utf-8")

    assert checker.main(["--packet", str(path), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_ready"
    assert "decision_rule.threshold" in payload["error"]


# ---------------------------------------------------------------------------
# Issue #5961: instantiability admission checks (re-break each condition).
#
# These tests resolve each declared arm through the REAL loaders
# (algorithm_readiness, _ppo_paper_gate_status, the model registry, the real
# algo-config YAML) -- not stubs -- and verify the admission gate fails closed
# on each of the five declared-roster != instantiated-roster conditions.
# ---------------------------------------------------------------------------


def _findings_for(packet: dict[str, Any], planner_id: str) -> list[ArmAdmissionFinding]:
    summary = check_campaign_arm_admission(packet, repo_root=REPO_ROOT)
    return [f for f in summary.findings if f.planner_id == planner_id]


def test_admission_fails_when_qualified_readiness_lacks_provenance() -> None:
    """Declaring artifact_qualified without provenance fails the readiness contract.

    Re-break condition 1: the PPO arm is declared artifact_qualified_only but its config lacks the
    paper-grade provenance block, so ``_ppo_paper_gate_status`` returns ``(False, ...)``. The
    admission gate must catch this through the real loader, not a stub.
    """
    packet = _packet()
    ppo = _arm(packet, "ppo")
    # Point the ppo arm at a config with a paper profile but no provenance, so the real
    # ``_ppo_paper_gate_status`` gate fails on missing provenance exactly as the issue describes.
    ppo["config_path"] = "configs/algos/ppo_v3_camera_ready.yaml"
    findings = _findings_for(packet, "ppo")
    readiness_findings = [f for f in findings if f.contract == "readiness"]
    assert readiness_findings, "expected a readiness finding for the unqualified PPO arm"
    assert any("paper-grade gate" in f.message for f in readiness_findings)


def test_admission_fails_when_checkpoint_points_at_nonexistent_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint pointing at a nonexistent model_id fails the checkpoint contract.

    Re-break condition 2: the arm declares a ``model_id`` that is not in the model registry. The
    admission gate resolves through ``get_registry_entry`` (the real loader) and fails closed.
    """
    packet = _packet()
    planner = _arm(packet, "prediction_planner")
    # Write a config that points at a model_id absent from the real registry, then repoint the arm.
    bad_config = {
        "predictive_model_id": "definitely_not_a_real_model_id_5961",
        "predictive_device": "cpu",
    }
    config_path = tmp_path / "bad_prediction_planner.yaml"
    config_path.write_text(yaml.safe_dump(bad_config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    planner["config_path"] = str(config_path)
    isolated_packet = {
        "execution_boundary": packet["execution_boundary"],
        "planner_roster": {"required": [planner]},
    }
    summary = check_campaign_arm_admission(isolated_packet, repo_root=tmp_path)
    findings = [f for f in summary.findings if f.planner_id == "prediction_planner"]
    checkpoint_findings = [f for f in findings if f.contract == "checkpoint"]
    assert checkpoint_findings, "expected a checkpoint finding for the unknown model_id"
    assert any("definitely_not_a_real_model_id_5961" in f.message for f in checkpoint_findings)


def test_admission_fails_on_fallback_contradiction(tmp_path: Path) -> None:
    """Reintroducing the fallback contradiction fails the fallback contract.

    Re-break condition 4: the packet forbids fallback/degraded success, but the arm config enables
    ``fallback_to_goal``. The admission gate must catch the policy-identity violation through the
    real config loader.
    """
    packet = _packet()
    # The shipped ppo config already sets fallback_to_goal: true; confirm the gate catches it.
    findings = _findings_for(packet, "ppo")
    fallback_findings = [f for f in findings if f.contract == "fallback"]
    assert fallback_findings, (
        "expected a fallback finding for fallback_to_goal under no-fallback boundary"
    )
    assert any("fallback_to_goal" in f.message for f in fallback_findings)


def test_admission_fails_when_leader_role_lacks_evidence() -> None:
    """A leader role without a cited registered result fails the evidence contract.

    Re-break condition 5: the arm's role asserts leadership/superiority but cites no evidence. The
    registered h500/h600/job-13282 evidence treats the four hybrid arms as a statistical tie, so a
    ``leader`` role cannot be admitted without a citation that establishes the separation.
    """
    packet = _packet()
    findings = _findings_for(packet, "scenario_adaptive_hybrid_orca_v1")
    evidence_findings = [f for f in findings if f.contract == "evidence"]
    assert evidence_findings, "expected an evidence finding for the leader role without a citation"
    assert any("scenario_adaptive_hybrid_leader" in f.message for f in evidence_findings)


def test_admission_passes_when_leader_role_cites_existing_evidence() -> None:
    """A leader role that cites a durable registered result is admissible on the evidence contract."""
    packet = _packet()
    hybrid = _arm(packet, "scenario_adaptive_hybrid_orca_v1")
    # Cite a real durable evidence file that backs the hybrid roster interpretation.
    hybrid["evidence"] = "docs/context/evidence/issue_3810_h600_interpretation_2026-07/README.md"
    findings = _findings_for(packet, "scenario_adaptive_hybrid_orca_v1")
    assert not [f for f in findings if f.contract == "evidence"]


def test_admission_passes_when_fallback_is_allowed() -> None:
    """When the execution boundary allows fallback, the fallback flag is not a contradiction."""
    packet = _packet()
    execution = packet["execution_boundary"]
    assert isinstance(execution, dict)
    execution["fallback_or_degraded_success_allowed"] = True
    findings = _findings_for(packet, "ppo")
    assert not [f for f in findings if f.contract == "fallback"]


def test_admission_reports_unknown_readiness_label() -> None:
    """An unrecognized readiness label is itself an admission finding."""
    packet = _packet()
    _arm(packet, "orca")["readiness"] = "totally_made_up_readiness"
    findings = _findings_for(packet, "orca")
    readiness_findings = [f for f in findings if f.contract == "readiness"]
    assert readiness_findings
    assert any("unrecognized readiness label" in f.message for f in readiness_findings)


def test_admission_summary_summary_shape() -> None:
    """The admission summary exposes admissibility, per-arm reports, and flat findings."""
    packet = _packet()
    summary = check_campaign_arm_admission(packet, repo_root=REPO_ROOT)
    assert summary.arm_count == 5
    assert len(summary.arms) == 5
    # orca and prediction_mpc instantiate cleanly as declared.
    clean_arms = {arm.planner_id for arm in summary.arms if not arm.findings}
    assert {"orca", "prediction_planner", "prediction_mpc"} <= clean_arms


def test_admission_malformed_roster_raises() -> None:
    """A roster that is not a mapping / list raises rather than silently passing."""
    packet = _packet()
    packet["planner_roster"] = "not-a-mapping"
    with pytest.raises(CampaignArmAdmissionError, match="planner_roster"):
        check_campaign_arm_admission(packet, repo_root=REPO_ROOT)


def test_main_reports_admission_failure_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI emits structured JSON for an admission failure instead of a traceback."""
    path = tmp_path / "packet.yaml"
    path.write_text(yaml.safe_dump(_packet()), encoding="utf-8")
    assert checker.main(["--packet", str(path), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_ready"
    assert "cannot instantiate as declared" in payload["error"]
