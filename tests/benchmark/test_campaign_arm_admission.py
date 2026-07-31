"""Focused unit tests for the generalized campaign-arm admission check (issue #5961).

These exercise :mod:`robot_sf.benchmark.campaign_arm_admission` -- the fail-closed packet-admission
gate that resolves every declared arm through the REAL loaders
(``algorithm_readiness``, ``_ppo_paper_gate_status``, the model registry, the real algo-config
YAML) so a packet whose declared roster cannot instantiate is rejected at admission time. The
issue explicitly requires tests to go through the real loaders, not stubs; these tests therefore
resolve real registry entries and real algorithm tiers rather than mocking the loaders.

Each admission contract (readiness, checkpoint, config_load, fallback, evidence) is exercised in
isolation by constructing a minimal ``planner_roster`` packet around a real algorithm/model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.benchmark.campaign_arm_admission import (
    CampaignArmAdmissionError,
    check_campaign_arm_admission,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _packet_with_arms(
    arms: list[dict[str, Any]], *, fallback_allowed: bool = False
) -> dict[str, Any]:
    """Build a minimal campaign packet wrapping the given roster arms."""
    return {
        "execution_boundary": {"fallback_or_degraded_success_allowed": fallback_allowed},
        "planner_roster": {"required": arms},
    }


def _admit(packet: dict[str, Any]) -> list[str]:
    """Return the flat list of admission failure messages for a packet."""
    summary = check_campaign_arm_admission(packet, repo_root=REPO_ROOT)
    return list(summary.failure_messages())


# --- readiness contract ----------------------------------------------------


def test_canonical_baseline_arm_with_baseline_tier_is_admissible() -> None:
    """An orca arm declared canonical_baseline resolves to baseline-ready and is admissible."""
    packet = _packet_with_arms(
        [{"planner_id": "orca", "role": "baseline_reactive", "readiness": "canonical_baseline"}]
    )
    assert _admit(packet) == []


def test_canonical_baseline_declared_for_experimental_tier_fails() -> None:
    """Declaring canonical_baseline for an experimental-tier algorithm fails the readiness gate."""
    packet = _packet_with_arms(
        [{"planner_id": "drl_vo", "role": "baseline_reactive", "readiness": "canonical_baseline"}]
    )
    messages = _admit(packet)
    assert any("canonical_baseline" in m and "tier" in m for m in messages)


def test_artifact_qualified_requires_benchmark_promoted_model() -> None:
    """artifact_qualified_only is rejected when the resolved model is only a benchmark_candidate.

    Uses the real PPO model ``ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200``, whose
    registry ``benchmark_promotion.claim_boundary`` is ``benchmark_candidate`` (not promoted), so a
    declared artifact_qualified arm overstates what would instantiate.
    """
    packet = _packet_with_arms(
        [
            {
                "planner_id": "ppo",
                "role": "learned_policy",
                "readiness": "artifact_qualified_only",
                "config_path": "configs/algos/ppo_v3_camera_ready.yaml",
            }
        ]
    )
    messages = _admit(packet)
    assert any("benchmark_candidate" in m and "not benchmark_promoted" in m for m in messages)


def test_experimental_opt_in_requires_allow_testing_flag() -> None:
    """experimental_explicit_opt_in is rejected when allow_testing_algorithms is not set.

    The hybrid_rule_local_planner is a real experimental algorithm (requires_explicit_opt_in); its
    shipped candidate config sets ``allow_testing_algorithms: true``, so it is admissible. A config
    that omits the flag fails through the real ``require_algorithm_allowed`` semantics.
    """
    # Shipped candidate config DOES set the flag -> admissible on readiness.
    packet_ok = _packet_with_arms(
        [
            {
                "planner_id": "scenario_adaptive_hybrid_orca_v1",
                "role": "candidate",
                "readiness": "experimental_explicit_opt_in",
                "config_path": "configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v1.yaml",
            }
        ]
    )
    assert not [m for m in _admit(packet_ok) if "[scenario_adaptive_hybrid_orca_v1] readiness" in m]


def test_experimental_opt_in_rejects_false_flag(tmp_path: Path) -> None:
    """A YAML false value is not an explicit opt-in for an experimental algorithm."""
    config_path = tmp_path / "experimental_false.yaml"
    config_path.write_text(
        yaml.safe_dump({"algo": "hybrid_rule_local_planner", "allow_testing_algorithms": False}),
        encoding="utf-8",
    )
    packet = _packet_with_arms(
        [
            {
                "planner_id": "hybrid_rule_local_planner",
                "role": "candidate",
                "readiness": "experimental_explicit_opt_in",
                "config_path": str(config_path),
            }
        ]
    )
    messages = list(check_campaign_arm_admission(packet, repo_root=tmp_path).failure_messages())
    assert any("allow_testing_algorithms not true" in message for message in messages)


# --- checkpoint contract ---------------------------------------------------


def test_checkpoint_contract_passes_for_stageable_real_model() -> None:
    """A real model_id with a durable remote source resolves in cheap (network-free) mode.

    The predictive planner's model ``predictive_proxy_selected_v2_full`` is absent locally but
    declares a github_release, so it is stageable by ``resolve_model_path`` and therefore
    resolvable. A naive ``local_path`` probe would report false-absent; this test confirms the
    admission gate resolves through the registry instead.
    """
    packet = _packet_with_arms(
        [
            {
                "planner_id": "prediction_planner",
                "role": "predictive",
                "readiness": "checkpoint_qualified_only",
                "config_path": "configs/algos/prediction_planner_camera_ready.yaml",
            }
        ]
    )
    assert not [m for m in _admit(packet) if "[prediction_planner] checkpoint" in m]


def test_checkpoint_contract_fails_for_unknown_model_id(tmp_path: Path) -> None:
    """An arm whose config points at an unknown model_id fails the checkpoint contract."""
    config_path = tmp_path / "unknown.yaml"
    config_path.write_text(
        yaml.safe_dump({"model_id": "no_such_model_id_5961_unit_test"}), encoding="utf-8"
    )
    packet = _packet_with_arms(
        [
            {
                "planner_id": "ppo",
                "role": "learned_policy",
                "readiness": "checkpoint_qualified_only",
                "config_path": str(config_path),
            }
        ]
    )
    # Resolve against tmp_path so the relative config_path is found.
    summary = check_campaign_arm_admission(packet, repo_root=tmp_path)
    messages = list(summary.failure_messages())
    assert any("no_such_model_id_5961_unit_test" in m and "[ppo] checkpoint" in m for m in messages)


# --- config_load contract --------------------------------------------------


def test_config_load_fails_when_algo_config_is_not_a_mapping(tmp_path: Path) -> None:
    """An arm whose algo_config loads as a non-mapping (e.g. a bare list) fails config_load."""
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    packet = _packet_with_arms(
        [
            {
                "planner_id": "ppo",
                "role": "learned_policy",
                "readiness": "checkpoint_qualified_only",
                "config_path": str(config_path),
            }
        ]
    )
    summary = check_campaign_arm_admission(packet, repo_root=tmp_path)
    messages = list(summary.failure_messages())
    assert any("[ppo] config_load" in m and "did not load as a mapping" in m for m in messages)


def test_config_load_reports_malformed_yaml(tmp_path: Path) -> None:
    """Malformed YAML is reported as a structured config finding, not raised to the caller."""
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text("algo: [\n", encoding="utf-8")
    packet = _packet_with_arms(
        [
            {
                "planner_id": "ppo",
                "role": "learned_policy",
                "readiness": "checkpoint_qualified_only",
                "config_path": str(config_path),
            }
        ]
    )
    messages = list(check_campaign_arm_admission(packet, repo_root=tmp_path).failure_messages())
    assert any("[ppo] config_load" in m and "malformed YAML" in m for m in messages)


def test_config_path_outside_repo_root_fails_closed(tmp_path: Path) -> None:
    """An absolute config path outside the trusted root cannot be loaded."""
    outside = tmp_path.parent / "outside_config_5961.yaml"
    outside.write_text("algo: orca\n", encoding="utf-8")
    packet = _packet_with_arms(
        [
            {
                "planner_id": "orca",
                "role": "baseline_reactive",
                "readiness": "canonical_baseline",
                "config_path": str(outside),
            }
        ]
    )
    messages = list(check_campaign_arm_admission(packet, repo_root=tmp_path).failure_messages())
    assert any("outside the trusted repository root" in m for m in messages)


def test_missing_config_path_raises_file_not_found() -> None:
    """A declared config_path that does not exist is a file error, not an apparently-valid packet."""
    packet = _packet_with_arms(
        [
            {
                "planner_id": "ppo",
                "role": "learned_policy",
                "readiness": "checkpoint_qualified_only",
                "config_path": "configs/algos/does_not_exist_5961.yaml",
            }
        ]
    )
    with pytest.raises(FileNotFoundError, match="planner config missing"):
        check_campaign_arm_admission(packet, repo_root=REPO_ROOT)


# --- fallback contract -----------------------------------------------------


def test_fallback_flag_blocked_when_boundary_forbids_fallback(tmp_path: Path) -> None:
    """fallback_to_goal under fallback_or_degraded_success_allowed=false is a contradiction."""
    config = {
        "algo": "ppo",
        "model_id": "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        "fallback_to_goal": True,
    }
    config_path = tmp_path / "ppo_fallback.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    packet = _packet_with_arms(
        [
            {
                "planner_id": "ppo",
                "role": "learned_policy",
                "readiness": "checkpoint_qualified_only",
                "config_path": str(config_path),
            }
        ]
    )
    # Resolve against tmp_path so the relative config_path is found.
    summary = check_campaign_arm_admission(packet, repo_root=tmp_path)
    messages = list(summary.failure_messages())
    assert any("[ppo] fallback" in m and "fallback_to_goal" in m for m in messages)


def test_nested_fallback_flag_is_surfaced(tmp_path: Path) -> None:
    """A fallback flag enabled in a nested block is still reported."""
    config = {
        "algo": "ppo",
        "model_id": "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        "scenario_overrides": {"some_scenario": {"fallback_to_stop": True}},
    }
    config_path = tmp_path / "ppo_nested_fallback.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    packet = _packet_with_arms(
        [
            {
                "planner_id": "ppo",
                "role": "learned_policy",
                "readiness": "checkpoint_qualified_only",
                "config_path": str(config_path),
            }
        ]
    )
    summary = check_campaign_arm_admission(packet, repo_root=tmp_path)
    messages = list(summary.failure_messages())
    assert any("[ppo] fallback" in m and "fallback_to_stop" in m for m in messages)


def test_string_true_fallback_boundary_fails_closed(tmp_path: Path) -> None:
    """A string that resembles true does not allow fallback in a boolean contract."""
    config = {
        "algo": "ppo",
        "model_id": "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        "fallback_to_goal": True,
    }
    config_path = tmp_path / "ppo_string_boundary.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    packet = _packet_with_arms(
        [
            {
                "planner_id": "ppo",
                "role": "learned_policy",
                "readiness": "checkpoint_qualified_only",
                "config_path": str(config_path),
            }
        ]
    )
    packet["execution_boundary"]["fallback_or_degraded_success_allowed"] = "true"
    messages = list(check_campaign_arm_admission(packet, repo_root=tmp_path).failure_messages())
    assert any("[ppo] fallback" in m and "fallback_to_goal" in m for m in messages)


# --- evidence contract -----------------------------------------------------


def test_leader_role_requires_cited_evidence() -> None:
    """A role asserting leadership without a citation fails the evidence contract."""
    packet = _packet_with_arms(
        [{"planner_id": "orca", "role": "roster_leader", "readiness": "canonical_baseline"}]
    )
    messages = _admit(packet)
    assert any("[orca] evidence" in m and "roster_leader" in m for m in messages)


def test_leader_role_with_cited_existing_evidence_is_admissible() -> None:
    """A leader role citing a real durable evidence path passes the evidence contract."""
    packet = _packet_with_arms(
        [
            {
                "planner_id": "orca",
                "role": "roster_leader",
                "readiness": "canonical_baseline",
                "evidence": "docs/context/evidence/issue_3810_h600_interpretation_2026-07/README.md",
            }
        ]
    )
    messages = _admit(packet)
    assert not [m for m in messages if "[orca] evidence" in m]


def test_evidence_path_escape_and_symlink_fail_closed(tmp_path: Path) -> None:
    """Evidence citations must be regular files below the repository root, without symlinks."""
    outside = tmp_path.parent / "outside_evidence_5961.txt"
    outside.write_text("not a registered result", encoding="utf-8")
    symlink = tmp_path / "evidence_link"
    symlink.symlink_to(outside)
    for evidence in (str(outside), "../outside_evidence_5961.txt", "evidence_link"):
        packet = _packet_with_arms(
            [
                {
                    "planner_id": "orca",
                    "role": "roster_leader",
                    "readiness": "canonical_baseline",
                    "evidence": evidence,
                }
            ]
        )
        messages = list(check_campaign_arm_admission(packet, repo_root=tmp_path).failure_messages())
        assert any("[orca] evidence" in message for message in messages)


def test_leader_role_with_missing_cited_evidence_fails() -> None:
    """A leader role citing a nonexistent evidence path fails the evidence contract."""
    packet = _packet_with_arms(
        [
            {
                "planner_id": "orca",
                "role": "roster_leader",
                "readiness": "canonical_baseline",
                "evidence": "docs/context/evidence/does_not_exist_5961",
            }
        ]
    )
    messages = _admit(packet)
    assert any("[orca] evidence" in m and "does not resolve" in m for m in messages)


def test_provenance_readiness_evidence_names_readiness_not_role() -> None:
    """A readiness-triggered evidence finding names the readiness label, not the role.

    Regression: when evidence is required by ``candidate_requires_existing_provenance`` (not by a
    leader role), the finding must label the claim with the readiness label and must not mislabel
    the unrelated role string as the readiness, nor call it a superiority/leadership claim.
    """
    # Missing citation: the finding must reference the readiness label.
    packet_missing = _packet_with_arms(
        [
            {
                "planner_id": "orca",
                "role": "plain_candidate",
                "readiness": "candidate_requires_existing_provenance",
            }
        ]
    )
    messages_missing = _admit(packet_missing)
    assert len(messages_missing) == 1
    missing = messages_missing[0]
    assert "readiness=candidate_requires_existing_provenance" in missing
    assert "candidate_requires_existing_provenance" in missing
    # The unrelated role string must not appear as the labelled claim.
    assert "readiness=plain_candidate" not in missing

    # Cited-but-missing evidence path: still a readiness claim, not superiority/leadership.
    packet_cited = _packet_with_arms(
        [
            {
                "planner_id": "orca",
                "role": "plain_candidate",
                "readiness": "candidate_requires_existing_provenance",
                "evidence": "docs/context/evidence/does_not_exist_5961",
            }
        ]
    )
    messages_cited = _admit(packet_cited)
    assert len(messages_cited) == 1
    cited = messages_cited[0]
    assert "readiness claim" in cited
    assert "superiority" not in cited


# --- roster shape ----------------------------------------------------------


def test_malformed_roster_raises() -> None:
    """A non-mapping roster raises rather than silently passing."""
    with pytest.raises(CampaignArmAdmissionError, match="planner_roster"):
        check_campaign_arm_admission({"planner_roster": []}, repo_root=REPO_ROOT)


def test_non_list_required_raises() -> None:
    """A non-list ``required`` roster raises."""
    with pytest.raises(CampaignArmAdmissionError, match="must be a list"):
        check_campaign_arm_admission(
            {"planner_roster": {"required": "not-a-list"}}, repo_root=REPO_ROOT
        )


def test_empty_roster_is_admissible() -> None:
    """An empty roster is trivially admissible."""
    packet = _packet_with_arms([])
    summary = check_campaign_arm_admission(packet, repo_root=REPO_ROOT)
    assert summary.admissible is True
    assert summary.arm_count == 0
