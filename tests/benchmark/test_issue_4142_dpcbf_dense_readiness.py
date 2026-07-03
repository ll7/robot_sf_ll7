"""Tests for the issue #4142 dense DPCBF comparison readiness surface.

The comparison packet predeclares three CBF arms (unfiltered, collision-cone, DPCBF) that
must stay predeclared, distinct, and fail-closed before any campaign is authorized. These
tests pin the read-only contract: the real on-disk packet reaches
``inputs_ready_campaign_gated`` with the campaign still gated, and every structural gap
(missing arm, duplicate/reused arm, drifted variant, disabled fallback exclusion, missing
adapter config) fails closed to ``prerequisites_incomplete``.
"""

from __future__ import annotations

import copy
import pathlib

import pytest
import yaml

from robot_sf.benchmark.issue_4142_dpcbf_dense_readiness import (
    CAMPAIGN_GATES,
    PACKET_PATH,
    REQUIRED_ARMS,
    DpcbfDenseReadinessError,
    evaluate_readiness,
    render_markdown,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_HAPPY_PACKET = {
    "schema_version": "robot_sf.issue_4142_dpcbf_dense_comparison.v1",
    "canonical_command": "uv run python -m robot_sf.benchmark.cli run --config packet.yaml",
    "scenario_manifest": "configs/scenarios/sets/dense.yaml",
    "algorithm": "prediction_mpc_cv",
    "algorithm_configs": {
        "cbf_off": "configs/algos/off.yaml",
        "cbf_collision_cone_on": "configs/algos/cone.yaml",
        "cbf_dynamic_parabolic_v1_on": "configs/algos/dpcbf.yaml",
    },
    "runtime_cbf_arms": [
        {"enabled": False, "arm_key": "cbf_off"},
        {"enabled": True, "arm_key": "cbf_collision_cone_on"},
        {
            "enabled": True,
            "arm_key": "cbf_dynamic_parabolic_v1_on",
            "variant": "dynamic_parabolic_cbf_v1",
        },
    ],
    "summary_contract": {
        "evidence_tier": "bounded_runtime_comparison",
        "fallback_rows_are_success_evidence": False,
        "excluded_row_statuses": ["fallback", "degraded", "failed", "ineligible"],
        "required_arms": [
            "cbf_off",
            "cbf_collision_cone_on",
            "cbf_dynamic_parabolic_v1_on",
        ],
    },
}


def _write_tree(root: pathlib.Path, packet: dict) -> pathlib.Path:
    """Materialize a minimal repo tree (packet + referenced configs) under ``root``."""
    (root / "configs/algos").mkdir(parents=True, exist_ok=True)
    (root / "configs/scenarios/sets").mkdir(parents=True, exist_ok=True)
    (root / "configs/research").mkdir(parents=True, exist_ok=True)

    (root / "configs/scenarios/sets/dense.yaml").write_text(
        "schema_version: robot_sf.scenario_matrix.v1\n", encoding="utf-8"
    )
    (root / "configs/algos/off.yaml").write_text("algorithm: prediction_mpc_cv\n", encoding="utf-8")
    (root / "configs/algos/cone.yaml").write_text(
        "cbf_safety_filter:\n  enabled: true\n", encoding="utf-8"
    )
    (root / "configs/algos/dpcbf.yaml").write_text(
        "cbf_safety_filter:\n  enabled: true\n  variant: dynamic_parabolic_cbf_v1\n",
        encoding="utf-8",
    )
    packet_path = root / "configs/research/packet.yaml"
    packet_path.write_text(yaml.safe_dump(packet, sort_keys=False), encoding="utf-8")
    return packet_path


def test_real_packet_reaches_inputs_ready_but_stays_gated() -> None:
    """The tracked packet validates fully yet remains campaign-gated (fail-closed)."""
    readiness = evaluate_readiness(repo_root=REPO_ROOT, packet_path=PACKET_PATH)

    assert readiness.status == "inputs_ready_campaign_gated"
    assert readiness.inputs_ready is True
    assert readiness.blockers == ()
    assert readiness.required_arms_present is True
    assert readiness.arms_distinct is True
    assert readiness.fallback_excluded is True
    assert {arm.arm_key for arm in readiness.arms} == set(REQUIRED_ARMS)
    assert all(arm.runtime_valid for arm in readiness.arms)
    assert all(arm.algorithm_config_consistent for arm in readiness.arms)
    # No 'ready-to-run' state: downstream gates are always surfaced.
    assert readiness.campaign_gates == CAMPAIGN_GATES
    assert len(readiness.campaign_gates) >= 1


def test_synthetic_happy_path_is_ready(tmp_path: pathlib.Path) -> None:
    """A minimal well-formed tree reaches inputs_ready_campaign_gated."""
    packet_path = _write_tree(tmp_path, copy.deepcopy(_HAPPY_PACKET))
    readiness = evaluate_readiness(repo_root=tmp_path, packet_path=packet_path)
    assert readiness.status == "inputs_ready_campaign_gated"
    assert readiness.blockers == ()


def test_missing_required_arm_fails_closed(tmp_path: pathlib.Path) -> None:
    """Dropping the DPCBF arm blocks the comparison."""
    packet = copy.deepcopy(_HAPPY_PACKET)
    packet["runtime_cbf_arms"] = packet["runtime_cbf_arms"][:2]
    packet["summary_contract"]["required_arms"] = ["cbf_off", "cbf_collision_cone_on"]
    packet_path = _write_tree(tmp_path, packet)
    readiness = evaluate_readiness(repo_root=tmp_path, packet_path=packet_path)
    assert readiness.status == "prerequisites_incomplete"
    assert readiness.required_arms_present is False
    assert any("missing required arms" in b for b in readiness.blockers)


def test_dpcbf_reusing_collision_cone_arm_key_fails_closed(tmp_path: pathlib.Path) -> None:
    """DPCBF must not reuse cbf_collision_cone_on; a duplicate arm_key is rejected."""
    packet = copy.deepcopy(_HAPPY_PACKET)
    packet["runtime_cbf_arms"][2]["arm_key"] = "cbf_collision_cone_on"
    packet_path = _write_tree(tmp_path, packet)
    readiness = evaluate_readiness(repo_root=tmp_path, packet_path=packet_path)
    assert readiness.status == "prerequisites_incomplete"
    assert readiness.arms_distinct is False
    assert any("duplicate arm_key" in b for b in readiness.blockers)


def test_dpcbf_variant_on_collision_cone_arm_rejected_by_canonical_validator(
    tmp_path: pathlib.Path,
) -> None:
    """Threshold/variant drift on the collision-cone arm fails via the canonical validator."""
    packet = copy.deepcopy(_HAPPY_PACKET)
    packet["runtime_cbf_arms"][1]["variant"] = "dynamic_parabolic_cbf_v1"
    packet_path = _write_tree(tmp_path, packet)
    readiness = evaluate_readiness(repo_root=tmp_path, packet_path=packet_path)
    assert readiness.status == "prerequisites_incomplete"
    cone = next(a for a in readiness.arms if a.arm_key == "cbf_collision_cone_on")
    assert cone.runtime_valid is False
    assert any("canonical validator" in b for b in readiness.blockers)


def test_fallback_rows_success_flag_true_fails_closed(tmp_path: pathlib.Path) -> None:
    """Flipping fallback_rows_are_success_evidence blocks the packet."""
    packet = copy.deepcopy(_HAPPY_PACKET)
    packet["summary_contract"]["fallback_rows_are_success_evidence"] = True
    packet_path = _write_tree(tmp_path, packet)
    readiness = evaluate_readiness(repo_root=tmp_path, packet_path=packet_path)
    assert readiness.status == "prerequisites_incomplete"
    assert readiness.fallback_excluded is False


def test_missing_excluded_status_fails_closed(tmp_path: pathlib.Path) -> None:
    """Dropping a required fail-closed excluded status blocks the packet."""
    packet = copy.deepcopy(_HAPPY_PACKET)
    packet["summary_contract"]["excluded_row_statuses"] = ["fallback", "failed", "ineligible"]
    packet_path = _write_tree(tmp_path, packet)
    readiness = evaluate_readiness(repo_root=tmp_path, packet_path=packet_path)
    assert readiness.status == "prerequisites_incomplete"
    assert readiness.fallback_excluded is False
    assert any("excluded_row_statuses" in b for b in readiness.blockers)


def test_missing_adapter_config_fails_closed(tmp_path: pathlib.Path) -> None:
    """A referenced adapter config that does not exist blocks the packet."""
    packet = copy.deepcopy(_HAPPY_PACKET)
    packet["algorithm_configs"]["cbf_dynamic_parabolic_v1_on"] = "configs/algos/missing.yaml"
    packet_path = _write_tree(tmp_path, packet)
    readiness = evaluate_readiness(repo_root=tmp_path, packet_path=packet_path)
    assert readiness.status == "prerequisites_incomplete"
    dpcbf = next(a for a in readiness.arms if a.arm_key == "cbf_dynamic_parabolic_v1_on")
    assert dpcbf.algorithm_config_exists is False


def test_adapter_config_variant_mismatch_fails_closed(tmp_path: pathlib.Path) -> None:
    """A DPCBF arm pointed at the collision-cone adapter config is inconsistent."""
    packet = copy.deepcopy(_HAPPY_PACKET)
    packet["algorithm_configs"]["cbf_dynamic_parabolic_v1_on"] = "configs/algos/cone.yaml"
    packet_path = _write_tree(tmp_path, packet)
    readiness = evaluate_readiness(repo_root=tmp_path, packet_path=packet_path)
    assert readiness.status == "prerequisites_incomplete"
    dpcbf = next(a for a in readiness.arms if a.arm_key == "cbf_dynamic_parabolic_v1_on")
    assert dpcbf.algorithm_config_consistent is False
    assert any("variant mismatch" in b for b in readiness.blockers)


def test_wrong_schema_version_fails_closed(tmp_path: pathlib.Path) -> None:
    """An unexpected packet schema_version blocks the packet."""
    packet = copy.deepcopy(_HAPPY_PACKET)
    packet["schema_version"] = "robot_sf.some_other_schema.v9"
    packet_path = _write_tree(tmp_path, packet)
    readiness = evaluate_readiness(repo_root=tmp_path, packet_path=packet_path)
    assert readiness.status == "prerequisites_incomplete"
    assert readiness.packet_schema_ok is False


def test_missing_packet_raises(tmp_path: pathlib.Path) -> None:
    """A missing packet file raises the dedicated loader error."""
    with pytest.raises(DpcbfDenseReadinessError):
        evaluate_readiness(repo_root=tmp_path, packet_path="configs/research/nope.yaml")


def test_render_markdown_states_claim_boundary_and_status() -> None:
    """The Markdown report leads with the claim boundary and shows the status."""
    readiness = evaluate_readiness(repo_root=REPO_ROOT, packet_path=PACKET_PATH)
    report = render_markdown(readiness)
    assert "Claim boundary" in report
    assert "authorizes no campaign" in report
    assert readiness.status in report
