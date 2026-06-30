"""Tests for the ORCA-residual learned-policy lane readiness surface (issue #1358).

The lane is a parent/coordination issue gated by child #1475. These tests pin the
read-only contract: missing lane metadata and route configs fail closed
(``prerequisites_incomplete``), the declared external blockers are always surfaced, and a
fully scaffolded repo reports ``blocked_on_followup`` (handoff-complete, still gated).
"""

from __future__ import annotations

import pathlib

import pytest

from robot_sf.benchmark.orca_residual_lane_readiness import (
    ISSUE_1475_LOCAL_PREFLIGHT_COMMANDS,
    ISSUE_1475_REQUIRED_SMOKE_EVIDENCE_FIELDS,
    LANE_BLOCKERS,
    LANE_ROUTES,
    REQUIRED_CANDIDATE_IDS,
    REQUIRED_PREREQUISITES,
    SCHEMA_VERSION,
    assess_lane_readiness,
)
from robot_sf.training.orca_residual_lineage_packet import REQUIRED_ORCA_RESIDUAL_DIAGNOSTICS

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _write_synthetic_lane(root: pathlib.Path, *, with_registry_candidates: bool = True) -> None:
    """Create a minimal repo tree with every required prerequisite file present.

    The lineage packet body is a placeholder; callers that exercise packet validation use
    ``validate_packet=False`` so these structural tests stay independent of the full
    lineage-packet schema (covered by its own validator tests).
    """
    for prereq in REQUIRED_PREREQUISITES:
        path = root / prereq.path
        path.parent.mkdir(parents=True, exist_ok=True)
        if prereq.key == "candidate_registry":
            continue
        path.write_text(f"# synthetic {prereq.key}\n", encoding="utf-8")

    registry_path = root / "docs/context/policy_search/candidate_registry.yaml"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["version: 1", "candidates:"]
    if with_registry_candidates:
        for candidate_id in REQUIRED_CANDIDATE_IDS:
            lines.append(f"  {candidate_id}:")
            lines.append("    status: prototype")
            lines.append("    training_required: true")
    else:
        lines.append("  some_other_candidate:")
        lines.append("    training_required: false")
    registry_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_fully_scaffolded_synthetic_repo_is_blocked_on_followup(tmp_path: pathlib.Path) -> None:
    """All prerequisites present + registry candidates -> handoff-complete, still gated."""
    _write_synthetic_lane(tmp_path)
    report = assess_lane_readiness(tmp_path, validate_packet=False)

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["issue"] == 1358
    assert report["overall_status"] == "blocked_on_followup"
    assert report["errors"] == []
    assert all(p["present"] for p in report["prerequisites"])


def test_missing_lane_metadata_fails_closed(tmp_path: pathlib.Path) -> None:
    """Removing a required surface (smoke config) reports prerequisites_incomplete."""
    _write_synthetic_lane(tmp_path)
    (
        tmp_path / "configs/training/orca_residual/orca_residual_bc_issue_1475_smoke_pretrain.yaml"
    ).unlink()

    report = assess_lane_readiness(tmp_path, validate_packet=False)

    assert report["overall_status"] == "prerequisites_incomplete"
    smoke = next(p for p in report["prerequisites"] if p["key"] == "smoke_pretrain_config")
    assert smoke["present"] is False
    assert any("smoke_pretrain" in err for err in report["errors"])


def test_missing_route_runner_fails_closed(tmp_path: pathlib.Path) -> None:
    """Removing the policy-search runner (route surface) fails closed."""
    _write_synthetic_lane(tmp_path)
    (tmp_path / "scripts/validation/run_policy_search_candidate.py").unlink()

    report = assess_lane_readiness(tmp_path, validate_packet=False)

    assert report["overall_status"] == "prerequisites_incomplete"
    runner = next(p for p in report["prerequisites"] if p["key"] == "policy_search_runner")
    assert runner["present"] is False


def test_registry_missing_candidate_is_flagged(tmp_path: pathlib.Path) -> None:
    """A registry without the residual candidates is an actionable error."""
    _write_synthetic_lane(tmp_path, with_registry_candidates=False)

    report = assess_lane_readiness(tmp_path, validate_packet=False)

    assert report["overall_status"] == "prerequisites_incomplete"
    registry = next(p for p in report["prerequisites"] if p["key"] == "candidate_registry")
    assert registry["present"] is True  # file exists, but content is incomplete
    joined = " ".join(registry["messages"])
    for candidate_id in REQUIRED_CANDIDATE_IDS:
        assert candidate_id in joined


def test_registry_training_required_false_is_flagged(tmp_path: pathlib.Path) -> None:
    """Residual candidates must be registered as learned (training_required: true)."""
    _write_synthetic_lane(tmp_path)
    registry_path = tmp_path / "docs/context/policy_search/candidate_registry.yaml"
    lines = ["version: 1", "candidates:"]
    for candidate_id in REQUIRED_CANDIDATE_IDS:
        lines.append(f"  {candidate_id}:")
        lines.append("    training_required: false")
    registry_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = assess_lane_readiness(tmp_path, validate_packet=False)

    assert report["overall_status"] == "prerequisites_incomplete"
    registry = next(p for p in report["prerequisites"] if p["key"] == "candidate_registry")
    assert any("training_required: true" in m for m in registry["messages"])


def test_invalid_lineage_packet_fails_closed(tmp_path: pathlib.Path) -> None:
    """An invalid lineage packet body fails closed when packet validation is enabled."""
    _write_synthetic_lane(tmp_path)
    packet_path = tmp_path / "configs/training/orca_residual/orca_residual_bc_issue_1428.yaml"
    packet_path.write_text("schema_version: wrong\ncampaign_id: ''\n", encoding="utf-8")

    report = assess_lane_readiness(tmp_path, validate_packet=True)

    assert report["overall_status"] == "prerequisites_incomplete"
    lineage = next(p for p in report["prerequisites"] if p["key"] == "lineage_packet")
    assert any("failed validation" in m for m in lineage["messages"])


def test_blockers_and_routes_always_present(tmp_path: pathlib.Path) -> None:
    """The declared external gates and informational routes are always reported."""
    _write_synthetic_lane(tmp_path)
    report = assess_lane_readiness(tmp_path, validate_packet=False)

    blocker_keys = {b["key"] for b in report["blockers"]}
    assert blocker_keys == {b.key for b in LANE_BLOCKERS}
    assert "slurm_training_required" in blocker_keys
    assert "child_classification_gate" in blocker_keys
    assert "durable_artifacts_pending" in blocker_keys

    route_keys = {r["key"] for r in report["routes"]}
    assert route_keys == {r.key for r in LANE_ROUTES}
    # The lane must never submit SLURM from this surface.
    slurm_route = next(r for r in report["routes"] if r["key"] == "slurm_handoff")
    assert "1475" in slurm_route["command_shape"] or "approved" in slurm_route["command_shape"]


def test_diagnostics_contract_reuses_canonical_owner(tmp_path: pathlib.Path) -> None:
    """The diagnostics contract is sourced from the lineage-packet canonical owner."""
    _write_synthetic_lane(tmp_path)
    report = assess_lane_readiness(tmp_path, validate_packet=False)
    assert report["required_diagnostics_contract"] == list(REQUIRED_ORCA_RESIDUAL_DIAGNOSTICS)


def test_issue_1475_decision_packet_is_read_only_and_fail_closed(
    tmp_path: pathlib.Path,
) -> None:
    """Issue #1475 packet exposes smoke-readiness gates without allowing submission."""
    _write_synthetic_lane(tmp_path)

    report = assess_lane_readiness(tmp_path, validate_packet=False)

    packet = report["issue_1475_decision_packet"]
    assert packet["issue"] == 1475
    assert packet["task_class"] == "orca_residual_bc_smoke_decision_packet"
    assert packet["decision_status"] == "ready_for_single_smoke_handoff"
    assert packet["submission_allowed_from_this_checker"] is False
    assert packet["run_nominal"] is False
    assert packet["nominal_escalation_gate"]["allowed_by_packet"] is False
    assert packet["required_smoke_evidence_fields"] == list(
        ISSUE_1475_REQUIRED_SMOKE_EVIDENCE_FIELDS
    )
    assert packet["local_preflight_commands"] == list(ISSUE_1475_LOCAL_PREFLIGHT_COMMANDS)
    assert any(
        "required smoke evidence fields present" in gate
        for gate in packet["nominal_escalation_gate"]["required_before_nominal"]
    )


def test_issue_1475_decision_packet_reflects_incomplete_local_surface(
    tmp_path: pathlib.Path,
) -> None:
    """Missing local scaffolding makes the #1475 packet incomplete, not runnable."""
    _write_synthetic_lane(tmp_path)
    (tmp_path / "scripts/validation/run_policy_search_candidate.py").unlink()

    report = assess_lane_readiness(tmp_path, validate_packet=False)

    packet = report["issue_1475_decision_packet"]
    assert report["overall_status"] == "prerequisites_incomplete"
    assert packet["decision_status"] == "local_packet_incomplete"
    assert packet["submission_allowed_from_this_checker"] is False
    assert packet["run_nominal"] is False
    assert packet["local_blockers"] == report["errors"]


def test_real_repo_lane_is_blocked_on_followup() -> None:
    """Against the real repo, the lane scaffolding is complete and validates."""
    report = assess_lane_readiness(REPO_ROOT, validate_packet=True)
    assert report["overall_status"] == "blocked_on_followup", report["errors"]
    assert report["errors"] == []


def test_cli_exit_codes(tmp_path: pathlib.Path) -> None:
    """CLI returns 0 when handoff-complete and 2 when a prerequisite is missing."""
    from scripts.tools.orca_residual_lane_readiness import main as cli_main

    _write_synthetic_lane(tmp_path)
    assert cli_main(["--repo-root", str(tmp_path), "--no-validate-packet", "--json"]) == 0

    (tmp_path / "scripts/validation/run_policy_search_candidate.py").unlink()
    assert cli_main(["--repo-root", str(tmp_path), "--no-validate-packet"]) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
