"""Tests for the issue #4142 dense DPCBF comparison run planner.

The planner consumes the predeclared comparison packet (schema
``robot_sf.issue_4142_dpcbf_dense_comparison.v1``) and resolves it into a concrete,
ordered three-arm run plan. These tests pin the fail-closed contract:

- the real on-disk packet resolves to ``plan_ready_campaign_gated`` with one job per arm,
  the fail-closed row-status exclusion carried into the plan, and execution still gated;
- every structural gap (missing arm, weakened fallback exclusion, missing algorithm) fails
  closed to ``prerequisites_incomplete`` with no executable arm jobs; and
- ``execute_run_plan`` always fails closed -- the planner never runs episodes.
"""

from __future__ import annotations

import copy
import pathlib

import pytest
import yaml

from robot_sf.benchmark.issue_4142_dpcbf_dense_readiness import (
    PACKET_PATH,
    REQUIRED_ARMS,
    REQUIRED_EXCLUDED_ROW_STATUSES,
)
from robot_sf.benchmark.issue_4142_dpcbf_dense_runner import (
    PLAN_SCHEMA_VERSION,
    RUNNER_GATES,
    DenseComparisonExecutionGatedError,
    build_run_plan,
    execute_run_plan,
    render_markdown,
    to_dict,
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


def test_real_packet_resolves_plan_but_stays_gated() -> None:
    """The tracked packet resolves a full three-arm plan yet stays execution-gated."""
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH)

    assert plan.status == "plan_ready_campaign_gated"
    assert plan.is_executable_in_principle is True
    assert plan.blockers == ()
    assert plan.schema_version == PLAN_SCHEMA_VERSION
    assert plan.readiness_status == "inputs_ready_campaign_gated"
    assert plan.algorithm == "prediction_mpc_cv"
    # One job per required arm, in packet order.
    assert tuple(job.arm_key for job in plan.arms) == REQUIRED_ARMS
    # Each job is pinned to the shared algorithm and a distinct adapter config + output path.
    assert all(job.algorithm == "prediction_mpc_cv" for job in plan.arms)
    assert len({job.algorithm_config for job in plan.arms}) == len(plan.arms)
    assert len({job.output_jsonl for job in plan.arms}) == len(plan.arms)
    # Execution stays gated even for a fully resolved plan.
    assert plan.runner_gates == RUNNER_GATES


def test_plan_carries_failclosed_exclusion_forward() -> None:
    """The plan carries the packet's fail-closed row-status exclusion verbatim."""
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH)

    assert plan.fallback_rows_are_success_evidence is False
    assert plan.fallback_excluded is True
    assert set(REQUIRED_EXCLUDED_ROW_STATUSES).issubset(set(plan.excluded_row_statuses))
    payload = to_dict(plan)
    assert set(REQUIRED_EXCLUDED_ROW_STATUSES).issubset(set(payload["excluded_row_statuses"]))
    assert payload["fallback_rows_are_success_evidence"] is False


def test_synthetic_happy_path_resolves_output_paths(tmp_path: pathlib.Path) -> None:
    """A minimal well-formed tree resolves per-arm JSONL paths under the output dir."""
    packet_path = _write_tree(tmp_path, copy.deepcopy(_HAPPY_PACKET))
    plan = build_run_plan(
        repo_root=tmp_path, packet_path=packet_path, output_dir="output/dense_test"
    )
    assert plan.status == "plan_ready_campaign_gated"
    assert plan.blockers == ()
    for job in plan.arms:
        assert job.output_jsonl == f"output/dense_test/{job.arm_key}.jsonl"
    # The cbf_off arm is disabled; the two CBF arms are enabled.
    enabled = {job.arm_key: job.enabled for job in plan.arms}
    assert enabled["cbf_off"] is False
    assert enabled["cbf_collision_cone_on"] is True
    assert enabled["cbf_dynamic_parabolic_v1_on"] is True


def test_missing_required_arm_fails_closed_with_no_jobs(tmp_path: pathlib.Path) -> None:
    """Dropping the DPCBF arm blocks the plan and resolves no executable jobs."""
    packet = copy.deepcopy(_HAPPY_PACKET)
    packet["runtime_cbf_arms"] = packet["runtime_cbf_arms"][:2]
    packet["summary_contract"]["required_arms"] = ["cbf_off", "cbf_collision_cone_on"]
    packet_path = _write_tree(tmp_path, packet)

    plan = build_run_plan(repo_root=tmp_path, packet_path=packet_path)
    assert plan.status == "prerequisites_incomplete"
    assert plan.is_executable_in_principle is False
    assert plan.arms == ()
    assert plan.blockers != ()


def test_weakened_fallback_exclusion_fails_closed(tmp_path: pathlib.Path) -> None:
    """Dropping a fail-closed row status from the exclusion blocks the plan."""
    packet = copy.deepcopy(_HAPPY_PACKET)
    packet["summary_contract"]["excluded_row_statuses"] = ["fallback", "degraded"]
    packet_path = _write_tree(tmp_path, packet)

    plan = build_run_plan(repo_root=tmp_path, packet_path=packet_path)
    assert plan.status == "prerequisites_incomplete"
    assert plan.arms == ()
    assert any("excluded_row_statuses" in b for b in plan.blockers)


def test_fallback_success_flag_true_fails_closed(tmp_path: pathlib.Path) -> None:
    """Declaring fallback rows as success evidence blocks the plan (fail-closed)."""
    packet = copy.deepcopy(_HAPPY_PACKET)
    packet["summary_contract"]["fallback_rows_are_success_evidence"] = True
    packet_path = _write_tree(tmp_path, packet)

    plan = build_run_plan(repo_root=tmp_path, packet_path=packet_path)
    assert plan.status == "prerequisites_incomplete"
    assert plan.arms == ()


def test_missing_algorithm_fails_closed(tmp_path: pathlib.Path) -> None:
    """A packet without a shared algorithm cannot resolve arm jobs."""
    packet = copy.deepcopy(_HAPPY_PACKET)
    del packet["algorithm"]
    packet_path = _write_tree(tmp_path, packet)

    plan = build_run_plan(repo_root=tmp_path, packet_path=packet_path)
    assert plan.status == "prerequisites_incomplete"
    assert plan.arms == ()
    assert any("algorithm" in b for b in plan.blockers)


def test_execute_run_plan_always_fails_closed() -> None:
    """Execution is authorization-gated: execute_run_plan never runs episodes."""
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH)
    assert plan.is_executable_in_principle is True  # even a ready plan stays gated
    with pytest.raises(DenseComparisonExecutionGatedError):
        execute_run_plan(plan)


def test_markdown_leads_with_claim_boundary() -> None:
    """The rendered report leads with the claim boundary before any plan detail."""
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH)
    md = render_markdown(plan)
    assert "Claim boundary:" in md
    assert md.index("Claim boundary:") < md.index("Resolved arm jobs")
    # Every resolved arm is listed in the report.
    for arm_key in REQUIRED_ARMS:
        assert arm_key in md
