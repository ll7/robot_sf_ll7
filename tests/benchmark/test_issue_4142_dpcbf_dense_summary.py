"""Tests for the issue #4142 dense DPCBF comparison summarizer.

The summarizer consumes the resolved three-arm run plan (schema
``robot_sf.issue_4142_dpcbf_dense_comparison_plan.v1``) and reads each arm's per-episode
JSONL output into a fail-closed comparison summary (schema
``robot_sf.issue_4142_dpcbf_dense_comparison_summary.v1``). These tests pin the
fail-closed contract:

- the real on-disk packet resolves a plan but, with execution gated, has no arm outputs, so
  the summary is ``results_incomplete`` with every arm artifact recorded as missing;
- an invalid plan fails closed to ``plan_blocked`` with no artifacts consumed;
- fallback/degraded/failed/ineligible rows are counted as caveats, never success evidence;
- ``complete`` requires all three arm artifacts to be present; and
- the rendered report leads with the claim boundary before any result detail.
"""

from __future__ import annotations

import copy
import json
import pathlib

import yaml

from robot_sf.benchmark.issue_4142_dpcbf_dense_readiness import (
    PACKET_PATH,
    REQUIRED_ARMS,
    REQUIRED_EXCLUDED_ROW_STATUSES,
)
from robot_sf.benchmark.issue_4142_dpcbf_dense_summary import (
    SUMMARY_SCHEMA_VERSION,
    render_markdown,
    summarize_dense_comparison,
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


def _write_arm_jsonl(root: pathlib.Path, output_dir: str, arm_key: str, rows: list[dict]) -> None:
    """Write a per-arm episode JSONL artifact under ``output_dir``."""
    arm_dir = root / output_dir
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / f"{arm_key}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_real_packet_summary_incomplete_no_artifacts() -> None:
    """The tracked packet resolves a plan, but with no arm outputs the summary is incomplete."""
    summary = summarize_dense_comparison(repo_root=REPO_ROOT, packet_path=PACKET_PATH)

    assert summary.schema_version == SUMMARY_SCHEMA_VERSION
    assert summary.plan_status == "plan_ready_campaign_gated"
    assert summary.status == "results_incomplete"
    # Every required arm is recorded, and each artifact is missing (execution is gated).
    assert tuple(arm.arm_key for arm in summary.arms) == REQUIRED_ARMS
    assert all(arm.missing_status == "missing" for arm in summary.arms)
    assert all(not arm.artifact_present for arm in summary.arms)
    assert summary.all_arms_have_success_evidence is False
    assert summary.blockers != ()
    # The fail-closed exclusion is carried from the plan/packet.
    assert set(REQUIRED_EXCLUDED_ROW_STATUSES).issubset(set(summary.excluded_row_statuses))
    assert summary.fallback_excluded is True


def test_invalid_plan_fails_closed_to_plan_blocked(tmp_path: pathlib.Path) -> None:
    """A packet missing a required arm yields a plan-blocked summary with no artifacts."""
    packet = copy.deepcopy(_HAPPY_PACKET)
    packet["runtime_cbf_arms"] = packet["runtime_cbf_arms"][:2]
    packet["summary_contract"]["required_arms"] = ["cbf_off", "cbf_collision_cone_on"]
    packet_path = _write_tree(tmp_path, packet)

    summary = summarize_dense_comparison(repo_root=tmp_path, packet_path=packet_path)
    assert summary.status == "plan_blocked"
    assert summary.arms == ()
    assert summary.blockers != ()


def test_all_arms_present_reaches_complete(tmp_path: pathlib.Path) -> None:
    """When all three arm artifacts are present with rows, the summary is complete."""
    packet_path = _write_tree(tmp_path, copy.deepcopy(_HAPPY_PACKET))
    output_dir = "output/dense_test"
    for arm_key in REQUIRED_ARMS:
        _write_arm_jsonl(tmp_path, output_dir, arm_key, [{"status": "ok"}, {"status": "ok"}])

    summary = summarize_dense_comparison(
        repo_root=tmp_path, packet_path=packet_path, output_dir=output_dir
    )
    assert summary.status == "complete"
    assert summary.blockers == ()
    assert all(arm.missing_status == "present" for arm in summary.arms)
    assert all(arm.success_evidence_rows == 2 for arm in summary.arms)
    assert all(arm.caveat_rows == 0 for arm in summary.arms)
    assert summary.all_arms_have_success_evidence is True


def test_one_missing_arm_keeps_incomplete(tmp_path: pathlib.Path) -> None:
    """A missing DPCBF artifact keeps the comparison out of the complete state."""
    packet_path = _write_tree(tmp_path, copy.deepcopy(_HAPPY_PACKET))
    output_dir = "output/dense_test"
    # Only two of three arms produced output.
    for arm_key in ("cbf_off", "cbf_collision_cone_on"):
        _write_arm_jsonl(tmp_path, output_dir, arm_key, [{"status": "ok"}])

    summary = summarize_dense_comparison(
        repo_root=tmp_path, packet_path=packet_path, output_dir=output_dir
    )
    assert summary.status == "results_incomplete"
    by_key = {arm.arm_key: arm for arm in summary.arms}
    assert by_key["cbf_dynamic_parabolic_v1_on"].missing_status == "missing"
    assert any("cbf_dynamic_parabolic_v1_on" in b for b in summary.blockers)


def test_fallback_and_degraded_rows_are_caveats_not_success(tmp_path: pathlib.Path) -> None:
    """Excluded row statuses are counted as caveats and never as success evidence."""
    packet_path = _write_tree(tmp_path, copy.deepcopy(_HAPPY_PACKET))
    output_dir = "output/dense_test"
    _write_arm_jsonl(tmp_path, output_dir, "cbf_off", [{"status": "ok"}])
    _write_arm_jsonl(tmp_path, output_dir, "cbf_collision_cone_on", [{"status": "ok"}])
    # The DPCBF arm ran only in fallback/degraded/ineligible modes plus one success row.
    _write_arm_jsonl(
        tmp_path,
        output_dir,
        "cbf_dynamic_parabolic_v1_on",
        [
            {"status": "fallback"},
            {"status": "degraded"},
            {"status": "ineligible"},
            {"status": "ok"},
        ],
    )

    summary = summarize_dense_comparison(
        repo_root=tmp_path, packet_path=packet_path, output_dir=output_dir
    )
    dpcbf = {arm.arm_key: arm for arm in summary.arms}["cbf_dynamic_parabolic_v1_on"]
    assert dpcbf.success_evidence_rows == 1
    assert dpcbf.caveat_rows == 3
    assert dpcbf.caveat_rows_by_status == {"degraded": 1, "fallback": 1, "ineligible": 1}
    # Artifact coverage is complete, but caveats are reported separately from success.
    assert summary.status == "complete"
    payload = to_dict(summary)
    dpcbf_payload = next(
        a for a in payload["artifact_manifest"] if a["arm_key"] == "cbf_dynamic_parabolic_v1_on"
    )
    assert dpcbf_payload["caveat_rows"] == 3
    assert dpcbf_payload["success_evidence_rows"] == 1


def test_unknown_and_unparseable_rows_fail_closed(tmp_path: pathlib.Path) -> None:
    """Unknown statuses are caveats; a malformed line marks the whole artifact unparseable."""
    packet_path = _write_tree(tmp_path, copy.deepcopy(_HAPPY_PACKET))
    output_dir = "output/dense_test"
    # Unknown status -> caveat, artifact still present/parseable.
    _write_arm_jsonl(tmp_path, output_dir, "cbf_off", [{"status": "mystery"}])
    _write_arm_jsonl(tmp_path, output_dir, "cbf_collision_cone_on", [{"status": "ok"}])
    # Malformed JSON line -> whole artifact fails closed as unparseable.
    (tmp_path / output_dir / "cbf_dynamic_parabolic_v1_on.jsonl").write_text(
        '{"status": "ok"}\nnot-json\n', encoding="utf-8"
    )

    summary = summarize_dense_comparison(
        repo_root=tmp_path, packet_path=packet_path, output_dir=output_dir
    )
    by_key = {arm.arm_key: arm for arm in summary.arms}
    assert by_key["cbf_off"].caveat_rows == 1
    assert by_key["cbf_off"].success_evidence_rows == 0
    assert by_key["cbf_dynamic_parabolic_v1_on"].missing_status == "unparseable"
    assert by_key["cbf_dynamic_parabolic_v1_on"].artifact_present is False
    assert summary.status == "results_incomplete"


def test_markdown_leads_with_claim_boundary() -> None:
    """The rendered report leads with the claim boundary before any result detail."""
    summary = summarize_dense_comparison(repo_root=REPO_ROOT, packet_path=PACKET_PATH)
    md = render_markdown(summary)
    assert "Claim boundary:" in md
    assert md.index("Claim boundary:") < md.index("Artifact manifest")
    for arm_key in REQUIRED_ARMS:
        assert arm_key in md
