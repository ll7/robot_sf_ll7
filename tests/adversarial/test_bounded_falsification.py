"""Contract tests for the default-disabled issue #8571 falsification slice."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.adversarial.bounded_falsification import (
    CLAIM_BOUNDARY,
    VERTICAL_SLICE_SCHEMA_VERSION,
    BoundedFalsificationError,
    build_bounded_falsification_preflight,
    validate_bounded_falsification_preflight,
    write_bounded_falsification_preflight,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/adversarial/issue_8570_bounded_falsification_answerability_v1.yaml"


def test_preflight_prepares_equal_budget_controls_without_compute() -> None:
    """The admitted slice preserves source custody while keeping all native work disabled."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)

    assert report["schema_version"] == VERTICAL_SLICE_SCHEMA_VERSION
    assert report["claim_boundary"] == CLAIM_BOUNDARY
    assert report["packet"]["issue"] == 8570
    assert report["source"]["base_commit"] == "c61b0f93683e1f9d2c83125f1d830b3d0162f3d1"
    assert report["zero_overlay_equivalence"]["status"] == "passed"
    assert report["execution"] == {
        "campaign_launched": False,
        "compute_authorized_by_packet": False,
        "default_disabled": True,
        "optimizer_instantiated": False,
        "planner_executed": False,
        "simulator_executed": False,
    }

    assert len(report["arms"]["random"]) == 3
    assert len(report["arms"]["halton"]) == 3
    assert all(row["candidate_budget"] == 64 for row in report["arms"]["random"])
    assert all(row["candidate_budget"] == 64 for row in report["arms"]["halton"])
    assert report["arms"]["cma_es"]["execution_status"] == "declared_not_executed"
    assert report["arms"]["cma_es"]["candidate_budget_per_seed"] == 64

    assert len(report["outcome_rows"]) == 3 * 2 * 64
    assert report["outcome_summary"]["blocked"] == 3 * 2 * 64
    assert report["outcome_summary"]["invalid"] == 0
    assert report["outcome_summary"]["result"] == 0
    assert report["outcome_summary"]["null"] == 0
    assert all(row["simulation_executed"] is False for row in report["outcome_rows"])
    assert all(row["native_outcome_digest"] is None for row in report["outcome_rows"])
    assert all(row["replay_digest"] is None for row in report["outcome_rows"])


def test_preflight_is_deterministic_and_persisted_report_round_trips(tmp_path: Path) -> None:
    """Repeated source-bound preparation emits byte-stable candidate and gate provenance."""
    first = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    second = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    assert first == second

    output = tmp_path / "preflight.json"
    persisted = write_bounded_falsification_preflight(
        PACKET,
        output,
        repo_root=REPO_ROOT,
    )
    assert persisted == first
    assert json.loads(output.read_text(encoding="utf-8")) == first


def test_preflight_validator_rejects_claim_result_rows() -> None:
    """Preparation must not be relabeled as an available result or null outcome."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    report["outcome_rows"][0]["status"] = "result"
    report["outcome_summary"]["blocked"] -= 1
    report["outcome_summary"]["result"] += 1

    with pytest.raises(BoundedFalsificationError, match="result or null"):
        validate_bounded_falsification_preflight(report)
