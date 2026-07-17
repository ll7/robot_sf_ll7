"""Contract tests for the issue #3078 job 13521 evidence registration."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "docs/context/evidence/issue_3078_package_a_job_13521_2026-07-16"


def _load_json(name: str) -> dict:
    """Load one registered JSON object."""
    payload = json.loads((BUNDLE / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_fullpilot_has_exact_predeclared_identity_scope() -> None:
    """The accepted rows exactly cover six cells by three planners at seed 111."""
    plan = _load_json("fullpilot_plan.json")
    acceptance = _load_json("row_acceptance.json")
    expected = {tuple(identity) for identity in plan["expected_identities"]}
    actual = {tuple(identity) for identity in acceptance["identities"]}

    assert plan["expected_episode_count"] == 18
    assert plan["cell_count"] == 6
    assert acceptance["episode_count"] == 18
    assert acceptance["unique_identity_count"] == 18
    assert actual == expected
    assert Counter(identity[1] for identity in actual) == {
        "goal": 6,
        "social_force": 6,
        "orca": 6,
    }
    assert {identity[2] for identity in actual} == {111}


def test_fullpilot_replaces_synthetic_heldout_input_fail_closed() -> None:
    """The transfer report consumed real accepted rows without degraded success."""
    row_acceptance = _load_json("row_acceptance.json")
    report_acceptance = _load_json("postrun_acceptance.json")
    decision = _load_json("package_a_decision_packet.json")

    assert row_acceptance["synthetic_fixture_used"] is False
    assert row_acceptance["fallback_degraded_rows"] == 0
    assert row_acceptance["row_status_counts"] == {"adapter": 12, "native": 6}
    assert report_acceptance == {
        "classification": "diagnostic_review_ready",
        "episode_count": 18,
        "heldout_table_episode_count": 18,
        "issue_result_classification": "diagnostic",
        "status": "postrun_accepted",
        "synthetic_fixture_used": False,
    }
    assert all(item["status"] == "satisfied" for item in decision["acceptance_criteria"])


def test_heldout_table_contains_all_eighteen_real_rows() -> None:
    """All planner/family aggregates are present and claim promotion stays disabled."""
    with (BUNDLE / "heldout_family_table.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (BUNDLE / "transfer_delta.csv").open(newline="", encoding="utf-8") as handle:
        deltas = list(csv.DictReader(handle))

    assert len(rows) == 6
    assert sum(int(row["episode_count"]) for row in rows) == 18
    assert all(int(row["eligible_episode_count"]) == 3 for row in rows)
    assert {row["planner"] for row in rows} == {"goal", "social_force", "orca"}
    assert all(row["claim_eligible"] == "false" for row in deltas)
    assert all(row["transfer_delta_snqi"] == "" for row in deltas)


def test_registration_preserves_private_row_store_checksum_without_committing_rows() -> None:
    """The compact source rows remain private while their exact checksum is registered."""
    registration = _load_json("registration.json")
    store = registration["source_episode_store"]

    assert store["uri"] == "private-campaign://job-13521/result_store/episodes.parquet"
    assert store["sha256"] == ("46466cd3db27d6f8a10181a8ec7c4676b24179bb97902aa8eec686d09a53942b")
    assert store["committed"] is False
    assert not (BUNDLE / "episodes.parquet").exists()
    assert registration["supersedes"]["job_id"] == 13506


def test_reproduction_documents_local_hydration_boundary() -> None:
    """The reproduction command hydrates private evidence before using local Path inputs."""
    reproduction = (BUNDLE / "reproduction.md").read_text(encoding="utf-8")

    assert "private-campaign://job-13521/result_store" in reproduction
    assert '"$JOB_13521_RESULT_STORE"' in reproduction
    assert "--result-store private-campaign://" not in reproduction
    assert "--output-dir output/issue_3078_package_a_job_13521_transfer_report" in reproduction
