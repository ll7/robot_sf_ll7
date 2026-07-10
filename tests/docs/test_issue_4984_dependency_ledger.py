"""Consistency checks for the issue #4984 improvement-batch dependency ledger."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "docs/context/issue_4984_improvement_batch_dependency_ledger.yaml"
EXPECTED_SEQUENCE = [
    4966,
    4967,
    4968,
    4969,
    4970,
    4971,
    4975,
    4972,
    4973,
    4974,
    4976,
    4977,
    4978,
    4979,
    4983,
    4980,
    4981,
    4982,
]


def test_issue_4984_ledger_is_complete_and_topologically_ordered() -> None:
    """Every child has a complete row and every dependency precedes its consumer."""
    ledger = yaml.safe_load(LEDGER_PATH.read_text(encoding="utf-8"))

    assert ledger["schema_version"] == "improvement-batch-dependency-ledger.v1"
    assert ledger["issue"] == 4984

    phases = ledger["phases"]
    phase_ids = [phase["phase"] for phase in phases]
    assert phase_ids == [1, 2, 3, 4, 5]
    known_phases = set(phase_ids)
    for phase in phases:
        assert set(phase["recommended_after"]) <= known_phases
        assert set(phase["parallel_with"]) <= known_phases
        assert all(parent < phase["phase"] for parent in phase["recommended_after"])

    items = ledger["items"]
    issues = [item["issue"] for item in items]
    assert issues == EXPECTED_SEQUENCE
    assert set(issues) == set(range(4966, 4984))
    assert [item["phase"] for item in items] == sorted(item["phase"] for item in items)
    assert "#4997" in items[0]["completion"]["evidence"]
    assert "#4975" in next(item for item in items if item["issue"] == 4972)["evidence_boundary"]

    positions = {issue: position for position, issue in enumerate(issues)}
    external_issues = {gate["issue"] for gate in ledger["cross_cutting_gates"]}
    for item in items:
        assert item["phase"] in known_phases
        assert item["owner"]["issue"] == item["issue"]
        assert item["owner"]["assignees"] == []
        assert item["evidence_boundary"].strip()

        completion = item["completion"]
        assert completion["github_state"] in {"open", "closed"}
        assert completion["delivery_state"].strip()
        assert completion["evidence"].strip()
        assert completion["remaining"].strip()

        dependencies = item["implementation_prerequisites"] + item["evidence_gates"]
        for dependency in dependencies:
            assert dependency in positions or dependency in external_issues
            if dependency in positions:
                assert positions[dependency] < positions[item["issue"]]
