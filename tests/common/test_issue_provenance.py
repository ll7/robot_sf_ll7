"""Contract tests for the issue-number provenance registry (issue #6355).

These identifiers are immutable code-owned breadcrumbs to the issue contracts that
introduced a diagnostic or report field. The tests lock the structural contract
(positive integer ids, non-empty purposes, matching compatibility aliases, and
identity-preserving lookup keys) and the immutability guarantee (frozen dataclass
plus a read-only ``MappingProxyType``). They exercise public constants in memory
only and never touch the network or GitHub state.
"""

from __future__ import annotations

import dataclasses

import pytest

from robot_sf.common.issue_provenance import (
    FALSE_POSITIVE_INJECTION_REPLAY_READINESS,
    FALSE_POSITIVE_INJECTION_REPLAY_READINESS_ISSUE,
    ISSUE_PROVENANCE_BY_KEY,
    LIVE_FORECAST_REPLAY_GATE_CONTRACT,
    LIVE_FORECAST_REPLAY_GATE_CONTRACT_ISSUE,
    SCENARIO_BELIEF_DESIGN_PARENT,
    SCENARIO_BELIEF_DESIGN_PARENT_ISSUE,
    IssueProvenance,
)

#: The full set of published provenance records, in declaration order.
ALL_RECORDS: tuple[IssueProvenance, ...] = (
    SCENARIO_BELIEF_DESIGN_PARENT,
    LIVE_FORECAST_REPLAY_GATE_CONTRACT,
    FALSE_POSITIVE_INJECTION_REPLAY_READINESS,
)

#: Expected issue numbers per record. Locking the exact values prevents an
#: accidental renumber from being silently absorbed by a shape-only test.
EXPECTED_ISSUE_BY_RECORD: dict[IssueProvenance, int] = {
    SCENARIO_BELIEF_DESIGN_PARENT: 1966,
    LIVE_FORECAST_REPLAY_GATE_CONTRACT: 2941,
    FALSE_POSITIVE_INJECTION_REPLAY_READINESS: 3300,
}

#: Lookup keys that must resolve to the identical record objects.
EXPECTED_KEY_BY_RECORD: dict[IssueProvenance, str] = {
    SCENARIO_BELIEF_DESIGN_PARENT: "scenario_belief_design_parent",
    LIVE_FORECAST_REPLAY_GATE_CONTRACT: "live_forecast_replay_gate_contract",
    FALSE_POSITIVE_INJECTION_REPLAY_READINESS: "false_positive_injection_replay_readiness",
}

#: Compatibility integer aliases that must equal each record's ``.issue``.
EXPECTED_ALIAS_BY_RECORD: dict[IssueProvenance, int] = {
    SCENARIO_BELIEF_DESIGN_PARENT: SCENARIO_BELIEF_DESIGN_PARENT_ISSUE,
    LIVE_FORECAST_REPLAY_GATE_CONTRACT: LIVE_FORECAST_REPLAY_GATE_CONTRACT_ISSUE,
    FALSE_POSITIVE_INJECTION_REPLAY_READINESS: FALSE_POSITIVE_INJECTION_REPLAY_READINESS_ISSUE,
}


@pytest.mark.parametrize("record", ALL_RECORDS, ids=lambda r: r.purpose[:24])
def test_record_has_positive_integer_issue(record: IssueProvenance) -> None:
    """Each provenance record carries a positive integer issue id (bool is rejected)."""
    assert isinstance(record.issue, int)
    assert not isinstance(record.issue, bool)
    assert record.issue > 0


@pytest.mark.parametrize("record", ALL_RECORDS, ids=lambda r: r.purpose[:24])
def test_record_has_non_empty_purpose(record: IssueProvenance) -> None:
    """Each provenance record carries a non-empty purpose string describing its contract."""
    assert isinstance(record.purpose, str)
    assert record.purpose.strip() != ""


def test_issue_numbers_match_expected_contract() -> None:
    """The published issue ids match the locked contract values (1966, 2941, 3300)."""
    for record, expected_issue in EXPECTED_ISSUE_BY_RECORD.items():
        assert record.issue == expected_issue


@pytest.mark.parametrize("record", ALL_RECORDS, ids=lambda r: r.purpose[:24])
def test_compatibility_alias_equals_record_issue(record: IssueProvenance) -> None:
    """Each ``*_ISSUE`` compatibility alias equals its record's ``.issue`` value."""
    assert EXPECTED_ALIAS_BY_RECORD[record] == record.issue


@pytest.mark.parametrize("record", ALL_RECORDS, ids=lambda r: r.purpose[:24])
def test_lookup_key_points_to_same_record_object(record: IssueProvenance) -> None:
    """Each expected lookup key resolves to the identical record object (identity check)."""
    key = EXPECTED_KEY_BY_RECORD[record]
    assert ISSUE_PROVENANCE_BY_KEY[key] is record


def test_lookup_mapping_has_exactly_expected_keys() -> None:
    """The lookup mapping exposes exactly the expected keys, with no drift in either direction."""
    expected_keys = set(EXPECTED_KEY_BY_RECORD.values())
    assert set(ISSUE_PROVENANCE_BY_KEY.keys()) == expected_keys


def test_record_field_mutation_is_rejected() -> None:
    """The dataclass is frozen: assigning to a field raises ``FrozenInstanceError``."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        SCENARIO_BELIEF_DESIGN_PARENT.issue = 0  # type: ignore[misc]


def test_lookup_mapping_rejects_item_assignment() -> None:
    """The lookup mapping is read-only: item assignment raises ``TypeError``."""
    with pytest.raises(TypeError):
        ISSUE_PROVENANCE_BY_KEY["scenario_belief_design_parent"] = SCENARIO_BELIEF_DESIGN_PARENT


def test_lookup_mapping_rejects_item_deletion() -> None:
    """The lookup mapping is read-only: deleting a key raises ``TypeError``."""
    with pytest.raises(TypeError):
        del ISSUE_PROVENANCE_BY_KEY["scenario_belief_design_parent"]
