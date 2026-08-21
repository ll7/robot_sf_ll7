"""Contract tests for the canonical algorithm contract registry (issue #7676).

The pre-refactor snapshot fixture pins the hard semantics of every migrated
algorithm; these tests prove the facades still expose byte-identical values
after the registry became the single owner of the migrated families.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from robot_sf.benchmark import algorithm_metadata as metadata_module
from robot_sf.benchmark import algorithm_readiness as readiness_module
from robot_sf.benchmark.algorithm_contract import (
    ALGORITHM_ALIAS_INDEX,
    CONTRACT_RECORDS_BY_NAME,
    MIGRATED_ALGORITHM_RECORDS,
    AlgorithmContractRecord,
    audit_contract_ownership,
    build_alias_index,
    get_contract_record,
    validate_builder_agreement,
)
from robot_sf.benchmark.algorithm_readiness import (
    get_algorithm_readiness,
    paper_baseline_algorithms,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "issue_7676_pre_refactor_snapshot.json"

_METADATA_SURFACES = {
    "_BASELINE_CATEGORY_BY_CANONICAL": "baseline_category",
    "_POLICY_SEMANTICS_BY_CANONICAL": "policy_semantics",
    "_OBSERVATION_SPEC_BY_CANONICAL": "observation_spec",
    "_UPSTREAM_REFERENCE_BY_CANONICAL": "upstream_reference",
    "_KINEMATICS_PROFILE_BY_CANONICAL": "kinematics_profile",
}


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_every_migrated_family_has_exactly_one_record() -> None:
    """Each migrated canonical algorithm owns exactly one registry record."""
    fixture = _fixture()
    assert sorted(CONTRACT_RECORDS_BY_NAME) == sorted(fixture["families"])
    assert len(MIGRATED_ALGORITHM_RECORDS) == len(fixture["families"])


@pytest.mark.parametrize("surface", sorted(_METADATA_SURFACES))
def test_snapshot_parity_metadata_facade(surface: str) -> None:
    """Facade metadata equals the pinned pre-refactor snapshot values."""
    fixture = _fixture()
    facade = getattr(metadata_module, surface)
    field = _METADATA_SURFACES[surface]
    for canonical in fixture["families"]:
        assert facade[canonical] == getattr(CONTRACT_RECORDS_BY_NAME[canonical], field), canonical


def test_snapshot_parity_readiness_and_paper_baseline() -> None:
    """Readiness entries and paper-baseline membership match the snapshot."""
    fixture = _fixture()
    by_name = {spec.canonical_name: spec for spec in readiness_module._ALGORITHMS}
    for canonical, expected in fixture["readiness"].items():
        spec = by_name[canonical]
        assert spec.tier == expected["tier"], canonical
        assert list(spec.aliases) == expected["aliases"], canonical
        assert spec.note == expected["note"], canonical
        assert spec.requires_explicit_opt_in == expected["requires_explicit_opt_in"]
    assert list(paper_baseline_algorithms()) == fixture["paper_baseline"]


def test_canonical_and_alias_lookup_parity() -> None:
    """Registry alias index and readiness facade resolve identically."""
    for record in MIGRATED_ALGORITHM_RECORDS:
        for alias in record.aliases:
            facade_spec = get_algorithm_readiness(alias)
            contract_record = get_contract_record(alias)
            assert facade_spec is not None
            assert contract_record is not None
            assert facade_spec.canonical_name == record.canonical_name
            assert contract_record.canonical_name == record.canonical_name


def test_from_mapping_rejects_unknown_fields() -> None:
    """Unknown fields fail closed instead of dropping contract metadata."""
    mapping = CONTRACT_RECORDS_BY_NAME["orca"].snapshot()
    mapping["unexpected_field"] = 1
    with pytest.raises(ValueError, match="Unknown algorithm-contract fields"):
        AlgorithmContractRecord.from_mapping(mapping)


def test_from_mapping_rejects_missing_required_fields() -> None:
    """Missing required fields fail closed."""
    mapping = CONTRACT_RECORDS_BY_NAME["orca"].snapshot()
    del mapping["observation_spec"]
    with pytest.raises(ValueError, match="Missing required algorithm-contract fields"):
        AlgorithmContractRecord.from_mapping(mapping)


def test_from_mapping_rejects_empty_provenance_payloads() -> None:
    """Unresolved upstream references are rejected at construction time."""
    for payload in ("observation_spec", "upstream_reference", "kinematics_profile"):
        mapping = CONTRACT_RECORDS_BY_NAME["gensafenav_ours_gst"].snapshot()
        mapping[payload] = {}
        with pytest.raises(ValueError, match="must be a non-empty mapping"):
            AlgorithmContractRecord.from_mapping(mapping)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("aliases", "orca", "aliases must be a list or tuple"),
        ("aliases", ("orca", "ORCA"), "aliases must be normalized lowercase"),
        ("tier", "ready", "unknown readiness tier"),
        ("requires_explicit_opt_in", 1, "requires_explicit_opt_in must be boolean"),
        ("paper_baseline_eligible", "false", "paper_baseline_eligible must be boolean"),
        ("policy_builder_owner", "unknown", "unknown policy_builder_owner"),
        ("observation_spec", [], "observation_spec must be a mapping"),
    ],
)
def test_from_mapping_rejects_invalid_schema_values(
    field: str, value: object, message: str
) -> None:
    """Strict construction rejects malformed values instead of coercing them."""
    mapping = CONTRACT_RECORDS_BY_NAME["orca"].snapshot()
    mapping[field] = value
    with pytest.raises(ValueError, match=message):
        AlgorithmContractRecord.from_mapping(mapping)


def test_direct_record_construction_is_strict() -> None:
    """The frozen record cannot bypass schema validation through its constructor."""
    with pytest.raises(ValueError, match="unknown readiness tier"):
        replace(CONTRACT_RECORDS_BY_NAME["orca"], tier="ready")


def test_build_alias_index_rejects_duplicate_aliases() -> None:
    """Duplicate aliases across records are rejected."""
    orca = AlgorithmContractRecord.from_mapping(
        CONTRACT_RECORDS_BY_NAME["orca"].snapshot()
        | {"aliases": ("orca", "social_nav_pyenvs_orca")},
    )
    pyenvs_orca = AlgorithmContractRecord.from_mapping(
        CONTRACT_RECORDS_BY_NAME["social_navigation_pyenvs_orca"].snapshot()
    )
    with pytest.raises(ValueError, match="duplicate algorithm alias"):
        build_alias_index((orca, pyenvs_orca))


def test_build_alias_index_rejects_duplicate_canonical_names() -> None:
    """Duplicate canonical names are rejected."""
    orca_a = AlgorithmContractRecord.from_mapping(CONTRACT_RECORDS_BY_NAME["orca"].snapshot())
    orca_b = AlgorithmContractRecord.from_mapping(CONTRACT_RECORDS_BY_NAME["orca"].snapshot())
    with pytest.raises(ValueError, match="duplicate canonical algorithm name"):
        build_alias_index((orca_a, orca_b))


def test_build_alias_index_rejects_alias_colliding_with_foreign_canonical() -> None:
    """An alias equal to another record's canonical name is rejected."""
    orca = AlgorithmContractRecord.from_mapping(
        CONTRACT_RECORDS_BY_NAME["orca"].snapshot() | {"aliases": ("orca", "goal")},
    )
    goal = AlgorithmContractRecord.from_mapping(
        CONTRACT_RECORDS_BY_NAME["socnav_hrvo"].snapshot()
        | {"canonical_name": "goal", "aliases": ("goal",)},
    )
    with pytest.raises(ValueError, match="collides with a canonical algorithm name"):
        build_alias_index((orca, goal))


def test_builder_agreement_passes_for_current_records() -> None:
    """Every record's declared builder owner actually builds that algorithm."""
    report = validate_builder_agreement(check_legacy_dispatch=True)
    assert sorted(report["checked"]) == sorted(CONTRACT_RECORDS_BY_NAME)
    assert report["skipped_legacy_checks"] == []


def test_builder_agreement_fails_closed_on_missing_socnav_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record whose socnav-family adapter vanished fails closed."""
    from robot_sf.benchmark.map_runner_policies import socnav_family

    monkeypatch.setattr(socnav_family, "_SOCNAV_FAMILY_LOOKUP", {}, raising=False)
    with pytest.raises(RuntimeError, match="no such registered adapter exists"):
        validate_builder_agreement()


def test_audit_reports_no_split_ownership() -> None:
    """No migrated canonical name is split across conflicting owners."""
    report = audit_contract_ownership()
    assert report["schema"] == "algorithm_contract_ownership_audit.v1"
    assert report["split_ownership_detected"] is False
    assert report["conflicts"] == []
    assert set(report["migrated"]) == set(CONTRACT_RECORDS_BY_NAME)
    assert not (set(report["migrated"]) & set(report["legacy_remaining"]))


def test_registry_order_is_stable_and_deterministic() -> None:
    """Record order and derived structures are deterministic across imports."""
    order = [record.canonical_name for record in MIGRATED_ALGORITHM_RECORDS]
    readiness_order = [
        spec.canonical_name
        for spec in readiness_module._ALGORITHMS
        if spec.canonical_name in CONTRACT_RECORDS_BY_NAME
    ]
    assert order == readiness_order
    assert list(ALGORITHM_ALIAS_INDEX) == [
        alias for record in MIGRATED_ALGORITHM_RECORDS for alias in record.aliases
    ]


def test_facade_error_behavior_unchanged() -> None:
    """Placeholder gating and unknown-algorithm behavior are untouched."""
    assert get_algorithm_readiness("not_in_catalog") is None
    with pytest.raises(ValueError, match="placeholder"):
        readiness_module.require_algorithm_allowed(
            algo="rvo",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
        )
    assert get_contract_record("not_in_catalog") is None


def test_record_snapshots_are_isolated_copies() -> None:
    """Mutating a returned snapshot cannot corrupt the shared record."""
    record = CONTRACT_RECORDS_BY_NAME["orca"]
    snapshot = record.snapshot()
    snapshot["observation_spec"]["default_mode"] = "mutated"
    snapshot["aliases"].append("mutated")
    assert record.observation_spec["default_mode"] != "mutated"
    assert "mutated" not in record.aliases
    # facade payloads stay independent objects, matching pre-refactor behavior
    facade_spec = metadata_module._OBSERVATION_SPEC_BY_CANONICAL["orca"]
    assert facade_spec is not record.observation_spec
    facade_spec_copy = copy.deepcopy(facade_spec)
    assert facade_spec == facade_spec_copy
