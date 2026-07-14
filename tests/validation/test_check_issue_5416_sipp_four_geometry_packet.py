"""Contract tests for the issue #5416 four-geometry SIPP packet."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validation import check_issue_5416_sipp_four_geometry_packet as checker

PACKET = Path("configs/benchmarks/issue_5416_sipp_four_geometry_preregistration.yaml")


def test_packet_passes_tracked_geometry_and_roster_gate() -> None:
    """The shipped packet resolves all four rows and exactly five opt-in planners."""
    result = checker.validate_packet(checker.load_packet(PACKET))

    assert result["status"] == "ready"
    assert result["planner_count"] == 5
    assert result["scenario_count"] == 4
    assert result["blocked_rows"] == []
    assert [row["scenario_id"] for row in result["certification"]] == [
        "classic_head_on_corridor_low",
        "classic_doorway_low",
        "classic_station_platform_medium",
        "classic_merging_low",
    ]


def test_packet_preserves_stress_only_geometry_as_visible_caveat() -> None:
    """Doorway remains visible as stress-only rather than being promoted or dropped."""
    result = checker.validate_packet(checker.load_packet(PACKET))
    doorway = next(
        row for row in result["certification"] if row["scenario_id"] == "classic_doorway_low"
    )

    assert doorway["classification"] == "knife_edge"
    assert doorway["benchmark_eligibility"] == "stress_only"
    assert doorway["gate"] == "pass"


def test_packet_rejects_transient_routing_state() -> None:
    """Tracked preregistration must not carry queue or host routing state."""
    packet = checker.load_packet(PACKET)
    packet["target_host"] = "imech156-u"

    with pytest.raises(checker.PacketError, match="transient routing state"):
        checker.validate_packet(packet)


def test_packet_requires_all_five_result_seeds() -> None:
    """The paired diagnostic seed budget cannot silently shrink."""
    packet = checker.load_packet(PACKET)
    packet["scenario_contract"]["result_producing_seeds"] = [111]

    with pytest.raises(checker.PacketError, match="seed set mismatch"):
        checker.validate_packet(packet)


def test_packet_rejects_excluded_geometry(monkeypatch) -> None:
    """An excluded certificate blocks the campaign gate and remains machine-visible."""

    def fake_certify(path, *, scenario_id=None):
        return [object()]

    def fake_to_dict(_certificate):
        return {
            "benchmark_eligibility": "excluded",
            "classification": "geometrically_infeasible",
            "reasons": ["test_exclusion"],
            "route_certificates": [{}],
        }

    monkeypatch.setattr(checker, "certify_scenario_file", fake_certify)
    monkeypatch.setattr(checker, "certificate_to_dict", fake_to_dict)

    result = checker.validate_packet(checker.load_packet(PACKET))
    assert result["status"] == "blocked"
    assert result["blocked_rows"] == [
        "classic_head_on_corridor_low",
        "classic_doorway_low",
        "classic_station_platform_medium",
        "classic_merging_low",
    ]
