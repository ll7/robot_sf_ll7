"""Tests for campaign expected-row ledger generation (scripts/validation/generate_campaign_row_ledger.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.validation.generate_campaign_row_ledger import (
    SCHEMA_VERSION,
    ExpectedRow,
    compare_observed_rows,
    expand_campaign_packet,
    generate_ledger,
    main,
    validate_schema,
)

FIXTURES_DIR = Path("tests/validation/fixtures/campaign_row_ledger")


def test_cartesian_grid_expansion() -> None:
    """Standard Cartesian grid generates exact ordered expected rows and valid schema."""
    res = generate_ledger(FIXTURES_DIR / "cartesian_grid.json")
    assert res["ok"] and res["schema"] == SCHEMA_VERSION and res["expected_count"] == 8
    assert len(res["identity_digest"]) == 64 and len(res["ledger_sha256"]) == 64
    first = res["rows"][0]
    assert first["arm"] == "orca" and first["scenario_id"] == "corridor_pass"
    assert first["seed"] == 101 and first["replicate"] == 0
    assert not validate_schema({k: v for k, v in res.items() if k != "ok"})


def test_paired_arms_expansion() -> None:
    """Paired arms expand both arms with pairing metadata."""
    packet = {
        "campaign_id": "c_paired",
        "packet_class": "benchmark",
        "paired_arms": [["baseline_orca", "learned_sf"]],
        "scenarios": ["cross_traffic"],
        "seeds": [201, 202],
        "replicates": 1,
    }
    rows, blockers, camp_id, _ = expand_campaign_packet(packet)
    assert not blockers and camp_id == "c_paired" and len(rows) == 4
    assert {r.arm for r in rows} == {"baseline_orca", "learned_sf"}
    assert all(r.metadata and "pair_partner" in r.metadata for r in rows)


def test_arrays_and_repeats_expansion() -> None:
    """Arrays with multiple replicates emit distinct replicate row identities."""
    packet = {
        "campaign_id": "c_rep",
        "packet_class": "benchmark",
        "arms": ["nmpc"],
        "scenarios": ["blind_corner"],
        "seeds": [301],
        "replicates": 3,
    }
    rows, blockers, _, _ = expand_campaign_packet(packet)
    assert not blockers and len(rows) == 3
    assert [r.replicate for r in rows] == [0, 1, 2]
    assert [r.row_id for r in rows] == [f"c_rep::nmpc::blind_corner::301::r{i}" for i in range(3)]


def test_excluded_cells_omitted() -> None:
    """Cells specified in excluded_cells must be omitted from the ledger."""
    packet = {
        "campaign_id": "c_ex",
        "packet_class": "benchmark",
        "arms": ["planner_a", "planner_b"],
        "scenarios": ["sc_1", "sc_2"],
        "seeds": [401],
        "excluded_cells": [{"arm": "planner_b", "scenario_id": "sc_2"}],
    }
    rows, blockers, _, _ = expand_campaign_packet(packet)
    assert not blockers and len(rows) == 3
    keys = {(r.arm, r.scenario_id) for r in rows}
    assert ("planner_b", "sc_2") not in keys and ("planner_a", "sc_1") in keys


def _p(**kw: Any) -> dict[str, Any]:
    base = {
        "campaign_id": "c",
        "packet_class": "benchmark",
        "arms": ["a"],
        "scenarios": ["s"],
        "seeds": [1],
    }
    return {**base, **kw}


@pytest.mark.parametrize(
    ("packet", "expected_blocker_fragment"),
    [
        (_p(arms=["a", "a"]), "duplicate_identity"),
        (_p(arms=[], scenarios=[]), "under_specified_dimension"),
        (_p(arms=["<TO_BE_CONFIGURED>"]), "unknown_alias"),
        (_p(declared_count=99), "inconsistent_count"),
        (_p(config_path="../../outside.yaml"), "mutable_input"),
    ],
)
def test_invalid_packets_fail_closed(
    packet: dict, expected_blocker_fragment: str, tmp_path: Path
) -> None:
    """Packets violating invariants fail closed and report descriptive blockers."""
    pfile = tmp_path / "packet.json"
    pfile.write_text(json.dumps(packet), encoding="utf-8")
    result = generate_ledger(pfile)
    assert result["ok"] is False
    assert any(expected_blocker_fragment in b for b in result.get("blockers", []))


def test_observed_rows_compliant() -> None:
    """Observed rows matching all expected identities pass with zero anomalies."""
    packet_path = FIXTURES_DIR / "cartesian_grid.json"
    obs_path = FIXTURES_DIR / "observed_compliant.jsonl"
    result = generate_ledger(packet_path, observed_rows_path=obs_path)

    assert result["ok"] is True
    obs = result["observed_comparison"]
    assert obs["status"] == "pass"
    assert obs["summary"]["present"] == 8
    assert all(obs["summary"][k] == 0 for k in obs["summary"] if k != "present")


def test_observed_rows_divergences() -> None:
    """Divergences (missing, unexpected, conflict, fallback, degraded, invalid provenance) fail."""
    result = generate_ledger(
        FIXTURES_DIR / "cartesian_grid.json",
        observed_rows_path=FIXTURES_DIR / "observed_divergent.jsonl",
    )
    assert result["ok"] is False
    obs = result["observed_comparison"]
    assert obs["status"] == "fail"
    for cat in (
        "missing",
        "unexpected",
        "conflict",
        "fallback",
        "degraded",
        "failed",
        "provenance_invalid",
    ):
        assert obs["summary"][cat] >= 1


def test_byte_stability(tmp_path: Path) -> None:
    """Multiple expansions of the same packet yield byte-identical ledger hashes."""
    packet_path = FIXTURES_DIR / "cartesian_grid.json"
    res1 = generate_ledger(packet_path)
    res2 = generate_ledger(packet_path)
    assert res1["identity_digest"] == res2["identity_digest"]
    assert res1["ledger_sha256"] == res2["ledger_sha256"]


def test_research_campaign_manifest_compatibility() -> None:
    """Canonical research campaign manifest parses cleanly without errors."""
    manifest_path = Path("configs/benchmarks/research_campaign_manifest.example.yaml")
    if manifest_path.is_file():
        res = generate_ledger(manifest_path)
        assert res["ok"] is True
        assert res["expected_count"] == 4
        assert res["packet_class"] == "benchmark"
        assert res["campaign_id"] == "issue_3062_example_research_campaign"


def test_cli_execution(tmp_path: Path) -> None:
    """CLI prints valid JSON and supports --check."""
    packet_path = FIXTURES_DIR / "cartesian_grid.json"
    out_file = tmp_path / "ledger.json"
    ret = main(["--packet", str(packet_path), "--output", str(out_file), "--check"])
    assert ret == 0
    assert out_file.is_file()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["ok"] is True


def test_compare_observed_rows_direct() -> None:
    """Unit test for compare_observed_rows edge cases."""
    expected = [
        ExpectedRow("c::a::s::1::r0", "c", "a", "s", 1, 0, "native"),
        ExpectedRow("c::a::s::2::r0", "c", "a", "s", 2, 0, "native"),
    ]
    comparison = compare_observed_rows(
        expected, [{"row_id": "c::a::s::1::r0", "row_status": "native"}]
    )
    assert comparison["status"] == "fail"
    assert comparison["summary"]["missing"] == 1
    assert comparison["summary"]["present"] == 1
