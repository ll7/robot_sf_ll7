"""Focused contract tests for the dissertation-coverage aggregate."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from scripts.analysis import build_dissertation_coverage_manifest as builder

if TYPE_CHECKING:
    from pathlib import Path


ROOT = builder.REPO_ROOT


def test_profile_and_release_identity_are_verified() -> None:
    """The generated consumer profile stays bound to the frozen release identity."""
    payload, summary, _ = builder.build_outputs()

    profile = payload["consumer_profile"]
    assert profile["consumer_id"] == "ll7-diss-submission-2026"
    assert profile["source_commit"] == "b1d5ab6de708385c0828c99501a9d1c29727ec11"
    assert profile["release_tag"] == "paper-matrix-v2-h600-s30-2026-08-cd831d7582c1"
    assert profile["release_doi"] == "10.5281/zenodo.22077448"
    assert profile["campaign_id"] == "paper_matrix_v2_h600_s30_2026_08_cd831d7582c1"
    assert (profile["planner_count"], profile["scenario_cell_count"]) == (14, 48)
    assert (profile["seed_count"], profile["expected_episode_count"]) == (30, 20160)
    assert "not benchmark evidence" in summary


def test_known_source_conflicts_are_explicitly_retained() -> None:
    """Expected roster and owner gaps remain visible instead of being inferred away."""
    payload, _, _ = builder.build_outputs()

    roster = payload["source_reconciliation"]["planner_roster"]
    assert roster["status"] == "conflict_explicitly_recorded"
    assert roster["source_only"] == ["scenario_adaptive_hybrid_orca_v1"]
    assert roster["release_manifest_only"] == ["scenario_adaptive_hybrid_orca_v2_bottleneck_yield"]
    stale_paths = {
        (entry["source_id"], entry["capability_id"], entry["path"])
        for entry in payload["source_reconciliation"]["owner_paths"]["stale_paths"]
    }
    assert (
        "post_anchor_capability_delta",
        "carla_cross_simulator_bridge",
        "scripts/carla/",
    ) in stale_paths
    assert (
        "future_work_card_incident_to_scenario_provenance",
        "incident_to_scenario_provenance",
        "robot_sf/provenance/",
    ) in stale_paths


def test_source_drift_fails_closed(tmp_path: Path) -> None:
    """A one-byte source drift cannot pass the pinned profile."""
    profile = builder._load_yaml(ROOT / builder.DEFAULT_PROFILE)
    source = profile["source_packages"][0]
    source_copy = tmp_path / "planner.json"
    source_copy.write_bytes((ROOT / source["path"]).read_bytes() + b"\n")
    profile["source_packages"] = copy.deepcopy(profile["source_packages"])
    profile["source_packages"][0]["path"] = str(source_copy)

    with pytest.raises(builder.CoverageContractError, match="source digest drift"):
        builder._source_records(profile, ROOT)


def test_duplicate_capability_ids_fail_closed(tmp_path: Path) -> None:
    """Duplicate semantic IDs in one source package are rejected."""
    profile = builder._load_yaml(ROOT / builder.DEFAULT_PROFILE)
    profile["source_packages"] = copy.deepcopy(profile["source_packages"])
    source = json.loads(
        (ROOT / "docs/context/evidence/planner_development_funnel.v1.json").read_text(
            encoding="utf-8"
        )
    )
    source["candidates"].append(copy.deepcopy(source["candidates"][0]))
    source_copy = tmp_path / "planner-duplicate.json"
    source_copy.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    profile["source_packages"][0]["path"] = str(source_copy)
    profile["source_packages"][0]["sha256"] = hashlib.sha256(source_copy.read_bytes()).hexdigest()

    with pytest.raises(
        builder.CoverageContractError,
        match="duplicate capability ID in planner_development_funnel",
    ):
        builder._source_records(profile, ROOT)


def test_unexpected_stale_owner_fails_closed() -> None:
    """A newly stale owner is not accepted merely because the source is review-marked."""
    profile = builder._load_yaml(ROOT / builder.DEFAULT_PROFILE)
    anchor, release_manifest, _ = builder._verify_anchor_and_release(profile, ROOT)
    del anchor
    _, records, _ = builder._source_records(profile, ROOT)
    target = next(
        item for item in records if item["record_key"] == "route_side_homotopy_observability"
    )
    target["record"]["owner_paths"].append("tests/does_not_exist_for_coverage.v1/")

    with pytest.raises(builder.CoverageContractError, match="owner-path reconciliation is stale"):
        builder._verify_reconciliation(profile, ROOT, release_manifest, records)


def test_conflicting_statuses_fail_closed() -> None:
    """Overlapping source rows must agree on evidence classification."""
    profile = builder._load_yaml(ROOT / builder.DEFAULT_PROFILE)
    _, records, _ = builder._source_records(profile, ROOT)
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in records:
        grouped.setdefault(item["record_key"], []).append(item)
    items = copy.deepcopy(grouped["carla_cross_simulator_bridge"])
    items[0]["record"]["evidence_status"] = "release_evaluated"

    with pytest.raises(builder.CoverageContractError, match="conflicting evidence_status"):
        builder._aggregate_row(items, ROOT)


def test_source_statuses_and_wording_are_preserved() -> None:
    """The aggregate carries raw source vocabularies alongside projections."""
    payload, _, _ = builder.build_outputs()
    rows = {row["capability_id"]: row for row in payload["capabilities"]}

    release_row = rows["prediction_planner"]
    assert release_row["evidence_status"] == "release_evaluated"
    assert release_row["source_evidence_statuses"] == [
        {"source_id": "planner_development_funnel", "value": "release_evaluated"}
    ]
    assert release_row["anchor_relation"] == "present_at_anchor"
    assert release_row["implementation_status"] == "unknown"

    bridge = rows["carla_cross_simulator_bridge"]
    assert bridge["evidence_status"] == "diagnostic_only"
    assert bridge["implementation_status"] == "partial"
    assert bridge["dissertation_relationship"] == "future_work_bridge"
    assert len(bridge["claim_boundary_variants"]) == 2
    assert all(entry["value"] == "diagnostic_only" for entry in bridge["source_evidence_statuses"])


def test_rebuild_and_checksums_are_deterministic() -> None:
    """Repeated builds match tracked bytes and the complete checksum inventory."""
    first = builder.build_outputs()
    second = builder.build_outputs()
    assert json.dumps(first[0], sort_keys=True) == json.dumps(second[0], sort_keys=True)
    assert first[1:] == second[1:]

    builder.check_outputs(
        *first,
        root=ROOT,
        manifest_path=builder.DEFAULT_MANIFEST,
        summary_path=builder.DEFAULT_SUMMARY,
        checksums_path=builder.DEFAULT_CHECKSUMS,
    )
    checksum_lines = (ROOT / builder.DEFAULT_CHECKSUMS).read_text(encoding="utf-8").splitlines()
    assert checksum_lines[0] == "# AI-GENERATED NEEDS-REVIEW"
    listed_paths = {line.split("  ", 1)[1] for line in checksum_lines[1:]}
    assert {
        "configs/publication/dissertation_coverage_v1.yaml",
        "docs/context/dissertation_coverage/coverage_manifest.v1.json",
        "docs/context/dissertation_coverage/coverage_summary.md",
    }.issubset(listed_paths)
