"""Contract tests for the issue #3078 job 13521 evidence registration."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "docs/context/evidence/issue_3078_package_a_job_13521_2026-07-16"
SYNTHETIC_BUNDLE = "docs/context/evidence/issue_3078_package_a_2026-07-08"
DIAGNOSTIC_NAME = "seed_rank_stability_diagnostic.json"
DIAGNOSTIC_REL = (
    f"docs/context/evidence/issue_3078_package_a_job_13521_2026-07-16/{DIAGNOSTIC_NAME}"
)
NO_COMPARATOR_SIDECAR_REL = (
    f"{DIAGNOSTIC_REL.removesuffix(DIAGNOSTIC_NAME)}no_eligible_comparator.json.review.json"
)
OVERLAY_NAME = "package_a_domain_decision_overlay.json"


def _load_json(name: str) -> dict:
    """Load one registered JSON object."""
    payload = json.loads((BUNDLE / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _parse_checksums(lines: list[str]) -> dict[str, str]:
    """Parse checksum lines while rejecting malformed or duplicate paths."""
    recorded: dict[str, str] = {}
    for line in lines:
        digest, separator, rel_path = line.partition("  ")
        assert separator == "  " and rel_path, line
        assert rel_path not in recorded, f"duplicate checksum entry: {rel_path}"
        recorded[rel_path] = digest
    return recorded


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
        "domain_decision_overlay": OVERLAY_NAME,
        "episode_count": 18,
        "heldout_table_episode_count": 18,
        "issue_result_classification": "diagnostic",
        "review_state": "renderer_output_not_domain_approved",
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
    with (BUNDLE / "baseline_table.csv").open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))

    assert len(rows) == 6
    assert sum(int(row["episode_count"]) for row in rows) == 18
    assert all(int(row["eligible_episode_count"]) == 3 for row in rows)
    assert {row["planner"] for row in rows} == {"goal", "social_force", "orca"}
    assert len(baseline_rows) == 0
    assert all(row["claim_eligible"] == "false" for row in deltas)
    assert all(row["benchmark_set_mean_snqi"] == "" for row in deltas)
    assert all(row["transfer_delta_snqi"] == "" for row in deltas)


def test_no_eligible_comparator_receipt_is_recorded() -> None:
    """Verify that a no_eligible_comparator receipt exists and documents eligibility audit."""
    receipt = _load_json("no_eligible_comparator.json")
    readiness_receipt = json.loads(
        (
            REPO_ROOT
            / "docs/context/evidence/issue_3078_package_a_readiness/no_eligible_comparator.json"
        ).read_text(encoding="utf-8")
    )

    assert receipt["status"] == "no_eligible_comparator"
    assert readiness_receipt["status"] == "no_eligible_comparator"
    assert receipt["validated_criteria"]["job_id"] == 13521
    assert receipt["validated_criteria"]["source_episode_store_sha256"] == (
        "46466cd3db27d6f8a10181a8ec7c4676b24179bb97902aa8eec686d09a53942b"
    )
    assert receipt["validated_criteria"]["baseline_table_empty"] is True
    assert receipt["validated_criteria"]["transfer_delta_benchmark_set_snqi_empty"] is True
    assert receipt["validated_criteria"]["transfer_delta_snqi_empty"] is True
    assert receipt["comparator_audit"]["matching_benchmark_set_campaign_found"] is False


def test_domain_decision_overlay_preserves_renderer_state_and_binds_receipt() -> None:
    """The reviewed decision overlays, rather than rewrites, generated evidence."""
    overlay = _load_json(OVERLAY_NAME)
    bundle_rel = BUNDLE.relative_to(REPO_ROOT).as_posix()
    assert overlay["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
    assert overlay["schema_version"] == "package_a_domain_decision_overlay.v1"
    assert overlay["issue"] == 3078
    assert overlay["bundle_path"] == bundle_rel
    assert (
        overlay["claim_boundary_ref"]
        == "Issue #6157 canonical diagnostic-only boundary, as carried by issue #6430"
    )
    assert overlay["approval"] == {
        "approval_source": "Issue #6430 approved decision, carried by PR #6444",
        "approved_at_utc": "2026-07-30",
        "review_marker": "DOMAIN-APPROVED (2026-07-30) - REVIEWED",
        "status": "domain_approved",
    }
    assert overlay["forbidden_actions_confirmed"] == {
        "benchmark_campaign_run": False,
        "compute_submit": False,
        "paper_claim_edits": False,
        "ranking_claim_promotion": False,
    }
    assert overlay["decision"] == {
        "claim_status": "reviewed",
        "classification": "diagnostic",
        "transfer_sub_classification": "no_eligible_comparator",
    }

    claim_card = yaml.safe_load((BUNDLE / "claim_card.yaml").read_text(encoding="utf-8"))
    assert claim_card["claim_status"] == "not_reviewed"
    assert claim_card["classification"] == "diagnostic_review_ready"
    packet = _load_json("package_a_decision_packet.json")
    assert packet["classification"] == "diagnostic_review_ready"

    for artifact in overlay["renderer_outputs"]:
        path = REPO_ROOT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]

    renderer_by_path = {artifact["path"]: artifact for artifact in overlay["renderer_outputs"]}
    assert set(renderer_by_path) == {
        f"{bundle_rel}/claim_card.yaml",
        f"{bundle_rel}/package_a_decision_packet.json",
        f"{bundle_rel}/postrun_acceptance.json",
        f"{bundle_rel}/registration.json",
        f"{bundle_rel}/seed_rank_stability_diagnostic.json",
    }
    assert renderer_by_path[f"{bundle_rel}/claim_card.yaml"]["claim_status"] == "not_reviewed"
    assert renderer_by_path[f"{bundle_rel}/claim_card.yaml"]["classification"] == (
        "diagnostic_review_ready"
    )
    assert renderer_by_path[f"{bundle_rel}/package_a_decision_packet.json"]["classification"] == (
        "diagnostic_review_ready"
    )
    assert renderer_by_path[f"{bundle_rel}/postrun_acceptance.json"]["classification"] == (
        "diagnostic_review_ready"
    )
    assert renderer_by_path[f"{bundle_rel}/registration.json"]["classification"] == (
        "diagnostic_review_ready"
    )
    seed_renderer = renderer_by_path[f"{bundle_rel}/seed_rank_stability_diagnostic.json"]
    assert seed_renderer["classification"] == "not_identifiable"

    receipt_binding = overlay["transfer_comparator_receipt"]
    assert receipt_binding["path"] == NO_COMPARATOR_SIDECAR_REL.removesuffix(".review.json")
    assert receipt_binding["review_sidecar_path"] == NO_COMPARATOR_SIDECAR_REL
    assert receipt_binding["status"] == "no_eligible_comparator"
    receipt_path = REPO_ROOT / receipt_binding["path"]
    sidecar_path = REPO_ROOT / receipt_binding["review_sidecar_path"]
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == receipt_binding["sha256"]
    assert (
        hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
        == receipt_binding["review_sidecar_sha256"]
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert "DOMAIN-APPROVED" in sidecar["review_marker"]
    assert sidecar["review_source"] == OVERLAY_NAME

    manifest = yaml.safe_load((BUNDLE / "artifact_manifest.yaml").read_text(encoding="utf-8"))
    manifest_paths = {entry["path"] for entry in manifest["files"]}
    assert NO_COMPARATOR_SIDECAR_REL in manifest_paths
    assert f"{BUNDLE.relative_to(REPO_ROOT).as_posix()}/{OVERLAY_NAME}" in manifest_paths
    for entry in manifest["files"]:
        path = REPO_ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_registration_preserves_private_row_store_checksum_without_committing_rows() -> None:
    """The compact source rows remain private while their exact checksum is registered."""
    registration = _load_json("registration.json")
    store = registration["source_episode_store"]

    assert store["uri"] == "private-campaign://job-13521/result_store/episodes.parquet"
    assert store["sha256"] == ("46466cd3db27d6f8a10181a8ec7c4676b24179bb97902aa8eec686d09a53942b")
    assert store["committed"] is False
    assert not (BUNDLE / "episodes.parquet").exists()
    assert registration["supersedes"]["job_id"] == 13506
    assert registration["domain_decision_overlay"] == OVERLAY_NAME
    assert registration["review_state"] == "renderer_output_not_domain_approved"


def test_reproduction_documents_local_hydration_boundary() -> None:
    """The reproduction command hydrates private evidence before using local Path inputs."""
    reproduction = (BUNDLE / "reproduction.md").read_text(encoding="utf-8")

    assert "private-campaign://job-13521/result_store" in reproduction
    assert '"$JOB_13521_RESULT_STORE"' in reproduction
    assert "--result-store private-campaign://" not in reproduction
    assert "--output-dir output/issue_3078_package_a_job_13521_transfer_report" in reproduction


def test_real_data_seed_rank_stability_diagnostic_records_not_identifiable() -> None:
    """The real-data diagnostic records both diagnostics as not_identifiable."""
    diagnostic = _load_json(DIAGNOSTIC_NAME)

    headline = diagnostic["headline_rank_stability_contract"]
    assert headline["label"] == "not_identifiable"
    assert headline["promotion_allowed"] is False
    assert headline["seed_count"] == 1
    assert "single evaluation seed (111)" in headline["reason"].lower()
    assert headline["claim_status"] == "not_identifiable_single_seed"

    transfer = diagnostic["heldout_transfer_delta_classification"]
    assert transfer["label"] == "not_identifiable"
    assert transfer["claim_eligible"] is False
    assert transfer["baseline_table_empty"] is True
    assert transfer["transfer_delta_snqi_empty"] is True
    assert "no eligible benchmark-set comparator" in transfer["reason"].lower()
    assert transfer["claim_status"] == "not_identifiable_no_eligible_comparator"

    rows = diagnostic["planner_rank_stability"]
    assert {row["planner"] for row in rows} == {"goal", "social_force", "orca"}
    assert all(row["rank_stability"] == "not_identifiable" for row in rows)
    assert all(row["promotion_allowed"] is False for row in rows)


def test_diagnostic_preserves_eighteen_identity_accounting() -> None:
    """The diagnostic records the frozen 18-identity, adapter/native accounting."""
    diagnostic = _load_json(DIAGNOSTIC_NAME)
    provenance = diagnostic["provenance"]

    assert provenance["unique_identity_count"] == 18
    assert provenance["cell_count"] == 6
    assert provenance["row_status_counts"] == {"adapter": 12, "native": 6}
    assert provenance["fallback_degraded_rows"] == 0
    assert provenance["synthetic_fixture_used"] is False
    assert provenance["source_episode_store_sha256"] == (
        "46466cd3db27d6f8a10181a8ec7c4676b24179bb97902aa8eec686d09a53942b"
    )
    assert diagnostic["seed_basis"]["seeds"] == [111]
    assert diagnostic["seed_basis"]["seed_count"] == 1


def test_diagnostic_never_relabels_adapter_rows_native_only() -> None:
    """Adapter planners (social_force, orca) stay labeled adapter in the diagnostic."""
    diagnostic = _load_json(DIAGNOSTIC_NAME)
    by_planner = {row["planner"]: row["row_status"] for row in diagnostic["planner_rank_stability"]}

    assert by_planner == {"goal": "native", "social_force": "adapter", "orca": "adapter"}


def test_decision_packet_references_real_data_diagnostic_not_synthetic() -> None:
    """The decision packet points at the real-data diagnostic, not the synthetic analysis."""
    decision = _load_json("package_a_decision_packet.json")

    seed_reports = decision["seed_analysis_reports"]
    assert len(seed_reports) == 1
    assert seed_reports[0]["path"] == DIAGNOSTIC_REL
    assert seed_reports[0]["ok"] is True
    artifact_criterion = next(
        item
        for item in decision["acceptance_criteria"]
        if item["criterion"].startswith("Produces baseline table")
    )
    assert DIAGNOSTIC_REL in artifact_criterion["evidence"]
    assert SYNTHETIC_BUNDLE not in json.dumps(decision)


def test_no_synthetic_bundle_reference_remains_in_regenerated_files() -> None:
    """Regenerated surfaces no longer consume the synthetic 2026-07-08 seed analysis."""
    for name in (
        "package_a_decision_packet.json",
        "claim_card.yaml",
        "reproduction.md",
        "README.md",
    ):
        content = (BUNDLE / name).read_text(encoding="utf-8")
        assert SYNTHETIC_BUNDLE not in content, name


def test_reproduction_documents_diagnostic_rebuild_and_checksum_checks() -> None:
    """The novel diagnostic artifacts have an executable exact-byte reproduction path."""
    reproduction = (BUNDLE / "reproduction.md").read_text(encoding="utf-8")

    assert "scripts/analysis/build_issue_3078_job_13521_diagnostic.py --check" in reproduction
    assert (
        "sha256sum -c "
        "docs/context/evidence/issue_3078_package_a_job_13521_2026-07-16/checksums.sha256"
        in reproduction
    )


def test_deterministic_figures_present_and_checksummed() -> None:
    """Both deterministic diagnostic figures exist and are non-empty PNG artifacts."""
    for name in ("fig_seed_rank_stability.png", "fig_transfer_delta.png"):
        path = BUNDLE / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name


def test_checksums_cover_every_primary_bundle_file() -> None:
    """checksums.sha256 lists and correctly hashes every primary bundle artifact."""
    lines = (BUNDLE / "checksums.sha256").read_text(encoding="utf-8").strip().splitlines()
    recorded = _parse_checksums(lines)

    # Every non-sidecar, non-checksum file under the bundle must be covered.
    expected = {
        f"docs/context/evidence/issue_3078_package_a_job_13521_2026-07-16/"
        f"{path.relative_to(BUNDLE).as_posix()}"
        for path in BUNDLE.rglob("*")
        if path.is_file()
        and not path.name.endswith(".review.json")
        and path.name != "checksums.sha256"
    }
    expected.add(NO_COMPARATOR_SIDECAR_REL)
    assert set(recorded) == expected, (
        f"missing={expected - set(recorded)} extra={set(recorded) - expected}"
    )

    for rel_path, digest in recorded.items():
        actual = hashlib.sha256((REPO_ROOT / rel_path).read_bytes()).hexdigest()
        assert actual == digest, rel_path

    # The new diagnostic and figures are explicitly covered.
    assert DIAGNOSTIC_REL in recorded
    assert (
        "docs/context/evidence/issue_3078_package_a_job_13521_2026-07-16/fig_seed_rank_stability.png"
        in recorded
    )
    assert (
        "docs/context/evidence/issue_3078_package_a_job_13521_2026-07-16/fig_transfer_delta.png"
        in recorded
    )


def test_checksum_parser_rejects_duplicate_paths() -> None:
    """A duplicate path cannot hide an earlier checksum entry."""
    lines = (BUNDLE / "checksums.sha256").read_text(encoding="utf-8").strip().splitlines()

    with pytest.raises(AssertionError, match="duplicate checksum entry"):
        _parse_checksums([*lines, lines[0]])
