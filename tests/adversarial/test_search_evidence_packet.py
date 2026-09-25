"""Tests for compact falsification-packet reconciliation in the replay gallery."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.adversarial import replay_gallery


def _write_packet(root: Path, *, critical: bool = False, unknown: bool = False) -> Path:
    """Create a one-row, internally checksummed-by-reference #9645 packet."""
    packet = root / "evidence" / "payload"
    manifests_dir = packet / "source_manifests"
    manifests_dir.mkdir(parents=True)
    candidate = {
        "start": {"x": 1.0, "y": 2.0, "theta": 0.0},
        "goal": {"x": 3.0, "y": 2.0, "theta": 0.0},
        "scenario_seed": 17,
    }
    scenario_id = "candidate_0000"
    outcome = {
        "collision_event": critical,
        "route_complete": not critical,
        "timeout_event": False,
    }
    primary_failure = "collision" if critical else "unknown" if unknown else "success"
    episode_status = "collision" if critical else "success"
    row_status_label = "unexpected_failure" if unknown else "successful_evidence"
    execution_evidence = not unknown
    effective_hash = "b" * 64
    manifest_candidate = {
        "candidate": candidate,
        "bundle_path": "output/raw/candidate_0000",
        "scenario_yaml_path": "output/raw/candidate_0000/scenario.yaml",
        "episode_record_path": "output/raw/candidate_0000/episode_records.jsonl",
        "effective_scenario_hash": effective_hash,
        "objective_value": 0.0,
        "error": None,
        "analysis_eligibility": {
            "schema_version": "search_analysis_eligibility.v1",
            "eligible": True,
            "certificate_ok": True,
            "execution_mode": "native",
            "objective_scored": True,
        },
        "certification_status": {
            "schema_version": "scenario_certification_status.v1",
            "status": "passed",
            "details": {
                "certificates": [
                    {
                        "scenario_id": scenario_id,
                        "classification": "valid",
                        "benchmark_eligibility": "eligible",
                    }
                ]
            },
        },
        "failure_attribution": {
            "status": "unsupported" if unknown else "attributed",
            "primary_failure": primary_failure,
            "details": {
                "status": episode_status,
                "termination_reason": episode_status,
                "execution_mode": "native",
                "readiness_status": "native",
                "availability_status": "available",
                "outcome": outcome,
            },
        },
    }
    source_manifest = {
        "schema_version": "adversarial-search-manifest.v1",
        "config": {
            "seed": 17,
            "budget": 1,
            "objective": "constraints_first_lexicographic_v1",
            "policy": "goal",
            "output_dir": "output/raw/random",
        },
        "candidates": [manifest_candidate],
    }
    manifest_path = manifests_dir / "random_seed_17.json"
    manifest_bytes = json.dumps(source_manifest, sort_keys=True).encode() + b"\n"
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    candidate_sha = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()

    csv_row = {
        "run_id": "random_17",
        "sampler": "random",
        "sampler_seed": "17",
        "evaluation_index": "1",
        "scenario_id": scenario_id,
        "candidate_bundle": manifest_candidate["bundle_path"],
        "candidate_parameters_json": json.dumps(
            candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ),
        "candidate_sha256": candidate_sha,
        "effective_scenario_sha256": effective_hash,
        "certification_status": "passed",
        "certification_classification": "valid",
        "benchmark_eligibility": "eligible",
        "analysis_eligibility": json.dumps(
            manifest_candidate["analysis_eligibility"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "row_status": row_status_label,
        "counts_as_execution_success_evidence": str(execution_evidence),
        "execution_mode": "native",
        "readiness_status": "native",
        "availability_status": "available",
        "episode_status": episode_status,
        "termination_reason": episode_status,
        "collision_event": str(critical),
        "route_complete": str(not critical),
        "timeout_event": "False",
        "failure_attribution": primary_failure,
        "objective_value": "0.0",
        "min_clearance_m": "8.0",
        "distance_to_human_min_m": "9.0",
        "near_miss_count": "0",
        "total_collision_count": str(int(critical)),
        "ped_collision_count": str(int(critical)),
        "steps": "12",
        "source_manifest_sha256": manifest_sha,
        "scenario_yaml_sha256": "c" * 64,
        "episode_records_sha256": "d" * 64,
        "error": "",
    }
    with (packet / "candidate_evaluations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_row))
        writer.writeheader()
        writer.writerow(csv_row)

    status_row = {
        "row_id": "random_17:01",
        "status": row_status_label,
        "counts_as_success_evidence": execution_evidence,
        "candidate_certification_status": "passed",
        "candidate_classification": "valid",
        "execution_mode": "native",
        "readiness_status": "native",
        "availability_status": "available",
        "fallback_or_degraded": False,
        "error": None,
    }
    _write_json(
        packet / "row_status.json",
        {
            "schema_version": "benchmark_row_status.v1",
            "issue": 9645,
            "counts": {row_status_label: 1},
            "rows": [status_row],
        },
    )
    _write_json(
        packet / "summary.json",
        {
            "schema_version": "issue_9645_bounded_pilot_summary.v1",
            "issue": 9645,
            "source_revision": "a" * 40,
            "pilot_budget": {
                "attempted": 1,
                "completed": 1,
                "failed": 0,
                "invalid": 0,
                "planned": 1,
                "unknown_or_scoreless": int(unknown),
            },
            "pilot_metrics": {
                "objective": "constraints_first_lexicographic_v1",
                "collision_events": int(critical),
                "timeouts": 0,
                "route_completions": int(not critical),
                "near_miss_count_total": 0,
                "min_clearance_m_range": [8.0, 8.0],
                "distance_to_human_min_m_range": [9.0, 9.0],
                "score_set": [0.0],
            },
            "candidate_outcomes": {
                "new_counterexamples_discovered": int(critical),
                "distinct_effective_scenarios": 1,
                "distinct_candidate_specs": 1,
                "duplicate_candidate_specs": 0,
            },
            "evaluation_budget_consumed": {"pilot_search_candidates": 1},
            "feasibility": {
                "pilot_empirical_goal_successes": int(not critical and not unknown),
                "pilot_scenario_certification": {"passed:valid": 1},
            },
            "runs": [
                {
                    "run_id": "random_17",
                    "sampler": "random",
                    "sampler_seed": 17,
                    "candidate_rows": 1,
                    "attempted_evaluations": 1,
                    "planned_evaluations": 1,
                    "failed_evaluations": 0,
                    "invalid_candidates": 0,
                    "best_objective": 0.0,
                }
            ],
        },
    )
    _write_json(
        packet / "run_metadata.json",
        {
            "schema_version": "issue_9645_execution_provenance.v1",
            "experiment_source_commit": "a" * 40,
        },
    )
    _write_json(
        packet / "convergence_report.json",
        {
            "schema_version": "adversarial-search-convergence-report.v1",
            "claim_scope": "finite budget only",
            "aggregates": [
                {
                    "sampler": "random",
                    "candidate_accounting": {
                        "attempted": 1,
                        "critical": int(critical),
                        "failed": 0,
                        "invalid": 0,
                        "missing": 0,
                        "duplicate": 0,
                        "valid_total_minus_invalid_minus_failed": 1,
                        "duplicate_rate_observed": 0.0,
                        "invalid_rate_observed": 0.0,
                    },
                }
            ],
        },
    )
    _refresh_bundle_receipts(packet)
    return packet


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_bundle_receipts(packet: Path) -> None:
    """Regenerate fixture receipts after a deliberate source-file mutation."""
    files = sorted(path for path in packet.rglob("*") if path.is_file())
    entries = []
    checksum_lines = []
    for path in files:
        relative = path.relative_to(packet).as_posix()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        entries.append({"path": relative, "sha256": digest, "size_bytes": len(content)})
        checksum_lines.append(f"{digest}  payload/{relative}")
    _write_json(
        packet.parent / "evidence_bundle_manifest.json",
        {
            "schema_version": "evidence_bundle.v1",
            "bundle_name": "test-search-packet",
            "totals": {
                "file_count": len(entries),
                "total_bytes": sum(e["size_bytes"] for e in entries),
            },
            "files": entries,
        },
    )
    (packet.parent / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )


def test_compact_packet_accounts_success_eligibility_and_missing_inputs_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _write_packet(tmp_path)
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: tmp_path)

    first = replay_gallery.build_replay_gallery(
        packet, tmp_path / "output" / "gallery-a", top_k=5, render=False, video=False
    )
    replay_gallery.build_replay_gallery(
        packet, tmp_path / "output" / "gallery-b", top_k=5, render=False, video=False
    )

    row = first["candidates"][0]
    assert first["summary"]["source_candidate_count"] == 1
    assert first["summary"]["critical_case_count"] == 0
    assert first["summary"]["zero_critical_result_verified"] is True
    assert "availability does not establish" in first["summary"]["replay_input_scope"]
    assert first["summary"]["scenario_eligibility_counts"] == {"eligible": 1}
    assert first["summary"]["execution_outcome_counts"] == {"successful_execution": 1}
    assert row["case_criticality"] == "noncritical_success"
    assert row["scenario_id"] == "candidate_0000"
    assert row["candidate_parameters"]["scenario_seed"] == 17
    assert row["replay_input_status"] == "missing"
    assert row["missing_replay_inputs"] == ["scenario_yaml", "episode_record"]
    assert first["cases"] == []
    assert (tmp_path / "output" / "gallery-a" / "gallery_manifest.json").read_bytes() == (
        tmp_path / "output" / "gallery-b" / "gallery_manifest.json"
    ).read_bytes()


def test_compact_packet_rejects_report_disagreement_and_critical_cases_require_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: tmp_path)
    contradictory = _write_packet(tmp_path / "contradictory")
    convergence_path = contradictory / "convergence_report.json"
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    convergence["aggregates"][0]["candidate_accounting"]["critical"] = 1
    _write_json(convergence_path, convergence)
    _refresh_bundle_receipts(contradictory)
    with pytest.raises(ValueError, match="convergence aggregate random critical conflicts"):
        replay_gallery.build_replay_gallery(
            contradictory,
            tmp_path / "output" / "contradictory-gallery",
            render=False,
            video=False,
        )
    assert not (tmp_path / "output" / "contradictory-gallery").exists()

    critical = _write_packet(tmp_path / "critical", critical=True)
    with pytest.raises(ValueError, match="pass the original adversarial-search-manifest"):
        replay_gallery.build_replay_gallery(
            critical,
            tmp_path / "output" / "critical-gallery",
            render=False,
            video=False,
        )
    assert not (tmp_path / "output" / "critical-gallery").exists()


def test_compact_packet_keeps_unknown_criticality_out_of_zero_case_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _write_packet(tmp_path, unknown=True)
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: tmp_path)

    result = replay_gallery.build_replay_gallery(
        packet, tmp_path / "output" / "unknown-gallery", render=False, video=False
    )

    assert result["summary"]["zero_critical_result_verified"] is False
    assert result["summary"]["criticality_unknown_count"] == 1
    assert result["selection"]["status"] == "criticality_unknown_no_candidate_selected"
    assert result["candidates"][0]["case_criticality"] == "unknown"
    assert result["candidates"][0]["selection_status"] == "no_case_selected"


def test_compact_packet_does_not_verify_zero_critical_for_incomplete_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _write_packet(tmp_path)
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: tmp_path)
    manifest_path = packet / "source_manifests" / "random_seed_17.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest["config"]["budget"] = 2
    _write_json(manifest_path, source_manifest)
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    csv_path = packet / "candidate_evaluations.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["source_manifest_sha256"] = manifest_digest
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = packet / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["pilot_budget"]["planned"] = 2
    summary["runs"][0]["planned_evaluations"] = 2
    _write_json(summary_path, summary)
    convergence_path = packet / "convergence_report.json"
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    convergence["aggregates"][0]["candidate_accounting"]["missing"] = 1
    _write_json(convergence_path, convergence)
    _refresh_bundle_receipts(packet)

    result = replay_gallery.build_replay_gallery(
        packet, tmp_path / "output" / "incomplete-budget", render=False, video=False
    )

    assert result["summary"]["planned_candidate_count"] == 2
    assert result["summary"]["source_candidate_count"] == 1
    assert result["summary"]["missing_planned_candidate_count"] == 1
    assert result["summary"]["zero_critical_result_verified"] is False
    assert result["selection"]["status"] == "incomplete_budget_no_candidate_selected"


def test_compact_packet_preserves_scoreless_candidate_without_zero_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _write_packet(tmp_path)
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: tmp_path)
    manifest_path = packet / "source_manifests" / "random_seed_17.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest["candidates"][0]["objective_value"] = None
    _write_json(manifest_path, source_manifest)
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    csv_path = packet / "candidate_evaluations.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["source_manifest_sha256"] = manifest_digest
    rows[0]["objective_value"] = ""
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = packet / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["pilot_budget"]["unknown_or_scoreless"] = 1
    summary["pilot_metrics"]["score_set"] = []
    summary["runs"][0]["best_objective"] = None
    _write_json(summary_path, summary)
    _refresh_bundle_receipts(packet)

    result = replay_gallery.build_replay_gallery(
        packet, tmp_path / "output" / "scoreless-candidate", render=False, video=False
    )

    assert result["summary"]["scoreless_candidate_count"] == 1
    assert result["summary"]["zero_critical_result_verified"] is False
    assert result["selection"]["status"] == "scoreless_candidate_no_candidate_selected"


def test_compact_packet_rejects_candidate_rows_omitted_from_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _write_packet(tmp_path)
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: tmp_path)
    manifest_path = packet / "source_manifests" / "random_seed_17.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["budget"] = 2
    manifest["candidates"].append(dict(manifest["candidates"][0]))
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    csv_path = packet / "candidate_evaluations.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["source_manifest_sha256"] = manifest_digest
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = packet / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["pilot_budget"]["planned"] = 2
    summary["runs"][0]["planned_evaluations"] = 2
    _write_json(summary_path, summary)
    convergence_path = packet / "convergence_report.json"
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    convergence["aggregates"][0]["candidate_accounting"]["missing"] = 1
    _write_json(convergence_path, convergence)
    _refresh_bundle_receipts(packet)

    with pytest.raises(ValueError, match="does not account for every candidate"):
        replay_gallery.build_replay_gallery(
            packet, tmp_path / "output" / "omitted-candidate-gallery", render=False, video=False
        )


def test_compact_packet_rejects_candidate_scenario_id_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _write_packet(tmp_path)
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: tmp_path)
    csv_path = packet / "candidate_evaluations.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["scenario_id"] = "tampered_scenario"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_bundle_receipts(packet)

    with pytest.raises(ValueError, match="scenario_id conflicts with source manifest"):
        replay_gallery.build_replay_gallery(
            packet, tmp_path / "output" / "tampered-id-gallery", render=False, video=False
        )


def test_compact_packet_keeps_unhashed_existing_inputs_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _write_packet(tmp_path)
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: tmp_path)
    candidate_dir = tmp_path / "output" / "raw" / "candidate_0000"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "scenario.yaml").write_text("scenario: test\n", encoding="utf-8")
    (candidate_dir / "episode_records.jsonl").write_text("{}\n", encoding="utf-8")
    csv_path = packet / "candidate_evaluations.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["scenario_yaml_sha256"] = ""
    rows[0]["episode_records_sha256"] = ""
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_bundle_receipts(packet)

    result = replay_gallery.build_replay_gallery(
        packet, tmp_path / "output" / "unhashed-inputs-gallery", render=False, video=False
    )

    candidate = result["candidates"][0]
    assert candidate["replay_input_status"] == "unknown"
    assert candidate["input_digest_status"] == "unknown"
    assert candidate["missing_replay_inputs"] == []


@pytest.mark.parametrize(
    ("file_name", "mutate", "message"),
    [
        (
            "summary.json",
            lambda payload: payload["pilot_budget"].update(planned=999),
            "summary pilot budget does not match candidate ledger",
        ),
        (
            "summary.json",
            lambda payload: payload["pilot_budget"].update(unknown_or_scoreless=999),
            "summary failed/invalid counts do not match candidate rows",
        ),
        (
            "summary.json",
            lambda payload: payload["runs"][0].update(planned_evaluations=999),
            "summary run budget or outcome counts conflict",
        ),
        (
            "summary.json",
            lambda payload: payload["runs"][0].update(failed_evaluations=16),
            "summary run budget or outcome counts conflict",
        ),
    ],
)
def test_compact_packet_rejects_inconsistent_planned_and_run_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_name: str,
    mutate: Any,
    message: str,
) -> None:
    packet = _write_packet(tmp_path)
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: tmp_path)
    path = packet / file_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(path, payload)
    _refresh_bundle_receipts(packet)

    with pytest.raises(ValueError, match=message):
        replay_gallery.build_replay_gallery(
            packet, tmp_path / "output" / "inconsistent-accounting", render=False, video=False
        )


def test_compact_packet_binds_summary_budget_to_source_manifest_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _write_packet(tmp_path)
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: tmp_path)
    manifest_path = packet / "source_manifests" / "random_seed_17.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["budget"] = 999
    _write_json(manifest_path, manifest)
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    csv_path = packet / "candidate_evaluations.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["source_manifest_sha256"] = manifest_digest
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_bundle_receipts(packet)

    with pytest.raises(ValueError, match="summary pilot budget does not match candidate ledger"):
        replay_gallery.build_replay_gallery(
            packet, tmp_path / "output" / "manifest-budget", render=False, video=False
        )


@pytest.mark.parametrize("field", ["failed", "invalid", "missing", "duplicate"])
def test_compact_packet_rejects_inconsistent_convergence_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    packet = _write_packet(tmp_path)
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: tmp_path)
    path = packet / "convergence_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["aggregates"][0]["candidate_accounting"][field] = 999
    _write_json(path, payload)
    _refresh_bundle_receipts(packet)

    with pytest.raises(ValueError, match=f"convergence aggregate random {field} conflicts"):
        replay_gallery.build_replay_gallery(
            packet, tmp_path / "output" / f"bad-convergence-{field}", render=False, video=False
        )


def test_compact_packet_rejects_duplicate_convergence_sampler_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _write_packet(tmp_path)
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: tmp_path)
    path = packet / "convergence_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["aggregates"].append(dict(payload["aggregates"][0]))
    _write_json(path, payload)
    _refresh_bundle_receipts(packet)

    with pytest.raises(ValueError, match="missing or duplicated sampler identity"):
        replay_gallery.build_replay_gallery(
            packet, tmp_path / "output" / "duplicate-sampler", render=False, video=False
        )


def test_compact_packet_checks_consumed_bytes_against_bundle_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _write_packet(tmp_path)
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: tmp_path)
    with (packet / "candidate_evaluations.csv").open(encoding="utf-8") as handle:
        content = handle.read()
    (packet / "candidate_evaluations.csv").write_text(content + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bundle artifact digest or size conflicts"):
        replay_gallery.build_replay_gallery(
            packet, tmp_path / "output" / "tampered-bundle", render=False, video=False
        )


def test_compact_packet_requires_bundle_checksum_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _write_packet(tmp_path)
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: tmp_path)
    (packet.parent / "checksums.sha256").unlink()

    with pytest.raises(ValueError, match="requires sibling evidence_bundle_manifest.json"):
        replay_gallery.build_replay_gallery(
            packet, tmp_path / "output" / "missing-bundle-anchor", render=False, video=False
        )
