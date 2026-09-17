"""BA-01 campaign inventory, cache, and offline component tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.audit_contracts import SourceRef, record_to_dict
from robot_sf.analysis_workbench.audit_scan import (
    AUDIT_REGISTRY_FILENAME,
    AUDIT_REPORT_FILENAME,
    AuditScanError,
    cache_is_current,
    main,
    run,
    scan_campaign,
)
from robot_sf.analysis_workbench.audit_store import AuditStore
from robot_sf.analysis_workbench.review_contracts import ComponentRequest

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "audit_campaign_v1"
    / "campaign.json"
)


def _frame(index: int, x: float = 0.0) -> dict[str, object]:
    return {
        "time_s": float(index),
        "robot": {"position": [x, 0.0], "heading": 0.0},
        "planner": {"selected_action": {"linear_velocity": 0.2, "angular_velocity": 0.0}},
    }


def _episode(episode_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "episode_id": episode_id,
        "execution_id": f"execution-{episode_id}",
        "planner_id": "planner-a",
        "scenario_id": "scenario-a",
        "seed": 1,
        "config": {"config_id": "config-a"},
        "metrics": {"clearance_m": 1.0, "speed_m_s": 0.2},
        "outcome": {"label": "success"},
        "provenance": {
            "campaign_id": "campaign-test",
            "source_commit": "a" * 40,
            "config_identity": "config-a",
        },
        "trace": {"frames": [_frame(0), _frame(1, 1.0)]},
    }
    row.update(overrides)
    return row


def _campaign(*episodes: object, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "campaign_id": "campaign-test",
        "config_identity": "config-a",
        "schema_version": "campaign-result.v1",
        "source_commit": "a" * 40,
        "execution_status": "native",
        "episodes": list(episodes),
    }
    payload.update(overrides)
    return payload


def test_fixture_accounts_each_expected_episode_once() -> None:
    report = scan_campaign(FIXTURE, root=FIXTURE.parents[4])
    statuses = {item.episode_id: item.status for item in report.inventory}
    assert statuses == {
        "fixture-duplicate": "duplicate",
        "fixture-invalid": "invalid",
        "fixture-missing": "missing",
        "fixture-readable": "readable",
        "fixture-unsupported": "unsupported",
    }
    coverage = report.counts["coverage"]
    assert coverage == {
        "expected": 5,
        "indexed": 4,
        "readable": 1,
        "missing": 1,
        "duplicate": 1,
        "invalid": 1,
        "unsupported": 1,
        "unexpected_observed": 0,
        "observed_rows": 5,
    }
    detectors = report.counts["detectors"]
    assert detectors["scheduled"] == len(report.inventory) * len(report.detector_registry)
    assert detectors["evaluable"] == detectors["flagged"] + detectors["clear"]
    assert detectors["unavailable"] > 0
    assert detectors["error"] == 0
    assert detectors["candidate_signals"] == detectors["flagged"]
    assert detectors["confirmed_findings"] == 0
    assert report.counts["review"]["denominator"] == 1
    assert report.counts["diagnostic"]["executed"] == 0
    assert all(item.episode_id for item in report.episode_refs)


def test_scan_is_logically_deterministic_and_cache_tracks_config_and_registry() -> None:
    payload = _campaign(_episode("episode"), expected_episode_ids=["episode"])
    first = scan_campaign(payload, config={"tail_steps": 2})
    second = scan_campaign(payload, config={"tail_steps": 2})
    assert first.to_dict() == second.to_dict()
    assert first.report_digest == second.report_digest
    source_digest = first.provenance["source"]["sha256_observed"]
    assert cache_is_current(first, source_digest=source_digest, config={"tail_steps": 2})
    assert cache_is_current(first.to_dict(), source_digest=source_digest, config={"tail_steps": 2})
    assert not cache_is_current(first, source_digest=source_digest, config={"tail_steps": 3})
    changed_registry = type(first.detector_registry)(
        version="audit-detector-registry.changed",
        detectors=first.detector_registry.detectors,
    )
    assert not cache_is_current(
        first,
        source_digest=source_digest,
        config={"tail_steps": 2},
        registry=changed_registry,
    )


def test_cache_matches_serialized_selective_detector_report() -> None:
    payload = _campaign(_episode("episode"), expected_episode_ids=["episode"])
    report = scan_campaign(payload, detector_ids=["goal_geometry"], include_advisory=False)
    source_digest = report.provenance["source"]["sha256_observed"]
    assert cache_is_current(report, source_digest=source_digest)
    assert cache_is_current(report.to_dict(), source_digest=source_digest)


def test_cache_uses_effective_embedded_source_config() -> None:
    payload = _campaign(
        _episode("episode"),
        config={"tail_steps": 2},
        expected_episode_ids=["episode"],
    )
    report = scan_campaign(payload)
    assert cache_is_current(report, source_digest=report.audit.source_digest)
    assert not cache_is_current(
        report,
        source_digest=report.audit.source_digest,
        config={"tail_steps": 3},
    )


def test_malformed_expected_episode_declaration_fails_closed() -> None:
    with pytest.raises(AuditScanError, match="expected episode IDs"):
        scan_campaign(_campaign(_episode("episode"), expected_episode_ids=[1]))


def test_inventory_handles_unexpected_and_malformed_anonymous_rows_once() -> None:
    payload = _campaign(
        _episode("expected"),
        {"metrics": {"clearance_m": 1.0}},
        expected_episode_ids=["expected", "missing"],
    )
    report = scan_campaign(payload)
    assert [item.episode_id for item in report.inventory].count("invalid-line-2") == 1
    assert report.counts["coverage"]["unexpected_observed"] == 1
    assert report.counts["coverage"]["expected"] == 2
    assert report.counts["coverage"]["missing"] == 1
    assert report.counts["coverage"]["invalid"] == 0


def test_inventory_rejects_malformed_identity_without_coercion() -> None:
    report = scan_campaign(
        _campaign(
            _episode("malformed", planner_id={"name": "planner-a"}),
            expected_episode_ids=["malformed"],
        )
    )
    assert report.inventory[0].status == "invalid"
    assert "planner_id" in report.inventory[0].reason


def test_direct_jsonl_preserves_bad_lines_for_inventory_accounting(tmp_path: Path) -> None:
    source = tmp_path / "episodes.jsonl"
    source.write_text(
        json.dumps(_episode("good")) + "\n" + "{not-json}\n" + json.dumps(_episode("good")) + "\n",
        encoding="utf-8",
    )
    report = scan_campaign(
        source,
        root=tmp_path,
        config={"expected_episode_ids": ["good", "missing"]},
    )
    assert [(item.episode_id, item.status) for item in report.inventory] == [
        ("good", "duplicate"),
        ("invalid-line-2", "invalid"),
        ("missing", "missing"),
    ]
    assert report.counts["coverage"]["observed_rows"] == 3
    assert report.counts["coverage"]["unexpected_observed"] == 1


def test_campaign_store_directory_is_bounded_and_unsupported_payload_is_visible(
    tmp_path: Path,
) -> None:
    store = tmp_path / "campaign-store"
    store.mkdir()
    (store / "manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": "campaign-test",
                "expected_episode_ids": ["good", "missing"],
                "schema_version": "campaign-result-store.v2",
            }
        ),
        encoding="utf-8",
    )
    (store / "episodes.jsonl").write_text(json.dumps(_episode("good")) + "\n", encoding="utf-8")
    report = scan_campaign(store, root=tmp_path)
    assert {item.episode_id: item.status for item in report.inventory} == {
        "good": "readable",
        "missing": "missing",
    }

    unsupported = tmp_path / "unsupported-store"
    unsupported.mkdir()
    (unsupported / "manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": "campaign-test",
                "expected_episode_ids": ["good"],
                "schema_version": "campaign-result-store.v2",
            }
        ),
        encoding="utf-8",
    )
    (unsupported / "episodes.parquet").write_bytes(b"not-a-supported-reader")
    unsupported_report = scan_campaign(unsupported, root=tmp_path)
    assert unsupported_report.inventory[0].status == "unsupported"
    assert unsupported_report.provenance["source"]["execution_status"] == "unsupported"


def test_unknown_file_format_is_indexed_as_unsupported(tmp_path: Path) -> None:
    source = tmp_path / "episodes.parquet"
    source.write_bytes(b"not-a-supported-reader")
    report = scan_campaign(
        source,
        root=tmp_path,
        config={"expected_episode_ids": ["episode"]},
    )
    assert [(item.episode_id, item.status) for item in report.inventory] == [
        ("episode", "unsupported")
    ]
    assert report.provenance["source"]["format"] == "unsupported"


def test_source_digest_binding_and_admitted_root_reject_tampering(tmp_path: Path) -> None:
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps(_campaign(_episode("episode"))), encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    good_ref = SourceRef(
        artifact_id="campaign",
        uri="campaign.json",
        format="campaign-result",
        schema="campaign-result.v1",
        sha256=digest,
    )
    report = scan_campaign(source, root=tmp_path, source_ref=good_ref)
    assert report.provenance["source"]["integrity_status"] == "digest_verified"
    bad_ref = SourceRef(
        artifact_id="campaign",
        uri="campaign.json",
        format="campaign-result",
        schema="campaign-result.v1",
        sha256="b" * 64,
    )
    with pytest.raises(AuditScanError, match="digest"):
        scan_campaign(source, root=tmp_path, source_ref=bad_ref)
    outside = tmp_path.parent / "outside-campaign.json"
    outside.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(AuditScanError, match="escapes"):
        scan_campaign(outside, root=tmp_path)


def test_source_ref_mapping_rejects_non_string_fields() -> None:
    with pytest.raises(AuditScanError, match="must be strings"):
        scan_campaign(
            _campaign(_episode("episode")),
            source_ref={
                "artifact_id": "campaign",
                "uri": "campaign.json",
                "format": "campaign-result",
                "sha256": 123,
            },
        )


def test_missing_capabilities_generate_selective_enrichment_without_execution() -> None:
    report = scan_campaign(
        _campaign(
            {"episode_id": "sparse", "outcome": {"label": "timeout"}},
            expected_episode_ids=["sparse"],
        ),
        detector_ids=["goal_adjacent_timeout", "actuator_mismatch", "telemetry_integrity"],
    )
    assert report.counts["diagnostic"]["executed"] == 0
    assert report.counts["diagnostic"]["requested"] == len(report.enrichment_requests)
    assert report.enrichment_requests
    assert {item["execution"] for item in report.enrichment_requests} == {"not_requested_here"}
    assert {item["capability"] for item in report.enrichment_requests} >= {"trace", "commands"}
    assert report.provenance["no_simulation_execution"] is True
    assert report.provenance["no_network_or_shell"] is True


def test_store_commit_reuses_ba03_transaction_boundary(tmp_path: Path) -> None:
    payload = _campaign(_episode("episode"), expected_episode_ids=["episode"])
    with AuditStore(tmp_path / "audit-store") as store:
        report = scan_campaign(payload, store=store)
        records = store.list_records()
        assert len(records) == 1 + len(report.signals)
        # Repeating the exact scan operation is idempotent in BA-03's store.
        scan_campaign(payload, store=store)
        assert len(store.list_records()) == len(records)


def test_component_run_writes_report_and_registry(tmp_path: Path) -> None:
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps(_campaign(_episode("episode"))), encoding="utf-8")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    request = ComponentRequest(
        request_id="request-1",
        component_id="ba01-audit-scan",
        sources=(
            SourceRef(
                artifact_id="campaign",
                uri="campaign.json",
                format="campaign-result",
                schema="campaign-result.v1",
                sha256=source_digest,
            ),
        ),
        output_directory="audit-output",
        config={"expected_episode_ids": ["episode"]},
    )
    result = run(request, base=tmp_path)
    assert result.status == "complete"
    assert (tmp_path / "audit-output" / AUDIT_REPORT_FILENAME).is_file()
    assert (tmp_path / "audit-output" / AUDIT_REGISTRY_FILENAME).is_file()
    report_payload = json.loads(
        (tmp_path / "audit-output" / AUDIT_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    assert report_payload["schema_version"] == "benchmark-audit.v1"
    assert record_to_dict(scan_campaign(_campaign(_episode("episode"))).audit)


def test_component_request_rejects_unsupported_required_capability(tmp_path: Path) -> None:
    request = ComponentRequest(
        request_id="request-capability",
        component_id="ba01-audit-scan",
        sources=(),
        output_directory="audit-output",
        required_capabilities=("simulator",),
    )
    result = run(request, base=tmp_path)
    assert result.status == "unavailable"
    assert "missing_required_capabilities" in result.reason


def test_cli_emits_machine_readable_report_and_refuses_existing_output(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps(_campaign(_episode("episode"))), encoding="utf-8")
    output = tmp_path / "report.json"
    assert (
        main(["--input", "campaign.json", "--base", str(tmp_path), "--output", "report.json"]) == 0
    )
    stdout = capsys.readouterr().out
    assert json.loads(stdout)["schema_version"] == "benchmark-audit.v1"
    assert output.is_file()
    assert (
        main(["--input", "campaign.json", "--base", str(tmp_path), "--output", "report.json"]) == 2
    )
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_scan_rejects_unsupported_source_schema_without_dropping_rows() -> None:
    report = scan_campaign(
        _campaign(_episode("episode")),
        source_ref={
            "artifact_id": "campaign",
            "uri": "campaign",
            "format": "campaign-result",
            "schema": "campaign-result.v99",
        },
    )
    assert report.inventory[0].status == "unsupported"
    assert report.counts["coverage"]["unsupported"] == 1
    assert all(signal.status == "unavailable" for signal in report.signals)
