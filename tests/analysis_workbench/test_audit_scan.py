"""BA-01 campaign inventory, cache, and offline component tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from robot_sf.analysis_workbench import audit_scan as audit_scan_module
from robot_sf.analysis_workbench import review_context
from robot_sf.analysis_workbench.audit_contracts import SourceRef, record_to_dict
from robot_sf.analysis_workbench.audit_detectors import detect
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
from robot_sf.benchmark import case_workbench
from robot_sf.benchmark.analysis_trace import build_analysis_trace
from robot_sf.benchmark.parquet_export import export_campaign_result_store_v2

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


def test_scan_marks_only_derived_config_digest_as_adapter_default() -> None:
    derived = scan_campaign(_campaign(_episode("derived")))
    derived_row = derived.inventory[0].row
    assert derived_row is not None
    assert (
        derived_row["_audit_scan_identity_defaults"]["config_digest"]
        == derived_row["config_digest"]
    )

    claimed_digest = "b" * 64
    claimed = scan_campaign(_campaign(_episode("claimed", config_digest=claimed_digest)))
    claimed_row = claimed.inventory[0].row
    assert claimed_row is not None
    assert claimed_row["config_digest"] == claimed_digest
    assert "config_digest" not in claimed_row.get("_audit_scan_identity_defaults", {})


def test_jsonl_admits_distinct_row_config_hashes_without_inventing_source_identity(
    tmp_path: Path,
) -> None:
    """A multi-run JSONL keeps each row hash instead of collapsing it at source scope."""

    source = tmp_path / "episodes.jsonl"
    rows = [
        {
            "episode_id": "run-a",
            "scenario_id": "scenario-a",
            "algo": "goal",
            "seed": 1,
            "git_hash": "a" * 40,
            "config_hash": "1" * 16,
            "metrics": {"success": False, "collisions": 1},
        },
        {
            "episode_id": "run-b",
            "scenario_id": "scenario-a",
            "algo": "goal",
            "seed": 2,
            "git_hash": "a" * 40,
            "config_hash": "2" * 16,
            "metrics": {"success": True, "collisions": 0},
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = scan_campaign(source, root=tmp_path)

    assert [item.status for item in report.inventory] == ["readable", "readable"]
    assert report.audit.source is not None
    assert report.audit.source.config_identity == ""
    assert [ref.config_digest for ref in report.episode_refs] == [
        hashlib.sha256(("1" * 16).encode()).hexdigest(),
        hashlib.sha256(("2" * 16).encode()).hexdigest(),
    ]

    pinned = scan_campaign(
        source,
        root=tmp_path,
        source_ref=SourceRef(
            artifact_id=source.name,
            uri=source.name,
            format="episode-jsonl",
            schema="episode-jsonl.v1",
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_commit="a" * 40,
            config_identity="1" * 16,
        ),
    )
    assert [(item.episode_id, item.status) for item in pinned.inventory] == [
        ("run-a", "readable"),
        ("run-b", "invalid"),
    ]
    assert "config_identity conflicts" in pinned.inventory[1].reason


@pytest.mark.parametrize(
    "container", ["config", "result_provenance", "cell_context", "algorithm_metadata"]
)
def test_scan_does_not_mark_nested_row_campaign_claim_as_default(container: str) -> None:
    row = _episode("nested-campaign")
    row["provenance"].pop("campaign_id")
    if container == "config":
        row["config"]["campaign"] = "campaign-test"
    else:
        row[container] = {"campaign": "campaign-test"}
    report = scan_campaign(_campaign(row))
    admitted = report.inventory[0].row
    assert admitted is not None
    assert admitted["campaign_id"] == "campaign-test"
    assert "campaign_id" not in admitted.get("_audit_scan_identity_defaults", {})


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


def test_canonical_v2_parquet_population_and_analysis_trace_are_admitted(tmp_path: Path) -> None:
    """The canonical case-workbench loader supplies rows when no IDs are passed."""

    pytest.importorskip("pyarrow")
    trace = build_analysis_trace(
        steps=[
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {
                    "position": [0.2, 0.0],
                    "heading": 0.0,
                    "velocity": [1.0, 0.0],
                },
                "pedestrians": [],
                "controls": {
                    "requested": {"linear_m_s": 0.2, "turn_rate_rad_s": 0.1},
                    "applied": {"linear_m_s": 0.2, "turn_rate_rad_s": 0.1},
                },
            }
        ],
        initial_robot_position=[0.0, 0.0],
        initial_robot_heading=0.0,
        initial_pedestrians=[],
        initial_robot_velocity=[0.0, 0.0],
        initial_pedestrian_velocities=[],
        initial_pedestrian_ids=[],
        dt=0.1,
        horizon=1,
        robot_radius_m=0.25,
        pedestrian_radius_m=0.25,
        scenario={
            "id": "canonical-scenario",
            "seed": 17,
            "map_file": "maps/svg_maps/classic_bottleneck.svg",
        },
        planner="canonical-planner",
        planner_commit="a" * 7,
        config_hash="canonical-config",
        git_hash="b" * 7,
        termination_reason="success",
        safety_events=[],
    )
    source = tmp_path / "episodes.jsonl"
    source.write_text(
        json.dumps(
            {
                "episode_id": " canonical-episode ",
                "scenario_id": "canonical-scenario",
                "seed": 17,
                "algo": "canonical-planner",
                "status": "success",
                "row_status": "native",
                "outcome": {"route_complete": True, "collision_event": False},
                "algorithm_metadata": {"analysis_trace": trace},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = export_campaign_result_store_v2(source, tmp_path / "store", overwrite=True)
    report = scan_campaign(result.output_dir, root=tmp_path)
    assert [(item.episode_id, item.status) for item in report.inventory] == [
        ("canonical-episode", "readable")
    ]
    assert report.counts["coverage"]["expected"] == 1
    retained = report.inventory[0].row
    assert retained is not None
    canonical_trace = retained["algorithm_metadata"]["analysis_trace"]
    assert canonical_trace["steps"][1]["controls"]["requested"]["linear_m_s"] == 0.2
    actuator = next(
        signal for signal in report.signals if signal.detector_id == "actuator_mismatch"
    )
    assert actuator.status == "clear"


@pytest.mark.parametrize(
    "field",
    [
        "row_status",
        "status",
        "availability_status",
        "readiness_status",
        "campaign_execution_status",
    ],
)
def test_canonical_status_surfaces_never_advertise_degraded_as_readable(field: str) -> None:
    report = scan_campaign(
        _campaign(_episode("blocked", **{field: "degraded"}), expected_episode_ids=["blocked"])
    )
    assert report.inventory[0].status == "unsupported"
    assert report.counts["coverage"]["readable"] == 0


def test_canonical_terminal_status_does_not_override_native_row_status() -> None:
    report = scan_campaign(
        _campaign(
            _episode("collision", row_status="native", status="collision"),
            expected_episode_ids=["collision"],
        )
    )
    assert report.inventory[0].status == "readable"


def test_nested_fallback_counter_and_malformed_status_fail_closed() -> None:
    fallback = scan_campaign(
        _campaign(
            _episode(
                "fallback",
                algorithm_metadata={"status": "native", "fallback_counters": {"steps": 1}},
            ),
            expected_episode_ids=["fallback"],
        )
    )
    assert fallback.inventory[0].status == "unsupported"
    metric_fallback = scan_campaign(
        _campaign(
            _episode("metric-fallback", metrics={"fallback_steps": 1}),
            expected_episode_ids=["metric-fallback"],
        )
    )
    assert metric_fallback.inventory[0].status == "unsupported"
    malformed = scan_campaign(
        _campaign(_episode("malformed", availability_status=3), expected_episode_ids=["malformed"])
    )
    assert malformed.inventory[0].status == "invalid"


def test_prose_not_available_status_is_not_readable() -> None:
    report = scan_campaign(
        _campaign(
            _episode("not-available", row_status="not available"),
            expected_episode_ids=["not-available"],
        )
    )
    assert report.inventory[0].status == "unsupported"
    signal = report.signals[0]
    assert signal.status == "unavailable"
    direct = detect("stuck_no_progress", {"episode_id": "direct", "status": "not available"})
    assert direct.status == "unavailable"


def test_progress_contract_error_is_preserved_when_waiting() -> None:
    signal = detect(
        "stuck_no_progress",
        {"episode_id": "waiting", "expected_waiting": True, "progress_m": "malformed"},
    )
    assert signal.status == "error"
    assert signal.reason_code == "progress_telemetry_malformed"


def test_row_source_digest_mismatch_is_invalid_and_not_exposed_by_signal() -> None:
    report = scan_campaign(
        _campaign(
            _episode("untrusted", source_digest="0" * 64),
            expected_episode_ids=["untrusted"],
        )
    )
    assert report.inventory[0].status == "invalid"
    assert "source_digest" in report.inventory[0].reason
    signal_payloads = [
        json.dumps(record_to_dict(signal), sort_keys=True) for signal in report.signals
    ]
    assert all("0" * 64 not in payload for payload in signal_payloads)


def test_zero_inventory_unsupported_store_is_not_complete(tmp_path: Path) -> None:
    store = tmp_path / "unsupported-empty"
    store.mkdir()
    (store / "manifest.json").write_text(
        json.dumps({"schema_version": "campaign-result-store.v2"}), encoding="utf-8"
    )
    (store / "episodes.parquet").write_bytes(b"invalid-parquet")
    result = run(
        ComponentRequest(
            request_id="unsupported-empty",
            component_id="ba01-audit-scan",
            sources=(
                SourceRef(
                    artifact_id="unsupported-empty",
                    uri="unsupported-empty",
                    format="campaign-result-store",
                    schema="campaign-result-store.v2",
                ),
            ),
            output_directory="audit-output",
        ),
        base=tmp_path,
    )
    assert result.status == "partial"
    assert "source_execution" in result.reason


def test_canonical_parquet_rows_are_bound_to_the_validated_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replacement after validation cannot change admitted owner rows."""

    pytest.importorskip("pyarrow")
    original_source = tmp_path / "original.jsonl"
    canonical_provenance = {
        "campaign_id": "campaign",
        "source_commit": "a" * 40,
        "config_identity": "config-a",
    }
    original_source.write_text(
        json.dumps(_episode("original", provenance=canonical_provenance)) + "\n", encoding="utf-8"
    )
    original_store = export_campaign_result_store_v2(
        original_source, tmp_path / "original-store", overwrite=True
    ).output_dir
    external_source = tmp_path / "external.jsonl"
    external_source.write_text(
        json.dumps(_episode("external", provenance=canonical_provenance)) + "\n", encoding="utf-8"
    )
    external_store = export_campaign_result_store_v2(
        external_source, tmp_path / "external-store", overwrite=True
    ).output_dir
    external_parquet = (external_store / "episodes.parquet").read_bytes()
    original_parquet = original_store / "episodes.parquet"
    owner_integrity = case_workbench._v2_integrity_errors

    def swap_after_integrity(path: Path) -> list[str]:
        errors = owner_integrity(path)
        original_parquet.write_bytes(external_parquet)
        return errors

    monkeypatch.setattr(case_workbench, "_v2_integrity_errors", swap_after_integrity)
    report = scan_campaign(original_store, root=tmp_path)
    assert [item.episode_id for item in report.inventory] == ["original"]
    assert all(item.episode_id != "external" for item in report.inventory)
    assert report.provenance["source"]["sha256_observed"] == report.audit.source_digest


def test_directory_special_file_is_rejected_before_digest(tmp_path: Path) -> None:
    store = tmp_path / "special-store"
    store.mkdir()
    (store / "manifest.json").write_text(
        json.dumps({"schema_version": "campaign-result-store.v2"}), encoding="utf-8"
    )
    os.mkfifo(store / "unexpected.fifo")
    with pytest.raises(AuditScanError, match="special file"):
        scan_campaign(store, root=tmp_path)


def test_nonfinite_campaign_metadata_is_rejected_but_bad_jsonl_row_is_indexed(
    tmp_path: Path,
) -> None:
    with pytest.raises(AuditScanError, match="non-finite"):
        scan_campaign(_campaign(_episode("metadata"), metadata={"value": float("nan")}))
    source = tmp_path / "episodes.jsonl"
    source.write_text(
        json.dumps({"episode_id": "good", "metrics": {"value": 1.0}})
        + "\n"
        + '{"episode_id":"bad","metrics":{"value":NaN}}\n',
        encoding="utf-8",
    )
    report = scan_campaign(source, root=tmp_path, config={"expected_episode_ids": ["good", "bad"]})
    assert {item.episode_id: item.status for item in report.inventory} == {
        "good": "readable",
        "bad": "invalid",
    }
    assert report.counts["coverage"]["invalid"] == 1


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


def test_source_ref_schema_and_identity_bindings_fail_closed() -> None:
    payload = _campaign(_episode("episode"), expected_episode_ids=["episode"])
    with pytest.raises(AuditScanError, match="schema is required"):
        scan_campaign(
            payload,
            source_ref={
                "artifact_id": "campaign",
                "uri": "campaign",
                "format": "campaign-result",
            },
        )
    with pytest.raises(AuditScanError, match="source_commit"):
        scan_campaign(
            payload,
            source_ref={
                "artifact_id": "campaign",
                "uri": "campaign",
                "format": "campaign-result",
                "schema": "campaign-result.v1",
                "source_commit": "b" * 40,
            },
        )
    conflicting = _campaign(
        _episode("episode"),
        git_hash="b" * 40,
        expected_episode_ids=["episode"],
    )
    with pytest.raises(AuditScanError, match="conflicting source_commit"):
        scan_campaign(conflicting)


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


def test_component_publication_rejects_output_directory_symlink_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps(_campaign(_episode("episode"))), encoding="utf-8")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    outside = tmp_path.parent / "ba01-output-race-target"
    outside.mkdir()
    original_reserve = audit_scan_module._reserve_output_directory

    def swap_before_reservation(output_directory: str, base: Path, **kwargs: object):
        (base / "audit-output").symlink_to(outside, target_is_directory=True)
        return original_reserve(output_directory, base, **kwargs)

    monkeypatch.setattr(audit_scan_module, "_reserve_output_directory", swap_before_reservation)
    result = run(
        ComponentRequest(
            request_id="output-race",
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
        ),
        base=tmp_path,
    )
    assert result.status == "failed"
    assert not (outside / AUDIT_REPORT_FILENAME).exists()
    assert not (outside / AUDIT_REGISTRY_FILENAME).exists()


def test_component_publication_rejects_temporary_inode_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replaced temporary pathname cannot become a published report."""
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps(_campaign(_episode("episode"))), encoding="utf-8")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    outside = tmp_path.parent / "ba01-temporary-race-target"
    outside.write_text("outside sentinel", encoding="utf-8")
    original_publish = review_context._publish_output

    def swap_temporary(
        directory_fd: int,
        temporary_name: str,
        final_name: str,
        output_directory,
        published_names: set[str],
        **kwargs: object,
    ) -> None:
        os.unlink(temporary_name, dir_fd=directory_fd)
        os.symlink(outside, temporary_name, dir_fd=directory_fd)
        original_publish(
            directory_fd,
            temporary_name,
            final_name,
            output_directory,
            published_names,
            **kwargs,
        )

    monkeypatch.setattr(review_context, "_publish_output", swap_temporary)
    result = run(
        ComponentRequest(
            request_id="temporary-race",
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
        ),
        base=tmp_path,
    )
    assert result.status == "failed"
    assert outside.read_text(encoding="utf-8") == "outside sentinel"
    assert not (tmp_path / "audit-output" / AUDIT_REPORT_FILENAME).exists()
    outside.unlink()


def test_component_publication_rejects_parent_move_after_fd_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API must not create output through a parent moved outside the root."""
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps(_campaign(_episode("episode"))), encoding="utf-8")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    outside = tmp_path.parent / "ba01-api-parent-race-target"
    original_assert = review_context._assert_output_parent_current
    moved = False

    def move_after_parent_open(guard) -> None:
        nonlocal moved
        original_assert(guard)
        if not moved:
            moved = True
            parent = guard.root / guard.parts[0]
            parent.rename(outside)
            parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(review_context, "_assert_output_parent_current", move_after_parent_open)
    result = run(
        ComponentRequest(
            request_id="api-parent-race",
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
            output_directory="nested/audit-output",
            config={"expected_episode_ids": ["episode"]},
        ),
        base=tmp_path,
    )
    assert result.status == "failed"
    assert not (outside / "audit-output" / AUDIT_REPORT_FILENAME).exists()
    assert not (outside / "audit-output" / AUDIT_REGISTRY_FILENAME).exists()
    assert (tmp_path / "nested").is_symlink()
    (tmp_path / "nested").unlink()
    outside.rmdir()


@pytest.mark.parametrize("entrypoint", ("api", "cli"))
def test_output_root_replacement_after_path_admission_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entrypoint: str
) -> None:
    """A replacement root inode cannot become the publication authority."""
    root = tmp_path / "root"
    root.mkdir()
    moved = tmp_path / "moved-root"
    source = root / "campaign.json"
    source.write_text(json.dumps(_campaign(_episode("episode"))), encoding="utf-8")
    original_safe_path = audit_scan_module._safe_path_for_output
    replaced = False

    def replace_root(value: str, *, root: Path):
        nonlocal replaced
        target = original_safe_path(value, root=root)
        if not replaced:
            replaced = True
            root.rename(moved)
            root.mkdir()
        return target

    monkeypatch.setattr(audit_scan_module, "_safe_path_for_output", replace_root)
    if entrypoint == "api":
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        result = run(
            ComponentRequest(
                request_id="root-replacement",
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
            ),
            base=root,
        )
        assert result.status == "failed"
    else:
        with pytest.raises(review_context.ReviewContractsValidationError):
            audit_scan_module._write_cli_output("report.json", "payload\n", root=root)
    assert not (root / "audit-output" / AUDIT_REPORT_FILENAME).exists()
    assert not (root / "report.json").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO support is platform-specific")
@pytest.mark.parametrize("entrypoint", ("api", "cli"))
@pytest.mark.parametrize("replacement", ("symlink", "fifo", "external"))
def test_temporary_replacement_never_deletes_external_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    replacement: str,
) -> None:
    """API and CLI cleanup must leave every replacement inode untouched."""
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps(_campaign(_episode("episode"))), encoding="utf-8")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    victim = tmp_path.parent / f"ba01-{entrypoint}-{replacement}-victim"
    victim_fd = -1
    if replacement in {"symlink", "external"}:
        victim.write_text("outside sentinel", encoding="utf-8")
        if replacement == "external":
            victim_fd = os.open(victim, os.O_RDONLY)

    original_publish = (
        review_context._publish_output if entrypoint == "api" else audit_scan_module._publish_output
    )

    def swap_temporary(
        directory_fd: int,
        temporary_name: str,
        final_name: str,
        output_directory,
        published_names: set[str],
        **kwargs: object,
    ) -> None:
        os.unlink(temporary_name, dir_fd=directory_fd)
        if replacement == "symlink":
            os.symlink(victim, temporary_name, dir_fd=directory_fd)
        elif replacement == "fifo":
            os.mkfifo(temporary_name, 0o600, dir_fd=directory_fd)
        else:
            os.rename(victim, temporary_name, dst_dir_fd=directory_fd)
        original_publish(
            directory_fd,
            temporary_name,
            final_name,
            output_directory,
            published_names,
            **kwargs,
        )

    if entrypoint == "api":
        monkeypatch.setattr(review_context, "_publish_output", swap_temporary)
        result = run(
            ComponentRequest(
                request_id=f"temporary-{entrypoint}-{replacement}",
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
            ),
            base=tmp_path,
        )
        assert result.status == "failed"
        report = tmp_path / "audit-output" / AUDIT_REPORT_FILENAME
    else:
        monkeypatch.setattr(audit_scan_module, "_publish_output", swap_temporary)
        with pytest.raises(review_context.ReviewContractsValidationError):
            audit_scan_module._write_cli_output("report.json", "payload\n", root=tmp_path)
        report = tmp_path / "report.json"

    assert not report.exists()
    if replacement == "external":
        assert os.fstat(victim_fd).st_nlink == 1
    if victim.exists() or victim.is_symlink():
        victim.unlink()
    if victim_fd >= 0:
        os.close(victim_fd)


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


def test_cli_output_rejects_parent_symlink_race(tmp_path: Path, monkeypatch, capsys) -> None:
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps(_campaign(_episode("episode"))), encoding="utf-8")
    outside = tmp_path.parent / "ba01-cli-output-race-target"
    outside.mkdir()
    original_safe_path = audit_scan_module._safe_path_for_output

    def swap_after_validation(value: str, root: Path):
        target = original_safe_path(value, root=root)
        if value == "nested/report.json":
            (root / "nested").symlink_to(outside, target_is_directory=True)
        return target

    monkeypatch.setattr(audit_scan_module, "_safe_path_for_output", swap_after_validation)
    code = main(
        [
            "--input",
            "campaign.json",
            "--base",
            str(tmp_path),
            "--output",
            "nested/report.json",
        ]
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert not (outside / "report.json").exists()


def test_cli_output_rejects_temporary_inode_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A replaced CLI temporary pathname cannot become the final report."""
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps(_campaign(_episode("episode"))), encoding="utf-8")
    outside = tmp_path.parent / "ba01-cli-temporary-race-target"
    outside.write_text("outside sentinel", encoding="utf-8")
    original_publish = audit_scan_module._publish_output

    def swap_temporary(
        directory_fd: int,
        temporary_name: str,
        final_name: str,
        output_directory,
        published_names: set[str],
        **kwargs: object,
    ) -> None:
        os.unlink(temporary_name, dir_fd=directory_fd)
        os.symlink(outside, temporary_name, dir_fd=directory_fd)
        original_publish(
            directory_fd,
            temporary_name,
            final_name,
            output_directory,
            published_names,
            **kwargs,
        )

    monkeypatch.setattr(audit_scan_module, "_publish_output", swap_temporary)
    code = main(
        [
            "--input",
            "campaign.json",
            "--base",
            str(tmp_path),
            "--output",
            "report.json",
        ]
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert outside.read_text(encoding="utf-8") == "outside sentinel"
    assert not (tmp_path / "report.json").exists()
    outside.unlink()


def test_cli_output_rejects_parent_move_then_symlink_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A parent moved outside the root is never used through its retained fd."""
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps(_campaign(_episode("episode"))), encoding="utf-8")
    outside = tmp_path.parent / "ba01-cli-parent-race-target"
    original_open_parent = review_context._open_output_parent

    def move_parent(root: Path, parts: tuple[str, ...]):
        parent_fd = original_open_parent(root, parts)
        parent = root / parts[0]
        parent.rename(outside)
        parent.symlink_to(outside, target_is_directory=True)
        return parent_fd

    monkeypatch.setattr(review_context, "_open_output_parent", move_parent)
    code = main(
        [
            "--input",
            "campaign.json",
            "--base",
            str(tmp_path),
            "--output",
            "nested/report.json",
        ]
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert not (outside / "report.json").exists()
    assert (tmp_path / "nested").is_symlink()
    (tmp_path / "nested").unlink()
    outside.rmdir()


def test_cli_parent_move_after_initial_guard_does_not_open_temp_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent moved after the first guard cannot reach temporary creation."""
    nested = tmp_path / "nested"
    nested.mkdir()
    outside = tmp_path.parent / "ba01-cli-initial-parent-race-target"
    original_assert = review_context._assert_output_parent_current
    original_open_temporary = review_context._open_output_temporary
    calls = 0
    temporary_targets: list[str] = []

    def move_after_initial_guard(guard) -> None:
        nonlocal calls
        original_assert(guard)
        calls += 1
        if calls == 2:
            nested.rename(outside)
            nested.symlink_to(outside, target_is_directory=True)

    def record_temporary_target(directory_fd: int, prefix: str):
        temporary_targets.append(os.readlink(f"/proc/self/fd/{directory_fd}"))
        return original_open_temporary(directory_fd, prefix)

    monkeypatch.setattr(review_context, "_assert_output_parent_current", move_after_initial_guard)
    monkeypatch.setattr(
        audit_scan_module, "_assert_output_parent_current", move_after_initial_guard
    )
    monkeypatch.setattr(review_context, "_open_output_temporary", record_temporary_target)
    monkeypatch.setattr(audit_scan_module, "_open_output_temporary", record_temporary_target)
    with pytest.raises(review_context.ReviewContractsValidationError):
        audit_scan_module._write_cli_output("nested/report.json", "payload\n", root=tmp_path)
    assert temporary_targets == []
    assert not (outside / "report.json").exists()
    assert nested.is_symlink()
    nested.unlink()
    outside.rmdir()


def test_cli_output_rejects_parent_move_after_final_identity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent moved after final identity validation cannot publish successfully."""
    outside = tmp_path.parent / "ba01-cli-final-parent-race-target"
    original_assert = review_context._assert_output_parent_current
    moved = False

    def move_after_final_check(guard) -> None:
        nonlocal moved
        original_assert(guard)
        final = guard.root.joinpath(*guard.parts)
        if not moved and final.is_file():
            moved = True
            parent = guard.root / guard.parts[0]
            parent.rename(outside)
            parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(review_context, "_assert_output_parent_current", move_after_final_check)
    monkeypatch.setattr(audit_scan_module, "_assert_output_parent_current", move_after_final_check)
    with pytest.raises(review_context.ReviewContractsValidationError):
        audit_scan_module._write_cli_output("nested/report.json", "payload\n", root=tmp_path)
    assert not (outside / "report.json").exists()
    assert (tmp_path / "nested").is_symlink()
    (tmp_path / "nested").unlink()
    outside.rmdir()


def test_cli_publication_rejects_parent_move_before_final_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CLI parent moved after temp admission cannot receive a final artifact."""
    nested = tmp_path / "nested"
    nested.mkdir()
    outside = tmp_path.parent / "ba01-cli-before-link-target"
    original_retained = review_context._retained_output_identity
    original_link = review_context.os.link
    link_calls = 0
    moved = False

    def move_after_final_identity(*args: object, **kwargs: object) -> tuple[int, int]:
        nonlocal moved
        identity = original_retained(*args, **kwargs)
        if not moved:
            nested.rename(outside)
            nested.symlink_to(outside, target_is_directory=True)
            moved = True
        return identity

    def record_link(*args: object, **kwargs: object) -> object:
        nonlocal link_calls
        link_calls += 1
        return original_link(*args, **kwargs)

    monkeypatch.setattr(review_context, "_retained_output_identity", move_after_final_identity)
    monkeypatch.setattr(review_context.os, "link", record_link)
    try:
        with pytest.raises(review_context.ReviewContractsValidationError):
            audit_scan_module._write_cli_output("nested/report.json", "payload\n", root=tmp_path)
        assert link_calls == 0
        assert not (outside / "report.json").exists()
        assert not any(outside.iterdir())
    finally:
        output_link = tmp_path / "nested"
        if output_link.is_symlink():
            output_link.unlink()
        outside.rmdir()


def test_cli_config_rejects_symlink_swap_after_path_admission(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps(_campaign(_episode("episode"))), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"tail_steps": 2}), encoding="utf-8")
    outside = tmp_path.parent / "ba01-cli-config-race.json"
    outside.write_text(json.dumps({"tail_steps": 99}), encoding="utf-8")
    original_safe_path = audit_scan_module._safe_path

    def swap_after_admission(value: str | Path, root: Path):
        target = original_safe_path(value, root=root)
        if str(value) == "config.json":
            config.unlink()
            config.symlink_to(outside)
        return target

    monkeypatch.setattr(audit_scan_module, "_safe_path", swap_after_admission)
    code = main(
        [
            "--input",
            "campaign.json",
            "--base",
            str(tmp_path),
            "--config",
            "config.json",
        ]
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_regular_source_parent_symlink_race_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regular-file sources use no-follow descriptor walks for every component."""
    root = tmp_path / "root"
    root.mkdir()
    nested = root / "nested"
    nested.mkdir()
    outside = tmp_path / "moved-source"
    old_payload = _campaign(_episode("old"), expected_episode_ids=["old"])
    new_payload = _campaign(_episode("outside"), expected_episode_ids=["outside"])
    (nested / "campaign.json").write_text(json.dumps(old_payload), encoding="utf-8")
    original_safe_path = audit_scan_module._safe_path
    moved = False

    def swap_after_admission(value: str | Path, *, root: Path):
        nonlocal moved
        target = original_safe_path(value, root=root)
        if not moved:
            moved = True
            (root / "nested").rename(outside)
            (outside / "campaign.json").write_text(json.dumps(new_payload), encoding="utf-8")
            (root / "nested").symlink_to(outside, target_is_directory=True)
        return target

    monkeypatch.setattr(audit_scan_module, "_safe_path", swap_after_admission)
    with pytest.raises(AuditScanError, match="replaced"):
        scan_campaign("nested/campaign.json", root=root)


def test_descriptor_read_rejects_root_move_after_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retained source root FD cannot admit bytes after its directory moves."""
    root = tmp_path / "root"
    root.mkdir()
    source = root / "campaign.json"
    source.write_text("original", encoding="utf-8")
    moved = tmp_path / "moved-source-root"
    original_open = review_context.os.open
    moved_once = False

    def move_after_root_open(path, flags, *args, **kwargs):
        nonlocal moved_once
        directory_fd = kwargs.get("dir_fd")
        if not moved_once and directory_fd is not None:
            try:
                anchored = Path(os.readlink(f"/proc/self/fd/{directory_fd}"))
            except OSError:
                anchored = None
            if anchored == root:
                root.rename(moved)
                root.mkdir()
                (root / "campaign.json").write_text("replacement", encoding="utf-8")
                moved_once = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(review_context.os, "open", move_after_root_open)
    with pytest.raises(review_context.ReviewContractsValidationError, match="ancestry"):
        review_context._read_descriptor_backed_file(
            ("campaign.json",),
            root,
            limit=1024,
            kind="campaign source",
        )
    assert (moved / "campaign.json").read_text(encoding="utf-8") == "original"


def test_cli_config_rejects_root_move_after_descriptor_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """CLI config bytes are rejected if the admitted root moves after open."""
    root = tmp_path / "root"
    root.mkdir()
    source = _campaign(_episode("episode"))
    (root / "campaign.json").write_text(json.dumps(source), encoding="utf-8")
    (root / "config.json").write_text(json.dumps({"tail_steps": 2}), encoding="utf-8")
    moved = tmp_path / "moved-cli-root"
    original_capture = audit_scan_module._capture_admitted_path
    original_open = review_context.os.open
    armed = False
    moved_once = False

    def arm_after_config_capture(*args, **kwargs):
        nonlocal armed
        admission = original_capture(*args, **kwargs)
        if kwargs.get("kind", "").endswith("config.json"):
            armed = True
        return admission

    def move_after_root_open(path, flags, *args, **kwargs):
        nonlocal moved_once
        directory_fd = kwargs.get("dir_fd")
        if armed and not moved_once and directory_fd is not None:
            try:
                anchored = Path(os.readlink(f"/proc/self/fd/{directory_fd}"))
            except OSError:
                anchored = None
            if anchored == root:
                root.rename(moved)
                root.mkdir()
                (root / "config.json").write_text(json.dumps({"tail_steps": 99}), encoding="utf-8")
                moved_once = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(audit_scan_module, "_capture_admitted_path", arm_after_config_capture)
    monkeypatch.setattr(review_context.os, "open", move_after_root_open)
    code = main(
        [
            "--input",
            "campaign.json",
            "--base",
            str(root),
            "--config",
            "config.json",
        ]
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert (moved / "config.json").read_text(encoding="utf-8") == '{"tail_steps": 2}'


def test_cli_scans_exact_admitted_input_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Input config and source bytes remain one immutable CLI admission."""
    source = tmp_path / "campaign.json"
    old_payload = _campaign(
        _episode("old"),
        expected_episode_ids=["old"],
        config={"tail_steps": 2},
    )
    new_payload = _campaign(
        _episode("new"),
        expected_episode_ids=["new"],
        config={"tail_steps": 99},
    )
    old_raw = json.dumps(old_payload).encode("utf-8")
    new_raw = json.dumps(new_payload).encode("utf-8")
    source.write_bytes(old_raw)
    original_snapshot = audit_scan_module._read_cli_json_snapshot
    replaced = False

    def replace_after_snapshot(*args, **kwargs):
        nonlocal replaced
        target, raw, payload = original_snapshot(*args, **kwargs)
        if not replaced:
            replaced = True
            source.write_bytes(new_raw)
        return target, raw, payload

    monkeypatch.setattr(audit_scan_module, "_read_cli_json_snapshot", replace_after_snapshot)
    assert main(["--input", "campaign.json", "--base", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    coverage = report["counts"]["coverage"]
    assert coverage["expected"] == 1
    assert coverage["readable"] == 1
    assert coverage["missing"] == 0
    assert coverage["unexpected_observed"] == 0
    assert report["provenance"]["source"]["sha256_observed"] == hashlib.sha256(old_raw).hexdigest()


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


def test_scan_schema_and_execution_status_helpers_fail_closed(tmp_path: Path) -> None:
    """Schema, status, and source-path adapters retain malformed input explicitly."""
    with pytest.raises(AuditScanError, match="finite"):
        audit_scan_module._finite(True, name="value")
    with pytest.raises(AuditScanError, match="finite"):
        audit_scan_module._finite(float("inf"), name="value")

    recursive: dict[str, object] = {}
    recursive["self"] = recursive
    with pytest.raises(AuditScanError, match="recursive"):
        audit_scan_module._strict_walk(recursive)
    with pytest.raises(AuditScanError, match="non-string"):
        audit_scan_module._strict_walk({1: "value"})
    with pytest.raises(AuditScanError, match="unsupported"):
        audit_scan_module._strict_walk({"value": object()})
    with pytest.raises(AuditScanError, match="non-finite"):
        audit_scan_module._strict_walk({"value": float("nan")})
    with pytest.raises(AuditScanError, match="maximum"):
        audit_scan_module._strict_walk(0, max_depth=-1)
    assert audit_scan_module._contains_nonfinite({"nested": [1.0]}) is False
    assert audit_scan_module._contains_nonfinite({"nested": [float("nan")]}) is True

    assert audit_scan_module._counter_status(False, path="fallback") == (None, None)
    assert audit_scan_module._counter_status(True, path="fallback")[0] == "unsupported"
    assert audit_scan_module._counter_status(-1, path="fallback")[1] == "fallback is malformed"
    assert audit_scan_module._counter_status({"steps": 0}, path="fallback") == (None, None)
    assert audit_scan_module._counter_status({"steps": [1]}, path="fallback")[0] == "unsupported"
    assert audit_scan_module._counter_status({1: 0}, path="fallback")[0] is None
    assert audit_scan_module._counter_status("bad", path="fallback")[1] == "fallback is malformed"

    assert audit_scan_module._execution_status({"status": "collision"}) == ("native", "")
    assert audit_scan_module._execution_status({"status": "future-status"})[0] == "unsupported"
    assert audit_scan_module._execution_status({"row_status": "future-status"})[0] == "invalid"
    assert audit_scan_module._execution_status({"algorithm_metadata": 1})[0] == "invalid"
    assert (
        audit_scan_module._execution_status({"algorithm_metadata": {"analysis_trace": 1}})[0]
        == "invalid"
    )
    assert audit_scan_module._execution_status({"metrics": []})[0] == "invalid"
    assert (
        audit_scan_module._execution_status({"fallback_counters": {"steps": 1}})[0] == "unsupported"
    )
    assert audit_scan_module._execution_status({"fallback_counters": {"steps": -1}})[0] == "invalid"

    with pytest.raises(AuditScanError, match="conflict"):
        audit_scan_module._payload_schema({"schema_version": "one", "schema": "two"})
    with pytest.raises(AuditScanError, match="schema_version is malformed"):
        audit_scan_module._payload_schema({"schema_version": 1})
    observed = "a" * 64
    with pytest.raises(AuditScanError, match="unknown fields"):
        audit_scan_module._source_ref_from_value(
            {
                "artifact_id": "campaign",
                "uri": "campaign.json",
                "format": "campaign-result",
                "extra": "x",
            },
            source_path=None,
            observed_digest=observed,
        )
    with pytest.raises(AuditScanError, match="missing fields"):
        audit_scan_module._source_ref_from_value(
            {"artifact_id": "campaign", "uri": "campaign.json"},
            source_path=None,
            observed_digest=observed,
        )
    with pytest.raises(AuditScanError, match="digest"):
        audit_scan_module._source_ref_from_value(
            {
                "artifact_id": "campaign",
                "uri": "campaign.json",
                "format": "campaign-result",
                "schema": "campaign-result.v1",
                "sha256": "b" * 64,
            },
            source_path=None,
            observed_digest=observed,
        )
    source = audit_scan_module._source_ref_from_value(
        None,
        source_path=tmp_path / "episodes.jsonl",
        observed_digest=observed,
        payload=_campaign(_episode("episode")),
    )
    assert source.format == "episode-jsonl"
    assert source.schema == "campaign-result.v1"


def test_scan_path_and_jsonl_boundaries_preserve_input_accounting(tmp_path: Path) -> None:
    """Path, strict-JSON, and JSONL boundaries reject unsafe or corrupt rows."""
    with pytest.raises(AuditScanError, match="scheme"):
        audit_scan_module._safe_path("https://example.invalid/input.json", root=tmp_path)
    with pytest.raises(AuditScanError, match="traversal"):
        audit_scan_module._safe_path("../input.json", root=tmp_path)
    with pytest.raises(AuditScanError, match="strict JSON"):
        audit_scan_module._strict_json(b'{"a": 1, "a": 2}', label="duplicate")
    with pytest.raises(AuditScanError, match="strict JSON"):
        audit_scan_module._strict_json(b"not-json", label="invalid")
    permissive = audit_scan_module._strict_json(b'{"value": NaN}', label="row", validate=False)
    assert permissive["value"] != permissive["value"]
    rows = audit_scan_module._jsonl_rows(
        b'{"episode_id":"good"}\nnot-json\n\n{"episode_id":"bad","value":NaN}\n',
        label="episodes",
    )
    assert [line for _row, line in rows] == [1, 2, 4]
    assert rows[1][0] is None
    assert rows[2][0]["episode_id"] == "bad"

    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    assert audit_scan_module._safe_path(source.name, root=tmp_path) == source
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(AuditScanError, match="symlink"):
        audit_scan_module._safe_path(link.name, root=tmp_path)
