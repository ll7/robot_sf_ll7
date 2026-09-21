"""Focused retained-native projection tests for the BA-06 workbench seam."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.analysis_workbench.audit_scan import scan_campaign
from robot_sf.analysis_workbench.audit_service import (
    AuditSelectionContext,
    AuditService,
    SessionPolicy,
)
from robot_sf.analysis_workbench.review_contracts import component_request_from_dict
from robot_sf.benchmark.analysis_trace import trace_artifact_sha256
from robot_sf.benchmark.runner import run_episode
from robot_sf.render.audit_trace_projection import project_retained_native_trace
from robot_sf.render.review_editor import build_editor_model

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def native_selected() -> dict[str, Any]:
    """Retain one real native runner trace, as the BA-05 materialization tests do."""

    original = run_episode(
        {"id": "ba06_projection_probe", "density": "low", "flow": "uni", "obstacle": "open"},
        42,
        horizon=8,
        dt=0.1,
        robot_start=(-4.0, 0.0),
        robot_goal=(4.0, 0.0),
        record_forces=False,
        algo="simple_policy",
        telemetry={"analysis_trace": "all"},
        provenance={"test": "ba06-trace-projection"},
    )
    assert original["algorithm_metadata"]["planner_kinematics"]["execution_mode"] == "native"
    trace = original["algorithm_metadata"]["analysis_trace"]
    return {
        "episode_id": original["episode_id"],
        "seed": original["seed"],
        "scenario_id": "ba06_projection_probe",
        "algo": "simple_policy",
        "source_commit": original["git_hash"],
        "config_identity": trace["config_digest"],
        "retained_trace": trace,
    }


def test_native_trace_projects_explicit_time_scene_and_truthful_missingness(
    native_selected: dict[str, Any],
) -> None:
    """A retained native trace is reviewable without fabricating media or metrics."""

    selected = copy.deepcopy(native_selected)
    trace = selected["retained_trace"]
    result = project_retained_native_trace(selected)

    # The canonical low-density probe intentionally has positional pedestrian
    # labels; geometry remains usable while stable actor identity is explicit
    # missingness, so the projection is partial rather than silently promoted.
    assert result["status"] == "partial"
    panel = result["panel_model"]
    editor = result["editor_model"]
    assert panel["schema_version"] == "review-panels.v1"
    assert editor["schema_version"] == "review-editor.v1"
    expected_times = [step["time_s"] for step in trace["steps"]]
    assert [sample["time_s"] for sample in panel["streams"]["scene"]["samples"]] == expected_times
    assert panel["time"]["origin_s"] == expected_times[0]
    assert panel["time"]["terminal_s"] == expected_times[-1]
    assert panel["time"]["rule"] == "explicit_trace_time_only; no_interpolation"
    assert panel["time"]["authority"] == "simulation_time"
    assert panel["streams"]["video"]["status"] == "unavailable"
    assert "media_uri" not in panel["streams"]["video"]
    assert panel["metrics"] == {}
    assert panel["missingness"]["metrics"]["derived"] is False
    assert panel["provenance"]["original_recording_presented"] is False
    assert panel["provenance"]["derived_media_presented_as_original"] is False
    assert editor["time"] == panel["time"]
    assert editor["streams"]["scene"] == panel["streams"]["scene"]


def test_native_projection_keeps_config_identity_separate_from_trace_digest(
    native_selected: dict[str, Any],
) -> None:
    """A source configuration identity is not the resolved trace config digest."""

    selected = copy.deepcopy(native_selected)
    selected["config_identity"] = "native-config-sha256:distinct-source-identity"
    selected["config_digest"] = selected["retained_trace"]["config_digest"]

    result = project_retained_native_trace(selected)

    assert result["status"] == "partial"
    identity = result["editor_model"]["source_identity"]
    assert identity["config_identity"] == selected["config_identity"]
    assert identity["config_digest"] == selected["config_digest"]
    trace_source = next(iter(identity["sources"].values()))
    assert trace_source["config_identity"] == selected["config_identity"]
    assert trace_source["config_digest"] == selected["config_digest"]


def test_projection_preserves_non_nominal_timestamps_and_optional_missingness(
    native_selected: dict[str, Any],
) -> None:
    """The projector copies trace time and reports absent optional state fields."""

    selected = copy.deepcopy(native_selected)
    trace = selected["retained_trace"]
    for index, step in enumerate(trace["steps"][1:], start=1):
        step["time_s"] = 0.37 + (index - 1) * 0.11
    trace["steps"][1]["robot"].pop("velocity", None)
    trace["artifact_sha256"] = trace_artifact_sha256(trace)

    result = project_retained_native_trace(selected)

    assert result["status"] == "partial"
    panel = result["panel_model"]
    assert panel["streams"]["scene"]["samples"][1]["time_s"] == 0.37
    assert panel["streams"]["scene"]["samples"][1]["value"]["robot"].get("velocity") is None
    assert panel["missingness"]["scene"]["status"] == "partial"
    assert any(
        "robot.velocity" in item["fields"] for item in panel["missingness"]["scene"]["fields"]
    )
    assert panel["time"]["terminal_s"] == trace["steps"][-1]["time_s"]


def test_projection_panel_is_the_editor_inline_wiring_boundary(
    native_selected: dict[str, Any],
) -> None:
    """The projected SREV-16 panel can be handed to SREV-17 without file paths."""

    projection = project_retained_native_trace(native_selected)
    request = component_request_from_dict(
        {
            "schema_version": "component-request.v1",
            "component_id": "srev17-review-editor",
            "request_id": "projection-editor",
            "sources": [
                {"artifact_id": "retained", "uri": "retained.json", "format": "review-panels.v1"}
            ],
            "config": {"panel_model": projection["panel_model"]},
            "output_directory": "unused",
        }
    )

    editor = build_editor_model(request)

    assert editor["schema_version"] == "review-editor.v1"
    assert editor["panel_model"]["schema_version"] == "review-panels.v1"
    assert editor["time"] == projection["panel_model"]["time"]
    assert editor["streams"]["scene"] == projection["panel_model"]["streams"]["scene"]


def test_projection_accepts_scanner_bound_native_trace_without_conflating_digests(
    tmp_path: Path, native_selected: dict[str, Any]
) -> None:
    """Campaign-file and trace-artifact digests remain distinct authorities."""

    trace = native_selected["retained_trace"]
    row = {
        "episode_id": native_selected["episode_id"],
        "scenario_id": native_selected["scenario_id"],
        "planner_id": "simple_policy",
        "seed": native_selected["seed"],
        "source_commit": native_selected["source_commit"],
        "config_identity": native_selected["config_identity"],
        "config_digest": trace["config_digest"],
        "retained_trace": trace,
        "metrics": {},
        "outcome": {"label": "success"},
    }
    source = tmp_path / "campaign.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "campaign-result.v1",
                "campaign_id": "ba06-projection-campaign",
                "episodes": [row],
            }
        ),
        encoding="utf-8",
    )
    report = scan_campaign(source, root=tmp_path)
    assert report.episode_refs
    service = AuditService(tmp_path / "store", campaign_source=source, source_root=tmp_path)
    try:
        session = service.open_session(
            AuditSelectionContext(campaign_id="ba06-projection-campaign"),
            policy=SessionPolicy(allowed_roots=(str(tmp_path),)),
        )
        campaign = service.read_campaign(session)
        assert campaign.value is not None
        episode = campaign.value.episode(native_selected["episode_id"])
        assert episode is not None and episode.row is not None
        scanned = episode.row
    finally:
        service.close()
    assert scanned["source_digest"] == report.audit.source_digest
    assert scanned["source_digest"] != trace["artifact_sha256"]

    scan_defaults = scanned.get("_audit_scan_identity_defaults")
    assert isinstance(scan_defaults, dict)
    projected = project_retained_native_trace(scanned, trusted_scan_identity_defaults=scan_defaults)

    assert projected["status"] == "partial", (
        projected["diagnostics"],
        scanned.get("config_identity"),
        scanned.get("config_digest"),
    )
    assert projected["panel_model"]["provenance"]["source_digest"] == trace["artifact_sha256"]

    forged = copy.deepcopy(scanned)
    forged["source_digest"] = "f" * 64
    forged["_audit_scan_identity_defaults"]["source_digest"] = "f" * 64
    rejected = project_retained_native_trace(forged)
    assert rejected["status"] == "unavailable"
    assert rejected["reason"] == "retained_native_trace_unavailable"


def test_projection_fails_closed_on_digest_or_selected_identity_change(
    native_selected: dict[str, Any],
) -> None:
    """A retained trace cannot be rebound to a different selected episode."""

    digest_changed = copy.deepcopy(native_selected)
    digest_changed["retained_trace"]["steps"][0]["robot"]["position"][0] += 1.0
    digest_result = project_retained_native_trace(digest_changed)
    assert digest_result["status"] == "unavailable"
    assert digest_result["reason"] == "retained_native_trace_unavailable"

    identity_changed = copy.deepcopy(native_selected)
    identity_changed["scenario_id"] = "foreign-scenario"
    identity_result = project_retained_native_trace(identity_changed)
    assert identity_result["status"] == "unavailable"
    assert identity_result["reason"] == "retained_native_trace_unavailable"


def test_projection_does_not_follow_retained_trace_path_or_media_label() -> None:
    """A path/recording declaration is not silently treated as retained trace bytes."""

    result = project_retained_native_trace(
        {
            "episode_id": "episode",
            "retained_trace_source": {"uri": "trace.json", "sha256": "a" * 64},
            "recording": {"uri": "trace.json", "format": "video/mp4"},
        }
    )

    assert result["status"] == "unavailable"
    assert result["reason"] == "retained_native_trace_unavailable"
    assert result["panel_model"]["provenance"]["original_recording_presented"] is False
