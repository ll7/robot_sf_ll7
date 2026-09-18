"""Focused offline SREV-17 annotation, reference, persistence, and browser tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.analysis_workbench.audit_contracts import (
    ActionRecord,
    Reference,
)
from robot_sf.analysis_workbench.audit_store import AuditConflictError, CommitResult
from robot_sf.analysis_workbench.review_contracts import component_request_from_dict
from robot_sf.render import review_editor

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "scenario_review" / "review_editor"
)


def _request(tmp_path: Path, *, output: str = "editor"):
    panel = tmp_path / "panel-model.json"
    shutil.copyfile(FIXTURE_ROOT / "panel-model.json", panel)
    payload = json.loads((FIXTURE_ROOT / "request.json").read_text(encoding="utf-8"))
    payload["output_directory"] = output
    payload["sources"][0]["uri"] = "panel-model.json"
    return component_request_from_dict(payload, source="request.json")


def _model(tmp_path: Path):
    return review_editor.build_editor_model(_request(tmp_path), base=tmp_path)


def test_fixture_emits_offline_editor_and_preserves_source(tmp_path: Path) -> None:
    request = _request(tmp_path)
    before = (tmp_path / "panel-model.json").read_bytes()
    result = review_editor.run(request, base=tmp_path)
    assert result.status == "complete"
    assert (tmp_path / "editor" / "review-editor.v1.json").is_file()
    html = (tmp_path / "editor" / "review-editor.v1.html").read_text(encoding="utf-8")
    javascript = (
        tmp_path / "editor" / "components" / "review_editor" / "review_editor.js"
    ).read_text(encoding="utf-8")
    assert "localStorage" not in html + javascript
    assert "http://" not in html + javascript
    assert "review-editor-root" in html
    assert "review-editor-data" in html
    assert 'getElementById("review-editor-root")' in javascript
    assert (tmp_path / "panel-model.json").read_bytes() == before


def test_three_speeds_and_quick_note_do_not_require_cause() -> None:
    model = {"context": {"episode_id": "ep", "execution_id": "run", "cursor": {"time_s": 2.5}}}
    triage = review_editor.make_one_click_annotation(model, "Bug")
    quick = review_editor.make_quick_annotation(model, "unclear", tags=("odd",))
    rich = review_editor.make_full_annotation(
        model,
        "planner_defect",
        observed_behavior="robot stops",
        actors=("robot",),
        measured_evidence=({"metric_id": "speed", "value": 0.0, "units": "m/s"},),
        hypothesis="blocked route",
        confidence=0.8,
        notes="inspect next event",
    )
    assert triage.mode == "one_click" and triage.classification == "planner_defect"
    assert quick.mode == "quick" and quick.suspected_cause == "" and quick.confidence is None
    assert rich.mode == "full" and rich.metadata["actors"] == ["robot"]


def test_source_bound_references_snap_and_distance_use_recorded_units(tmp_path: Path) -> None:
    model = _model(tmp_path)
    refs = [
        review_editor.snap_reference(model, "actor", target_id="robot"),
        review_editor.snap_reference(model, "goal"),
    ]
    assert refs[0].timestamp_s == 10.5
    assert refs[0].seek_identity == "execution-17"
    commands = review_editor.build_overlay_commands(model, refs, overlays={"distances": True})
    distance = next(item for item in commands if item["kind"] == "distance")
    assert distance["units"] == "m"
    image = Reference(
        reference_id="image", coordinate_frame="image", point=(30, 40), timestamp_s=10.5
    )
    assert not any(
        item["kind"] == "distance"
        for item in review_editor.build_overlay_commands(
            model, [image, image], overlays={"distances": True}
        )
    )


def test_image_reference_preserves_source_point_across_crop_and_resize() -> None:
    source = review_editor.SourceRef(
        artifact_id="video",
        uri="video.mp4",
        format="video/mp4",
        sha256="a" * 64,
        source_commit="capture-4",
    )
    model = {
        "context": {"execution_id": "run-4", "cursor": {"time_s": 4.0}},
        "source_identity": {"coordinate_frame": "image"},
    }
    reference = review_editor.create_reference(
        model,
        (50.0, 25.0),
        coordinate_frame="image",
        source=source,
        source_revision="capture-4",
        source_point=(200.0, 100.0),
        calibration={
            "source_width": 640,
            "source_height": 360,
            "display_width": 320,
            "display_height": 180,
            "crop_x": 80,
            "crop_y": 40,
            "crop_width": 480,
            "crop_height": 270,
        },
    )
    command = review_editor.build_overlay_commands(model, [reference], overlays={"numbered": True})[
        0
    ]
    assert command["point"] == [50.0, 25.0]
    assert command["source_point"] == [200.0, 100.0]
    assert reference.seek_identity == "run-4"


def test_storyboard_rejects_invalid_intervals_and_round_trips_exactly(tmp_path: Path) -> None:
    editor = review_editor.StoryboardEditor(duration_s=12.0)
    editor.add_interval("a", 1.0, 2.0, caption="A")
    editor.add_interval("b", 3.0, 4.0, caption="B")
    editor.reorder(["b", "a"])
    editor.set_caption("a", "updated")
    with pytest.raises(review_editor.InvalidIntervalError):
        editor.update_interval("a", 8.0, 2.0)
    editor.undo()
    editor.redo()
    path = editor.save(tmp_path / "storyboard.json")
    raw = path.read_bytes()
    loaded = review_editor.StoryboardEditor.load(path)
    loaded.save(tmp_path / "storyboard-copy.json")
    assert (tmp_path / "storyboard-copy.json").read_bytes() == raw


def test_store_adapter_cas_conflict_and_failed_autosave_preserve_local_edit(tmp_path: Path) -> None:
    model = _model(tmp_path)
    with review_editor.AuditStoreAdapter(tmp_path / "store") as adapter:
        first_session = review_editor.ReviewEditorSession(model, adapter=adapter)
        second_session = review_editor.ReviewEditorSession(model, adapter=adapter)
        first = first_session.create_quick("unclear", observed_behavior="human edit")
        receipt = first_session.save_annotation(first, operation_id="op-human", expected_revision=0)
        stale = second_session.create_quick(
            "unclear", annotation_id=first.annotation_id, observed_behavior="agent edit"
        )
        with pytest.raises(AuditConflictError):
            second_session.save_annotation(stale, operation_id="op-agent", expected_revision=0)
        assert second_session.autosave_status.state == "error"
        assert second_session.autosave_status.conflict is not None
        assert second_session.pending_record == stale
        assert adapter.get(first.annotation_id).revision == receipt.revision


def test_save_does_not_fill_missing_expected_revision_from_current_store(tmp_path: Path) -> None:
    model = {"context": {"episode_id": "ep", "execution_id": "run"}}
    with review_editor.AuditStoreAdapter(tmp_path / "store") as adapter:
        session = review_editor.ReviewEditorSession(model, adapter=adapter)
        first = session.create_quick("unclear", annotation_id="same")
        session.save_annotation(first, operation_id="first", expected_revision=0)
        replacement = session.create_quick("unclear", annotation_id="same")
        with pytest.raises(AuditConflictError, match="expected_revision is required"):
            session.save_annotation(replacement, operation_id="replacement")


def test_source_origin_resolution_and_schema_are_fail_closed(tmp_path: Path) -> None:
    model = {
        "context": {"episode_id": "ep"},
        "time": {"origin_s": 10.0, "terminal_s": 12.0},
        "streams": {
            "scene": {
                "resolution_s": 0.1,
                "samples": [{"time_s": 10.0, "value": {"robot": {"id": "r", "position": [1, 2]}}}],
            }
        },
    }
    with pytest.raises(review_editor.InvalidIntervalError):
        review_editor.make_quick_annotation(model, "unclear", interval=(1.0, 2.0))
    with pytest.raises(review_editor.ReviewEditorError, match="geometry unavailable"):
        review_editor.snap_reference(model, "actor", target_id="r", timestamp_s=11.0)
    with pytest.raises(review_editor.ReviewEditorError, match="schema_version"):
        review_editor.StoryboardEditor.from_mapping(
            {
                "schema_version": "review-storyboard-edit.v0",
                "intervals": [],
                "order": [],
                "captions": {},
            }
        )
    with pytest.raises(review_editor.ReviewEditorError, match="schema_version is required"):
        review_editor.StoryboardEditor.from_mapping({"intervals": [], "order": [], "captions": {}})
    complete_storyboard = {
        "schema_version": review_editor.STORYBOARD_SCHEMA_VERSION,
        "intervals": [],
        "order": [],
        "captions": {},
    }
    for missing in ("intervals", "order", "captions"):
        incomplete = dict(complete_storyboard)
        incomplete.pop(missing)
        with pytest.raises(review_editor.ReviewEditorError, match="fields are required"):
            review_editor.StoryboardEditor.from_mapping(incomplete)
    with pytest.raises(review_editor.InvalidIntervalError, match="before source origin"):
        review_editor.create_reference(model, (0.0, 0.0), timestamp_s=9.9)
    with pytest.raises(review_editor.InvalidIntervalError, match="after source terminal"):
        review_editor.create_reference(model, (0.0, 0.0), timestamp_s=12.1)

    panel = tmp_path / "panel.json"
    panel.write_text('{"schema_version":"review-panels.v0"}', encoding="utf-8")
    payload = json.loads((FIXTURE_ROOT / "request.json").read_text(encoding="utf-8"))
    payload["sources"][0].update(
        uri="panel.json", sha256=hashlib.sha256(panel.read_bytes()).hexdigest()
    )
    request = component_request_from_dict(payload, source="request.json")
    with pytest.raises(review_editor.ReviewEditorError, match="review-panels.v1"):
        review_editor.build_editor_model(request, base=tmp_path)


def test_storyboard_adapter_persistence_round_trips_with_cas(tmp_path: Path) -> None:
    model = {
        "context": {"episode_id": "ep", "execution_id": "run"},
        "time": {"origin_s": 10.0, "terminal_s": 12.0},
    }
    with review_editor.AuditStoreAdapter(tmp_path / "store") as adapter:
        session = review_editor.ReviewEditorSession(model, adapter=adapter)
        session.storyboard.add_interval("a", 10.2, 10.5, caption="near miss")
        receipt = session.save_storyboard(operation_id="story-1", expected_revision=0)
        reloaded = review_editor.ReviewEditorSession(model, adapter=adapter)
        reloaded.reload_storyboard()
        assert receipt.revision == 1
        assert reloaded.storyboard.snapshot()["intervals"][0]["caption"] == "near miss"


def test_stale_selection_and_failed_service_save_preserve_local_edit(tmp_path: Path) -> None:
    model = _model(tmp_path)
    session = review_editor.ReviewEditorSession(
        model, adapter=review_editor.AuditStoreAdapter(tmp_path / "store")
    )
    annotation = session.create_quick("unclear", observed_behavior="before seek")
    session.select(time_s=11.0)
    with pytest.raises(review_editor.StaleSelectionError):
        session.save_annotation(annotation, operation_id="stale-selection")
    session.adapter.close()

    class FailingAdapter:
        def get_revision(self, _record_id: str) -> int:
            return 0

        def save(self, *_args, **_kwargs):
            raise RuntimeError("simulated service outage")

        def get(self, *_args, **_kwargs):
            return None

    failed = review_editor.ReviewEditorSession(model, adapter=FailingAdapter())
    local = failed.create_quick("unclear", observed_behavior="keep this")
    with pytest.raises(RuntimeError, match="simulated service outage"):
        failed.save_annotation(local, operation_id="failed-save")
    assert failed.autosave_status.state == "error"
    assert failed.pending_record == local


def test_blocking_adapter_serializes_selection_and_durable_save(tmp_path: Path) -> None:
    model = _model(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    selector_started = threading.Event()
    selector_finished = threading.Event()

    class BlockingAdapter:
        def get(self, _record_id: str, *, include_deleted: bool = False):
            del include_deleted

        def save(self, record, **_kwargs):
            entered.set()
            assert release.wait(timeout=2.0)
            return CommitResult("blocked", record.annotation_id, "annotation", 1, 1)

    session = review_editor.ReviewEditorSession(model, adapter=BlockingAdapter())
    annotation = session.create_quick("unclear", observed_behavior="captured")
    errors: list[BaseException] = []

    def persist() -> None:
        try:
            session.save_annotation(annotation, operation_id="blocked-save", expected_revision=0)
        except BaseException as error:  # pragma: no cover - diagnostic propagation
            errors.append(error)

    writer = threading.Thread(target=persist)
    writer.start()
    assert entered.wait(timeout=2.0)

    def seek() -> None:
        selector_started.set()
        session.select(time_s=11.0)
        selector_finished.set()

    selector = threading.Thread(target=seek)
    selector.start()
    assert selector_started.wait(timeout=2.0)
    assert not selector_finished.wait(timeout=0.1)
    release.set()
    writer.join(timeout=2.0)
    selector.join(timeout=2.0)
    assert not errors
    assert not writer.is_alive() and not selector.is_alive()
    assert selector_finished.is_set()
    assert session.autosave_status.state == "saved"
    assert session.autosave_status.selection_revision == session.selection_revision - 1


def test_verified_provenance_requires_matching_nonempty_identity_and_revision() -> None:
    ref = review_editor.SourceRef(
        artifact_id="scene",
        uri="scene.json",
        format="scene/json",
        schema="scene.v1",
        sha256="a" * 64,
        source_commit="commit-1",
        config_identity="config-1",
    )
    model = {"context": {"episode_id": "ep"}}
    verified = review_editor.make_quick_annotation(
        model,
        "unclear",
        source_refs={"scene": ref},
        source_digests={"scene": "a" * 64},
    )
    assert verified.provenance_status == "verified"
    assert verified.source_identity == "a" * 64
    assert verified.source_revision == "commit-1"
    unavailable = review_editor.make_quick_annotation(
        model,
        "unclear",
        source_refs={"scene": replace(ref, source_commit="")},
        source_digests={"scene": "a" * 64},
    )
    assert unavailable.provenance_status == "unavailable"
    first_session = review_editor.ReviewEditorSession(
        model, source_refs={"scene": ref}, source_digests={"scene": "a" * 64}
    )
    changed_session = review_editor.ReviewEditorSession(
        model,
        source_refs={"scene": replace(ref, schema="scene.v2")},
        source_digests={"scene": "a" * 64},
    )
    config_session = review_editor.ReviewEditorSession(
        model,
        source_refs={"scene": replace(ref, config_identity="config-2")},
        source_digests={"scene": "a" * 64},
    )
    assert first_session._source_identity_token() != changed_session._source_identity_token()
    assert first_session._source_revision_token() != changed_session._source_revision_token()
    assert first_session._source_identity_token() != config_session._source_identity_token()
    assert first_session._source_revision_token() != config_session._source_revision_token()
    rebound = review_editor._rebind_annotation_provenance(
        {
            "provenance_status": "verified",
            "source_ref": {"artifact_id": "scene"},
            "source_identity": "foreign",
            "source_revision": "commit-1",
        },
        {"scene": ref},
        {"scene": "a" * 64},
    )
    assert rebound["provenance_status"] == "stale"


def test_storyboard_reload_rejects_tombstones_revisions_types_and_missing_provenance(
    tmp_path: Path,
) -> None:
    model = {"context": {"episode_id": "ep", "execution_id": "run"}}
    with review_editor.AuditStoreAdapter(tmp_path / "store") as adapter:
        session = review_editor.ReviewEditorSession(model, adapter=adapter)
        session.save_storyboard(operation_id="reload-contract", expected_revision=0)
        stored = adapter.get(session.storyboard_record_id(), include_deleted=True)
        assert stored is not None and isinstance(stored.record, ActionRecord)

        cases = (
            (replace(stored, deleted=True), "tombstone"),
            (replace(stored, revision=-1), "revision"),
            (replace(stored, record_type="annotation"), "record type"),
            (
                replace(stored, record=replace(stored.record, action_type="annotation")),
                "identity",
            ),
        )
        details = dict(stored.record.details)
        details.pop("source_revision")
        cases += ((replace(stored, record=replace(stored.record, details=details)), "provenance"),)

        class FixedAdapter:
            def __init__(self, value):
                self.value = value

            def get(self, _record_id: str, *, include_deleted: bool = False):
                del include_deleted
                return self.value

        for value, message in cases:
            candidate = review_editor.ReviewEditorSession(model, adapter=FixedAdapter(value))
            with pytest.raises(review_editor.ReviewEditorError, match=message):
                candidate.reload_storyboard()


def test_full_coverage_excludes_triage_intervals_and_agents(tmp_path: Path) -> None:
    model = _model(tmp_path)
    triage = review_editor.make_one_click_annotation(model, "normal")
    agent = review_editor.make_full_annotation(
        model, "unclear", author_kind="agent", full_episode=True
    )
    interval = review_editor.make_quick_annotation(model, "unclear")
    receipt = review_editor.make_full_episode_review(model)
    summary = review_editor.coverage_summary([triage, agent, interval, receipt])
    assert summary["human_full_episode_count"] == 1
    assert summary["triage_count"] == 1  # quick/interval rows are not triage or coverage.
    assert summary["agent_annotation_count"] == 1


def test_stale_source_is_explicit_and_not_rebound(tmp_path: Path) -> None:
    panel = tmp_path / "panel-model.json"
    shutil.copyfile(FIXTURE_ROOT / "panel-model.json", panel)
    payload = json.loads((FIXTURE_ROOT / "request.json").read_text())
    payload["sources"][0]["uri"] = "panel-model.json"
    payload["sources"][0]["sha256"] = "0" * 64
    request = component_request_from_dict(payload, source="request.json")
    model = review_editor.build_editor_model(request, base=tmp_path)
    assert model["source_identity_status"]["panel"]["status"] == "stale"
    annotation = review_editor.make_quick_annotation(
        model,
        "unclear",
        source_refs={"panel": request.sources[0]},
        source_digests=model["source_identity_status"]
        and {"panel": hashlib.sha256(panel.read_bytes()).hexdigest()},
    )
    assert annotation.provenance_status == "stale"


def test_existing_annotation_revision_is_marked_stale_on_source_revision_change(
    tmp_path: Path,
) -> None:
    panel = tmp_path / "panel-model.json"
    shutil.copyfile(FIXTURE_ROOT / "panel-model.json", panel)
    payload = json.loads((FIXTURE_ROOT / "request.json").read_text(encoding="utf-8"))
    payload["sources"][0].update(
        uri="panel-model.json",
        sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),
        source_commit="b" * 40,
    )
    payload["config"] = {
        "annotations": [
            {
                "record_type": "annotation",
                "annotation_id": "old",
                "episode_id": "episode-17",
                "classification": "unclear",
                "mode": "quick",
                "author_kind": "human",
                "source_identity": payload["sources"][0]["sha256"],
                "source_revision": "a" * 40,
                "provenance_status": "verified",
            }
        ]
    }
    model = review_editor.build_editor_model(
        component_request_from_dict(payload, source="request.json"), base=tmp_path
    )
    assert model["annotations"][0]["provenance_status"] == "stale"


def test_untrusted_config_and_storyboard_exports_reject_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text('{"schema_version":"review-panels.v1"}', encoding="utf-8")
    config = tmp_path / "config.json"
    config.symlink_to(outside)
    with pytest.raises(review_editor.ReviewEditorError):
        review_editor._strict_cli_json(str(config))
    export = tmp_path / "storyboard.json"
    export.symlink_to(outside)
    with pytest.raises(review_editor.ReviewEditorError):
        review_editor.StoryboardEditor().save(export, overwrite=True)


def test_snapping_uses_srev16_status_missingness_and_earlier_ties() -> None:
    model = {
        "context": {"cursor": {"time_s": 10.5}},
        "time": {"origin_s": 10.0, "terminal_s": 11.0},
        "streams": {
            "scene": {
                "status": "available",
                "resolution_s": 0.5,
                "samples": [
                    {"time_s": 10.0, "value": {"robot": {"id": "r", "position": [1, 2]}}},
                    {"time_s": 11.0, "value": {"robot": {"id": "r", "position": [9, 9]}}},
                ],
            }
        },
        "metrics": {
            "clearance": {
                "stream": {
                    "status": "available",
                    "resolution_s": 0.5,
                    "samples": [
                        {"time_s": 10.0, "value": 1.0, "missing": True},
                        {"time_s": 11.0, "value": 2.0},
                    ],
                }
            }
        },
    }
    actor = review_editor.snap_reference(model, "actor", target_id="r")
    assert actor.point == (1.0, 2.0)
    with pytest.raises(review_editor.ReviewEditorError, match="sample_missing"):
        review_editor.snap_reference(
            {
                **model,
                "streams": {
                    "scene": {
                        "status": "available",
                        "resolution_s": 0.5,
                        "samples": [
                            {
                                "time_s": 10.0,
                                "value": {"robot": {"id": "r", "position": [1, 2]}},
                                "missing_reason": "gap",
                            },
                            {"time_s": 11.0, "value": {"robot": {"id": "r", "position": [9, 9]}}},
                        ],
                    }
                },
            },
            "actor",
            target_id="r",
        )
    with pytest.raises(review_editor.ReviewEditorError, match="sample_missing"):
        review_editor.snap_reference(model, "metric", target_id="clearance")
    unavailable = {
        **model,
        "streams": {"scene": {"status": "unavailable", "reason": "capture_missing"}},
    }
    with pytest.raises(review_editor.ReviewEditorError, match="capture_missing"):
        review_editor.snap_reference(unavailable, "actor", target_id="r")


def test_verified_annotation_without_a_resolvable_binding_is_unavailable() -> None:
    result = review_editor._rebind_annotation_provenance(
        {"provenance_status": "verified", "source_identity": "missing-source"}, {}, {}
    )
    assert result["provenance_status"] == "unavailable"
    assert "binding" in result["provenance_reason"]


def test_node_browser_controller_honours_typing_shortcut_suppression() -> None:
    script = Path(__file__).resolve().parent / "review_editor_runtime.mjs"
    completed = subprocess.run(["node", str(script)], check=True, capture_output=True, text=True)
    assert "review_editor_runtime: ok" in completed.stdout


def test_ba05_service_adapter_forwards_save_and_commit_variants() -> None:
    record = SimpleNamespace(record_id="row-1")
    receipt = CommitResult("op", "row-1", "annotation", 1, 1)

    class SaveService:
        def __init__(self) -> None:
            self.calls = []

        def get(self, record_id: str, **kwargs):
            self.calls.append(("get", record_id, kwargs))

        def save(self, saved, **kwargs):
            self.calls.append(("save", saved, kwargs))
            return receipt

    service = SaveService()
    adapter = review_editor.ServicePersistenceAdapter(service)
    assert adapter.get("row-1") is None
    assert adapter.get_revision("row-1") == 0
    assert (
        adapter.save(
            record,
            operation_id="op",
            expected_revision=0,
            actor="human",
            actor_id="h1",
        )
        == receipt
    )
    assert service.calls[-1][2]["expected_revision"] == 0

    class CommitService:
        def load(self, _record_id: str, **_kwargs):
            return SimpleNamespace(revision=4)

        def commit(self, records, **_kwargs):
            assert records == [record]
            return receipt

    commit_adapter = review_editor.ServicePersistenceAdapter(CommitService())
    assert commit_adapter.get_revision("row-1") == 4
    assert (
        commit_adapter.save(
            record,
            operation_id="op",
            expected_revision=0,
            actor="human",
        )
        == receipt
    )

    with pytest.raises(review_editor.ReviewEditorError, match="requires get/load"):
        review_editor.ServicePersistenceAdapter(object()).get("row-1")
    with pytest.raises(review_editor.ReviewEditorError, match="requires save/commit"):
        review_editor.ServicePersistenceAdapter(
            SimpleNamespace(get=lambda *_args, **_kwargs: None)
        ).save(record, operation_id="op", expected_revision=0, actor="human")
    with pytest.raises(review_editor.ReviewEditorError, match="commit receipt"):
        review_editor.ServicePersistenceAdapter(
            SimpleNamespace(save=lambda *_args, **_kwargs: object())
        ).save(record, operation_id="op", expected_revision=0, actor="human")


def test_reference_targets_overlay_flags_and_storyboard_boundaries(tmp_path: Path) -> None:
    model = _model(tmp_path)
    waypoint = review_editor.snap_reference(model, "waypoint", target_id="wp-1")
    map_object = review_editor.snap_reference(model, "map", target_id="wall-1")
    event = review_editor.snap_reference(model, "event", target_id="near-miss-17")
    metric = review_editor.snap_reference(model, "metric", target_id="clearance")
    assert waypoint.waypoint_id == "wp-1"
    assert map_object.object_id == "wall-1"
    assert event.event_id == "near-miss-17"
    assert metric.metric_id == "clearance"

    commands = review_editor.build_overlay_commands(
        model,
        [waypoint, event],
        overlays={"numbered": True, "highlights": True, "arrows": True, "rings": True},
    )
    assert {item["kind"] for item in commands} >= {"numbered", "ring"}

    editor = review_editor.StoryboardEditor(duration_s=12.0)
    editor.add_interval("a", 1.0, 2.0)
    editor.update_interval("a", 2.0, 3.0)
    editor.set_caption("a", "updated")
    with pytest.raises(review_editor.ReviewEditorError, match="duplicate"):
        editor.add_interval("a", 4.0, 5.0)
    with pytest.raises(review_editor.ReviewEditorError, match="unknown storyboard interval"):
        editor.update_interval("missing", 1.0, 2.0)
    with pytest.raises(review_editor.ReviewEditorError, match="unknown storyboard interval"):
        editor.set_caption("missing", "nope")
    with pytest.raises(review_editor.ReviewEditorError, match="unknown storyboard interval"):
        editor.remove("missing")
    editor.remove("a")
    assert editor.undo().intervals
    assert editor.redo().intervals == ()
    destination = editor.save(tmp_path / "overwrite.json")
    editor.save(destination, overwrite=True)


def test_input_contract_rejects_invalid_values_and_malformed_model_bindings() -> None:
    with pytest.raises(review_editor.ReviewEditorError, match="finite"):
        review_editor.validate_interval({"start_s": float("nan")})
    with pytest.raises(review_editor.ReviewEditorError, match="exactly two"):
        review_editor.create_reference({}, (1.0,))
    with pytest.raises(review_editor.ReviewEditorError, match="optional text"):
        review_editor.make_quick_annotation({}, "unclear", observed_behavior=1)  # type: ignore[arg-type]
    with pytest.raises(review_editor.ReviewEditorError, match="author_kind"):
        review_editor.make_full_annotation({}, "unclear", author_kind="robot")
    with pytest.raises(review_editor.ReviewEditorError, match="measured_evidence"):
        review_editor.make_full_annotation({}, "unclear", measured_evidence=(1,))  # type: ignore[arg-type]

    model = {
        "source_identity": {
            "sources": {
                "bad": 3,
                "missing": {"artifact_id": "missing", "uri": "", "format": "scene"},
            }
        }
    }
    refs, digests = review_editor._source_bindings_from_model(model)
    assert refs == {} and digests == {}
