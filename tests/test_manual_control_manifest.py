"""Tests for manual-control session manifests."""

import json

from robot_sf.manual_control.baseline import BaselineMetric, MetricDirection, PolicyBaseline
from robot_sf.manual_control.manifest import ManualSessionManifest, write_manual_session_manifest
from robot_sf.manual_control.recording import ManualSessionMetadata
from robot_sf.manual_control.session import AttemptKey, AttemptProgress


def test_manual_session_manifest_serializes_baseline_and_progress():
    """Manifest should identify baseline, completed attempts, and artifacts."""
    manifest = ManualSessionManifest(
        session=ManualSessionMetadata(
            session_id="session-1",
            input_mapping_version="manual_keyboard_diff_drive_hold_v1",
            policy_to_beat="best-policy",
            policy_to_beat_source="model/registry.yaml",
        ),
        baseline=PolicyBaseline(
            policy_id="best-policy",
            source="model/registry.yaml",
            primary_metric="snqi",
            metrics={
                "snqi": BaselineMetric(
                    name="snqi",
                    value=0.5,
                    direction=MetricDirection.HIGHER_IS_BETTER,
                )
            },
        ),
        completed_attempts=(
            AttemptProgress(
                key=AttemptKey("scenario-a", 7),
                retry_count=1,
                beat_baseline=True,
                success=True,
            ),
        ),
        artifacts={"records_jsonl": "manual.jsonl"},
    )

    payload = manifest.to_json_dict()

    assert payload["manifest_schema"] == "manual_control_session_manifest_v1"
    assert payload["baseline"]["policy_id"] == "best-policy"
    assert payload["completed_attempts"][0]["scenario_id"] == "scenario-a"
    assert payload["artifacts"] == {"records_jsonl": "manual.jsonl"}


def test_write_manual_session_manifest(tmp_path):
    """Manifest writer should persist sorted JSON."""
    manifest = ManualSessionManifest(
        session=ManualSessionMetadata(
            session_id="session-1",
            input_mapping_version="manual_keyboard_diff_drive_hold_v1",
        )
    )
    path = tmp_path / "manifest.json"

    written_path = write_manual_session_manifest(manifest, path)

    assert written_path == path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["session"]["session_id"] == "session-1"
