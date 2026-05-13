"""Manual-control helpers for human-driven Robot SF sessions."""

from robot_sf.manual_control.baseline import (
    BaselineComparison,
    BaselineMetric,
    MetricDirection,
    PolicyBaseline,
)
from robot_sf.manual_control.export import (
    DemonstrationSample,
    export_demonstration_samples,
    export_demonstration_samples_from_jsonl,
    write_demonstration_samples_jsonl,
)
from robot_sf.manual_control.input_mapping import (
    DifferentialDriveKeyboardMapper,
    ManualKeyState,
    mapper_for_robot_config,
)
from robot_sf.manual_control.manifest import (
    ManualSessionManifest,
    write_manual_session_manifest,
)
from robot_sf.manual_control.modes import (
    ManualControlMode,
    ManualViewMode,
    ensure_supported_mvp_mode,
)
from robot_sf.manual_control.profile import ManualRecordingProfile, profile_manual_jsonl_recording
from robot_sf.manual_control.recording import (
    ManualControlRecord,
    ManualJsonlRecorder,
    ManualSessionMetadata,
    load_manual_jsonl_records,
)
from robot_sf.manual_control.replay import (
    ManualAttemptReplay,
    ManualReplayEvent,
    group_records_by_attempt,
    iter_replay_events,
    write_attempt_replay_json,
)
from robot_sf.manual_control.session import (
    AttemptKey,
    AttemptProgress,
    ManualSessionController,
    ManualSessionState,
)

__all__ = [
    "AttemptKey",
    "AttemptProgress",
    "BaselineComparison",
    "BaselineMetric",
    "DemonstrationSample",
    "DifferentialDriveKeyboardMapper",
    "ManualAttemptReplay",
    "ManualControlMode",
    "ManualControlRecord",
    "ManualJsonlRecorder",
    "ManualKeyState",
    "ManualRecordingProfile",
    "ManualReplayEvent",
    "ManualSessionController",
    "ManualSessionManifest",
    "ManualSessionMetadata",
    "ManualSessionState",
    "ManualViewMode",
    "MetricDirection",
    "PolicyBaseline",
    "ensure_supported_mvp_mode",
    "export_demonstration_samples",
    "export_demonstration_samples_from_jsonl",
    "group_records_by_attempt",
    "iter_replay_events",
    "load_manual_jsonl_records",
    "mapper_for_robot_config",
    "profile_manual_jsonl_recording",
    "write_attempt_replay_json",
    "write_demonstration_samples_jsonl",
    "write_manual_session_manifest",
]
