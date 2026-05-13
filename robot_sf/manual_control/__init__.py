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
    DifferentialDriveCruiseKeyboardMapper,
    DifferentialDriveKeyboardMapper,
    DifferentialDriveMouseTargetMapper,
    ManualKeyState,
    ManualMouseTarget,
    mapper_for_manual_mode,
    mapper_for_robot_config,
)
from robot_sf.manual_control.manifest import (
    ManualSessionManifest,
    write_manual_session_manifest,
)
from robot_sf.manual_control.modes import (
    CONTROL_MODE_REGISTRY,
    VIEW_MODE_REGISTRY,
    ManualControlMode,
    ManualControlModeSpec,
    ManualViewMode,
    ManualViewModeSpec,
    control_mode_spec,
    ensure_supported_manual_mode,
    ensure_supported_mvp_mode,
    parse_manual_control_mode,
    parse_manual_view_mode,
    view_mode_spec,
)
from robot_sf.manual_control.recording import (
    ManualControlRecord,
    ManualJsonlRecorder,
    ManualSessionMetadata,
    load_manual_jsonl_records,
)
from robot_sf.manual_control.replay import ManualAttemptReplay, group_records_by_attempt
from robot_sf.manual_control.session import (
    AttemptKey,
    AttemptProgress,
    ManualSessionController,
    ManualSessionState,
)

__all__ = [
    "CONTROL_MODE_REGISTRY",
    "VIEW_MODE_REGISTRY",
    "AttemptKey",
    "AttemptProgress",
    "BaselineComparison",
    "BaselineMetric",
    "DemonstrationSample",
    "DifferentialDriveCruiseKeyboardMapper",
    "DifferentialDriveKeyboardMapper",
    "DifferentialDriveMouseTargetMapper",
    "ManualAttemptReplay",
    "ManualControlMode",
    "ManualControlModeSpec",
    "ManualControlRecord",
    "ManualJsonlRecorder",
    "ManualKeyState",
    "ManualMouseTarget",
    "ManualSessionController",
    "ManualSessionManifest",
    "ManualSessionMetadata",
    "ManualSessionState",
    "ManualViewMode",
    "ManualViewModeSpec",
    "MetricDirection",
    "PolicyBaseline",
    "control_mode_spec",
    "ensure_supported_manual_mode",
    "ensure_supported_mvp_mode",
    "export_demonstration_samples",
    "export_demonstration_samples_from_jsonl",
    "group_records_by_attempt",
    "load_manual_jsonl_records",
    "mapper_for_manual_mode",
    "mapper_for_robot_config",
    "parse_manual_control_mode",
    "parse_manual_view_mode",
    "view_mode_spec",
    "write_demonstration_samples_jsonl",
    "write_manual_session_manifest",
]
