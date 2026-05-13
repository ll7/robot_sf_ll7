"""Session manifest helpers for manual-control benchmark runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot_sf.manual_control.baseline import PolicyBaseline
    from robot_sf.manual_control.recording import ManualSessionMetadata
    from robot_sf.manual_control.session import AttemptProgress


@dataclass(frozen=True)
class ManualSessionManifest:
    """JSON-safe manifest summarizing one manual-control session."""

    session: ManualSessionMetadata
    baseline: PolicyBaseline | None = None
    completed_attempts: tuple[AttemptProgress, ...] = ()
    unresolved_attempts: tuple[AttemptProgress, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        """Return the manifest as a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Serializable session manifest.
        """
        return {
            "manifest_schema": "manual_control_session_manifest_v1",
            "session": {
                "session_id": self.session.session_id,
                "input_mapping_version": self.session.input_mapping_version,
                "view_mode": self.session.view_mode,
                "policy_to_beat": self.session.policy_to_beat,
                "policy_to_beat_source": self.session.policy_to_beat_source,
                "extra": self.session.extra,
            },
            "baseline": self.baseline.to_manifest_dict() if self.baseline else None,
            "completed_attempts": [
                _attempt_progress_to_json(attempt) for attempt in self.completed_attempts
            ],
            "unresolved_attempts": [
                _attempt_progress_to_json(attempt) for attempt in self.unresolved_attempts
            ],
            "artifacts": dict(self.artifacts),
            "notes": list(self.notes),
        }


def write_manual_session_manifest(manifest: ManualSessionManifest, path: str | Path) -> Path:
    """Write a manual-control session manifest as sorted JSON.

    Returns
    -------
    Path
        Output manifest path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _attempt_progress_to_json(attempt: AttemptProgress) -> dict[str, Any]:
    """Return JSON-compatible attempt progress metadata.

    Returns
    -------
    dict[str, Any]
        Serializable attempt progress metadata.
    """
    return {
        "scenario_id": attempt.key.scenario_id,
        "seed": attempt.key.seed,
        "retry_count": attempt.retry_count,
        "beat_baseline": attempt.beat_baseline,
        "success": attempt.success,
        "failure_reason": attempt.failure_reason,
    }
