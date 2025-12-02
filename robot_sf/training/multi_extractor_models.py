"""Data structures shared by the multi-extractor training workflow.

These dataclasses describe configuration inputs, hardware metadata, and
per-extractor results so downstream modules can convert them into summaries
and JSON artifacts without bespoke knowledge of the training script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class ExtractorConfigurationProfile:
    """Static metadata describing a feature extractor under evaluation."""

    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None
    expected_resources: str = "cpu"
    priority: Optional[int] = None
    preset: Optional[str] = None

    def merged_parameters(self) -> dict[str, Any]:
        """Return parameters with a defensive copy for downstream mutation guard."""

        return dict(self.parameters or {})


@dataclass(slots=True)
class HardwareProfile:
    """Snapshot of the machine executing a training run."""

    platform: str
    arch: str
    python_version: str
    workers: int
    gpu_model: Optional[str] = None
    cuda_version: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """To dict.

        Returns:
            dict[str, Any]: Auto-generated placeholder description.
        """
        payload = {
            "platform": self.platform,
            "arch": self.arch,
            "python_version": self.python_version,
            "workers": self.workers,
        }
        if self.gpu_model is not None:
            payload["gpu_model"] = self.gpu_model
        if self.cuda_version is not None:
            payload["cuda_version"] = self.cuda_version
        return payload


@dataclass(slots=True)
class ExtractorRunRecord:
    """Runtime information captured for a single extractor execution."""

    config_name: str
    status: str
    start_time: str
    end_time: Optional[str]
    duration_seconds: Optional[float]
    hardware_profile: HardwareProfile
    worker_mode: str
    training_steps: int
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """To dict.

        Returns:
            dict[str, Any]: Auto-generated placeholder description.
        """
        payload = {
            "config_name": self.config_name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "hardware_profile": self.hardware_profile.to_dict(),
            "worker_mode": self.worker_mode,
            "training_steps": self.training_steps,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


@dataclass(slots=True)
class TrainingRunSummary:
    """Aggregated overview for a multi-extractor training comparison."""

    run_id: str
    created_at: str
    output_root: str
    hardware_overview: list[HardwareProfile]
    extractor_results: list[ExtractorRunRecord]
    aggregate_metrics: dict[str, float]
    notes: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        """To dict.

        Returns:
            dict[str, Any]: Auto-generated placeholder description.
        """
        payload = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "output_root": self.output_root,
            "hardware_overview": [profile.to_dict() for profile in self.hardware_overview],
            "extractor_results": [record.to_dict() for record in self.extractor_results],
            "aggregate_metrics": self.aggregate_metrics,
        }
        if self.notes:
            payload["notes"] = self.notes
        return payload
