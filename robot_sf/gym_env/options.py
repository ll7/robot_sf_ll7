"""Option dataclasses for environment factory ergonomics.

Purpose
-------
Encapsulate advanced rendering & recording configuration to keep public factory
signatures focused and explicit. Introduced in tasks T004–T006 with validation
and normalization helpers.

Design Principles
-----------------
* Validation methods (`validate`) surface misconfiguration early.
* Instances are lightweight (``slots=True``) to minimize per‑creation overhead.
* Convenience flag normalization (``RecordingOptions.from_bool_and_path``) allows
    gradual migration from legacy boolean parameters.

Precedence (summarized)
-----------------------
1. Explicit options objects override boolean convenience flags.
2. For robot/image factories: ``record_video=True`` upgrades ``record`` to True if False.
3. For pedestrian factory: explicit ``record=False`` is preserved (no upgrade).

See also: :mod:`robot_sf.gym_env.environment_factory` for detailed precedence narrative.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from robot_sf.telemetry import DEFAULT_TELEMETRY_METRICS


@dataclass(slots=True)
class RenderOptions:
    """Rendering related advanced options.

    Attributes
    ----------
    enable_overlay : bool
        Display debug overlay (perf metrics, entity counts) when rendering.
    max_fps_override : int | None
        Optional FPS cap for the rendering loop (>0). If set via convenience ``video_fps``
        and this field is already non-``None`` the existing value wins.
    ped_velocity_scale : float
        Scale multiplier for pedestrian velocity vectors; must be > 0.
    headless_ok : bool
        Permit creation without an available display (CI/headless servers).
    """

    enable_overlay: bool = False
    max_fps_override: int | None = None
    ped_velocity_scale: float = 1.0
    headless_ok: bool = True

    def validate(self) -> None:
        """Validate invariants.

        Rules
        -----
        * max_fps_override must be > 0 when provided.
        * ped_velocity_scale must be > 0.
        """
        if self.max_fps_override is not None and self.max_fps_override <= 0:
            raise ValueError("max_fps_override must be > 0")
        if self.ped_velocity_scale <= 0:
            raise ValueError("ped_velocity_scale must be > 0")


@dataclass(slots=True)
class RecordingOptions:
    """Video / frame recording configuration.

    Attributes
    ----------
    record : bool
        Enable frame capture (may still be gated by factory ``recording_enabled`` flag).
    video_path : str | None
        Output file path (``.mp4``). If ``None`` frames may be buffered/transient only.
    max_frames : int | None
        Hard cap on captured frames; ``None`` uses internal default.
    codec : str
        Video codec identifier (default libx264). Must be non-empty when recording.
    bitrate : str | None
        Target bitrate hint passed to encoder; non-empty string when provided.
    """

    record: bool = False
    video_path: str | None = None
    max_frames: int | None = None
    codec: str = "libx264"
    bitrate: str | None = None

    def validate(self) -> None:
        """Validate invariants.

        Rules
        -----
        * If record is True and video_path provided it must end with .mp4
          (project standard container) unless explicitly None.
        * max_frames must be > 0 when provided.
        * codec must be a non-empty string when recording.
        * bitrate, if provided, must be a non-empty string.
        """
        if self.record and self.video_path and not self.video_path.lower().endswith(".mp4"):
            raise ValueError("video_path must end with .mp4")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be > 0")
        if self.record and not self.codec:
            raise ValueError("codec must be non-empty when recording")
        if self.bitrate is not None and not self.bitrate:
            raise ValueError("bitrate must be non-empty when provided")

    @classmethod
    def from_bool_and_path(
        cls,
        record_video: bool,
        video_path: str | None,
        existing: RecordingOptions | None,
    ) -> RecordingOptions:
        """Create or normalize a RecordingOptions instance from convenience flags.

        If an existing options object is provided it is returned unchanged
        unless record_video=True and existing.record is False, in which case
        the returned object is a shallow copy with record flipped to True.
        (Final precedence rules may adjust in T006.)

        Returns:
            RecordingOptions instance with normalized settings.
        """
        if existing is not None:
            if record_video and not existing.record:
                # Shallow copy with record toggled on and optional path precedence.
                return replace(
                    existing,
                    record=True,
                    video_path=existing.video_path or video_path,
                )
            # Return existing (already validated) instance unchanged.
            return existing
        if record_video:
            return cls(record=True, video_path=video_path)
        return cls()


@dataclass(slots=True)
class JsonlRecordingOptions:
    """Per-episode JSONL recording metadata for environment factories.

    This option object groups the JSONL recorder settings that used to travel
    through the robot factory as separate keyword arguments. It intentionally
    does not replace ``recording_enabled`` or :class:`RecordingOptions`:
    ``recording_enabled`` remains the runtime master gate and
    :class:`RecordingOptions` remains the video/frame capture intent.
    """

    enabled: bool = False
    recording_dir: str = "recordings"
    suite_name: str = "robot_sim"
    scenario_name: str = "default"
    algorithm_name: str = "manual"
    recording_seed: int | None = None

    def validate(self) -> None:
        """Validate JSONL metadata fields before passing them to the recorder."""
        if not self.recording_dir:
            raise ValueError("recording_dir must be non-empty")
        if not self.suite_name:
            raise ValueError("suite_name must be non-empty")
        if not self.scenario_name:
            raise ValueError("scenario_name must be non-empty")
        if not self.algorithm_name:
            raise ValueError("algorithm_name must be non-empty")


@dataclass(slots=True)
class TelemetryOptions:
    """Live telemetry panel and recording options for robot factories."""

    enable_panel: bool = False
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_TELEMETRY_METRICS))
    record: bool = False
    refresh_hz: float = 1.0
    pane_layout: str = "vertical_split"
    decimation: int = 1

    def validate(self) -> None:
        """Validate telemetry options using the same rules as environment configs."""
        if self.refresh_hz <= 0:
            raise ValueError("refresh_hz must be > 0")
        if self.decimation <= 0:
            raise ValueError("decimation must be >= 1")
        if self.pane_layout not in {"vertical_split", "horizontal_split"}:
            raise ValueError("pane_layout must be 'vertical_split' or 'horizontal_split'")
        self.metrics = [
            metric for metric in (self.metrics or []) if isinstance(metric, str) and metric.strip()
        ]
        if not self.metrics:
            self.metrics = list(DEFAULT_TELEMETRY_METRICS)
