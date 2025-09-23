"""Option dataclasses for environment factory ergonomics (stub for T004).

These dataclasses will encapsulate advanced rendering / recording parameters so
the primary factory signatures remain concise. Full validation logic will be
added in T006; currently only structure and docstrings are provided.

Do not rely on runtime validation yet (placeholder implementation).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(slots=True)
class RenderOptions:
    """Rendering related advanced options.

    Attributes
    ----------
    enable_overlay: Show debug overlay information.
    max_fps_override: Optional FPS cap for rendering loop (>0 when set).
    ped_velocity_scale: Scale multiplier for velocity vectors (>0).
    headless_ok: Allow creation when display not available.
    """

    enable_overlay: bool = False
    max_fps_override: Optional[int] = None
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
    record: Enable frame capture.
    video_path: Output mp4 path (optional: buffer only if None).
    max_frames: Override internal frame cap (>0 when set; None keeps default behavior).
    codec: Preferred video codec (non-empty when recording).
    bitrate: Optional bitrate specification string (non-empty when provided).
    """

    record: bool = False
    video_path: Optional[str] = None
    max_frames: Optional[int] = None
    codec: str = "libx264"
    bitrate: Optional[str] = None

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
        if self.record and self.video_path:
            if not self.video_path.lower().endswith(".mp4"):
                raise ValueError("video_path must end with .mp4")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be > 0")
        if self.record and not self.codec:
            raise ValueError("codec must be non-empty when recording")
        if self.bitrate is not None and not self.bitrate:
            raise ValueError("bitrate must be non-empty when provided")

    @classmethod
    def from_bool_and_path(
        cls, record_video: bool, video_path: Optional[str], existing: Optional["RecordingOptions"]
    ) -> "RecordingOptions":
        """Create or normalize a RecordingOptions instance from convenience flags.

        If an existing options object is provided it is returned unchanged
        unless record_video=True and existing.record is False, in which case
        the returned object is a shallow copy with record flipped to True.
        (Final precedence rules may adjust in T006.)
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
