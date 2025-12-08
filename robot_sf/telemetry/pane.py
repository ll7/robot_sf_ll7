"""Lightweight telemetry pane for live blitting inside Pygame."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from robot_sf.common.artifact_paths import get_artifact_category_path
from robot_sf.telemetry.visualization import (
    DEFAULT_TELEMETRY_METRICS,
    make_surface_from_rgba,
    render_metric_panel,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence
    from pathlib import Path

    import pygame


def _timestamp_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class TelemetryPane:
    """Holds telemetry history and renders a chart panel for blitting."""

    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_TELEMETRY_METRICS))
    max_points: int = 512
    width: int = 320
    height: int = 240
    refresh_hz: float = 1.0
    decimation: int = 1

    _history: MutableMapping[str, deque[float]] = field(init=False, default_factory=dict)
    _last_render_ms: int = field(init=False, default=0)
    _last_surface: pygame.Surface | None = field(
        init=False, default=None
    )  # Cached surface for display persistence

    def update(
        self, values: Mapping[str, float | int | None], frame_idx: int | None = None
    ) -> None:
        """Append metric values to history."""
        for metric in self.metrics:
            val = values.get(metric)
            hist = self._history.setdefault(metric, deque(maxlen=self.max_points))
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            try:
                hist.append(float(val))
            except (TypeError, ValueError):
                continue

    def render_surface(self):
        """Render the pane as a pygame.Surface, or return cached surface if throttled.

        Returns:
            pygame.Surface or None: Freshly rendered surface when due, else last cached surface.
        """
        now = _timestamp_ms()
        min_interval_ms = 1000.0 / max(self.refresh_hz, 0.1)
        if self._last_render_ms and now - self._last_render_ms < min_interval_ms:
            # Return cached surface if available for display persistence
            return self._last_surface
        self._last_render_ms = now
        series = {name: list(history) for name, history in self._history.items()}
        try:
            rgba = render_metric_panel(series, self.metrics, width=self.width, height=self.height)
        except (ValueError, RuntimeError) as exc:  # pragma: no cover - defensive
            logger.warning("Failed to render telemetry pane: {}", exc)
            return self._last_surface  # Return cached surface on error
        surface = make_surface_from_rgba(rgba)
        if surface is not None:
            self._last_surface = surface
        return surface


class TelemetrySession:
    """Append-only telemetry JSONL writer with in-memory history for the pane."""

    def __init__(
        self,
        *,
        run_id: str,
        record: bool,
        metrics: Sequence[str] | None,
        refresh_hz: float,
        decimation: int,
        pane_size: tuple[int, int] = (320, 240),
    ) -> None:
        """Initialize a telemetry session for live rendering and optional persistence."""
        self.run_id = run_id
        self.record = record
        self.decimation = max(decimation, 1)
        self._last_append_ms = 0
        self._warned_drops = 0
        pane_width, pane_height = pane_size
        self.pane = TelemetryPane(
            metrics=list(metrics) if metrics is not None else list(DEFAULT_TELEMETRY_METRICS),
            refresh_hz=refresh_hz,
            width=pane_width,
            height=pane_height,
            decimation=self.decimation,
        )
        root = get_artifact_category_path("telemetry")
        self.run_dir = root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.run_dir / "telemetry.jsonl"
        self._samples = 0

    def append(self, payload: dict) -> None:
        """Write a telemetry payload (if recording) and update pane history."""
        metrics = payload.get("metrics", {})
        frame_idx = payload.get("frame_idx")
        self.pane.update(metrics, frame_idx=frame_idx)
        self._samples += 1
        now_ms = _timestamp_ms()
        if self._last_append_ms > 0:
            interval_ms = now_ms - self._last_append_ms
            expected_ms = max(int(1000.0 / max(self.pane.refresh_hz, 0.1)), 100)
            if interval_ms > expected_ms * 3:
                self._warned_drops += 1
                logger.warning(
                    "Telemetry append lag detected: interval={}ms (expected <= {}ms), run_id={}, total_warnings={}",
                    interval_ms,
                    expected_ms,
                    self.run_id,
                    self._warned_drops,
                )
                payload.setdefault("notes", "lagged")
                self._write_health_snapshot()
        self._last_append_ms = now_ms
        if not self.record:
            return
        if self._samples % self.decimation != 0:
            return
        line = payload.copy()
        line["timestamp_ms"] = line.get("timestamp_ms", now_ms)
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{json.dumps(line)}\n")

    def render_surface(self):
        """Render the telemetry pane surface (may return None for throttling).

        Returns:
            pygame.Surface or None: Rendered surface, or None if throttled or pygame unavailable.
        """
        return self.pane.render_surface()

    def surface_size(self) -> tuple[int, int]:
        """Return pane dimensions.

        Returns:
            tuple[int, int]: Width and height of the telemetry pane in pixels.
        """
        return self.pane.width, self.pane.height

    def summary_paths(self) -> tuple[Path, ...]:
        """Return telemetry artifact paths emitted by this session.

        Returns:
            tuple[Path, ...]: Path(s) to telemetry artifacts (JSONL).
        """
        return (self.telemetry_path,)

    def write_summary(self) -> tuple[Path, ...]:
        """Write summary JSON and PNG for the recorded telemetry history.

        Returns:
            tuple[Path, ...]: Paths to summary artifacts (JSON and optional PNG).
        """

        self.run_dir.mkdir(parents=True, exist_ok=True)
        histories = {name: list(hist) for name, hist in self.pane._history.items()}
        summary = {
            "run_id": self.run_id,
            "samples": self._samples,
            "warned_drops": self._warned_drops,
            "metrics": {},
        }
        for name, values in histories.items():
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            summary["metrics"][name] = {
                "last": float(arr[-1]),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
            }
        json_path = self.run_dir / "telemetry_summary.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        png_path = self.run_dir / "telemetry_summary.png"
        try:
            rgba = render_metric_panel(
                histories,
                self.pane.metrics,
                width=self.pane.width,
                height=self.pane.height,
            )
            plt.imsave(png_path, rgba)
        except (OSError, ValueError, RuntimeError) as exc:  # pragma: no cover - defensive
            logger.warning("Failed to write telemetry summary image: {}", exc)
            return (json_path,)

        return (json_path, png_path)

    @property
    def warned_drops(self) -> int:
        """Number of lag/delay warnings emitted."""

        return self._warned_drops

    def _write_health_snapshot(self) -> None:
        """Persist a lightweight health summary alongside telemetry."""

        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            data = {"run_id": self.run_id, "warned_drops": self._warned_drops}
            (self.run_dir / "telemetry_health.json").write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError as exc:  # pragma: no cover - defensive
            logger.debug("Failed to write telemetry health snapshot: {}", exc)
