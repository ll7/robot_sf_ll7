"""Telemetry configuration mixin shared across environment config dataclasses."""

from dataclasses import dataclass, field

from robot_sf.telemetry import DEFAULT_TELEMETRY_METRICS

_TELEMETRY_PANE_LAYOUTS = {"vertical_split", "horizontal_split"}


@dataclass(kw_only=True)
class TelemetryConfigMixin:
    """Shared telemetry configuration for environment configs."""

    enable_telemetry_panel: bool = False
    telemetry_record: bool = False
    telemetry_metrics: list[str] = field(
        default_factory=lambda: list(DEFAULT_TELEMETRY_METRICS),
    )
    telemetry_refresh_hz: float = 1.0
    telemetry_pane_layout: str = "vertical_split"
    telemetry_decimation: int = 1

    def _validate_telemetry(self) -> None:
        """Validate and normalize telemetry options."""
        if self.telemetry_refresh_hz <= 0:
            raise ValueError("telemetry_refresh_hz must be > 0")
        if self.telemetry_decimation <= 0:
            raise ValueError("telemetry_decimation must be >= 1")
        if self.telemetry_pane_layout not in _TELEMETRY_PANE_LAYOUTS:
            raise ValueError(
                "telemetry_pane_layout must be 'vertical_split' or 'horizontal_split'",
            )

        self.telemetry_metrics = [
            metric
            for metric in (self.telemetry_metrics or [])
            if isinstance(metric, str) and metric.strip()
        ]
        if not self.telemetry_metrics:
            self.telemetry_metrics = list(DEFAULT_TELEMETRY_METRICS)
