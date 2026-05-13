"""CARLA bridge helpers that remain import-safe without CARLA installed."""

from robot_sf.carla_bridge.parity import (
    DEFAULT_PARITY_METRICS,
    MetricParityRow,
    compare_oracle_replay_metrics,
)

__all__ = ["DEFAULT_PARITY_METRICS", "MetricParityRow", "compare_oracle_replay_metrics"]
