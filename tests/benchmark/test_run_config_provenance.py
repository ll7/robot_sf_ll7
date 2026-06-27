"""Serialization tests for metric-affecting run-config provenance (issue #3701).

These tests use synthetic config fixtures only; they do not start the simulator,
load maps, or run any benchmark campaign.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark.map_runner_env import representative_metric_affecting_config
from robot_sf.benchmark.run_config_provenance import (
    COLLISION_REGIME_BOUNCE_BACK,
    COLLISION_REGIME_TERMINATE_ON_CONTACT,
    METRIC_AFFECTING_CONFIG_SCHEMA,
    metric_affecting_run_config,
)


def _fake_config(scan_noise, *, static_forces=True, robot_repulsion=None) -> SimpleNamespace:
    """Build a minimal duck-typed env config fixture."""
    return SimpleNamespace(
        lidar_config=SimpleNamespace(scan_noise=scan_noise),
        peds_have_static_obstacle_forces=static_forces,
        peds_have_robot_repulsion=robot_repulsion,
    )


def test_records_default_noisy_sensor_config() -> None:
    """A default noisy sensor config is recorded as noise-enabled."""
    block = metric_affecting_run_config(_fake_config([0.005, 0.002]))

    assert block["schema"] == METRIC_AFFECTING_CONFIG_SCHEMA
    assert block["sensor_noise"]["scan_noise"] == [0.005, 0.002]
    assert block["sensor_noise"]["scan_noise_enabled"] is True
    assert block["collision_regime"]["regime"] == COLLISION_REGIME_TERMINATE_ON_CONTACT
    assert block["collision_regime"]["peds_have_static_obstacle_forces"] is True
    assert block["collision_regime"]["peds_have_robot_repulsion"] is None
    assert "interpretation" in block


def test_records_deterministic_noise_free_config() -> None:
    """A deterministic ``[0.0, 0.0]`` sensor config is recorded as noise-disabled."""
    block = metric_affecting_run_config(_fake_config([0.0, 0.0]))

    assert block["sensor_noise"]["scan_noise"] == [0.0, 0.0]
    assert block["sensor_noise"]["scan_noise_enabled"] is False


def test_noisy_and_deterministic_blocks_differ() -> None:
    """The recorded provenance distinguishes noisy from noise-free runs."""
    noisy = metric_affecting_run_config(_fake_config([0.005, 0.002]))
    deterministic = metric_affecting_run_config(_fake_config([0.0, 0.0]))

    assert noisy["sensor_noise"] != deterministic["sensor_noise"]


def test_coerces_numpy_scan_noise_array() -> None:
    """A read-only numpy ``scan_noise`` array (as LidarScannerSettings exposes) is JSON-safe."""
    arr = np.array([0.01, 0.0], dtype=np.float64)
    arr.flags.writeable = False
    block = metric_affecting_run_config(_fake_config(arr))

    assert block["sensor_noise"]["scan_noise"] == [0.01, 0.0]
    assert block["sensor_noise"]["scan_noise_enabled"] is True
    # The block must round-trip through JSON without custom encoders.
    assert json.loads(json.dumps(block)) == block


def test_real_lidar_scanner_settings_default() -> None:
    """The helper reads ``scan_noise`` from a real LidarScannerSettings object."""
    from robot_sf.sensor.range_sensor import LidarScannerSettings

    config = SimpleNamespace(lidar_config=LidarScannerSettings())
    block = metric_affecting_run_config(config)

    assert block["sensor_noise"]["scan_noise"] == [0.005, 0.002]
    assert block["sensor_noise"]["scan_noise_enabled"] is True


def test_bounce_back_regime_is_recorded() -> None:
    """A bounce-back regime is recorded distinctly from terminate-on-contact."""
    block = metric_affecting_run_config(
        _fake_config([0.0, 0.0]),
        collision_regime=COLLISION_REGIME_BOUNCE_BACK,
    )

    assert block["collision_regime"]["regime"] == COLLISION_REGIME_BOUNCE_BACK


def test_invalid_collision_regime_raises() -> None:
    """An unknown collision regime is rejected rather than silently recorded."""
    with pytest.raises(ValueError, match="collision_regime"):
        metric_affecting_run_config(_fake_config([0.0, 0.0]), collision_regime="warp")


def test_missing_lidar_config_records_none() -> None:
    """A config without a lidar_config records None rather than raising."""
    block = metric_affecting_run_config(SimpleNamespace())

    assert block["sensor_noise"]["scan_noise"] is None
    assert block["sensor_noise"]["scan_noise_enabled"] is None
    assert block["collision_regime"]["peds_have_static_obstacle_forces"] is None


def test_unparseable_scan_noise_records_none() -> None:
    """A non-sequence scan_noise value degrades to None instead of raising."""
    block = metric_affecting_run_config(_fake_config("noisy"))

    assert block["sensor_noise"]["scan_noise"] is None
    assert block["sensor_noise"]["scan_noise_enabled"] is None


def test_representative_config_without_scenarios_is_not_available() -> None:
    """The batch helper is fail-soft when there are no scenarios (no simulator needed)."""
    block = representative_metric_affecting_config([], scenario_path=Path("."))

    assert block == {"status": "not_available", "reason": "no scenarios"}
