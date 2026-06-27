"""Provenance for metric-affecting run configuration (issue #3701).

Some run-configuration toggles change *what the reported benchmark metrics mean*
rather than just the numbers. Two result sets are only comparable when these
toggles match, yet they are not part of the emitted results, so a reader of a
results table cannot tell which setting produced the numbers.

This module provides a small, pure serialization helper that records those
toggles so each result artifact is self-describing. Two metric-affecting
settings are captured:

* **LiDAR ``scan_noise``** -- sensor observation noise probabilities
  ``[loss, corruption]`` (see ``robot_sf/sensor/range_sensor.py``). The default
  is noisy (``[0.005, 0.002]``); deterministic/reproduction configs set
  ``[0.0, 0.0]``. Noise-free vs noisy observations change collision and
  near-miss outcomes.
* **Collision-handling regime** -- ``terminate_on_contact`` vs ``bounce_back``.
  The robot benchmark environment ends an episode on collision
  (``RobotState.is_terminal`` in ``robot_sf/robot/robot_state.py`` includes the
  collision flags), so a collision is an episode-ending failure rather than a
  recoverable event. The collision-force toggles
  (``peds_have_static_obstacle_forces``, ``peds_have_robot_repulsion``) are
  recorded alongside the regime because they also change collision outcomes.

The output is a JSON-safe dictionary intended for embedding in a benchmark run
manifest or episode metadata. This module deliberately depends only on the
standard library so it can be imported without simulator dependencies.
"""

from __future__ import annotations

from typing import Any

#: Collision-handling regime where any collision ends the episode (the robot
#: benchmark contract). See ``robot_sf/robot/robot_state.py``.
COLLISION_REGIME_TERMINATE_ON_CONTACT = "terminate_on_contact"
#: Collision-handling regime where a collision is recoverable and the episode
#: continues (e.g. bounce-back / respawn behavior).
COLLISION_REGIME_BOUNCE_BACK = "bounce_back"
#: All recognized collision-handling regimes.
COLLISION_REGIMES = (
    COLLISION_REGIME_TERMINATE_ON_CONTACT,
    COLLISION_REGIME_BOUNCE_BACK,
)

#: Human-readable note clarifying how this block should be interpreted.
METRIC_AFFECTING_CONFIG_INTERPRETATION = (
    "metric_affecting_run_config: run settings that change what reported metrics mean; "
    "two result sets are only directly comparable when these settings match"
)

#: Schema tag for the emitted block, so downstream readers can version it.
METRIC_AFFECTING_CONFIG_SCHEMA = "metric_affecting_run_config.v1"


def _coerce_scan_noise(value: Any) -> list[float] | None:
    """Coerce ``value`` into a JSON-safe list of floats.

    Accepts any sequence of numbers (e.g. a list or a read-only numpy array such
    as ``LidarScannerSettings.scan_noise``).

    Returns:
        A list of floats, or ``None`` when the value is missing or cannot be
        interpreted as a sequence of numbers.
    """
    if value is None:
        return None
    try:
        items = list(value)
    except TypeError:
        return None
    coerced: list[float] = []
    for item in items:
        try:
            coerced.append(float(item))
        except (TypeError, ValueError):
            return None
    return coerced


def _resolve_scan_noise(config: Any) -> list[float] | None:
    """Resolve the LiDAR ``scan_noise`` probabilities from a config-like object.

    Reads ``config.lidar_config.scan_noise`` via duck typing so the helper works
    on any environment config exposing that attribute.

    Returns:
        The scan-noise probabilities as a list of floats, or ``None`` when the
        attribute chain is absent or unparseable.
    """
    lidar_config = getattr(config, "lidar_config", None)
    if lidar_config is None:
        return None
    return _coerce_scan_noise(getattr(lidar_config, "scan_noise", None))


def metric_affecting_run_config(
    config: Any,
    *,
    collision_regime: str = COLLISION_REGIME_TERMINATE_ON_CONTACT,
) -> dict[str, Any]:
    """Extract a JSON-safe metric-affecting run-config provenance block.

    Args:
        config: An environment config-like object. The function reads
            ``config.lidar_config.scan_noise`` and the collision-force toggles
            (``peds_have_static_obstacle_forces``, ``peds_have_robot_repulsion``)
            via duck typing, so any object exposing those attributes works.
            Missing attributes are recorded as ``None`` rather than raising.
        collision_regime: The collision-handling regime in effect for the run.
            Defaults to ``terminate_on_contact``, which reflects the robot
            benchmark environment contract (collisions end the episode).

    Returns:
        A dictionary describing the metric-affecting settings, suitable for
        embedding in a benchmark run manifest or episode metadata.

    Raises:
        ValueError: If ``collision_regime`` is not one of ``COLLISION_REGIMES``.
    """
    if collision_regime not in COLLISION_REGIMES:
        raise ValueError(
            f"collision_regime must be one of {COLLISION_REGIMES!r} (got {collision_regime!r})"
        )

    scan_noise = _resolve_scan_noise(config)
    scan_noise_enabled: bool | None
    if scan_noise is None:
        scan_noise_enabled = None
    else:
        scan_noise_enabled = any(prob > 0.0 for prob in scan_noise)

    static_obstacle_forces = getattr(config, "peds_have_static_obstacle_forces", None)
    robot_repulsion = getattr(config, "peds_have_robot_repulsion", None)

    return {
        "schema": METRIC_AFFECTING_CONFIG_SCHEMA,
        "sensor_noise": {
            "scan_noise": scan_noise,
            "scan_noise_enabled": scan_noise_enabled,
        },
        "collision_regime": {
            "regime": collision_regime,
            "peds_have_static_obstacle_forces": (
                None if static_obstacle_forces is None else bool(static_obstacle_forces)
            ),
            "peds_have_robot_repulsion": (
                None if robot_repulsion is None else bool(robot_repulsion)
            ),
        },
        "interpretation": METRIC_AFFECTING_CONFIG_INTERPRETATION,
    }
