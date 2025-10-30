"""Sensor package initialization.

Registers built-in sensors automatically on import.
"""

from robot_sf.sensor.dummy_constant import factory as dummy_constant_factory
from robot_sf.sensor.registry import register_sensor

# Register built-in sensors
register_sensor("dummy_constant", dummy_constant_factory, override=False)

__all__ = ["dummy_constant_factory", "register_sensor"]
