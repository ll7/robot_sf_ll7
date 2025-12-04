"""Observation mode options for Gym environments."""

from enum import Enum


class ObservationMode(str, Enum):
    """Selectable observation encodings for Gym environments."""

    DEFAULT_GYM = "default_gym"
    SOCNAV_STRUCT = "socnav_struct"
