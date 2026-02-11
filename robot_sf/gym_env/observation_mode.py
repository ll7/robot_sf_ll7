"""Observation mode options for Gym environments."""

from enum import StrEnum


class ObservationMode(StrEnum):
    """Selectable observation encodings for Gym environments."""

    DEFAULT_GYM = "default_gym"
    SOCNAV_STRUCT = "socnav_struct"
