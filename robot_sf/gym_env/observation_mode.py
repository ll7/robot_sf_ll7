"""Observation mode options for Gym environments."""

from enum import StrEnum

__all__ = ["ObservationMode"]


class ObservationMode(StrEnum):
    """Selectable observation encodings for Gym environments."""

    DEFAULT_GYM = "default_gym"
    SOCNAV_STRUCT = "socnav_struct"
    SOCIAL_GRAPH = "social_graph"
