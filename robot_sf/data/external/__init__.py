"""License-safe accessors for bring-your-own external datasets."""

from robot_sf.data.external.sdd_trajectories import (
    SDD_TRAJECTORY_SCHEMA_VERSION,
    SddTrajectoryDataError,
    SddTrajectoryTrack,
    SddTrajectoryTrackSet,
    load_sdd_track_set,
    parse_sdd_annotations,
)

__all__ = [
    "SDD_TRAJECTORY_SCHEMA_VERSION",
    "SddTrajectoryDataError",
    "SddTrajectoryTrack",
    "SddTrajectoryTrackSet",
    "load_sdd_track_set",
    "parse_sdd_annotations",
]
