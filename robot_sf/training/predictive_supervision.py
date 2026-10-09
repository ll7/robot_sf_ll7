"""Identity and provenance contract for predictive trajectory supervision.

The public SOCNAV observation deliberately contains no oracle identities. Predictive
collectors instead read observation-aligned simulator source indices through the
SocNavObservationFusion *private* side channel. The PySocialForce population uses
fixed rows for the episode; namespacing those indices at every reset prevents
cross-episode joins. Do not substitute observation-derived tracking IDs here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREDICTIVE_DATASET_SCHEMA = "predictive_planner_dataset_v2_identity"
COLLECTOR_VERSION = "predictive_collectors_identity_v2"
MATCHING_METHOD = "episode_scoped_simulator_source_slot"
IDENTITY_SOURCE = "socnav_observation_source_indices"
VELOCITY_FRAME = "robot_ego_xy_rotation_only"
CONTRACT_KEYS = (
    "dataset_schema",
    "collector_version",
    "matching_method",
    "identity_source",
    "velocity_coordinate_frame",
)


def supervision_metadata(collector_id: str) -> dict[str, Any]:
    """Version a corrected collector without impersonating immutable v1 artifacts."""
    if collector_id not in {"base", "hardcase", "mixed"}:
        raise ValueError(f"Unexpected predictive collector: {collector_id!r}")
    return {
        "dataset_schema": PREDICTIVE_DATASET_SCHEMA,
        "collector_version": COLLECTOR_VERSION,
        "collector_id": collector_id,
        "matching_method": MATCHING_METHOD,
        "identity_source": IDENTITY_SOURCE,
        "velocity_coordinate_frame": VELOCITY_FRAME,
        "pedestrian_id_scope": "reset_episode",
    }


def observation_episode_ids(env: Any, *, episode_id: str) -> tuple[str, ...]:
    """Return identity rows aligned with the *last emitted* SOCNAV observation.

    An absent or malformed producer side channel is an error, not permission to
    fall back to row index in presentation order or nearest-position inference.
    """
    if not episode_id:
        raise ValueError("Predictive collection requires a nonempty reset-episode ID")
    sensor = getattr(getattr(env, "state", None), "sensors", None)
    sensor = getattr(sensor, "wrapped_adapter", sensor)
    settings = getattr(getattr(sensor, "env_config", None), "observation_visibility", None)
    if bool(getattr(settings, "memory_for_lost_pedestrians", False)):
        raise ValueError(
            "Predictive identity collection requires observation memory for lost pedestrians "
            "to be disabled: remembered estimates are not observed target positions."
        )
    indices = getattr(sensor, "current_source_indices", None)
    if indices is None:
        raise ValueError(
            "Predictive collection requires SocNavObservationFusion.current_source_indices; "
            "cannot infer identity from observation position or row order."
        )
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices):
        raise ValueError(f"Invalid simulator pedestrian source indices: {indices!r}")
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate simulator pedestrian source indices in observation")
    simulator = getattr(env, "simulator", None)
    positions = getattr(simulator, "ped_pos", None)
    if positions is None or any(index >= len(positions) for index in indices):
        raise ValueError("Pedestrian source indices exceed the current simulator population")
    # Current simulator rows are fixed within an episode, unlike the sorted SOCNAV rows.
    return tuple(f"{episode_id}:simulator-slot-{index}" for index in indices)


def identity_match_indices(
    source_ids: tuple[str, ...],
    target_ids: tuple[str, ...],
) -> dict[int, int]:
    """Join by episode-scoped actor ID only; absent IDs have no future target."""
    for ids in (source_ids, target_ids):
        if any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("Missing or invalid pedestrian identity in predictive frame")
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate pedestrian identities in predictive frame")
    lookup = {identity: i for i, identity in enumerate(target_ids)}
    return {i: lookup[identity] for i, identity in enumerate(source_ids) if identity in lookup}


def _parse_embedded_metadata(raw: Any, *, path: Path) -> dict[str, Any] | None:
    if "supervision_metadata_json" not in raw:
        return None
    field = raw["supervision_metadata_json"]
    if getattr(field, "shape", ()) != ():
        raise ValueError(f"Non-scalar supervision_metadata_json in {path}")
    value = field.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"Invalid supervision_metadata_json type in {path}")
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid supervision_metadata_json in {path}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Predictive supervision metadata must be an object: {path}")
    return result


def validate_supervision_metadata(
    raw: Any,
    *,
    path: Path,
    allow_legacy: bool = False,
) -> dict[str, Any] | None:
    """Verify the NPZ and, when present, its sidecar. Legacy is opt-in only."""
    metadata = _parse_embedded_metadata(raw, path=path)
    if metadata is None:
        if allow_legacy:
            return None
        raise ValueError(
            f"Predictive dataset {path} has no supervision_metadata_json; "
            "unmarked/legacy rows are not identity-corrected. "
            "Use --allow-legacy-supervision only to reproduce historical training."
        )
    expected = supervision_metadata(str(metadata.get("collector_id", "")))
    for key in (*CONTRACT_KEYS, "pedestrian_id_scope"):
        if metadata.get(key) != expected[key]:
            raise ValueError(
                f"Predictive supervision contract mismatch in {path}: "
                f"{key}={metadata.get(key)!r}, expected {expected[key]!r}"
            )
    if metadata["collector_id"] == "mixed":
        sources = metadata.get("source_collectors")
        if not isinstance(sources, list) or len(sources) != 2 or any(
            source not in {"base", "hardcase"} for source in sources
        ):
            raise ValueError(f"Invalid mixed predictive source_collectors in {path}")
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid predictive dataset manifest {manifest_path}") from exc
        if not isinstance(manifest, dict) or manifest.get("supervision_metadata") != metadata:
            raise ValueError(
                f"Predictive dataset manifest metadata differs from embedded NPZ: {manifest_path}"
            )
    return metadata


def compatible_mixed_supervision(
    base: dict[str, Any] | None,
    hardcase: dict[str, Any] | None,
    *,
    allow_legacy: bool = False,
) -> dict[str, Any] | None:
    """Refuse legacy/corrected mixing even with historical opt-in enabled."""
    if base is None or hardcase is None:
        if base is None and hardcase is None and allow_legacy:
            return None
        raise ValueError(
            "Incompatible predictive supervision: mixed input has missing/legacy "
            "identity metadata. Corrected and legacy rows must never be mixed."
        )
    for key in (*CONTRACT_KEYS, "pedestrian_id_scope"):
        if base.get(key) != hardcase.get(key):
            raise ValueError(
                f"Incompatible predictive supervision {key}: "
                f"base={base.get(key)!r}, hardcase={hardcase.get(key)!r}"
            )
    if base.get("collector_id") == "mixed" or hardcase.get("collector_id") == "mixed":
        raise ValueError("Mixed predictive datasets cannot be used as input components")
    result = supervision_metadata("mixed")
    result["source_collectors"] = [base["collector_id"], hardcase["collector_id"]]
    return result
