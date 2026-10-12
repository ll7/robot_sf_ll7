"""Identity and provenance contract for predictive trajectory supervision.

The public SOCNAV observation deliberately contains no oracle identities. Predictive
collectors instead read observation-aligned simulator source indices through the
SocNavObservationFusion *private* side channel. The PySocialForce population uses
fixed rows for the episode; namespacing those indices at every reset prevents
cross-episode joins. Do not substitute observation-derived tracking IDs here.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

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
    """Version a corrected collector without impersonating immutable v1 artifacts.

    Returns:
        The versioned supervision contract mapping for this collector.
    """
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


def observation_episode_ids(  # noqa: C901
    env: Any, *, episode_id: str, observation: dict | None = None
) -> tuple[str, ...]:
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
    if observation is not None:
        block = observation.get("pedestrians", observation)
        rows = np.asarray(block.get("positions", block.get("pedestrians_positions", [])))
        count = np.asarray(block.get("count", block.get("pedestrians_count", []))).reshape(-1)
        if count.size != 1 or count[0] != len(indices) or rows.ndim != 2:
            raise ValueError("Observation/source identity count mismatch")
        expected = np.asarray(positions, dtype=np.float32)[list(indices)]
        position_cap = getattr(sensor, "_position_cap", None)
        if not callable(position_cap):
            raise ValueError("Predictive observation identity requires the sensor position cap")
        cap = position_cap()
        expected = np.clip(expected, 0.0, cap)
        if not np.array_equal(rows[: len(indices)], expected):
            raise ValueError("Borrowed or stale observation/source pedestrian identity alignment")
    # Current simulator rows are fixed within an episode, unlike the sorted SOCNAV rows.
    # A route respawn teleports the row to a new start, so it gets a new identity.
    epochs = _respawn_epochs(simulator)
    return tuple(
        f"{episode_id}:simulator-slot-{index}"
        + (f":respawn-{epochs[index]}" if epochs.get(index, 0) else "")
        for index in indices
    )


def _respawn_epochs(simulator: Any) -> dict[int, int]:
    """Collect per-simulator-row route respawn counts from the pedestrian behaviors.

    Returns:
        Mapping from simulator pedestrian row to respawns this episode (absent means 0).
    """
    epochs: dict[int, int] = {}
    for behavior in getattr(simulator, "peds_behaviors", ()) or ():
        offset = int(getattr(behavior, "global_ped_offset", 0) or 0)
        counts = getattr(behavior, "respawn_epochs", None)
        if isinstance(counts, dict):
            for local_id, count in counts.items():
                epochs[offset + int(local_id)] = int(count)
    return epochs


def identity_match_indices(
    source_ids: tuple[str, ...],
    target_ids: tuple[str, ...],
) -> dict[int, int]:
    """Join by episode-scoped actor ID only; absent IDs have no future target.

    Returns:
        Mapping from source row to target row for identities present in both frames.
    """
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


def validate_supervision_metadata(  # noqa: C901
    raw: Any,
    *,
    path: Path,
    allow_legacy: bool = False,
) -> dict[str, Any] | None:
    """Verify the NPZ and, when present, its sidecar. Legacy is opt-in only.

    Returns:
        The validated metadata, or None for an explicitly allowed legacy dataset.
    """
    metadata = _parse_embedded_metadata(raw, path=path)
    if metadata is None:
        if allow_legacy:
            return None
        raise ValueError(
            f"Predictive dataset {path} has no supervision_metadata_json; "
            "unmarked/legacy rows are not identity-corrected. "
            "Use --allow-legacy-supervision only to reproduce historical training."
        )
    required = {"state", "target", "mask", "target_mask"}
    missing = required - set(raw.files)
    if missing:
        raise ValueError(
            f"Identity-corrected predictive dataset {path} requires explicit state, target, "
            f"mask and target_mask; missing: {sorted(missing)}"
        )
    target = raw["target"]
    state = raw["state"]
    mask = raw["mask"]
    target_mask = raw["target_mask"]
    if (
        state.ndim != 3
        or target.ndim != 4
        or target.shape[-1] != 2
        or mask.shape != state.shape[:2]
        or target.shape[:2] != state.shape[:2]
        or target_mask.shape != target.shape[:3]
    ):
        raise ValueError(
            f"Malformed identity-corrected predictive supervision array shapes: {path}"
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
        if (
            not isinstance(sources, list)
            or len(sources) != 2
            or any(source not in {"base", "hardcase"} for source in sources)
        ):
            raise ValueError(f"Invalid mixed predictive source_collectors in {path}")
    sample_schema = metadata.get("sample_identity_schema")
    if sample_schema is None:
        warnings.warn(
            "Historical identity-v2 dataset lacks checkable sample scenario/episode/seed/step "
            "provenance; it is not sample-identity verified.",
            stacklevel=2,
        )
    elif sample_schema != SAMPLE_IDENTITY_SCHEMA:
        raise ValueError(f"Unsupported predictive sample identity schema in {path}")
    else:
        validate_sample_identities(raw)
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
    """Refuse legacy/corrected mixing even with historical opt-in enabled.

    Returns:
        The mixed-dataset metadata, or None when both inputs are explicitly legacy.
    """
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
    if base.get("sample_identity_schema") != hardcase.get("sample_identity_schema"):
        raise ValueError(
            "Incompatible predictive sample identity provenance: never mix marked/unmarked rows"
        )
    result = supervision_metadata("mixed")
    if base.get("sample_identity_schema") is not None:
        result["sample_identity_schema"] = base["sample_identity_schema"]
    result["source_collectors"] = [base["collector_id"], hardcase["collector_id"]]
    return result


SAMPLE_IDENTITY_SCHEMA = "predictive_sample_identity_v1"
SAMPLE_IDENTITY_KEYS = (
    "sample_scenario_id",
    "sample_episode_id",
    "sample_seed",
    "sample_step",
    "sample_pedestrian_id",
    "sample_target_pedestrian_id",
    "sample_target_step",
)


def collection_metadata(collector_id: str) -> dict[str, Any]:
    """Mark newly collected rows without relabelling historical identity-v2 data.

    Returns:
        Supervision metadata with the independently versioned sample identity marker.
    """
    return {**supervision_metadata(collector_id), "sample_identity_schema": SAMPLE_IDENTITY_SCHEMA}


def sample_identity_arrays(
    frames: list[Any],
    *,
    scenario_id: str,
    episode_id: str,
    seed: int,
    mask: Any,
    target_mask: Any,
) -> dict[str, Any]:
    """Serialize the source/future actor IDs alongside the exact supervision rows.

    Returns:
        NPZ arrays aligned with the source masks and each target horizon.
    """
    n, agents = mask.shape
    horizon = target_mask.shape[-1]
    identities = [[*frame.ped_ids, *([""] * (agents - len(frame.ped_ids)))] for frame in frames]
    width = max((len(value) for row in identities for value in row), default=1)
    source = np.asarray(identities[:n], dtype=f"U{max(width, 1)}").reshape(n, agents)
    future = np.full((n, agents, horizon), "", dtype=source.dtype)
    for t in range(n):
        for k in range(horizon):
            matches = identity_match_indices(frames[t].ped_ids, frames[t + k + 1].ped_ids)
            for i, j in matches.items():
                if target_mask[t, i, k]:
                    future[t, i, k] = frames[t + k + 1].ped_ids[j]
    step = np.arange(n, dtype=np.int64)
    result = {
        "sample_scenario_id": np.full(n, scenario_id),
        "sample_episode_id": np.full(n, episode_id),
        "sample_seed": np.full(n, seed, dtype=np.int64),
        "sample_step": step,
        "sample_pedestrian_id": source,
        "sample_target_pedestrian_id": future,
        "sample_target_step": step[:, None] + np.arange(1, horizon + 1, dtype=np.int64),
    }
    validate_sample_identities({**result, "mask": mask, "target_mask": target_mask})
    return result


def validate_sample_identities(raw: Any) -> None:  # noqa: C901, PLR0912
    """Reject missing, ambiguous, padded or borrowed persisted source/target identities."""
    missing = set(SAMPLE_IDENTITY_KEYS) - set(raw)
    if missing:
        raise ValueError(f"Missing predictive sample identity fields: {sorted(missing)}")
    payload = raw
    raw = {key: payload[key] for key in (*SAMPLE_IDENTITY_KEYS, "mask", "target_mask")}
    mask, target_mask = np.asarray(raw["mask"]), np.asarray(raw["target_mask"])
    n, agents = mask.shape
    horizon = target_mask.shape[-1]
    shapes = {
        "sample_scenario_id": (n,),
        "sample_episode_id": (n,),
        "sample_seed": (n,),
        "sample_step": (n,),
        "sample_pedestrian_id": (n, agents),
        "sample_target_pedestrian_id": (n, agents, horizon),
        "sample_target_step": (n, horizon),
    }
    for key, shape in shapes.items():
        value = np.asarray(raw[key])
        if value.shape != shape:
            raise ValueError(f"Malformed predictive sample identity shape: {key}")
        if key in {"sample_seed", "sample_step", "sample_target_step"}:
            if value.dtype.kind not in "iu" or np.any(value < 0):
                raise ValueError(f"Invalid predictive sample identity integer: {key}")
        elif value.dtype.kind != "U":
            raise ValueError(f"Invalid predictive sample identity string: {key}")
    if not np.all(np.isin(mask, [0, 1])) or not np.all(np.isin(target_mask, [0, 1])):
        raise ValueError("Predictive identity masks must be binary")
    active, future_active = mask.astype(bool), target_mask.astype(bool)
    source, future = raw["sample_pedestrian_id"], raw["sample_target_pedestrian_id"]
    if np.any(future_active & ~active[..., None]):
        raise ValueError("Predictive target identity belongs to a padded source")
    if np.any(source[active] == "") or np.any(source[~active] != ""):
        raise ValueError("Missing or padded predictive source identity")
    if np.any(future[~future_active] != ""):
        raise ValueError("Masked predictive targets must not carry borrowed identities")
    expected = np.broadcast_to(source[..., None], future.shape)
    if np.any(future[future_active] != expected[future_active]):
        raise ValueError("Borrowed predictive target pedestrian identity")
    expected_steps = raw["sample_step"][:, None] + np.arange(1, horizon + 1)
    if not np.array_equal(raw["sample_target_step"], expected_steps):
        raise ValueError("Predictive target steps do not follow source steps")
    _validate_scenario_definitions(raw["sample_scenario_id"], payload)
    episodes: dict[str, tuple[str, int]] = {}
    samples: dict[tuple[str, int], int] = {}
    for i in range(n):
        episode = str(raw["sample_episode_id"][i])
        scenario = str(raw["sample_scenario_id"][i])
        if not episode or not scenario:
            raise ValueError("Missing predictive scenario or episode identity")
        identity = (scenario, int(raw["sample_seed"][i]))
        if episodes.setdefault(episode, identity) != identity:
            raise ValueError("Ambiguous predictive episode scenario/seed association")
        sample_key = (episode, int(raw["sample_step"][i]))
        previous = samples.setdefault(sample_key, i)
        # Oversampling can repeat a sample, but cannot attach different data to its key.
        for key in (
            ()
            if previous == i
            else (
                "sample_pedestrian_id",
                "sample_target_pedestrian_id",
                "mask",
                "target_mask",
                "state",
                "target",
            )
        ):
            if key not in raw and key in payload:
                raw[key] = payload[key]
            if key in raw and not np.array_equal(raw[key][i], raw[key][previous]):
                raise ValueError("Ambiguous predictive sample identity reused with different rows")
        ids = source[i, active[i]].tolist()
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate predictive sample pedestrian identity")
        if any(
            re.fullmatch(re.escape(episode) + r":simulator-slot-\d+(?::respawn-[1-9]\d*)?", value)
            is None
            for value in ids
        ):
            raise ValueError("Borrowed predictive source episode identity")


def _validate_scenario_definitions(scenarios: Any, payload: Any) -> None:
    """Verify the preserved preimage of each base scenario configuration digest."""
    for scenario in np.unique(scenarios):
        if str(scenario).startswith("base-config-"):
            if "scenario_definitions_json" not in payload:
                raise ValueError("Missing predictive scenario identity definition")
            definitions = json.loads(payload["scenario_definitions_json"].item())
            definition = definitions.get(str(scenario))
            digest = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
            if definition is None or str(scenario) != f"base-config-{digest}":
                raise ValueError("Borrowed predictive scenario identity definition")
