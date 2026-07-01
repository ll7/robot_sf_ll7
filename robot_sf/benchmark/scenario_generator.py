"""Deterministic scenario generator for the Social Navigation Benchmark.

This module converts high-level scenario parameter dictionaries into
`pysocialforce.Simulator` instances (or raw state + obstacles) in a
deterministic fashion (seeded), ensuring reproducibility.

Scenario parameter keys (expected):
    id: str (scenario id)
    density: one of {"low","med","high"}
    flow: one of {"uni","bi","cross","merge"}
    obstacle: one of {"open","bottleneck","maze"}
    groups: float fraction in {0.0,0.2,0.4}
    speed_var: {"low","high"}
    goal_topology: {"point","swap","circulate"}
    robot_context: {"ahead","behind","embedded"}
    repeats: int (ignored here, used by runner)

Returned structure:
    {
        "simulator": pysocialforce.Simulator,
        "state": np.ndarray shape (N,7),
        "obstacles": list[tuple[float,float,float,float]],
        "groups": list[int] (group id per agent, -1 if none),
        "metadata": { original params + derived fields }
    }

Notes:
 - We keep geometry simple initially (rectangular area 10m x 6m).
 - Densities map to approximate counts: low=10, med=25, high=40.
 - Obstacles layouts are coarse placeholders (can be evolved later).
 - Group assignment is random among eligible fraction, groups of size 2-4.
 - All randomness uses a local numpy Generator seeded with `seed`.
 - Initial velocities set to zero; policies/env will update.
 - Desired relaxation time (tau) set to 1.0 placeholder.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # Optional heavy import delayed until needed
    import pysocialforce as pysf
except ImportError:  # pragma: no cover - allow import failure during docs builds
    pysf = None  # type: ignore


AREA_WIDTH = 10.0
AREA_HEIGHT = 6.0

SCENARIO_GENERATION_PARAMS_SCHEMA_VERSION = "scenario_generation_params.v1"
SCENARIO_INITIAL_DIFFICULTY_SCHEMA_VERSION = "scenario_initial_difficulty.v1"
PARAMETERIZED_SCENARIO_PARAMS_SCHEMA_VERSION = "parameterized_scenario_params.v1"

GENERATION_PARAMETER_DEFAULTS = {
    "density": "med",
    "flow": "uni",
    "obstacle": "open",
    "groups": 0.0,
    "speed_var": "low",
    "goal_topology": "point",
    "robot_context": "embedded",
    "repeats": 1,
}

_KNOWN_GENERATION_PARAM_KEYS = frozenset(
    {
        "density",
        "flow",
        "obstacle",
        "groups",
        "speed_var",
        "goal_topology",
        "robot_context",
        "repeats",
        "id",
        "preset",
    }
)
_DENSITY_OPTIONS = ("low", "med", "high")
_DENSITY_ALIASES = {"medium": "med"}
_FLOW_OPTIONS = ("uni", "bi", "cross", "merge")
_FLOW_ALIASES = {
    "unidirectional": "uni",
    "bidirectional": "bi",
    "head_on": "bi",
    "head-on": "bi",
    "crossing": "cross",
    "bidirectional_crossing": "cross",
}
_OBSTACLE_OPTIONS = ("open", "bottleneck", "maze")
_SPEED_VARIATION_OPTIONS = ("low", "high")
_SPEED_VARIATION_ALIASES = {"med": "high", "medium": "high"}
_GOAL_TOPOLOGY_OPTIONS = ("point", "swap", "circulate")
_ROBOT_CONTEXT_OPTIONS = ("ahead", "behind", "embedded")

PARAMETERIZED_SCENARIO_PARAMETER_DEFAULTS = {
    "sidewalk_width": 4.0,
    "obstacle_density": 0.0,
    "pedestrian_density": 0.06,
    "bottleneck_width": 2.0,
    "crossing_angle": 90.0,
    "occlusion_probability": 0.0,
}

_DIFFICULTY_WEIGHT = {
    "density": {"low": 0.18, "med": 0.42, "high": 0.72},
    "flow": {"uni": 0.14, "bi": 0.20, "cross": 0.30, "merge": 0.24},
    "obstacle": {"open": 0.00, "bottleneck": 0.10, "maze": 0.14},
    "speed_var": {"low": 0.00, "high": 0.06},
    "goal_topology": {"point": 0.00, "swap": 0.04, "circulate": 0.07},
}


def normalize_generation_parameters(  # noqa: C901, PLR0912
    raw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize and validate generation inputs.

    Returns:
        Deterministic generation parameters with only supported keys.

    Raises:
        ValueError: for unknown keys, invalid scalars, unsupported enums, or empty strings.
    """

    if raw is None:
        return dict(GENERATION_PARAMETER_DEFAULTS)
    if not isinstance(raw, Mapping):
        raise ValueError("Generation parameters must be provided as a mapping.")

    generation_profile = raw.get("generation_profile")
    strict_profile = isinstance(generation_profile, Mapping)
    if strict_profile and "parameters" in generation_profile:
        schema_version = generation_profile.get("schema_version")
        if schema_version != SCENARIO_GENERATION_PARAMS_SCHEMA_VERSION:
            raise ValueError(
                "generation_profile.schema_version must equal "
                f"{SCENARIO_GENERATION_PARAMS_SCHEMA_VERSION}"
            )
        parameters = generation_profile.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("generation_profile.parameters must be a mapping.")
        source: Mapping[str, Any] = parameters
    else:
        source = generation_profile if strict_profile else raw

    payload: dict[str, Any] = dict(GENERATION_PARAMETER_DEFAULTS)
    for key, value in source.items():
        if key in _KNOWN_GENERATION_PARAM_KEYS:
            if value is None:
                value = GENERATION_PARAMETER_DEFAULTS.get(key)
            payload[key] = value
            continue
        if strict_profile:
            raise ValueError(f"Unsupported generation parameter: {key}")

    density = _DENSITY_ALIASES.get(str(payload["density"]).strip(), str(payload["density"]).strip())
    if density not in _DENSITY_OPTIONS:
        raise ValueError(
            f"Unsupported density '{payload['density']}', expected one of {_DENSITY_OPTIONS}"
        )
    flow = _FLOW_ALIASES.get(str(payload["flow"]).strip(), str(payload["flow"]).strip())
    if flow not in _FLOW_OPTIONS:
        raise ValueError(f"Unsupported flow '{payload['flow']}', expected one of {_FLOW_OPTIONS}")
    obstacle = str(payload["obstacle"]).strip()
    if obstacle not in _OBSTACLE_OPTIONS:
        raise ValueError(
            f"Unsupported obstacle '{payload['obstacle']}', expected one of {_OBSTACLE_OPTIONS}"
        )
    speed_var = _SPEED_VARIATION_ALIASES.get(
        str(payload["speed_var"]).strip(),
        str(payload["speed_var"]).strip(),
    )
    if speed_var not in _SPEED_VARIATION_OPTIONS:
        raise ValueError(
            f"Unsupported speed_var '{payload['speed_var']}', expected one of {_SPEED_VARIATION_OPTIONS}"
        )
    goal_topology = str(payload["goal_topology"]).strip()
    if goal_topology not in _GOAL_TOPOLOGY_OPTIONS:
        raise ValueError(
            f"Unsupported goal_topology '{payload['goal_topology']}', expected one of {_GOAL_TOPOLOGY_OPTIONS}"
        )
    robot_context = str(payload["robot_context"]).strip()
    if robot_context not in _ROBOT_CONTEXT_OPTIONS:
        raise ValueError(
            f"Unsupported robot_context '{payload['robot_context']}', expected one of {_ROBOT_CONTEXT_OPTIONS}"
        )

    try:
        repeats = int(payload["repeats"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"repeats must be a non-negative integer: {payload['repeats']}") from exc
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1: {payload['repeats']}")

    group_value = float(payload["groups"])
    if group_value < 0 or group_value > 1:
        raise ValueError(f"groups must be between 0.0 and 1.0 inclusive: {payload['groups']}")
    preset = payload.get("preset")
    if preset is not None and (not isinstance(preset, str) or not preset.strip()):
        raise ValueError("preset must be an optional non-empty string when provided")
    scenario_id = payload.get("id")
    if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id.strip()):
        raise ValueError("id must be an optional non-empty string when provided")

    return {
        "id": scenario_id,
        "density": density,
        "flow": flow,
        "obstacle": obstacle,
        "groups": group_value,
        "speed_var": speed_var,
        "goal_topology": goal_topology,
        "robot_context": robot_context,
        "repeats": repeats,
        **({"preset": str(preset).strip()} if isinstance(preset, str) else {}),
    }


def normalize_parameterized_scenario_parameters(  # noqa: C901
    raw: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Normalize the physical parameter slice used by draft scenario authoring.

    Returns:
        Validated parameter mapping with all values coerced to floats.
    """

    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("Parameterized scenario parameters must be provided as a mapping.")

    profile = raw.get("parameterized_profile")
    if isinstance(profile, Mapping) and "parameters" in profile:
        schema_version = profile.get("schema_version")
        if schema_version != PARAMETERIZED_SCENARIO_PARAMS_SCHEMA_VERSION:
            raise ValueError(
                "parameterized_profile.schema_version must equal "
                f"{PARAMETERIZED_SCENARIO_PARAMS_SCHEMA_VERSION}"
            )
        parameters = profile.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("parameterized_profile.parameters must be a mapping.")
        source = parameters
    else:
        source = profile if isinstance(profile, Mapping) else raw

    unknown = set(source) - set(PARAMETERIZED_SCENARIO_PARAMETER_DEFAULTS)
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown parameterized scenario keys: {keys}")

    values = dict(PARAMETERIZED_SCENARIO_PARAMETER_DEFAULTS)
    for key, value in source.items():
        if value is not None:
            values[key] = float(value)

    sidewalk_width = values["sidewalk_width"]
    obstacle_density = values["obstacle_density"]
    pedestrian_density = values["pedestrian_density"]
    bottleneck_width = values["bottleneck_width"]
    crossing_angle = values["crossing_angle"]
    occlusion_probability = values["occlusion_probability"]

    if sidewalk_width <= 0.0:
        raise ValueError(f"sidewalk_width must be > 0: {sidewalk_width}")
    if not 0.0 <= obstacle_density <= 1.0:
        raise ValueError(f"obstacle_density must be in [0, 1]: {obstacle_density}")
    if not 0.0 <= pedestrian_density <= 1.0:
        raise ValueError(f"pedestrian_density must be in [0, 1]: {pedestrian_density}")
    if bottleneck_width <= 0.0 or bottleneck_width > sidewalk_width:
        raise ValueError(
            "bottleneck_width must be > 0 and <= sidewalk_width: "
            f"{bottleneck_width} > {sidewalk_width}"
        )
    if not 0.0 <= crossing_angle <= 180.0:
        raise ValueError(f"crossing_angle must be in [0, 180]: {crossing_angle}")
    if not 0.0 <= occlusion_probability <= 1.0:
        raise ValueError(f"occlusion_probability must be in [0, 1]: {occlusion_probability}")

    return values


def _density_tier_from_pedestrian_density(pedestrian_density: float) -> str:
    if pedestrian_density < 0.04:
        return "low"
    if pedestrian_density < 0.12:
        return "med"
    return "high"


def _flow_from_crossing_angle(crossing_angle: float) -> str:
    if crossing_angle < 30.0:
        return "uni"
    if crossing_angle < 75.0:
        return "merge"
    if crossing_angle <= 120.0:
        return "cross"
    return "bi"


def _obstacle_profile_from_parameters(params: Mapping[str, float]) -> str:
    if params["obstacle_density"] >= 0.35:
        return "maze"
    if params["bottleneck_width"] < params["sidewalk_width"] * 0.75:
        return "bottleneck"
    return "open"


def select_map_id_for_parameterized_scenario(params: Mapping[str, Any] | None = None) -> str:
    """Choose a registered map id for the normalized physical parameter slice.

    Returns:
        Map id from ``maps/registry.yaml`` suitable for scenario loader smoke checks.
    """

    normalized = normalize_parameterized_scenario_parameters(params)
    obstacle_profile = _obstacle_profile_from_parameters(normalized)
    if obstacle_profile == "maze":
        return "classic_cross_trap"
    if obstacle_profile == "bottleneck":
        ratio = normalized["bottleneck_width"] / normalized["sidewalk_width"]
        if ratio < 0.40:
            return "classic_bottleneck_high"
        if ratio < 0.65:
            return "classic_bottleneck_medium"
        return "classic_bottleneck"
    if 30.0 <= normalized["crossing_angle"] <= 150.0:
        return "classic_crossing"
    return "classic_head_on_corridor"


def derive_generation_parameters_from_physical_slice(
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Map physical scenario knobs onto existing benchmark generation parameters.

    Returns:
        Existing generator profile fields derived from the physical issue #3970 knobs.
    """

    normalized = normalize_parameterized_scenario_parameters(params)
    obstacle = _obstacle_profile_from_parameters(normalized)
    return normalize_generation_parameters(
        {
            "density": _density_tier_from_pedestrian_density(normalized["pedestrian_density"]),
            "flow": _flow_from_crossing_angle(normalized["crossing_angle"]),
            "obstacle": obstacle,
            "groups": min(1.0, normalized["pedestrian_density"] * 4.0),
            "speed_var": "high" if normalized["occlusion_probability"] >= 0.5 else "low",
            "goal_topology": "swap" if normalized["crossing_angle"] >= 120.0 else "point",
            "robot_context": "ahead" if normalized["occlusion_probability"] >= 0.5 else "embedded",
            "repeats": 1,
        }
    )


def resolve_agent_count(params: Mapping[str, Any] | None = None) -> int:
    """Return deterministic agent count from generation parameters."""

    normalized = normalize_generation_parameters(params or {})
    return _select_counts(normalized)


def estimate_initial_difficulty(
    params: Mapping[str, Any] | None = None,
    *,
    n_agents: int | None = None,
) -> dict[str, Any]:
    """Build an initial scenario difficulty estimate used in draft metadata.

    Returns:
        Small deterministic difficulty metadata payload.
    """

    normalized = normalize_generation_parameters(dict(params or {}))
    if n_agents is None:
        n_agents = resolve_agent_count(normalized)
    if not isinstance(n_agents, int) or n_agents < 0:
        raise ValueError(f"n_agents must be a non-negative int: {n_agents}")
    crowding = n_agents / (AREA_WIDTH * AREA_HEIGHT)

    components = {
        "density": _DIFFICULTY_WEIGHT["density"][normalized["density"]],
        "flow": _DIFFICULTY_WEIGHT["flow"][normalized["flow"]],
        "obstacle": _DIFFICULTY_WEIGHT["obstacle"][normalized["obstacle"]],
        "speed_var": _DIFFICULTY_WEIGHT["speed_var"][normalized["speed_var"]],
        "goal_topology": _DIFFICULTY_WEIGHT["goal_topology"][normalized["goal_topology"]],
        "groups": max(0.0, min(0.30, float(normalized["groups"]) * 0.30)),
        "crowding": max(0.0, min(0.30, crowding / 20.0)),
    }
    score = sum(components.values())
    score = min(score, 1.0)
    score = max(score, 0.0)

    if score < 0.35:
        band = "low"
    elif score < 0.70:
        band = "medium"
    else:
        band = "high"

    return {
        "schema_version": SCENARIO_INITIAL_DIFFICULTY_SCHEMA_VERSION,
        "score": round(score, 3),
        "band": band,
        "components": components,
        "n_agents": n_agents,
        "area": AREA_WIDTH * AREA_HEIGHT,
    }


_DENSITY_COUNTS = {"low": 10, "med": 25, "high": 40}


@dataclass
class GeneratedScenario:
    """Container for a generated scenario and its metadata."""

    simulator: Any
    state: np.ndarray
    obstacles: list[tuple[float, float, float, float]]
    groups: list[int]
    metadata: dict[str, Any]


def _select_counts(params: dict[str, Any]) -> int:
    """Choose agent count based on density.

    Returns:
        Number of agents for the scenario.
    """
    density = params.get("density", "med")
    return int(_DENSITY_COUNTS.get(density, _DENSITY_COUNTS["med"]))


def _sample_positions(rng: np.random.Generator, n: int) -> np.ndarray:
    # Uniform in central bounding box with small margin
    """Sample initial positions uniformly within the arena bounds.

    Returns:
        Array of shape (n, 2) with positions.
    """
    margin = 0.5
    xs = rng.uniform(margin, AREA_WIDTH - margin, size=n)
    ys = rng.uniform(margin, AREA_HEIGHT - margin, size=n)
    return np.stack([xs, ys], axis=1)


def _build_obstacles(kind: str) -> list[tuple[float, float, float, float]]:
    """Construct obstacle segments for a named obstacle layout.

    Returns:
        List of obstacle line segments.
    """
    if kind == "open":
        return []
    if kind == "bottleneck":
        # Two vertical walls with a small gap centered
        gap_y1, gap_y2 = 2.5, 3.5
        x = AREA_WIDTH / 2
        return [
            (x, 0.0, x, gap_y1),  # lower segment
            (x, gap_y2, x, AREA_HEIGHT),  # upper segment
        ]
    if kind == "maze":
        # Simple grid-like three segments
        w2 = AREA_WIDTH / 3
        return [
            (w2, 0.0, w2, AREA_HEIGHT * 0.6),
            (2 * w2, AREA_HEIGHT * 0.4, 2 * w2, AREA_HEIGHT),
            (w2, AREA_HEIGHT * 0.6, 2 * w2, AREA_HEIGHT * 0.6),
        ]
    return []


def _assign_goals(flow: str, goal_topology: str, pos: np.ndarray) -> np.ndarray:
    """Assign goal positions based on flow and topology settings.

    Returns:
        Array of goal positions aligned with ``pos``.
    """
    n = pos.shape[0]
    goals = np.zeros_like(pos)
    if flow == "uni":
        # Move left->right: goals at right boundary
        goals[:, 0] = AREA_WIDTH - 0.2
        goals[:, 1] = pos[:, 1]
    elif flow == "bi":
        half = n // 2
        goals[:half, 0] = AREA_WIDTH - 0.2
        goals[:half, 1] = pos[:half, 1]
        goals[half:, 0] = 0.2
        goals[half:, 1] = pos[half:, 1]
    elif flow == "cross":
        # Half move horizontally, half vertically
        half = n // 2
        goals[:half, 0] = AREA_WIDTH - pos[:half, 0]
        goals[:half, 1] = pos[:half, 1]
        goals[half:, 0] = pos[half:, 0]
        goals[half:, 1] = AREA_HEIGHT - pos[half:, 1]
    elif flow == "merge":
        # All toward a central x then exit right
        cx = AREA_WIDTH * 0.6
        goals[:, 0] = cx
        goals[:, 1] = AREA_HEIGHT / 2
    # Adjust for goal topology variants
    if goal_topology == "swap":
        goals = pos[::-1].copy()
    elif goal_topology == "circulate":
        # shift positions circularly
        goals = np.roll(pos, shift=1, axis=0)
    return goals


def _assign_groups(rng: np.random.Generator, n: int, fraction: float) -> list[int]:
    """Assign group IDs for a fraction of agents.

    Returns:
        List of group IDs per agent (or -1 for ungrouped).
    """
    if fraction <= 0:
        return [-1] * n
    num_grouped = round(n * fraction)
    indices = rng.permutation(n)[:num_grouped]
    group_ids = [-1] * n
    current_gid = 0
    i = 0
    while i < len(indices):
        group_size = int(rng.integers(2, 5))  # 2-4
        members = indices[i : i + group_size]
        for m in members:
            group_ids[m] = current_gid
        current_gid += 1
        i += group_size
    return group_ids


def _speed_variation(speed_var: str) -> float:
    """Return speed variation scale for the requested setting.

    Returns:
        Speed variation scalar.
    """
    return 0.2 if speed_var == "low" else 0.5


def generate_scenario(params: dict[str, Any], seed: int) -> GeneratedScenario:
    """Generate a deterministic scenario.

    Parameters
    ----------
    params : dict
        Scenario parameter dictionary (see module docstring).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    GeneratedScenario
        Object containing generated state, map definition, and robot configuration.
    """
    normalized = normalize_generation_parameters(params)
    # Special preset for testing/validation: guaranteed contact at t=0
    # Places one pedestrian exactly at the default robot start (0.3, 3.0)
    # with goal equal to its position (no desired motion). This ensures
    # min distance < D_COLL at the first timestep, exercising the collision
    # counting pipeline end-to-end.
    preset = str(normalized.get("preset", "")).strip().lower()
    if preset == "collision_sanity":
        n = 1
        pos = np.array([[0.3, 3.0]], dtype=float)
        goals = pos.copy()  # no movement desired
        state = np.zeros((n, 7), dtype=float)
        state[:, 0:2] = pos
        state[:, 4:6] = goals
        state[:, 6] = 1.0
        obstacles: list[tuple[float, float, float, float]] = []
        groups: list[int] = [-1]
        metadata = {
            **normalized,
            "n_agents": n,
            "area": AREA_WIDTH * AREA_HEIGHT,
            "seed": seed,
            "group_count": 0,
            "generation_profile": {
                "schema_version": SCENARIO_GENERATION_PARAMS_SCHEMA_VERSION,
                "seed": seed,
                "seed_signature": f"seed={seed}",
                "parameters": dict(normalized),
            },
            "initial_difficulty": estimate_initial_difficulty(normalized, n_agents=n),
        }
        simulator = None if pysf is None else pysf.Simulator(state=state, obstacles=None)  # type: ignore[arg-type]
        return GeneratedScenario(
            simulator=simulator,
            state=state,
            obstacles=obstacles,
            groups=groups,
            metadata=metadata,
        )

    rng = np.random.default_rng(seed)
    n = _select_counts(normalized)
    pos = _sample_positions(rng, n)
    goals = _assign_goals(
        normalized.get("flow", "uni"), normalized.get("goal_topology", "point"), pos
    )
    speed_std = _speed_variation(normalized.get("speed_var", "low"))
    # Desired speeds around 1.3 m/s with variation
    desired_speeds = rng.normal(loc=1.3, scale=speed_std, size=n)
    desired_speeds = np.clip(desired_speeds, 0.2, 2.0)

    # State: [x,y,vx,vy,goalx,goaly,tau]
    state = np.zeros((n, 7), dtype=float)
    state[:, 0:2] = pos
    state[:, 4:6] = goals
    state[:, 6] = 1.0  # tau placeholder

    obstacles = _build_obstacles(normalized.get("obstacle", "open"))
    groups = _assign_groups(rng, n, float(normalized.get("groups", 0.0)))

    # Attach group influence via metadata; downstream can use groups list
    metadata = {
        **normalized,
        "n_agents": n,
        "area": AREA_WIDTH * AREA_HEIGHT,
        "seed": seed,
        "speed_std": speed_std,
        "group_count": len({g for g in groups if g >= 0}),
        "generation_profile": {
            "schema_version": SCENARIO_GENERATION_PARAMS_SCHEMA_VERSION,
            "seed": seed,
            "seed_signature": f"seed={seed}",
            "parameters": dict(normalized),
        },
        "initial_difficulty": estimate_initial_difficulty(normalized),
    }

    if pysf is None:
        simulator = None  # pragma: no cover
    else:
        # pysocialforce expects None (not empty list) for no obstacles; empty list triggers
        # a broadcasting issue inside EnvState._update_obstacles_raw.
        sim_obstacles = obstacles if len(obstacles) > 0 else None
        simulator = pysf.Simulator(state=state, obstacles=sim_obstacles)  # type: ignore[arg-type]

    return GeneratedScenario(
        simulator=simulator,
        state=state,
        obstacles=obstacles,
        groups=groups,
        metadata=metadata,
    )


__all__ = [
    "PARAMETERIZED_SCENARIO_PARAMS_SCHEMA_VERSION",
    "SCENARIO_GENERATION_PARAMS_SCHEMA_VERSION",
    "SCENARIO_INITIAL_DIFFICULTY_SCHEMA_VERSION",
    "GeneratedScenario",
    "derive_generation_parameters_from_physical_slice",
    "estimate_initial_difficulty",
    "generate_scenario",
    "normalize_generation_parameters",
    "normalize_parameterized_scenario_parameters",
    "resolve_agent_count",
    "select_map_id_for_parameterized_scenario",
]
