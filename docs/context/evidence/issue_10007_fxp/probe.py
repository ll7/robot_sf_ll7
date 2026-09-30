"""Serial FXP dev probes; direct planner/collector paths on real wall maps.

From a repository root, pass output JSON and a directory containing the two
released .pt assets. Only dev seeds 1001-1003 run, one environment at a time.
Wall-facing controls change observation yaw without advancing the simulation;
these are scorer diagnostics, not navigation-success evidence.
"""

import copy
import gzip
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from robot_sf.benchmark.map_runner.map_runner import _build_env_config
from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import _build_socnav_config
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.planner.predictive_mppi import PredictiveMPPIAdapter, build_predictive_mppi_config
from robot_sf.planner.socnav_prediction import PredictionPlannerAdapter
from robot_sf.training.scenario_loader import load_scenarios
from scripts.training import collect_predictive_hardcase_data as hard
from scripts.training import collect_predictive_planner_data as base


def observation_digest(observation):
    """Hash actual observation values and array dtype/shape for matched-input proof.

    Returns:
        str: SHA-256 digest of the recursively ordered observation bytes.
    """
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, np.ndarray):
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        else:
            digest.update(repr(value).encode())

    visit(observation)
    return digest.hexdigest()


logger.remove()
root = Path.cwd()
output = Path(sys.argv[1])
rows = []
start = time.monotonic()
models = Path(sys.argv[2])
params = yaml.safe_load((root / "configs/algos/prediction_planner_release_v0_0_8.yaml").read_text())
params["predictive_model_id"] = None
params["predictive_checkpoint_path"] = str(
    models / "predictive_proxy_selected_v2_full-predictive_model.pt"
)
planner = PredictionPlannerAdapter(_build_socnav_config(params), allow_fallback=False)
mp = yaml.safe_load((root / "configs/algos/predictive_mppi_release_v0_0_8.yaml").read_text())
mp["predictive_model_id"] = None
mp["predictive_checkpoint_path"] = str(models / "predictive_proxy_selected_v1-predictive_model.pt")
mppi = PredictiveMPPIAdapter(build_predictive_mppi_config(mp), allow_fallback=False)


def encode_observation(value):
    """Encode arrays with explicit dtype/shape in a portable JSON snapshot.

    Returns:
        object: JSON-serializable observation value.
    """
    if isinstance(value, np.ndarray):
        return {"__fxp_array__": str(value.dtype), "shape": value.shape, "values": value.tolist()}
    if isinstance(value, dict):
        return {key: encode_observation(item) for key, item in value.items()}
    return value


def decode_observation(value):
    """Restore dtype-preserving arrays from this probe's own JSON snapshot.

    Returns:
        object: Original observation value.
    """
    if isinstance(value, dict):
        if "__fxp_array__" in value:
            return np.asarray(value["values"], dtype=value["__fxp_array__"]).reshape(value["shape"])
        return {key: decode_observation(item) for key, item in value.items()}
    return value


def probe_collectors(obs, state):
    """Compare actual collector features with serving velocity bytes.

    Returns:
        dict: Mean velocity-feature error per collector.
    """
    a1 = {}
    for label, collector in (("base", base), ("hardcase", hard)):
        frame = collector._extract_frame(obs, max_agents=16)
        st, _, ma, _ = collector._frames_to_samples(
            [frame, frame], max_agents=16, horizon_steps=1, ego_conditioning=False
        )
        # independent observable oracle: current model serving consumes same ego velocity bytes
        count = int(ma[0].sum())
        a1[label] = (
            float(np.linalg.norm(st[0, :count, 2:4] - state[:count, 2:4], axis=1).mean())
            if count
            else None
        )
    return a1


def probe_walls(obs, future, mask, candidates):
    """Compare original and obstacle-cleared costs at sampled dev poses.

    Returns:
        tuple: Ordinary candidate deltas and controlled wall-facing deltas.
    """
    a2 = []
    # Score matched actions against original wall channels and a counterfactual clearing only obstacles.
    grid, meta = planner._extract_grid_payload(obs)
    clear = copy.deepcopy(obs)
    cleared = np.array(grid, copy=True)
    obstacles = planner._grid_channel_index(meta, "obstacles")
    combined = planner._grid_channel_index(meta, "combined")
    peds = planner._grid_channel_index(meta, "pedestrians")
    cleared[obstacles] = 0
    if combined >= 0:
        cleared[combined] = cleared[peds] if peds >= 0 else 0
    clear["occupancy_grid"] = cleared
    for v, w in candidates:
        if v <= 0:
            continue
        kw = {"future_peds": future, "mask": mask, "steps": 8}
        delta = planner._score_action(observation=obs, v=v, w=w, **kw) - planner._score_action(
            observation=clear, v=v, w=w, **kw
        )
        seqdelta = planner._score_action_sequence(
            observation=obs, sequence=[(v, w)], **kw
        ) - planner._score_action_sequence(observation=clear, sequence=[(v, w)], **kw)
        a2.append([float(v), float(w), float(delta), float(seqdelta)])
    # Controlled wall-facing yaw at the same sampled dev pose; no pedestrian cost.
    # Keep the original grid robot_pose metadata so world-to-grid mapping is unchanged.
    robot = planner._socnav_fields(obs)[0]
    pos = np.asarray(robot["position"], dtype=float)[:2]
    occupied = np.argwhere(grid[obstacles] > 0.5)
    cell_local = np.asarray(meta["origin"]).reshape(-1)[:2] + (occupied[:, [1, 0]] + 0.5) * float(
        np.asarray(meta["resolution"]).reshape(-1)[0]
    )
    if float(np.asarray(meta.get("use_ego_frame", [0])).reshape(-1)[0]) > 0.5:
        pose = np.asarray(meta["robot_pose"]).reshape(-1)
        ch, sh = np.cos(pose[2]), np.sin(pose[2])
        cell_world = cell_local @ np.array([[ch, sh], [-sh, ch]]) + pose[:2]
    else:
        cell_world = cell_local
    delta = cell_world - pos
    closest = int(np.argmin(np.linalg.norm(delta, axis=1)))
    yaw = float(np.arctan2(delta[closest, 1], delta[closest, 0]))
    aimed = copy.deepcopy(obs)
    aimed_clear = copy.deepcopy(clear)
    for ao in (aimed, aimed_clear):
        if "robot" in ao:
            ao["robot"]["heading"] = np.array([yaw])
        else:
            ao["robot_heading"] = np.array([yaw])
    wk = {"future_peds": np.zeros((0, 8, 2)), "mask": np.zeros(0), "steps": 8}
    vv = float(planner.config.max_linear_speed)
    wall_facing = [
        float(np.linalg.norm(delta[closest])),
        float(
            planner._score_action(observation=aimed, v=vv, w=0.0, **wk)
            - planner._score_action(observation=aimed_clear, v=vv, w=0.0, **wk)
        ),
        float(
            planner._score_action_sequence(observation=aimed, sequence=[(vv, 0.0)], **wk)
            - planner._score_action_sequence(observation=aimed_clear, sequence=[(vv, 0.0)], **wk)
        ),
    ]
    return a2, wall_facing


def score_observation(name, seed, tick, obs):
    """Run all four findings against one original or replayed observation.

    Returns:
        dict: Feature, scoring, horizon and command-lattice probe results.
    """
    state, mask, _, _ = planner._build_model_input(obs)
    future = planner._predict_trajectories(state, mask)
    candidates = planner._candidate_set(future_peds=future, mask=mask)
    a1 = probe_collectors(obs, state)
    a2, wall_facing = probe_walls(obs, future, mask, candidates)
    a3 = {}
    configured = mppi.config.horizon_steps
    for requested in (configured, 24, 4):
        mppi.config.horizon_steps = requested
        try:
            a3[str(requested)] = {"steps": mppi._predict_future(obs)[2], "error": None}
        except ValueError as exc:
            a3[str(requested)] = {"steps": None, "error": str(exc)}
    mppi.config.horizon_steps = configured
    ms, mm, _, _ = mppi._predictor._build_model_input(obs)
    mf = mppi._predictor._predict_trajectories(ms, mm)
    mrates = sorted({w for _, w in mppi._predictor._candidate_set(future_peds=mf, mask=mm)})
    return {
        "scenario": name,
        "observation_sha256": observation_digest(obs),
        "seed": seed,
        "tick": tick,
        "heading": float(np.asarray(planner._socnav_fields(obs)[0]["heading"]).reshape(-1)[0]),
        "ped_count": int(mask.sum()),
        "collector_velocity_error": a1,
        "wall_cost_deltas": a2,
        "wall_facing_control": wall_facing,
        "heading_rates": sorted({w for _, w in candidates}),
        "mppi_heading_rates": mrates,
        "mppi_horizon": a3,
    }


def collect_observations():
    """Yield dev observations from one environment at a time.

    Yields:
        tuple: Scenario, seed, tick and actual SOCNAV observation.
    """
    for name in ("classic_doorway_low", "classic_bottleneck_medium"):
        scenario = next(
            s
            for s in load_scenarios(root / "configs/scenarios/classic_interactions.yaml")
            if s["name"] == name
        )
        for seed in (1001, 1002, 1003):
            assert 1001 <= seed <= 1030
            # Seed legacy backend RNG consumers too, not only Gym's generator.
            np.random.seed(seed)
            random.seed(seed)
            cfg = _build_env_config(
                scenario, scenario_path=root / "configs/scenarios/classic_interactions.yaml"
            )
            cfg.observation_mode = ObservationMode.SOCNAV_STRUCT
            cfg.use_occupancy_grid = True
            cfg.include_grid_in_observation = True
            cfg._init_grid_config()
            env = make_robot_env(config=cfg, seed=seed, debug=False, recording_enabled=False)
            try:
                obs, _ = env.reset(seed=seed)
                for tick in range(160):
                    yield name, seed, tick, obs
                    action = base._goal_policy(obs, max_speed=1.0)
                    obs, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
            finally:
                env.close()


def replay_observations(snapshot):
    """Yield exactly the arrays captured in a base dev rollout.

    Yields:
        tuple: Scenario, seed, tick and original SOCNAV bytes.
    """
    with gzip.open(snapshot, "rt", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            assert 1001 <= item["seed"] <= 1030
            obs = decode_observation(item["observation"])
            assert observation_digest(obs) == item["observation_sha256"]
            yield item["scenario"], item["seed"], item["tick"], obs


snapshot = Path(sys.argv[3]) if len(sys.argv) > 3 else output.with_suffix(".observations.jsonl.gz")
if len(sys.argv) > 3:
    for name, seed, tick, obs in replay_observations(snapshot):
        rows.append(score_observation(name, seed, tick, obs))
else:
    with gzip.open(snapshot, "wt", encoding="utf-8") as stream:
        for name, seed, tick, obs in collect_observations():
            stream.write(
                json.dumps(
                    {
                        "scenario": name,
                        "seed": seed,
                        "tick": tick,
                        "observation_sha256": observation_digest(obs),
                        "observation": encode_observation(obs),
                    }
                )
                + "\n"
            )
            rows.append(score_observation(name, seed, tick, obs))
output.write_text(
    json.dumps(
        {
            "elapsed_s": time.monotonic() - start,
            "rows": rows,
            "provenance": {
                "seeds": [1001, 1002, 1003],
                "scenarios": ["classic_doorway_low", "classic_bottleneck_medium"],
                "trajectory_policy": "collector goal policy; comparator replays exact captured observation arrays",
                "observation_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "predictor_fallback": planner.foresight_degraded(),
                "mppi_predictor_fallback": mppi.foresight_degraded(),
            },
        },
        indent=2,
    )
)
print(len(rows), output)
