"""Feasibility-smoke adapter for the CrowdNav_Prediction_AttnGraph learned baseline.

Issue #4871 asked for a go/no-go feasibility smoke for reusing
``Shuijing725/CrowdNav_Prediction_AttnGraph`` (ICRA 2023, intention-aware crowd
navigation with an attention-based interaction graph) as an external *learned*
baseline planner. This module is the thinnest model-only adapter that proves the
shipped checkpoint loads and produces an action on synthetic Robot SF
observations, plus the per-step wall-clock measurement the issue asks for.

Scope guardrails (carry into every downstream doc):

* This is a **smoke**, not a roster addition. It is intentionally NOT registered
  in ``algorithm_metadata`` / ``algorithm_readiness`` and must not enter benchmark
  sweeps without a separate maintainer decision.
* Only the **PyTorch RL navigation policy** path is exercised. The shipped
  ``41200.pt`` checkpoint ("Ours without randomized humans") trains/runs as the
  ``CrowdSimPred-v0`` attention-graph policy with a 5-step future-position edge.
  The smoke reconstructs those futures with a constant-velocity model, so the
  checkpoint needs nothing beyond PyTorch at inference (no TensorFlow/GST model).
  Note: the bundled test config mismatches the weights (it declares
  ``CrowdSimVarNum-v0`` / ``predict_method='none'``); the state_dict is
  authoritative — see ``docs/context/issue_4871_crowdnav_pred_attng_smoke.md``.
* The **TensorFlow GST trajectory-predictor** variant (``CrowdSimPredRealGST-v0``,
  ``41665.pt``) and the full ``crowd_sim`` / OpenAI Baselines / Python-RVO2 stack
  are out of scope for this smoke and are documented as a higher-fidelity
  follow-up cost class in ``docs/context/issue_4871_crowdnav_pred_attng_smoke.md``.

The adapter mirrors the established CrowdNav_HEIGHT pattern (``crowdnav_height.py``):
reconstruct the upstream dict observation from world-frame Robot SF state, load the
upstream ``Policy`` checkpoint with ``torch.load``, and call ``policy.act`` directly.
No simulator, no ORCA/RVO2, no OpenAI Baselines install is required to run the
network forward.
"""

from __future__ import annotations

import importlib
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from gymnasium import spaces

if TYPE_CHECKING:
    from collections.abc import Iterator

# Staged checkout written by ``scripts/tools/manage_external_repos.py stage
# crowdnav_pred_attng`` (gitignored under third_party/external_repos/).
_DEFAULT_REPO_ROOT = Path("third_party/external_repos/crowdnav_pred_attng")
# Shipped "Ours without randomized humans" checkpoint. The state_dict is the
# CrowdSimPred-v0 attention-graph policy (5-step future-position edge, width 12);
# the bundled test config mismatches it (declares CrowdSimVarNum-v0 / width 2).
# The smoke reconstructs the 5 futures with a constant-velocity model, so no
# TensorFlow/GST model is needed at inference.
_DEFAULT_MODEL_SUBDIR = Path("trained_models/GST_predictor_non_rand")
_DEFAULT_CHECKPOINT_NAME = "41200.pt"

# Architecture + sim constants frozen by the pinned commit and the bundled
# trained_models/GST_predictor_non_rand/{arguments.py,configs/config.py}.
# Cite those files as the source of truth; they are fixed for a pinned SHA.
_DEFAULT_HUMAN_NUM = 20
_DEFAULT_TIME_STEP = 0.25
_DEFAULT_V_PREF = 1.0
_DEFAULT_SENSOR_RANGE = 5.0
_DEFAULT_ROBOT_RADIUS = 0.3
# Number of predicted future steps baked into each spatial edge. The checkpoint's
# spatial_attn input is 2*(predict_steps+1) = 12, so predict_steps=5.
_DEFAULT_PREDICT_STEPS = 5
# Sentinel the upstream env uses to pad invisible / missing humans in the sorted
# spatial-edge tensor (see crowd_sim/envs/crowd_sim_var_num.py generate_ob).
_PAD_DISTANCE = 15.0


@dataclass(frozen=True)
class CrowdNavPredAttnGraphConfig:
    """Configuration for loading one upstream CrowdNav_Prediction_AttnGraph checkpoint."""

    repo_root: Path = _DEFAULT_REPO_ROOT
    model_subdir: Path = _DEFAULT_MODEL_SUBDIR
    checkpoint_name: str = _DEFAULT_CHECKPOINT_NAME
    device: str = "cpu"
    human_num: int = _DEFAULT_HUMAN_NUM
    time_step: float = _DEFAULT_TIME_STEP
    v_pref: float = _DEFAULT_V_PREF
    sensor_range: float = _DEFAULT_SENSOR_RANGE
    robot_radius: float = _DEFAULT_ROBOT_RADIUS
    predict_steps: int = _DEFAULT_PREDICT_STEPS


def build_crowdnav_pred_attng_config(data: dict[str, Any] | None) -> CrowdNavPredAttnGraphConfig:
    """Build adapter config from a benchmark algo-config payload.

    Returns:
        CrowdNavPredAttnGraphConfig resolved from the payload with smoke defaults.
    """
    payload = data or {}
    device = str(payload.get("device", "cpu")).strip() or "cpu"
    human_num = int(payload.get("human_num", _DEFAULT_HUMAN_NUM))
    time_step = float(payload.get("time_step", _DEFAULT_TIME_STEP))
    v_pref = float(payload.get("v_pref", _DEFAULT_V_PREF))
    if human_num <= 0:
        raise ValueError("human_num must be positive")
    if time_step <= 0.0:
        raise ValueError("time_step must be positive")
    if v_pref <= 0.0:
        raise ValueError("v_pref must be positive")
    return CrowdNavPredAttnGraphConfig(
        repo_root=Path(str(payload.get("repo_root", _DEFAULT_REPO_ROOT))),
        model_subdir=Path(str(payload.get("model_subdir", _DEFAULT_MODEL_SUBDIR))),
        checkpoint_name=str(payload.get("checkpoint_name", _DEFAULT_CHECKPOINT_NAME)),
        device=device,
        human_num=human_num,
        time_step=time_step,
        v_pref=v_pref,
        sensor_range=float(payload.get("sensor_range", _DEFAULT_SENSOR_RANGE)),
        robot_radius=float(payload.get("robot_radius", _DEFAULT_ROBOT_RADIUS)),
        predict_steps=int(payload.get("predict_steps", _DEFAULT_PREDICT_STEPS)),
    )


@lru_cache(maxsize=1)
def _default_args_namespace() -> Any:
    """Return the upstream ``args`` namespace for the no_rand checkpoint.

    Values are frozen by the pinned commit and the bundled
    ``trained_models/GST_predictor_non_rand/arguments.py``. They reconstruct the
    ``selfAttn_merge_SRNN`` architecture that the ``41200.pt`` state_dict expects.

    Returns:
        A ``types.SimpleNamespace`` matching the upstream args contract.
    """
    return _build_args_namespace(
        # env_name=CrowdSimPred-v0 selects SpatialEdgeSelfAttn.input_size=12
        # (current + 5 const-velocity predicted positions), which is what the
        # 41200.pt state_dict was trained with. The bundled test config's
        # env_name=CrowdSimVarNum-v0 / predict_method='none' does NOT match the
        # shipped weights (input_size=2); the checkpoint is authoritative.
        # CrowdSimPred-v0 uses constant-velocity OR ground-truth prediction and
        # needs NO TensorFlow/GST model at inference.
        env_name="CrowdSimPred-v0",
        use_self_attn=True,
        use_hr_attn=True,
        sort_humans=True,
        no_cuda=True,
        num_processes=1,
    )


def _build_args_namespace(
    *,
    env_name: str,
    use_self_attn: bool,
    use_hr_attn: bool,
    sort_humans: bool,
    no_cuda: bool,
    num_processes: int,
) -> Any:
    """Build an upstream-compatible args namespace for the SRNN policy family.

    Returns:
        ``types.SimpleNamespace`` carrying the network-architecture fields read
        by ``rl/networks/{model,srnn_model,selfAttn_srnn_temp_node}.py``.
    """
    # env_type is read by the plain SRNN base (robot_size branch); the
    # selfAttn_merge_SRNN network hardcodes robot_size=9, so this only matters
    # for faithful args completeness.
    return SimpleNamespace(
        env_name=env_name,
        env_type="crowd_sim",
        use_self_attn=use_self_attn,
        use_hr_attn=use_hr_attn,
        sort_humans=sort_humans,
        no_cuda=no_cuda,
        cuda=False,
        # Recurrent graph sizes.
        human_node_rnn_size=128,
        human_human_edge_rnn_size=256,
        human_node_output_size=256,
        # Input / output / embedding sizes.
        human_node_input_size=3,
        human_human_edge_input_size=2,
        human_node_embedding_size=64,
        human_human_edge_embedding_size=64,
        attention_size=64,
        # Rollout bookkeeping (only nenv=1 matters at infer time).
        seq_length=30,
        num_processes=num_processes,
        num_mini_batch=2,
    )


@contextmanager
def _pred_attng_import_context(repo_root: Path) -> Iterator[None]:
    """Expose the upstream checkout and short-circuit the heavy env import.

    ``rl/networks/network_utils.py`` imports ``VecNormalize`` from
    ``rl.networks.envs``, which otherwise drags in ``gym`` + OpenAI Baselines +
    the crowd_sim stack. The model-only smoke never touches that runtime path, so
    we pre-inject a minimal stub for ``rl.networks.envs`` and let the real
    ``rl.networks.model`` / ``srnn_model`` / ``selfAttn_srnn_temp_node`` modules
    load against PyTorch only.

    Yields:
        None; the upstream package is importable for the duration of the block.
    """
    repo_str = str(repo_root)
    original_path = list(sys.path)
    injected: set[str] = set()
    stub_keys: tuple[str, ...] = ("rl.networks.envs",)
    original_modules = {key: sys.modules[key] for key in stub_keys if key in sys.modules}
    for key in stub_keys:
        sys.modules.pop(key, None)
    sys.path.insert(0, repo_str)
    # Minimal stub: network_utils only needs the VecNormalize name bound.
    envs_stub = ModuleType("rl.networks.envs")

    class _VecNormalize:  # pragma: no cover - never instantiated by the smoke
        """Placeholder satisfying ``from rl.networks.envs import VecNormalize``."""

    envs_stub.VecNormalize = _VecNormalize  # type: ignore[attr-defined]
    sys.modules["rl.networks.envs"] = envs_stub
    injected.add("rl.networks.envs")
    try:
        yield
    finally:
        for key in injected:
            sys.modules.pop(key, None)
        sys.modules.update(original_modules)
        sys.path[:] = original_path


def _build_observation_space(human_num: int, predict_steps: int) -> spaces.Dict:
    """Mirror the upstream CrowdSimPred dict observation layout.

    Returns:
        Gymnasium dict space matching the checkpoint input contract. Only
        ``spatial_edges`` shape[0] (human_num) is read by the network
        constructor; the per-edge width 2*(predict_steps+1) is recorded for
        faithfulness.
    """
    edge_width = 2 * (predict_steps + 1)
    return spaces.Dict(
        {
            # robot_node: px, py, r, gx, gy, v_pref, theta
            "robot_node": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7), dtype=np.float32),
            "temporal_edges": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32),
            "spatial_edges": spaces.Box(
                low=-np.inf, high=np.inf, shape=(human_num, edge_width), dtype=np.float32
            ),
            "detected_human_num": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            "visible_masks": spaces.Box(
                low=-np.inf, high=np.inf, shape=(human_num,), dtype=np.float32
            ),
        }
    )


def _xy_rows(value: Any) -> np.ndarray:
    """Normalize arbitrary XY payloads to an ``(N, 2)`` array.

    Returns:
        Two-column float array with one row per vector (empty when no data).
    """
    arr = np.asarray([] if value is None else value, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    arr = arr.reshape(-1, arr.shape[-1] if arr.ndim > 1 else 1)
    if arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=float)
    return arr[:, :2]


def _require_xy(value: Any, field: str) -> np.ndarray:
    """Return a required 2-vector or raise a contract error.

    Accepts flat ``[x, y]`` or nested ``[[x, y]]`` payloads uniformly.

    Returns:
        Two-element float array.
    """
    arr = np.asarray([] if value is None else value, dtype=float).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Missing or malformed required field: {field}")
    return arr[:2]


class CrowdNavPredAttnGraphAdapter:
    """Stateful model-only adapter around one upstream AttnGraph checkpoint.

    The adapter reconstructs the upstream dict observation from world-frame
    Robot SF state and runs one recurrent forward pass per step. It emits a
    holonomic ``(vx, vy)`` velocity command (the upstream ``ActionXY``), which is
    a valid Robot SF ``velocity``-space action; projection to a unicycle command
    is an explicit transfer caveat documented in the smoke note, not a silent
    transform.
    """

    upstream_policy = "rl.networks.model.Policy[selfAttn_merge_SRNN]"
    projection_policy = "holonomic_velocity_xy_clipped_to_v_pref"

    def __init__(self, config: CrowdNavPredAttnGraphConfig | None = None) -> None:
        """Initialize the adapter and load the upstream checkpoint."""
        self.config = config or CrowdNavPredAttnGraphConfig()
        self.repo_root = self.config.repo_root.resolve()
        self.checkpoint_path = (
            self.repo_root / self.config.model_subdir / "checkpoints" / self.config.checkpoint_name
        ).resolve()
        if not self.repo_root.exists():
            raise FileNotFoundError(
                "CrowdNav_Prediction_AttnGraph checkout not found: "
                f"{self.config.repo_root}. Run "
                "`uv run python scripts/tools/manage_external_repos.py stage crowdnav_pred_attng`."
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "Checkpoint not found in staged checkout: "
                f"{self.checkpoint_path}. Restage the pinned commit."
            )

        self._device = torch.device(self.config.device)
        self._human_num = int(self.config.human_num)
        self._args = _default_args_namespace()
        observation_space = _build_observation_space(self._human_num, self.config.predict_steps)
        # Holonomic ActionXY -> 2D continuous Box; DiagGaussian head in the Policy.
        action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        with _pred_attng_import_context(self.repo_root):
            policy_mod = importlib.import_module("rl.networks.model")
            self._policy = policy_mod.Policy(
                observation_space.spaces,
                action_space,
                base="selfAttn_merge_srnn",  # upstream Policy selects the base CLASS by this name
                base_kwargs=self._args,
            )
        # Match the upstream test entrypoint: load weights then pin nenv=1.
        # weights_only=False because these are legacy torch.save checkpoints
        # (saved under torch 1.12.1) that may carry non-tensor objects.
        state_dict = torch.load(self.checkpoint_path, map_location=self._device, weights_only=False)
        self._policy.load_state_dict(state_dict)
        self._policy.base.nenv = 1
        self._policy.to(self._device)
        self._policy.eval()
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        """Reset the two SRNN recurrent hidden-state tensors to zeros."""
        del seed
        base = self._policy.base
        self._hidden_state = {
            "human_node_rnn": torch.zeros(
                (1, 1, int(base.human_node_rnn_size)), dtype=torch.float32, device=self._device
            ),
            "human_human_edge_rnn": torch.zeros(
                (1, self._human_num + 1, int(base.human_human_edge_rnn_size)),
                dtype=torch.float32,
                device=self._device,
            ),
        }
        self._mask = torch.zeros((1, 1), dtype=torch.float32, device=self._device)

    def _extract_robot_sf_fields(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        """Extract world-frame Robot SF state from a benchmark observation payload.

        Accepts either the canonical ``robot_sf.baselines.interface.Observation``
        field layout (``robot``/``agents``/``dt``) or the planner nested
        ``robot``/``goal``/``pedestrians`` layout used elsewhere in the tree.

        Returns:
            Tuple of (robot_position, robot_velocity, robot_goal, robot_radius,
            pedestrian ``(N, 2)`` positions, pedestrian ``(N, 2)`` velocities).
        """
        robot = observation.get("robot", {})
        agents = observation.get("agents")
        if robot and (agents is not None or "pedestrians" in observation):
            robot_pos = _require_xy(robot.get("position"), "robot.position")
            robot_vel = _require_xy(robot.get("velocity"), "robot.velocity")
            robot_goal = _require_xy(robot.get("goal", robot.get("current")), "robot.goal")
            robot_radius = float(
                np.asarray(robot.get("radius", self.config.robot_radius)).reshape(-1)[0]
            )
            if agents is not None:
                ped_positions = _xy_rows([a.get("position") for a in agents])
                ped_velocities = _xy_rows([a.get("velocity", [0.0, 0.0]) for a in agents])
            else:
                ped = observation.get("pedestrians", {})
                ped_positions = _xy_rows(ped.get("positions"))
                ped_velocities = _xy_rows(ped.get("velocities"))
            return robot_pos, robot_vel, robot_goal, robot_radius, ped_positions, ped_velocities

        # Flat benchmark fallback.
        robot_pos = _require_xy(observation.get("robot_position"), "robot_position")
        robot_vel = _require_xy(observation.get("robot_velocity_xy"), "robot_velocity_xy")
        robot_goal = _require_xy(observation.get("goal_current"), "goal_current")
        robot_radius = float(
            np.asarray(observation.get("robot_radius", self.config.robot_radius)).reshape(-1)[0]
        )
        ped_positions = _xy_rows(observation.get("pedestrians_positions"))
        ped_velocities = _xy_rows(observation.get("pedestrians_velocities"))
        return robot_pos, robot_vel, robot_goal, robot_radius, ped_positions, ped_velocities

    def _build_model_inputs(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reconstruct the upstream dict observation as batched torch tensors.

        Spatial edges follow the upstream ``crowd_sim_pred.CrowdSimPred.generate_ob``
        layout: per human, the current position followed by ``predict_steps``
        constant-velocity future positions (all world-frame deltas relative to the
        robot), sorted by current-position distance ascending, capped at
        ``human_num`` and padded with the sentinel.

        Returns:
            Batched torch tensors (num_processes=1) and a metadata dict for tracing.
        """
        (
            robot_pos,
            robot_vel,
            robot_goal,
            robot_radius,
            ped_positions,
            ped_velocities,
        ) = self._extract_robot_sf_fields(observation)

        # Holonomic heading from velocity (upstream robot_node carries theta but
        # the spatial edges are world-frame deltas, so theta only informs the
        # robot node, not a frame rotation).
        speed = float(np.linalg.norm(robot_vel))
        theta = float(np.arctan2(robot_vel[1], robot_vel[0])) if speed > 1e-8 else 0.0
        robot_node = np.array(
            [
                [
                    float(robot_pos[0]),
                    float(robot_pos[1]),
                    float(robot_radius),
                    float(robot_goal[0]),
                    float(robot_goal[1]),
                    float(self.config.v_pref),
                    theta,
                ]
            ],
            dtype=np.float32,
        )
        temporal_edges = np.array([[float(robot_vel[0]), float(robot_vel[1])]], dtype=np.float32)

        # Build (current + predict_steps) const-velocity positions per human,
        # relative to the robot, then sort by current distance and cap/pad.
        n = int(ped_positions.shape[0])
        if n > 0:
            steps = np.arange(0, self.config.predict_steps + 1, dtype=float)  # [0..5]
            # pos_k = pos_0 + vel * k * time_step (pred_interval=1).
            future = ped_positions[:, None, :] + ped_velocities[:, None, :] * (
                steps[None, :, None] * float(self.config.time_step)
            )  # (n, predict_steps+1, 2)
            rel = future - np.asarray(robot_pos, dtype=float).reshape(1, 1, 2)  # robot-relative
            order = np.argsort(np.linalg.norm(rel[:, 0, :], axis=1))  # by current position
            rel = rel[order][: self._human_num].reshape(-1, 2 * (self.config.predict_steps + 1))
        else:
            rel = np.zeros((0, 2 * (self.config.predict_steps + 1)), dtype=np.float32)
        detected = max(1, int(rel.shape[0]))
        spatial = np.full(
            (self._human_num, 2 * (self.config.predict_steps + 1)),
            _PAD_DISTANCE,
            dtype=np.float32,
        )
        spatial[: rel.shape[0]] = rel.astype(np.float32)

        payload = {
            "robot_node": robot_node,
            "temporal_edges": temporal_edges,
            "spatial_edges": spatial,
            "detected_human_num": np.asarray([float(detected)], dtype=np.float32),
            "visible_masks": np.zeros((self._human_num,), dtype=np.float32),
        }
        tensors = {
            key: torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0).to(self._device)
            for key, value in payload.items()
        }
        meta = {
            "detected_human_num": detected,
            "raw_pedestrian_count": int(ped_positions.shape[0]),
            "checkpoint_path": str(self.checkpoint_path),
        }
        return tensors, meta

    def act(
        self, observation: dict[str, Any], *, time_step: float
    ) -> tuple[float, float, dict[str, Any]]:
        """Run one recurrent inference step and return a holonomic ``(vx, vy)`` command.

        The command is clipped to the upstream preferred-speed envelope
        (magnitude <= ``v_pref``) to match the upstream ``clip_action`` behavior.

        Returns:
            Holonomic ``(vx, vy)`` command plus debug metadata.
        """
        if self._hidden_state is None:
            self.reset()
        if not np.isfinite(time_step) or time_step <= 0.0:
            raise ValueError(f"Invalid time_step: {time_step}")
        # The recurrent policy was trained at a fixed 0.25s cadence; surface a
        # contract mismatch rather than silently running at another dt.
        expected = float(self.config.time_step)
        if not math.isclose(float(time_step), expected, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "CrowdNav_Prediction_AttnGraph adapter expects a fixed time_step of "
                f"{expected:.6f}s (training cadence), got {float(time_step):.6f}s"
            )
        obs_tensors, meta = self._build_model_inputs(observation)
        with torch.no_grad():
            _value, action, _log_prob, self._hidden_state = self._policy.act(
                obs_tensors,
                self._hidden_state,
                self._mask,
                deterministic=True,
            )
        # First done step keeps mask=0 (fresh episode); subsequent steps mask=1.
        self._mask = torch.ones((1, 1), dtype=torch.float32, device=self._device)
        vx_raw = float(action.reshape(-1)[0].item())
        vy_raw = float(action.reshape(-1)[1].item())
        vx, vy = _clip_holonomic_to_v_pref(vx_raw, vy_raw, self.config.v_pref)
        meta.update(
            {
                "raw_action_xy": [vx_raw, vy_raw],
                "clipped_action_xy": [vx, vy],
                "projection_policy": self.projection_policy,
            }
        )
        return vx, vy, meta

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Map-runner entrypoint returning only the holonomic ``(vx, vy)`` command.

        Returns:
            Clipped holonomic velocity command pair.
        """
        dt = float(np.asarray(observation.get("dt", self.config.time_step)).reshape(-1)[0])
        vx, vy, _meta = self.act(observation, time_step=dt)
        return vx, vy


def _clip_holonomic_to_v_pref(vx: float, vy: float, v_pref: float) -> tuple[float, float]:
    """Replicate the upstream holonomic clip_action (normalize to v_pref).

    Returns:
        Velocity pair whose magnitude does not exceed ``v_pref``.
    """
    norm = float(np.hypot(vx, vy))
    if norm > v_pref and norm > 0.0:
        scale = v_pref / norm
        return vx * scale, vy * scale
    return vx, vy


__all__ = [
    "CrowdNavPredAttnGraphAdapter",
    "CrowdNavPredAttnGraphConfig",
    "build_crowdnav_pred_attng_config",
]
