"""SA-CADRL planner-family implementation extracted from the SocNav facade."""

import hashlib
import re
from math import atan2, pi
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from robot_sf.models import resolve_model_path
from robot_sf.planner import socnav as _socnav

SamplingPlannerAdapter = _socnav.SamplingPlannerAdapter
SocNavPlannerConfig = _socnav.SocNavPlannerConfig
SocNavPlannerPolicy = _socnav.SocNavPlannerPolicy


_SACADRL_STATE_ORDER = (
    "num_other_agents",
    "dist_to_goal",
    "heading_ego_frame",
    "pref_speed",
    "radius",
    "other_agents_states",
)


def _sacadrl_actions() -> np.ndarray:
    """Return the discrete GA3C-CADRL action set (speed scale, delta heading)."""
    actions = np.mgrid[1.0:1.1:0.5, -np.pi / 6 : np.pi / 6 + 0.01 : np.pi / 12].reshape(2, -1).T
    actions = np.vstack(
        [
            actions,
            np.mgrid[0.5:0.6:0.5, -np.pi / 6 : np.pi / 6 + 0.01 : np.pi / 6].reshape(2, -1).T,
        ]
    )
    actions = np.vstack(
        [
            actions,
            np.mgrid[0.0:0.1:0.5, -np.pi / 6 : np.pi / 6 + 0.01 : np.pi / 6].reshape(2, -1).T,
        ]
    )
    return actions


def _sacadrl_session_config(tf_module: Any, *, device: str):
    """Build a TensorFlow session config for SA-CADRL inference.

    Returns:
        TensorFlow ConfigProto: Session configuration matching the requested device.
    """
    kwargs: dict[str, Any] = {
        "allow_soft_placement": True,
        "log_device_placement": False,
        "gpu_options": tf_module.GPUOptions(allow_growth=True),
    }
    normalized_device = device.lower().replace(" ", "")
    if re.search(r"(^|/)(device:)?cpu(?::|$)", normalized_device):
        kwargs["device_count"] = {"GPU": 0}
    return tf_module.ConfigProto(**kwargs)


class _SACADRLModel:
    """Tensorflow checkpoint wrapper for GA3C-CADRL policy inference."""

    def __init__(self, checkpoint_prefix: Path, *, device: str = "/cpu:0"):
        """Load the GA3C-CADRL model from the provided checkpoint prefix."""
        if _socnav.tf is None:  # pragma: no cover - optional dependency
            raise RuntimeError("TensorFlow is required to run the GA3C-CADRL (SA-CADRL) baseline.")

        self._tf = _socnav.tf
        self._actions = _sacadrl_actions()
        self._graph = self._tf.Graph()
        with self._graph.as_default():
            with self._tf.device(device):
                self._sess = self._tf.Session(
                    graph=self._graph,
                    config=_sacadrl_session_config(self._tf, device=device),
                )
                saver = self._tf.train.import_meta_graph(
                    f"{checkpoint_prefix}.meta", clear_devices=True
                )
                self._sess.run(self._tf.global_variables_initializer())
                saver.restore(self._sess, str(checkpoint_prefix))
                self._softmax = self._graph.get_tensor_by_name("Softmax:0")
                self._x = self._graph.get_tensor_by_name("X:0")
        self._input_dim = int(self._x.shape[-1])

    @property
    def actions(self) -> np.ndarray:
        """Discrete action table of [speed_scale, delta_heading]."""
        return self._actions

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return softmax action probabilities for the provided observations."""
        obs = self._crop(obs)
        return self._sess.run(self._softmax, feed_dict={self._x: obs})

    def _crop(self, obs: np.ndarray) -> np.ndarray:
        """Pad or crop observations to match the expected input dimension.

        Returns:
            np.ndarray: Observation array sized to the network input dimension.
        """
        if obs.shape[-1] > self._input_dim:
            return obs[:, : self._input_dim]
        if obs.shape[-1] < self._input_dim:
            padded = np.zeros((obs.shape[0], self._input_dim), dtype=obs.dtype)
            padded[:, : obs.shape[1]] = obs
            return padded
        return obs


class SACADRLPlannerAdapter(SamplingPlannerAdapter):
    """GA3C-CADRL (SA-CADRL) planner adapter backed by a TensorFlow checkpoint.

    Set ``allow_fallback=True`` to permit heuristic behavior when the checkpoint
    or TensorFlow dependency is unavailable.
    """

    def __init__(self, config: SocNavPlannerConfig | None = None, *, allow_fallback: bool = False):
        """Initialize the adapter and configure optional heuristic fallback."""
        self.config = config or SocNavPlannerConfig()
        self._allow_fallback = allow_fallback
        self._model: _SACADRLModel | None = None
        self._load_error: Exception | None = None
        self._fallback_warned = False
        self._checkpoint_provenance: dict[str, Any] = {
            "model_id": self.config.sacadrl_model_id,
            "checkpoint_path": self.config.sacadrl_checkpoint_path,
            "checkpoint_sha256": None,
            "hash_source": None,
            "load_succeeded": None,
            "fallback_triggered": False,
            "load_status": "not_attempted",
            "load_error": None,
        }

    def diagnostics(self) -> dict[str, Any]:
        """Return runtime checkpoint provenance for benchmark episode metadata."""
        return {"checkpoint_provenance": dict(self._checkpoint_provenance)}

    def plan(self, observation: dict) -> tuple[float, float]:
        """
        Compute (v, w) using the GA3C-CADRL model (or fallback heuristics).

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        model = self._ensure_model()
        if model is None:
            return self._heuristic_plan(observation)

        obs_vec, pref_speed, dist_to_goal = self._build_network_input(observation)
        if dist_to_goal <= self.config.goal_tolerance:
            return 0.0, 0.0

        predictions = model.predict(obs_vec)[0]
        action_idx = int(np.argmax(predictions))
        raw_action = model.actions[action_idx]
        linear = float(pref_speed * raw_action[0])
        delta_heading = float(raw_action[1])

        time_step = float(
            np.asarray(observation.get("sim", {}).get("timestep", [0.1]), dtype=float)[0]
        )
        if time_step <= 1e-6:
            logger.warning(
                "Invalid timestep ({}) for SACADRLPlannerAdapter; defaulting to 0.1s.",
                time_step,
            )
            time_step = 0.1

        angular = float(delta_heading / time_step)
        linear = float(np.clip(linear, 0.0, self.config.max_linear_speed))
        angular = float(
            np.clip(angular, -self.config.max_angular_speed, self.config.max_angular_speed)
        )
        return linear, angular

    def _ensure_model(self) -> _SACADRLModel | None:
        """Load the model checkpoint on demand and honor fallback settings.

        Returns:
            _SACADRLModel | None: Loaded model instance or ``None`` when falling back.
        """
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None if self._allow_fallback else self._raise_cached_error()
        try:
            self._model = self._build_model()
            self._checkpoint_provenance.update(
                {
                    "load_succeeded": True,
                    "fallback_triggered": False,
                    "load_status": "loaded",
                    "load_error": None,
                }
            )
        except Exception as exc:
            self._checkpoint_provenance.update(
                {
                    "load_succeeded": False,
                    "fallback_triggered": bool(self._allow_fallback),
                    "load_status": "fallback" if self._allow_fallback else "failed",
                    "load_error": f"{type(exc).__name__}: {exc}",
                }
            )
            if self._allow_fallback:
                self._load_error = exc
                if not self._fallback_warned:
                    logger.warning(
                        "Falling back to heuristic SACADRL behavior: {}. "
                        "Set allow_fallback=False to fail fast.",
                        exc,
                    )
                    self._fallback_warned = True
                return None
            raise
        return self._model

    def _raise_cached_error(self) -> None:
        """Re-raise cached initialization error when fallback is disabled."""
        assert self._load_error is not None
        raise self._load_error

    def _build_model(self) -> _SACADRLModel:
        """Resolve the GA3C-CADRL checkpoint and construct the TF model wrapper.

        Returns:
            _SACADRLModel: Loaded GA3C-CADRL model wrapper.
        """
        checkpoint_prefix = self._resolve_checkpoint_prefix()
        return _SACADRLModel(checkpoint_prefix, device="/cpu:0")

    def _resolve_checkpoint_prefix(self) -> Path:
        """Resolve the model checkpoint prefix for the GA3C-CADRL checkpoint.

        Returns:
            Path: Checkpoint prefix path without file extensions.
        """
        if self.config.sacadrl_checkpoint_path:
            checkpoint_path = Path(self.config.sacadrl_checkpoint_path).expanduser()
        else:
            checkpoint_path = resolve_model_path(self.config.sacadrl_model_id)

        if checkpoint_path.suffix == ".meta":
            prefix = checkpoint_path.with_suffix("")
        else:
            prefix = checkpoint_path

        meta_path = prefix.with_suffix(".meta")
        if not meta_path.exists():
            raise FileNotFoundError(f"GA3C-CADRL checkpoint meta file not found: {meta_path}")
        index_path = prefix.with_suffix(".index")
        if not index_path.exists():
            raise FileNotFoundError(f"GA3C-CADRL checkpoint index file not found: {index_path}")
        data_files = list(prefix.parent.glob(f"{prefix.name}.data*"))
        if not data_files:
            raise FileNotFoundError(
                f"GA3C-CADRL checkpoint data file not found for prefix: {prefix}"
            )
        checkpoint_files = sorted([meta_path, index_path, *data_files], key=lambda path: path.name)
        digest = hashlib.sha256()
        for path in checkpoint_files:
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        self._checkpoint_provenance.update(
            {
                "checkpoint_path": str(meta_path.resolve()),
                "checkpoint_sha256": digest.hexdigest(),
                "hash_source": "computed_tensorflow_checkpoint_bundle",
            }
        )
        return prefix

    def _compute_goal_frame(
        self, robot_pos: np.ndarray, robot_heading: float, goal: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, float]:
        """Compute the goal-aligned frame and heading in ego coordinates.

        Returns:
            tuple[float, np.ndarray, np.ndarray, float]: Distance to goal, parallel unit
            vector, orthogonal unit vector, and heading in ego frame.
        """
        to_goal = goal - robot_pos
        dist_to_goal = float(np.linalg.norm(to_goal))
        if dist_to_goal > 1e-8:
            ref_prll = to_goal / dist_to_goal
        else:
            ref_prll = np.array([1.0, 0.0], dtype=float)
        ref_orth = np.array([-ref_prll[1], ref_prll[0]], dtype=float)
        ref_angle = atan2(ref_prll[1], ref_prll[0])
        heading_ego_frame = self._wrap_angle(robot_heading - ref_angle)
        return dist_to_goal, ref_prll, ref_orth, heading_ego_frame

    def _normalize_pedestrians(  # noqa: C901
        self, ped_state: dict
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Normalize pedestrian positions/velocities and radius from observation.

        Returns:
            tuple[np.ndarray, np.ndarray, float]: Positions array, velocities array,
            and shared pedestrian radius.
        """
        ped_positions = np.asarray(ped_state.get("positions", []), dtype=float)
        if ped_positions.size == 0:
            ped_positions = np.zeros((0, 2), dtype=float)
        elif ped_positions.ndim == 1:
            ped_positions = (
                ped_positions.reshape(-1, 2)
                if ped_positions.size % 2 == 0
                else np.zeros((0, 2), dtype=float)
            )
        elif ped_positions.ndim == 2 and ped_positions.shape[1] != 2:
            if ped_positions.shape[1] > 2:
                ped_positions = ped_positions[:, :2]
            else:
                ped_positions = np.pad(
                    ped_positions,
                    ((0, 0), (0, 2 - ped_positions.shape[1])),
                    constant_values=0.0,
                )
        elif ped_positions.ndim != 2:
            ped_positions = np.zeros((0, 2), dtype=float)

        count_arr = np.asarray(
            ped_state.get("count", [ped_positions.shape[0]]), dtype=float
        ).reshape(-1)
        ped_count = int(count_arr[0]) if count_arr.size else int(ped_positions.shape[0])
        ped_count = max(0, min(ped_count, int(ped_positions.shape[0])))
        ped_positions = ped_positions[:ped_count]

        ped_velocities = np.asarray(ped_state.get("velocities", []), dtype=float)
        if ped_velocities.size == 0:
            ped_velocities = np.zeros_like(ped_positions, dtype=float)
        elif ped_velocities.ndim == 1:
            ped_velocities = (
                ped_velocities.reshape(-1, 2)
                if ped_velocities.size % 2 == 0
                else np.zeros((0, 2), dtype=float)
            )
        elif ped_velocities.ndim == 2 and ped_velocities.shape[1] != 2:
            if ped_velocities.shape[1] > 2:
                ped_velocities = ped_velocities[:, :2]
            else:
                ped_velocities = np.pad(
                    ped_velocities,
                    ((0, 0), (0, 2 - ped_velocities.shape[1])),
                    constant_values=0.0,
                )
        elif ped_velocities.ndim != 2:
            ped_velocities = np.zeros((0, 2), dtype=float)

        if ped_velocities.shape[0] < ped_count:
            pad_rows = ped_count - ped_velocities.shape[0]
            ped_velocities = np.pad(
                ped_velocities,
                ((0, pad_rows), (0, 0)),
                constant_values=0.0,
            )
        ped_velocities = ped_velocities[:ped_count]

        radius_arr = np.asarray(ped_state.get("radius", [0.3]), dtype=float).reshape(-1)
        ped_radius = float(radius_arr[0]) if radius_arr.size else 0.3
        return ped_positions, ped_velocities, ped_radius

    def _ego_to_global_velocities(
        self, robot_heading: float, ped_velocities: np.ndarray
    ) -> np.ndarray:
        """Convert ego-frame pedestrian velocities to global-frame velocities.

        Returns:
            np.ndarray: Global-frame velocities with the same shape as input.
        """
        cos_h = np.cos(robot_heading)
        sin_h = np.sin(robot_heading)
        v_global = np.zeros_like(ped_velocities, dtype=float)
        if ped_velocities.size:
            v_global[:, 0] = cos_h * ped_velocities[:, 0] - sin_h * ped_velocities[:, 1]
            v_global[:, 1] = sin_h * ped_velocities[:, 0] + cos_h * ped_velocities[:, 1]
        return v_global

    def _build_other_agents_states(
        self,
        ped_positions: np.ndarray,
        ped_velocities: np.ndarray,
        robot_pos: np.ndarray,
        robot_radius: float,
        ped_radius: float,
        ref_prll: np.ndarray,
        ref_orth: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Build the ordered other-agent state tensor for the GA3C-CADRL input.

        Returns:
            tuple[np.ndarray, float]: Other-agent state matrix and count.
        """
        max_other = max(0, int(self.config.sacadrl_max_other_agents))
        other_states = np.zeros((max_other, 7), dtype=float)
        sorting = []
        for idx in range(ped_positions.shape[0]):
            rel = ped_positions[idx] - robot_pos
            dist_center = float(np.linalg.norm(rel))
            dist_2_other = dist_center - robot_radius - ped_radius
            p_orth = float(np.dot(rel, ref_orth))
            sorting.append((idx, dist_2_other, p_orth))

        if sorting:
            if self.config.sacadrl_sorting_method == "closest_last":
                sorted_ids = sorted(sorting, key=lambda x: (-x[1], x[2]))
            else:
                sorted_ids = sorted(sorting, key=lambda x: (x[1], x[2]))
            selected = [idx for idx, _dist, _orth in sorted_ids[:max_other]]
        else:
            selected = []

        for slot, idx in enumerate(selected):
            rel = ped_positions[idx] - robot_pos
            p_parallel = float(np.dot(rel, ref_prll))
            p_orth = float(np.dot(rel, ref_orth))
            v_parallel = float(np.dot(ped_velocities[idx], ref_prll))
            v_orth = float(np.dot(ped_velocities[idx], ref_orth))
            dist_2_other = float(np.linalg.norm(rel) - robot_radius - ped_radius)
            combined_radius = robot_radius + ped_radius
            other_states[slot] = np.array(
                [
                    p_parallel,
                    p_orth,
                    v_parallel,
                    v_orth,
                    ped_radius,
                    combined_radius,
                    dist_2_other,
                ],
                dtype=float,
            )
        return other_states, float(len(selected))

    def _build_network_input(self, observation: dict) -> tuple[np.ndarray, float, float]:
        """Convert a SocNav observation into the GA3C-CADRL network input vector.

        Returns:
            tuple[np.ndarray, float, float]: Batched observation vector, preferred speed,
            and distance to goal.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
        robot_radius = float(np.asarray(robot_state.get("radius", [0.3]), dtype=float)[0])

        goal = np.asarray(goal_state["current"], dtype=float)
        dist_to_goal, ref_prll, ref_orth, heading_ego_frame = self._compute_goal_frame(
            robot_pos, robot_heading, goal
        )

        ped_positions, ped_velocities, ped_radius = self._normalize_pedestrians(ped_state)
        v_global = self._ego_to_global_velocities(robot_heading, ped_velocities)
        other_states, num_other_agents = self._build_other_agents_states(
            ped_positions,
            v_global,
            robot_pos,
            robot_radius,
            ped_radius,
            ref_prll,
            ref_orth,
        )
        pref_speed = float(self.config.sacadrl_pref_speed)

        obs_dict = {
            "num_other_agents": np.array([num_other_agents], dtype=np.float32),
            "dist_to_goal": np.array([dist_to_goal], dtype=np.float32),
            "heading_ego_frame": np.array([heading_ego_frame], dtype=np.float32),
            "pref_speed": np.array([pref_speed], dtype=np.float32),
            "radius": np.array([robot_radius], dtype=np.float32),
            "other_agents_states": other_states.astype(np.float32),
        }
        vec_obs = np.array([], dtype=np.float32)
        for state in _SACADRL_STATE_ORDER:
            vec_obs = np.hstack([vec_obs, obs_dict[state].flatten()])
        vec_obs = np.expand_dims(vec_obs, axis=0)
        return vec_obs, pref_speed, dist_to_goal

    def _heuristic_plan(self, observation: dict) -> tuple[float, float]:
        """Fallback heuristic that biases toward the goal while repulsing pedestrians.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
        goal = np.asarray(goal_state["current"], dtype=float)

        to_goal = goal - robot_pos
        goal_vec = to_goal / (np.linalg.norm(to_goal) + 1e-6)

        ped_positions, _ped_velocities, _ped_radius = self._normalize_pedestrians(ped_state)
        if ped_positions.shape[0] > 0:
            dists = np.linalg.norm(ped_positions - robot_pos, axis=1)
            neighbor_count = max(0, int(self.config.sacadrl_neighbors))
            nearest_idx = np.argsort(dists)[:neighbor_count]
            bias = np.zeros(2, dtype=float)
            for idx in nearest_idx:
                delta = robot_pos - ped_positions[idx]
                dist = dists[idx] + 1e-6
                bias += delta / dist**1.5
            combined = goal_vec + self.config.sacadrl_bias_weight * bias
        else:
            combined = goal_vec

        if np.linalg.norm(combined) > 1e-6:
            combined = combined / np.linalg.norm(combined)

        combined, occ_penalty = self._get_safe_heading(robot_pos, combined, observation)

        desired_heading = atan2(combined[1], combined[0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)
        angular = float(
            np.clip(
                self.config.angular_gain * heading_error,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            ),
        )
        linear = float(
            np.clip(
                np.linalg.norm(to_goal),
                0.0,
                self.config.max_linear_speed
                * max(0.0, 1.0 - abs(heading_error) / pi)
                * max(0.0, 1.0 - occ_penalty),
            ),
        )
        return linear, angular


def make_sacadrl_policy(
    config: SocNavPlannerConfig | None = None, *, allow_fallback: bool = False
) -> SocNavPlannerPolicy:
    """
    Convenience constructor for GA3C-CADRL (SA-CADRL) planner policy.

    Set ``allow_fallback=True`` to use heuristic behavior when the checkpoint
    cannot be loaded (e.g., missing TensorFlow dependency).

    Returns:
        SocNavPlannerPolicy: Policy wrapping SACADRLPlannerAdapter.
    """

    return SocNavPlannerPolicy(
        adapter=SACADRLPlannerAdapter(config=config, allow_fallback=allow_fallback)
    )
