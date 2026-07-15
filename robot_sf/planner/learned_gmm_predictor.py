"""Learned K-mode GMM pedestrian predictor for chance-constrained MPC #5307.

Implements the ``GaussianMixturePedestrianPredictor`` protocol (defined in
``chance_constrained_mpc.py``) with a small MLP that outputs K-mode GMM
forecast parameters.  An untrained (zero-initialised) smoke mode lets the
integration be validated on CPU without a trained checkpoint.  The real
checkpoint-loaded mode expects a model trained by the #2844 forecast lane.

Architecture
------------
The original ``mlp`` model type encodes the current scene (robot state, goal
offset, relative pedestrian positions and velocities) into a fixed-size feature
vector, feeds it through a small MLP, and reshapes the output to produce per-mode
position deltas (relative to a constant-velocity baseline), log-variances,
correlation parameters, and unnormalised mode-weight logits for each pedestrian.

The opt-in ``graph_gru`` model type is the bounded #2844 progression slice. It
encodes each pedestrian as a graph node, mean-pools valid node embeddings into
scene context, and uses a GRU decoder to emit K trajectory modes with a diagonal
Gaussian head. Mode weights remain equal in this scaffold; learning mode weights
or correlations belongs to a later checkpoint/training slice.

When either backend is zero-initialised (untrained smoke mode), every mode
predicts the constant-velocity trajectory with unit isotropic covariances and
equal mode weights, so the end-to-end chance-constrained MPC control law is
exercisable.
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.models.registry import resolve_model_path
from robot_sf.planner.chance_constrained_mpc import GaussianMixturePedestrianForecast
from robot_sf.planner.nmpc_social import _parse_bool

GRAPH_NODE_FEATURE_DIM: int = 4
GRAPH_GLOBAL_FEATURE_DIM: int = 6
GRAPH_GAUSSIAN_PARAM_DIM: int = 4

# ── helpers reused from learned_short_horizon_predictor ──────────────────────


def _as_1d(value: Any, *, pad: int) -> np.ndarray:
    """Return one-dimensional float array padded to ``pad`` length."""
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < pad:
        arr = np.pad(arr, (0, pad - arr.size), mode="constant")
    return arr


def _as_xy(value: Any) -> np.ndarray:
    """Return first two numeric values as a (2,) vector."""
    return _as_1d(value, pad=2)[:2].astype(float)


def _as_xy_matrix(value: Any) -> np.ndarray:
    """Return (N, 2) float matrix or empty for invalid inputs."""
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    if arr.ndim == 1 and arr.size % 2 == 0:
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[-1] != 2:
        return np.zeros((0, 2), dtype=float)
    return arr.astype(float)


def _optional_str(value: Any) -> str | None:
    """Normalise optional string values from YAML.

    Returns:
        Stripped string, or ``None`` for empty/None inputs.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_torch() -> tuple[Any, Any]:
    """Load PyTorch lazily so ordinary planner imports stay dependency-light.

    Returns:
        ``(torch, torch.nn)`` module tuple.
    """
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "learned GMM predictor requires torch for checkpoint or smoke mode"
        ) from exc
    return torch, nn


def _pedestrian_world_state(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Extract pedestrian positions and rotate ego-frame velocities into world frame.

    Returns:
        (N, 2) positions and (N, 2) world-frame velocities.
    """
    ped_state = observation.get("pedestrians", {})
    robot = observation.get("robot", {})
    robot_heading = float(_as_1d(robot.get("heading", [0.0]), pad=1)[0])
    positions = _as_xy_matrix(ped_state.get("positions", []))
    velocities_ego = _as_xy_matrix(ped_state.get("velocities", []))
    count = int(_as_1d(ped_state.get("count", [positions.shape[0]]), pad=1)[0])
    count = max(0, min(count, positions.shape[0]))
    positions = positions[:count]
    if velocities_ego.shape[0] < count:
        velocities_ego = np.zeros_like(positions)
    else:
        velocities_ego = velocities_ego[:count]
    cos_h = float(np.cos(robot_heading))
    sin_h = float(np.sin(robot_heading))
    velocities_world = np.empty_like(velocities_ego)
    velocities_world[:, 0] = cos_h * velocities_ego[:, 0] - sin_h * velocities_ego[:, 1]
    velocities_world[:, 1] = sin_h * velocities_ego[:, 0] + cos_h * velocities_ego[:, 1]
    return positions, velocities_world


# ── config ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LearnedGmmPredictorConfig:
    """Configuration for the learned K-mode GMM pedestrian predictor.

    The historical ``mlp`` backend outputs per-mode position deltas,
    log-variances, correlation parameters, and weight logits. The opt-in
    ``graph_gru`` backend uses masked graph-node pooling and emits diagonal
    Gaussian parameters with equal mode weights. When ``allow_untrained_smoke``
    is true and no checkpoint is provided, either backend is zero-initialised so
    it behaves as a constant-velocity baseline with isotropic unit covariances —
    enabling end-to-end CPU validation of the chance-constrained MPC control law
    without a trained model.

    Attributes:
        checkpoint_path: Path to a PyTorch state-dict checkpoint (``.pt``).
        model_id: Registry ID from ``robot_sf.models.registry``.
        device: Torch device string (default ``"cpu"``).
        max_pedestrians: Maximum number of tracked pedestrians (for fixed-size
            feature encoding).
        horizon_steps: Forecast horizon steps (must match the MPC horizon).
        rollout_dt: Timestep in seconds (must match MPC ``rollout_dt``).
        hidden_dim: Hidden-layer width for the tiny MLP.
        mode_count: Number of Gaussian modes per pedestrian (K >= 1).
        model_type: ``"mlp"`` preserves the existing flat-feature backend. The
            opt-in ``"graph_gru"`` backend is the #2844 graph-pooled GRU scaffold
            with K diagonal-Gaussian modes.
        allow_untrained_smoke: If true and no checkpoint, zero-initialise the
            network for diagnostic smoke tests.
    """

    checkpoint_path: str | None = None
    model_id: str | None = None
    device: str = "cpu"
    max_pedestrians: int = 16
    horizon_steps: int = 6
    rollout_dt: float = 0.25
    hidden_dim: int = 128
    mode_count: int = 3
    model_type: str = "mlp"
    allow_untrained_smoke: bool = False


# ── GMM output shape helpers ────────────────────────────────────────────────


def predictor_io_dims(config: LearnedGmmPredictorConfig) -> tuple[int, int]:
    """Return ``(input_dim, output_dim)`` implied by a predictor config.

    The input feature vector encodes the robot state (4), goal offset (2), and
    per-pedestrian relative position + ego-frame velocity (4 each) for up to
    ``max_pedestrians`` pedestrians.

    The output contains, for each mode, two position deltas (relative to CV),
    two log-variances, one correlation (atanh(rho)), and one weight logit:
    K * (2 + 2 + 1 + 1) = K * 6 parameters per pedestrian per horizon step.

    Returns:
        (int, int): Feature dimension and flattened output dimension.
    """
    input_dim = 4 + 2 + int(config.max_pedestrians) * 4
    output_dim = (
        int(config.max_pedestrians)
        * int(config.horizon_steps)
        * int(config.mode_count)
        * 6  # delta_x, delta_y, log_std_x, log_std_y, atanh_rho, weight_logit
    )
    return input_dim, output_dim


def graph_predictor_io_dims(config: LearnedGmmPredictorConfig) -> tuple[int, int, int]:
    """Return graph-node, graph-global, and flattened output dimensions.

    The graph backend consumes fixed-width node features and one scene-level context
    vector. Its head emits two mean deltas and two log standard deviations per mode
    and timestep. Mode weights are deliberately fixed to equal values in this
    scaffold.

    Returns:
        ``(node_feature_dim, global_feature_dim, output_dim)``.
    """
    output_dim = (
        int(config.max_pedestrians)
        * int(config.horizon_steps)
        * int(config.mode_count)
        * GRAPH_GAUSSIAN_PARAM_DIM
    )
    return GRAPH_NODE_FEATURE_DIM, GRAPH_GLOBAL_FEATURE_DIM, output_dim


# ── feature encoding ─────────────────────────────────────────────────────────


def encode_gmm_predictor_features(
    observation: dict[str, Any],
    ped_positions: np.ndarray,
    ped_velocities_world: np.ndarray,
    *,
    max_pedestrians: int,
) -> np.ndarray:
    """Encode observation and world-frame pedestrian state into a feature vector.

    The encoding mirrors ``learned_short_horizon_predictor.encode_predictor_features``
    for feature-level compatibility: robot position (2), heading (1), speed (1),
    goal delta (2), and per-pedestrian relative position + world-frame velocity
    (4 each).

    Args:
        observation: SocNav-structured observation dict.
        ped_positions: (N, 2) world-frame pedestrian positions.
        ped_velocities_world: (N, 2) world-frame pedestrian velocities.
        max_pedestrians: Fixed array size for batched inference.

    Returns:
        (input_dim,) float feature vector.
    """
    robot = observation.get("robot", {})
    goal = observation.get("goal", {})
    robot_pos = _as_xy(robot.get("position", [0.0, 0.0]))
    heading = float(_as_1d(robot.get("heading", [0.0]), pad=1)[0])
    speed = float(_as_1d(robot.get("speed", [0.0]), pad=1)[0])
    goal_pos = _as_xy(goal.get("current", goal.get("next", [0.0, 0.0])))

    max_p = int(max_pedestrians)
    features = np.zeros((4 + 2 + max_p * 4,), dtype=float)
    features[:4] = np.asarray([robot_pos[0], robot_pos[1], heading, speed], dtype=float)
    features[4:6] = goal_pos - robot_pos
    offset = 6
    count = min(ped_positions.shape[0], max_p)
    for idx in range(count):
        features[offset : offset + 2] = ped_positions[idx] - robot_pos
        features[offset + 2 : offset + 4] = ped_velocities_world[idx]
        offset += 4
    return features


def encode_graph_predictor_features(
    observation: dict[str, Any],
    ped_positions: np.ndarray,
    ped_velocities_world: np.ndarray,
    *,
    max_pedestrians: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode valid pedestrian nodes and scene context for ``graph_gru``.

    Returns:
        Tuple of ``(node_features, node_mask, global_features)``. Node features are
        relative position and world-frame velocity with shape ``(N, 4)``. The global
        vector contains robot position, heading, speed, and goal offset with shape
        ``(6,)``. Invalid padded nodes are zero-filled and masked out.
    """
    robot = observation.get("robot", {})
    goal = observation.get("goal", {})
    robot_pos = _as_xy(robot.get("position", [0.0, 0.0]))
    heading = float(_as_1d(robot.get("heading", [0.0]), pad=1)[0])
    speed = float(_as_1d(robot.get("speed", [0.0]), pad=1)[0])
    goal_pos = _as_xy(goal.get("current", goal.get("next", [0.0, 0.0])))

    node_count = int(max_pedestrians)
    if node_count <= 0:
        raise ValueError("max_pedestrians must be strictly positive")
    node_features = np.zeros((node_count, GRAPH_NODE_FEATURE_DIM), dtype=float)
    node_mask = np.zeros((node_count,), dtype=float)
    positions = np.asarray(ped_positions, dtype=float)
    velocities = np.asarray(ped_velocities_world, dtype=float)
    count = min(node_count, positions.shape[0], velocities.shape[0])
    if count:
        node_features[:count, :2] = positions[:count] - robot_pos
        node_features[:count, 2:] = velocities[:count]
        node_mask[:count] = 1.0

    global_features = np.asarray(
        [
            robot_pos[0],
            robot_pos[1],
            heading,
            speed,
            goal_pos[0] - robot_pos[0],
            goal_pos[1] - robot_pos[1],
        ],
        dtype=float,
    )
    return node_features, node_mask, global_features


# ── Torch wrapper ────────────────────────────────────────────────────────────


class _TinyGmmMlp:
    """Lazy PyTorch MLP that outputs K-mode GMM parameters.

    The network is a single-hidden-layer MLP with tanh activation.  Its output
    is reshaped to ``(max_pedestrians, horizon_steps, mode_count, 6)`` where the
    last axis holds (delta_x, delta_y, log_std_x, log_std_y, atanh_rho,
    weight_logit) per mode per timestep per pedestrian.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        device: str,
    ) -> None:
        """Build the MLP module on the requested device."""
        torch, nn = _load_torch()
        self.torch = torch
        self.module = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        ).to(device)

    def zero_initialize(self) -> None:
        """Zero all parameters so untrained output is the identity (CV + unit cov)."""
        for param in self.module.parameters():
            param.data.zero_()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load checkpoint weights."""
        self.module.load_state_dict(state_dict)

    def predict_numpy(self, features: np.ndarray) -> np.ndarray:
        """Run a single feature vector through the model.

        Returns:
            (max_pedestrians, horizon_steps, mode_count, 6) float array.
        """
        torch = self.torch
        self.module.eval()
        device = next(self.module.parameters()).device
        with torch.no_grad():
            tensor = torch.as_tensor(features[None, :], dtype=torch.float32, device=device)
            output = self.module(tensor).detach().cpu().numpy()
        return np.asarray(output[0], dtype=float)


class _TinyGraphGmmGru:
    """Lazy graph-pooled gated recurrent unit (GRU) with a diagonal GMM head."""

    def __init__(
        self,
        *,
        max_pedestrians: int,
        horizon_steps: int,
        mode_count: int,
        hidden_dim: int,
        device: str,
    ) -> None:
        """Build the graph encoder, masked pool, and GRU decoder on ``device``."""
        torch, nn = _load_torch()
        self.torch = torch
        self.max_pedestrians = int(max_pedestrians)
        node_feature_dim = GRAPH_NODE_FEATURE_DIM
        global_feature_dim = GRAPH_GLOBAL_FEATURE_DIM
        output_dim = int(mode_count) * GRAPH_GAUSSIAN_PARAM_DIM
        horizon = int(horizon_steps)

        class GraphGruModule(nn.Module):
            """Torch implementation kept local so importing this module stays lazy."""

            def __init__(self) -> None:
                super().__init__()
                self.node_encoder = nn.Sequential(
                    nn.Linear(node_feature_dim, hidden_dim),
                    nn.Tanh(),
                )
                self.global_encoder = nn.Sequential(
                    nn.Linear(global_feature_dim, hidden_dim),
                    nn.Tanh(),
                )
                self.decoder = nn.GRU(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    batch_first=True,
                )
                self.head = nn.Linear(hidden_dim, output_dim)

            def forward(
                self,
                node_features: Any,
                node_mask: Any,
                global_features: Any,
            ) -> Any:
                mask = node_mask.float().clamp(0.0, 1.0)
                encoded = self.node_encoder(node_features) * mask.unsqueeze(-1)
                pooled = encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                context = (
                    encoded
                    + pooled.unsqueeze(1)
                    + self.global_encoder(global_features).unsqueeze(1)
                )
                batch_size, node_count, _ = context.shape
                decoder_input = (
                    context.unsqueeze(2)
                    .expand(batch_size, node_count, horizon, hidden_dim)
                    .reshape(batch_size * node_count, horizon, hidden_dim)
                )
                decoded, _ = self.decoder(decoder_input)
                raw = self.head(decoded).reshape(
                    batch_size,
                    node_count,
                    horizon,
                    int(mode_count),
                    GRAPH_GAUSSIAN_PARAM_DIM,
                )
                return raw * mask[:, :, None, None, None]

        self.module = GraphGruModule().to(device)

    def zero_initialize(self) -> None:
        """Make untrained smoke output exactly the constant-velocity baseline."""
        for param in self.module.parameters():
            param.data.zero_()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a graph-GRU checkpoint state dict."""
        self.module.load_state_dict(state_dict)

    def predict_numpy(
        self,
        node_features: np.ndarray,
        node_mask: np.ndarray,
        global_features: np.ndarray,
    ) -> np.ndarray:
        """Run one padded graph through the model and return raw head parameters.

        Returns:
            Raw graph-head parameters with shape ``(N, T, K, 4)``.
        """
        if node_features.shape[0] != self.max_pedestrians:
            raise ValueError(
                "graph node count must match the configured max_pedestrians: "
                f"{node_features.shape[0]} != {self.max_pedestrians}"
            )
        torch = self.torch
        self.module.eval()
        device = next(self.module.parameters()).device
        with torch.no_grad():
            output = self.module(
                torch.as_tensor(node_features[None, :], dtype=torch.float32, device=device),
                torch.as_tensor(node_mask[None, :], dtype=torch.float32, device=device),
                torch.as_tensor(global_features[None, :], dtype=torch.float32, device=device),
            )
        return np.asarray(output.detach().cpu().numpy()[0], dtype=float)


# ── GMM decoder: raw MLP output → GaussianMixturePedestrianForecast ──────────


def decode_gmm_forecast(
    raw_output: np.ndarray,
    ped_positions: np.ndarray,
    ped_velocities_world: np.ndarray,
    *,
    dt: float,
    horizon_steps: int,
    mode_count: int,
    max_pedestrians: int,
    source: str = "learned_gmm_predictor",
) -> GaussianMixturePedestrianForecast:
    """Decode raw MLP output into a validated GMM forecast.

    The raw output has shape ``(max_pedestrians * horizon_steps * mode_count * 6,)``
    and must be reshaped to ``(actual_count, horizon, K, 6)`` where the last axis
    holds:
      0: delta_x     (position offset relative to CV baseline, world-frame)
      1: delta_y     (position offset relative to CV baseline)
      2: log_std_x   (log of per-axis standard deviation)
      3: log_std_y   (log of per-axis standard deviation)
      4: atanh_rho   (inverse-hyperbolic-tangent of the correlation coefficient)
      5: weight_logit (unnormalised mode weight)

    When the network is zero-initialised (untrained smoke mode), every mode
    produces zero deltas, zero log-stds (= std=1), zero correlation, and equal
    logits (= equal weights), which recovers a constant-velocity isotropic
    unit-covariance baseline.

    Args:
        raw_output: Flat MLP output vector.
        ped_positions: (N, 2) world-frame pedestrian positions for this step.
        ped_velocities_world: (N, 2) world-frame velocities.
        dt: Forecast timestep.
        horizon_steps: Number of forecast steps.
        mode_count: Number of Gaussian modes (K).
        max_pedestrians: Max pedestrians the network was built for.
        source: Name or identifier of the forecast source.

    Returns:
        A validated ``GaussianMixturePedestrianForecast``.
    """
    k = int(mode_count)
    steps = max(int(horizon_steps), 1)
    count = ped_positions.shape[0]

    # Reshape raw output to (max_pedestrians, horizon, K, 6)
    expected_elements = int(max_pedestrians) * steps * k * 6
    if raw_output.size != expected_elements:
        raise ValueError(
            f"MLP output size {raw_output.size} does not match "
            f"expected {expected_elements} (max_peds={max_pedestrians}, "
            f"horizon={steps}, modes={k})"
        )

    params = raw_output.reshape(int(max_pedestrians), steps, k, 6)

    # Build per-mode arrays for the actual pedestrian count
    means = np.zeros((count, k, steps, 2), dtype=float)
    covariances = np.tile(np.eye(2), (count, k, steps, 1, 1))
    weight_logits = np.zeros((count, k), dtype=float)

    for step in range(steps):
        tau = float(step + 1) * float(dt)
        for mode in range(k):
            # CV baseline
            cv_positions = ped_positions + ped_velocities_world * tau

            # Read mode parameters for active pedestrians
            deltas = params[:count, step, mode, :2]  # (N, 2)
            log_stds = params[:count, step, mode, 2:4]  # (N, 2)
            atanh_rhos = params[:count, step, mode, 4]  # (N,)
            wl = params[:count, step, mode, 5]  # (N,) weight logit

            means[:, mode, step, :] = cv_positions + deltas

            stds = np.exp(np.clip(log_stds, -10.0, 10.0))  # (N, 2)
            rho = np.tanh(atanh_rhos)  # (N,)  correlation in (-1, 1)
            for p_idx in range(count):
                sxx = stds[p_idx, 0] ** 2
                syy = stds[p_idx, 1] ** 2
                sxy = rho[p_idx] * stds[p_idx, 0] * stds[p_idx, 1]
                covariances[p_idx, mode, step] = np.asarray([[sxx, sxy], [sxy, syy]], dtype=float)

            weight_logits[:count, mode] += wl

    # Convert logits to normalised weights
    weights = np.zeros((count, k), dtype=float)
    if k > 1:
        # Numerically stable softmax over modes
        logits = weight_logits[:count, :]
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits)
        weights = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    else:
        weights[:] = 1.0

    return GaussianMixturePedestrianForecast(
        means_world=means.astype(float),
        covariances_world=covariances.astype(float),
        mode_weights=weights.astype(float),
        dt=float(dt),
        source=source,
    )


def decode_graph_gmm_forecast(
    raw_output: np.ndarray,
    ped_positions: np.ndarray,
    ped_velocities_world: np.ndarray,
    *,
    dt: float,
    horizon_steps: int,
    mode_count: int,
    max_pedestrians: int,
    source: str = "learned_graph_gru_predictor",
) -> GaussianMixturePedestrianForecast:
    """Decode graph-GRU output into equal-weight diagonal Gaussian modes.

    The four head values are ``delta_x``, ``delta_y``, ``log_std_x``, and
    ``log_std_y``. The decoder intentionally does not infer correlations or
    learned mode weights, so the scaffold's uncertainty contract stays explicit
    and easy to validate before a trained checkpoint exists.

    Returns:
        Validated K-mode Gaussian forecast with diagonal covariance matrices.
    """
    steps = max(int(horizon_steps), 1)
    modes = int(mode_count)
    if modes < 1:
        raise ValueError("mode_count must be >= 1")
    count = int(np.asarray(ped_positions).shape[0])
    if count > int(max_pedestrians):
        raise ValueError("ped_positions cannot exceed max_pedestrians")
    expected_elements = int(max_pedestrians) * steps * modes * GRAPH_GAUSSIAN_PARAM_DIM
    if raw_output.size != expected_elements:
        raise ValueError(
            f"graph-GRU output size {raw_output.size} does not match expected "
            f"{expected_elements} (max_peds={max_pedestrians}, horizon={steps}, modes={modes})"
        )
    params = np.asarray(raw_output, dtype=float).reshape(
        int(max_pedestrians),
        steps,
        modes,
        GRAPH_GAUSSIAN_PARAM_DIM,
    )
    positions = np.asarray(ped_positions, dtype=float)[:count]
    velocities = np.asarray(ped_velocities_world, dtype=float)[:count]
    means = np.zeros((count, modes, steps, 2), dtype=float)
    covariances = np.zeros((count, modes, steps, 2, 2), dtype=float)
    for step in range(steps):
        cv_positions = positions + velocities * ((step + 1) * float(dt))
        for mode in range(modes):
            mode_params = params[:count, step, mode]
            means[:, mode, step, :] = cv_positions + mode_params[:, :2]
            std = np.exp(np.clip(mode_params[:, 2:4], -10.0, 10.0))
            covariances[:, mode, step, 0, 0] = std[:, 0] ** 2
            covariances[:, mode, step, 1, 1] = std[:, 1] ** 2
    weights = np.full((count, modes), 1.0 / modes, dtype=float)
    return GaussianMixturePedestrianForecast(
        means_world=means,
        covariances_world=covariances,
        mode_weights=weights,
        dt=float(dt),
        source=source,
    )


# ── predictor module builder ────────────────────────────────────────────────


def build_gmm_predictor_module(
    *, input_dim: int, output_dim: int, hidden_dim: int, device: str
) -> Any:
    """Build the canonical tiny MLP for GMM pedestrian forecasting.

    Returns:
        A ``torch.nn.Sequential`` module on the requested device.
    """
    _, nn = _load_torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    ).to(device)


# ── learned GMM predictor ────────────────────────────────────────────────────


class LearnedGmmPedestrianPredictor:
    """Learned K-mode GMM pedestrian predictor for chance-constrained MPC.

    Implements the ``GaussianMixturePedestrianPredictor`` protocol.  When a
    checkpoint is provided the selected model loads its weights; otherwise, if
    ``allow_untrained_smoke`` is set, the network is zero-initialised so every
    mode predicts a constant-velocity baseline with isotropic unit covariances
    and equal weights.  This enables end-to-end CPU validation of the
    chance-constrained MPC control law without a trained model.
    """

    def __init__(self, config: LearnedGmmPredictorConfig) -> None:
        """Initialise the predictor, loading a checkpoint or entering smoke mode."""
        if config.max_pedestrians <= 0:
            raise ValueError("max_pedestrians must be strictly positive")
        if config.horizon_steps <= 0:
            raise ValueError("horizon_steps must be strictly positive")
        if config.mode_count < 1:
            raise ValueError("mode_count must be >= 1")
        if config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be strictly positive")
        try:
            rollout_dt = float(config.rollout_dt)
        except (TypeError, ValueError) as exc:
            raise ValueError("rollout_dt must be finite and strictly positive") from exc
        if not np.isfinite(rollout_dt) or rollout_dt <= 0.0:
            raise ValueError("rollout_dt must be finite and strictly positive")

        self.config = config
        self._calls = 0
        self._last_source = "not_run"
        self._checkpoint_path = self._resolve_checkpoint_path(config)
        self._model: Any = None

        if self._checkpoint_path is not None:
            self._model = self._build_model()
            self._load_checkpoint(self._checkpoint_path)
            self._evidence_tier = "checkpoint_loaded"
        elif config.allow_untrained_smoke:
            self._model = self._build_model()
            self._model.zero_initialize()
            self._evidence_tier = "diagnostic_untrained_smoke"
        else:
            raise ValueError(
                "learned GMM predictor requires checkpoint_path or model_id; "
                "set allow_untrained_smoke=true only for diagnostic smoke tests."
            )

    # ── public protocol ──────────────────────────────────────────────────────

    def predict(
        self,
        observation: dict[str, Any],
        *,
        horizon_steps: int,
        dt: float,
    ) -> GaussianMixturePedestrianForecast:
        """Return a K-mode GMM forecast aligned with the MPC rollout horizon.

        Args:
            observation: SocNav-structured observation.
            horizon_steps: Forecast horizon (must not exceed config horizon).
            dt: Timestep in seconds (must match config rollout_dt).

        Returns:
            A validated ``GaussianMixturePedestrianForecast``.
        """
        requested_steps = int(horizon_steps)
        if requested_steps <= 0:
            raise ValueError("horizon_steps must be strictly positive")
        if requested_steps > int(self.config.horizon_steps):
            raise ValueError(
                "requested horizon_steps exceeds the configured learned GMM horizon: "
                f"{requested_steps} > {self.config.horizon_steps}"
            )
        requested_dt = float(dt)
        if not np.isfinite(requested_dt) or requested_dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive")
        if not np.isclose(requested_dt, float(self.config.rollout_dt)):
            raise ValueError(
                "learned GMM predictor dt must match config.rollout_dt: "
                f"{requested_dt} != {self.config.rollout_dt}"
            )
        self._calls += 1
        steps = requested_steps
        ped_positions, ped_velocities_world = _pedestrian_world_state(observation)
        count = min(ped_positions.shape[0], int(self.config.max_pedestrians))

        if self._model is None:
            raise RuntimeError("learned GMM predictor model is unavailable")

        model_type = self.config.model_type.strip().lower()
        if model_type == "graph_gru":
            node_features, node_mask, global_features = encode_graph_predictor_features(
                observation,
                ped_positions[:count],
                ped_velocities_world[:count],
                max_pedestrians=int(self.config.max_pedestrians),
            )
            raw_output = self._model.predict_numpy(
                node_features,
                node_mask,
                global_features,
            )
            forecast = decode_graph_gmm_forecast(
                raw_output,
                ped_positions[:count],
                ped_velocities_world[:count],
                dt=float(dt),
                horizon_steps=steps,
                mode_count=int(self.config.mode_count),
                max_pedestrians=int(self.config.max_pedestrians),
                source=self._evidence_tier,
            )
        else:
            features = encode_gmm_predictor_features(
                observation,
                ped_positions[:count],
                ped_velocities_world[:count],
                max_pedestrians=int(self.config.max_pedestrians),
            )
            raw_output = self._model.predict_numpy(features)
            forecast = decode_gmm_forecast(
                raw_output,
                ped_positions[:count],
                ped_velocities_world[:count],
                dt=float(dt),
                horizon_steps=steps,
                mode_count=int(self.config.mode_count),
                max_pedestrians=int(self.config.max_pedestrians),
                source=self._evidence_tier,
            )
        self._last_source = self._evidence_tier
        return forecast

    def diagnostics(self) -> dict[str, Any]:
        """Return claim-boundary metadata."""
        return {
            "backend": "learned_graph_gru"
            if self.config.model_type.strip().lower() == "graph_gru"
            else "learned_gmm",
            "mode_count": int(self.config.mode_count),
            "evidence_tier": self._evidence_tier,
            "diagnostic_only": self._evidence_tier.startswith("diagnostic_"),
            "checkpoint_path": str(self._checkpoint_path) if self._checkpoint_path else None,
            "calls": self._calls,
            "last_source": self._last_source,
            "max_pedestrians": int(self.config.max_pedestrians),
            "horizon_steps": int(self.config.horizon_steps),
            "model_type": self.config.model_type.strip().lower(),
            "feature_schema": (
                "socnav_graph_nodes_v1"
                if self.config.model_type.strip().lower() == "graph_gru"
                else "socnav_flat_features_v1"
            ),
        }

    def reset(self) -> None:
        """Reset per-episode diagnostics."""
        self._calls = 0
        self._last_source = "not_run"

    # ── internal helpers ─────────────────────────────────────────────────────

    def _resolve_checkpoint_path(self, config: LearnedGmmPredictorConfig) -> Path | None:
        """Resolve configured checkpoint or model registry ID.

        Returns:
            Path to the resolved checkpoint file, or None if not configured.
        """
        if config.checkpoint_path and config.model_id:
            raise ValueError("configure either checkpoint_path or model_id, not both")
        if config.checkpoint_path:
            path = Path(config.checkpoint_path)
            if not path.exists():
                raise FileNotFoundError(f"learned GMM predictor checkpoint not found: {path}")
            return path
        if config.model_id:
            return resolve_model_path(config.model_id, allow_download=False)
        return None

    def _build_model(self) -> Any:
        """Build the configured tiny MLP or graph-GRU model.

        Returns:
            The instantiated predictor model wrapper.
        """
        model_type = self.config.model_type.strip().lower()
        if model_type == "mlp":
            input_dim, output_dim = predictor_io_dims(self.config)
            return _TinyGmmMlp(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=int(self.config.hidden_dim),
                device=self.config.device,
            )
        if model_type == "graph_gru":
            return _TinyGraphGmmGru(
                max_pedestrians=int(self.config.max_pedestrians),
                horizon_steps=int(self.config.horizon_steps),
                mode_count=int(self.config.mode_count),
                hidden_dim=int(self.config.hidden_dim),
                device=self.config.device,
            )
        raise ValueError("model_type must be 'mlp' or 'graph_gru'")

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a PyTorch state-dict checkpoint."""
        torch, _ = _load_torch()
        payload = torch.load(checkpoint_path, map_location=self.config.device)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            raise ValueError(
                f"learned GMM predictor checkpoint has unsupported format: {checkpoint_path}"
            )
        self._model.load_state_dict(state_dict)


# ── config builder ───────────────────────────────────────────────────────────


def build_learned_gmm_predictor_config(
    cfg: dict[str, Any] | None,
) -> LearnedGmmPredictorConfig:
    """Build predictor config from algorithm YAML mapping.

    Returns:
        Parsed ``LearnedGmmPredictorConfig`` with malformed fields restored to
        defaults.
    """
    raw = dict(cfg or {})
    defaults = LearnedGmmPredictorConfig()
    converters = {
        "checkpoint_path": _optional_str,
        "model_id": _optional_str,
        "device": lambda value: str(value).strip() if value else "cpu",
        "max_pedestrians": int,
        "horizon_steps": int,
        "rollout_dt": float,
        "hidden_dim": int,
        "mode_count": int,
        "model_type": str,
        "allow_untrained_smoke": _parse_bool,
    }
    kwargs: dict[str, Any] = {}
    for field in fields(LearnedGmmPredictorConfig):
        value = raw.get(field.name, getattr(defaults, field.name))
        try:
            kwargs[field.name] = converters[field.name](value)
        except (TypeError, ValueError):
            default_value = getattr(defaults, field.name)
            warnings.warn(
                (
                    f"Invalid learned GMM predictor config value '{field.name}': {value!r}; "
                    f"falling back to default {default_value!r}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            kwargs[field.name] = default_value
    return LearnedGmmPredictorConfig(**kwargs)


__all__ = [
    "GRAPH_GAUSSIAN_PARAM_DIM",
    "GRAPH_GLOBAL_FEATURE_DIM",
    "GRAPH_NODE_FEATURE_DIM",
    "LearnedGmmPedestrianPredictor",
    "LearnedGmmPredictorConfig",
    "build_learned_gmm_predictor_config",
    "decode_gmm_forecast",
    "decode_graph_gmm_forecast",
    "encode_gmm_predictor_features",
    "encode_graph_predictor_features",
    "graph_predictor_io_dims",
    "predictor_io_dims",
]
