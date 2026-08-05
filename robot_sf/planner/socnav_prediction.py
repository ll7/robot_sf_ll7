"""Prediction planner-family implementation extracted from the SocNav facade."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from robot_sf.common.forecast_variants import FORECAST_VARIANT_CHOICES
from robot_sf.common.math_utils import wrap_angle_pi
from robot_sf.models import get_registry_entry
from robot_sf.planner import socnav as _socnav
from robot_sf.planner.obstacle_features import (
    PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
    LocalObstacleFeatureExtractor,
    infer_predictive_feature_schema,
    normalize_obstacle_lines,
    obstacle_lines_from_map,
    obstacle_lines_from_observation,
    validate_predictive_runtime_feature_schema,
)

SamplingPlannerAdapter = _socnav.SamplingPlannerAdapter
SocNavPlannerConfig = _socnav.SocNavPlannerConfig
SocNavPlannerPolicy = _socnav.SocNavPlannerPolicy
PredictiveTrajectoryModel = _socnav.PredictiveTrajectoryModel

_DEFAULT_FORECAST_VARIANT_RISK_DISTANCE_M = 3.0


class PredictionPlannerAdapter(SamplingPlannerAdapter):
    """Predictive local planner with deterministic sampled-rollout search.

    The planner predicts pedestrian futures, builds a finite control lattice
    ``(v, omega)``, rolls out each candidate over a short horizon, and selects
    the minimum-cost command.

    Reference:
    - `docs/training/predictive_planner_complete_tutorial.md`

    Forecast variant integration (issue #2960):
    When ``forecast_variant`` is set to a non-``none`` value, this planner builds
    a :class:`BaselineProbabilisticPredictor` and uses its output as the predicted
    pedestrian futures scored by the normal sampled-rollout planner.
    """

    _EPS = 1e-6

    def __init__(self, config: SocNavPlannerConfig | None = None, *, allow_fallback: bool = False):
        """Initialize predictive planner adapter and deferred model loading."""
        self.config = config or SocNavPlannerConfig()
        self._allow_fallback = bool(allow_fallback)
        self._model: PredictiveTrajectoryModel | None = None
        self._load_error: Exception | None = None
        self._fallback_warned = False
        self._device = self._resolve_device()
        self._bound_obstacle_lines: list = []
        self._obstacle_feature_extractor = LocalObstacleFeatureExtractor()
        self._baseline_predictor: Any | None = None
        self._forecast_variant_execution_mode = self._init_forecast_variant()
        # Issue #6190: structured foresight-model-load provenance so a silent
        # constant-velocity fallback is observable and checker-actionable. The
        # benchmark metadata layer derives ``evidence_eligible`` and a degraded
        # ``status`` from this block; without it the fallback is invisible and a
        # degraded run can be promoted to a determinism verdict.
        self._foresight_provenance = self._init_foresight_provenance()

    def _init_forecast_variant(self) -> str:
        """Initialize baseline predictor when forecast_variant is configured.

        Returns:
            Execution mode for the configured forecast variant.
        """
        self._baseline_predictor = None
        configured_variant = getattr(self.config, "forecast_variant", "none")
        if configured_variant is None:
            variant = "none"
        else:
            variant = str(configured_variant).strip().lower() or "none"
        if variant == "none":
            return "native"

        if variant not in FORECAST_VARIANT_CHOICES:
            message = (
                f"PredictionPlannerAdapter: unsupported forecast_variant {variant!r}; "
                f"must be one of {FORECAST_VARIANT_CHOICES}"
            )
            logger.warning(message)
            if not self._allow_fallback:
                raise RuntimeError(message)
            return "blocked"

        try:
            from robot_sf.nav.baseline_probabilistic_predictor import (  # noqa: PLC0415
                BaselineProbabilisticPredictor,
            )

            self._baseline_predictor = BaselineProbabilisticPredictor(
                variant=variant,
                horizons_s=tuple(
                    getattr(self.config, "forecast_variant_horizons_s", (0.5, 1.0, 2.0))
                ),
                dt_s=float(getattr(self.config, "forecast_variant_dt_s", 0.1)),
                risk_distance_m=float(
                    getattr(
                        self.config,
                        "forecast_variant_risk_distance_m",
                        _DEFAULT_FORECAST_VARIANT_RISK_DISTANCE_M,
                    )
                ),
            )
            logger.info(
                "PredictionPlannerAdapter: built BaselineProbabilisticPredictor for variant {!r}",
                variant,
            )
            return "native"
        except (TypeError, ValueError) as exc:
            logger.warning(
                "PredictionPlannerAdapter: failed to build baseline predictor for {!r}: {}",
                variant,
                exc,
            )
            if self._allow_fallback:
                return "degraded"
            raise RuntimeError(
                f"PredictionPlannerAdapter: forecast predictor initialization failed for {variant!r}"
            ) from exc
        except ImportError as exc:
            logger.warning(
                "PredictionPlannerAdapter: forecast predictor unavailable for {!r}: {}",
                variant,
                exc,
            )
            if self._allow_fallback:
                return "degraded"
            raise RuntimeError(
                f"PredictionPlannerAdapter: forecast predictor unavailable for {variant!r}"
            ) from exc

    def get_forecast_variant_execution_mode(self) -> str:
        """Return the forecast variant execution mode.

        Returns:
            One of ``native``, ``degraded``, or ``blocked``.
        """
        return self._forecast_variant_execution_mode

    def _init_foresight_provenance(self) -> dict[str, Any]:
        """Initialize structured foresight-model-load provenance (issue #6190).

        Records the requested model asset plus its load outcome so a silent
        constant-velocity fallback is observable. The benchmark metadata layer
        derives ``evidence_eligible`` and a degraded ``status`` from this block.

        Returns:
            dict[str, Any]: Initial provenance block with a ``not_attempted`` load status.
        """
        model_id = getattr(self.config, "predictive_model_id", None)
        checkpoint_path = getattr(self.config, "predictive_checkpoint_path", None)
        requested_checkpoint = (
            str(checkpoint_path) if checkpoint_path else (str(model_id) if model_id else None)
        )
        return {
            "requested_model_id": model_id,
            "requested_checkpoint_path": requested_checkpoint,
            # This is the registry's expected digest, not a digest computed from
            # a locally resolved file. A missing/unloadable asset must retain the
            # requested provenance even when no local bytes exist to hash.
            "requested_checkpoint_sha256": self._expected_checkpoint_sha256(),
            "observed_checkpoint_sha256": None,
            "load_status": "not_attempted",
            "effective_prediction_mode": "not_attempted",
            "fallback_used": False,
            "fallback_reason": None,
            "load_error": None,
        }

    def _expected_checkpoint_sha256(self) -> str | None:
        """Return the configured model asset's expected registry digest, if declared."""
        model_id = getattr(self.config, "predictive_model_id", None)
        if not isinstance(model_id, str) or not model_id.strip():
            return None
        try:
            entry = get_registry_entry(model_id)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            return None
        release = entry.get("github_release")
        if not isinstance(release, dict):
            return None
        digest = release.get("sha256")
        if not isinstance(digest, str) or not digest.strip():
            return None
        return digest.strip().lower()

    def _record_foresight_load_success(self, checkpoint_sha256: str | None) -> None:
        """Record a successful predictive-model load in foresight provenance."""
        self._foresight_provenance.update(
            {
                "load_status": "loaded",
                "effective_prediction_mode": "predictive_foresight",
                "fallback_used": False,
                "fallback_reason": None,
                "load_error": None,
                "observed_checkpoint_sha256": checkpoint_sha256,
            }
        )

    def _record_foresight_load_failure(
        self,
        exc: Exception,
        checkpoint_sha256: str | None,
    ) -> None:
        """Record a predictive-model load failure in foresight provenance."""
        self._foresight_provenance.update(
            {
                "load_status": "failed",
                "effective_prediction_mode": "constant_velocity",
                "fallback_used": bool(self._allow_fallback),
                "fallback_reason": "predictive_model_load_failed",
                "load_error": f"{type(exc).__name__}: {exc}",
                "observed_checkpoint_sha256": checkpoint_sha256,
            }
        )

    def _record_foresight_constant_velocity_used(self) -> None:
        """Mark that a prediction step actually used the constant-velocity fallback."""
        provenance = self._foresight_provenance
        # Only transition from ``not_attempted`` to a degraded effective mode; a
        # recorded load failure already carries the constant-velocity mode.
        if provenance.get("load_status") == "not_attempted":
            provenance.update(
                {
                    "load_status": "not_attempted",
                    "effective_prediction_mode": "constant_velocity",
                    "fallback_used": bool(self._allow_fallback),
                    "fallback_reason": "predictive_model_not_loaded",
                }
            )
        else:
            provenance["effective_prediction_mode"] = "constant_velocity"

    def foresight_diagnostics(self) -> dict[str, Any]:
        """Return structured foresight-model-load provenance for benchmark metadata.

        The block is the structured source for ``evidence_eligible`` and the
        degraded ``status`` derivation in ``enrich_algorithm_metadata``.
        """
        return {"foresight_prediction": dict(self._foresight_provenance)}

    def foresight_degraded(self) -> bool:
        """Return True when the adapter fell back to constant-velocity prediction."""
        provenance = self._foresight_provenance
        if provenance.get("fallback_used") is True:
            return True
        return provenance.get("effective_prediction_mode") == "constant_velocity" and (
            provenance.get("load_status") in {"failed", "not_attempted"}
        )

    def configure(self, config: SocNavPlannerConfig | None = None) -> None:
        """Replace configuration and refresh forecast-variant runtime state."""
        self.config = config or SocNavPlannerConfig()
        self._device = self._resolve_device()
        self._model = None
        self._load_error = None
        self._fallback_warned = False
        self._forecast_variant_execution_mode = self._init_forecast_variant()
        self._foresight_provenance = self._init_foresight_provenance()

    def bind_obstacle_lines(self, obstacle_lines: Any) -> None:
        """Bind explicit runtime obstacle-line geometry for obstacle-feature inputs."""
        self._bound_obstacle_lines = normalize_obstacle_lines(obstacle_lines)
        self._obstacle_feature_extractor.precompute(self._bound_obstacle_lines)

    def bind_env(self, env: Any) -> None:
        """Bind static map obstacle geometry from a live Robot SF environment."""
        simulator = getattr(env, "simulator", None)
        map_def = getattr(env, "map_def", None)
        if map_def is None:
            map_def = getattr(simulator, "map_def", None)
        lines = obstacle_lines_from_map(map_def)
        if not lines and simulator is not None:
            iter_segments = getattr(simulator, "iter_obstacle_segments", None)
            if callable(iter_segments):
                lines = normalize_obstacle_lines(iter_segments())
        self._bound_obstacle_lines = lines
        self._obstacle_feature_extractor.precompute(lines)

    def _resolve_device(self) -> str:
        """Resolve runtime device string for predictive model inference.

        Returns:
            str: Torch device identifier.
        """
        requested = str(self.config.predictive_device).strip().lower()
        if requested.startswith("cuda"):
            runtime_torch = _socnav.torch
            if runtime_torch is not None and runtime_torch.cuda.is_available():
                return requested
            logger.warning(
                "Predictive planner requested device '{}' but CUDA is unavailable; using CPU.",
                requested,
            )
        return "cpu"

    def _ensure_model(self) -> PredictiveTrajectoryModel | None:
        """Load predictive model checkpoint on-demand.

        Returns:
            PredictiveTrajectoryModel | None: Model instance or None when fallback is enabled.
        """
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None if self._allow_fallback else self._raise_cached_error()
        # Hash a readable requested checkpoint before deserializing it. A corrupt
        # or schema-incompatible artifact still needs provenance in the fallback
        # record even though model construction will fail.
        checkpoint_sha256 = self._compute_checkpoint_sha256()
        try:
            self._model = self._build_model()
        except Exception as exc:  # broad catch: load surface unknown; fall back or fail
            self._record_foresight_load_failure(exc, checkpoint_sha256)
            if self._allow_fallback:
                self._load_error = exc
                if not self._fallback_warned:
                    logger.warning(
                        "Falling back to constant-velocity predictive planner behavior: {}. "
                        "Set allow_fallback=False to fail fast.",
                        exc,
                    )
                    self._fallback_warned = True
                return None
            raise
        self._record_foresight_load_success(checkpoint_sha256)
        return self._model

    def _compute_checkpoint_sha256(self) -> str | None:
        """Return the SHA-256 digest of the resolved predictive checkpoint, if available.

        Returns:
            str | None: Lowercase-hex SHA-256 digest, or ``None`` when the
            checkpoint cannot be resolved or read.
        """
        try:
            checkpoint = self._resolve_checkpoint_path()
        except (FileNotFoundError, KeyError, OSError, ValueError, RuntimeError):
            return None
        try:
            import hashlib  # noqa: PLC0415

            digest = hashlib.sha256()
            with checkpoint.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError:
            return None

    def _raise_cached_error(self) -> None:
        """Re-raise cached predictive-model initialization error."""
        # Load error must be cached before re-raise
        assert self._load_error is not None
        raise self._load_error

    def _resolve_checkpoint_path(self) -> Path:
        """Resolve predictive model checkpoint path.

        Returns:
            Path: Checkpoint path.
        """
        if self.config.predictive_checkpoint_path:
            checkpoint = Path(self.config.predictive_checkpoint_path).expanduser()
        else:
            checkpoint = _socnav.resolve_model_path(self.config.predictive_model_id)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Predictive planner checkpoint not found: {checkpoint}")
        return checkpoint

    def _build_model(self) -> PredictiveTrajectoryModel:
        """Construct predictive model from a checkpoint.

        Returns:
            PredictiveTrajectoryModel: Loaded model instance.
        """
        if _socnav.torch is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "PyTorch is required for PredictionPlannerAdapter. Install torch dependency."
            )
        checkpoint_loader = _socnav.load_predictive_checkpoint
        if checkpoint_loader is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Predictive model checkpoint loading is unavailable. Install torch dependency."
            )
        checkpoint_path = self._resolve_checkpoint_path()
        model, _payload = checkpoint_loader(
            checkpoint_path,
            map_location=self._device,
            expected_feature_schema_name=str(self.config.predictive_feature_schema_name),
        )
        self._validate_runtime_feature_schema(_payload.get("feature_schema"))
        model.to(self._device)
        model.eval()
        return model

    def _validate_runtime_feature_schema(self, feature_schema: Any) -> None:
        """Reject explicit standalone-producer checkpoints against runtime speed semantics."""
        validate_predictive_runtime_feature_schema(feature_schema)

    def _normalize_pedestrians(self, ped_state: dict) -> tuple[np.ndarray, np.ndarray]:
        """Normalize pedestrian positions and ego-frame velocities.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(positions_world, velocities_ego)`` arrays.
        """
        ped_positions = np.asarray(ped_state.get("positions", []), dtype=float)
        if ped_positions.ndim == 1:
            ped_positions = (
                ped_positions.reshape(-1, 2) if ped_positions.size % 2 == 0 else np.zeros((0, 2))
            )
        elif ped_positions.ndim != 2:
            ped_positions = np.zeros((0, 2), dtype=float)
        if ped_positions.ndim == 2 and ped_positions.shape[1] != 2:
            ped_positions = (
                ped_positions[:, :2]
                if ped_positions.shape[1] > 2
                else np.pad(
                    ped_positions,
                    ((0, 0), (0, 2 - ped_positions.shape[1])),
                    constant_values=0.0,
                )
            )

        ped_count = int(
            self._as_1d_float(ped_state.get("count", [ped_positions.shape[0]]), pad=1)[0]
        )
        ped_count = max(0, min(ped_count, int(ped_positions.shape[0])))
        ped_positions = ped_positions[:ped_count]

        ped_velocities = np.asarray(ped_state.get("velocities", []), dtype=float)
        if ped_velocities.ndim == 1:
            ped_velocities = (
                ped_velocities.reshape(-1, 2)
                if ped_velocities.size % 2 == 0
                else np.zeros((0, 2), dtype=float)
            )
        elif ped_velocities.ndim != 2:
            ped_velocities = np.zeros((0, 2), dtype=float)
        if ped_velocities.ndim == 2 and ped_velocities.shape[1] != 2:
            ped_velocities = (
                ped_velocities[:, :2]
                if ped_velocities.shape[1] > 2
                else np.pad(
                    ped_velocities,
                    ((0, 0), (0, 2 - ped_velocities.shape[1])),
                    constant_values=0.0,
                )
            )
        if ped_velocities.shape[0] < ped_count:
            ped_velocities = np.pad(
                ped_velocities,
                ((0, ped_count - ped_velocities.shape[0]), (0, 0)),
                constant_values=0.0,
            )
        ped_velocities = ped_velocities[:ped_count]
        return ped_positions, ped_velocities

    def _build_model_input(
        self, observation: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Build predictive-model input from SocNav structured observation.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            ``(state, mask, robot_pos, robot_heading)``.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state.get("position", [0.0, 0.0]), dtype=float)[:2]
        robot_heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        robot_speed = self._as_1d_float(robot_state.get("speed", [0.0, 0.0]), pad=2)[:2]
        ped_positions, ped_velocities_ego = self._normalize_pedestrians(ped_state)

        max_agents = max(1, int(self.config.predictive_max_agents))
        expected_dim = 4
        if bool(self.config.predictive_ego_conditioning):
            expected_dim = 9
        model = self._ensure_model()
        if model is not None:
            expected_dim = int(getattr(model.config, "input_dim", expected_dim))
        schema_metadata = infer_predictive_feature_schema(expected_dim)
        base_feature_dim = int(schema_metadata["base_feature_dim"])
        state = np.zeros((max_agents, expected_dim), dtype=np.float32)
        mask = np.zeros((max_agents,), dtype=np.float32)
        count = min(max_agents, ped_positions.shape[0])
        if count > 0:
            rel = ped_positions[:count] - robot_pos.reshape(1, 2)
            cos_h = float(np.cos(robot_heading))
            sin_h = float(np.sin(robot_heading))
            rel_x = cos_h * rel[:, 0] + sin_h * rel[:, 1]
            rel_y = -sin_h * rel[:, 0] + cos_h * rel[:, 1]
            state[:count, 0] = rel_x
            state[:count, 1] = rel_y
            state[:count, 2:4] = ped_velocities_ego[:count]
            if base_feature_dim >= 9:
                goal_current = self._as_1d_float(goal_state.get("current", [0.0, 0.0]), pad=2)[:2]
                goal_rel_world = goal_current - robot_pos
                goal_rel = np.array(
                    [
                        cos_h * goal_rel_world[0] + sin_h * goal_rel_world[1],
                        -sin_h * goal_rel_world[0] + cos_h * goal_rel_world[1],
                    ],
                    dtype=np.float32,
                )
                goal_dist = float(np.linalg.norm(goal_rel))
                goal_dir = goal_rel / max(goal_dist, 1e-6)
                state[:count, 4] = float(robot_speed[0]) if robot_speed.size > 0 else 0.0
                state[:count, 5] = float(robot_speed[1]) if robot_speed.size > 1 else 0.0
                state[:count, 6] = float(goal_dir[0])
                state[:count, 7] = float(goal_dir[1])
                state[:count, 8] = goal_dist
            if schema_metadata["name"] == PREDICTIVE_OBSTACLE_FEATURE_SCHEMA:
                extractor = self._obstacle_feature_extractor
                obstacle_lines = obstacle_lines_from_observation(observation)
                if not obstacle_lines:
                    obstacle_lines = self._bound_obstacle_lines
                obstacle_rows = extractor.extract_many(
                    [tuple(point) for point in ped_positions[:count]],
                    obstacle_lines,
                )
                end = min(expected_dim, base_feature_dim + extractor.feature_dim)
                state[:count, base_feature_dim:end] = obstacle_rows[:, : end - base_feature_dim]
            mask[:count] = 1.0
        return state, mask, robot_pos, robot_heading

    def _constant_velocity_prediction(self, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Generate constant-velocity fallback trajectories.

        Returns:
            np.ndarray: Predicted trajectories ``(N, T, 2)`` in robot frame.
        """
        steps = max(1, int(self.config.predictive_horizon_steps))
        dt = max(float(self.config.predictive_rollout_dt), 1e-3)
        future = np.zeros((state.shape[0], steps, 2), dtype=np.float32)
        for t in range(steps):
            tau = float(t + 1) * dt
            future[:, t, 0] = state[:, 0] + tau * state[:, 2]
            future[:, t, 1] = state[:, 1] + tau * state[:, 3]
        future *= mask[:, None, None]
        return future

    def _predict_trajectories(self, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Predict future pedestrian trajectories in robot frame.

        When ``forecast_variant`` is configured and a baseline predictor is available,
        this method consumes the baseline forecast instead of the learned model.

        Returns:
            np.ndarray: Predicted trajectories ``(N, T, 2)``.
        """
        if self._baseline_predictor is not None:
            return self._predict_with_baseline(state, mask)

        model = self._ensure_model()
        if model is None:
            self._record_foresight_constant_velocity_used()
            return self._constant_velocity_prediction(state, mask)
        runtime_torch = _socnav.torch
        if runtime_torch is None:
            raise RuntimeError(
                "PyTorch is required for predictive model inference but is not available",
            )
        with runtime_torch.no_grad():
            state_t = runtime_torch.from_numpy(state[None]).to(self._device)
            mask_t = runtime_torch.from_numpy(mask[None]).to(self._device)
            out = model(state_t, mask_t)
            future = out["future_positions"][0].detach().cpu().numpy().astype(np.float32)
        return future

    def _predict_with_baseline(self, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Predict using BaselineProbabilisticPredictor when forecast_variant is configured.

        Args:
            state: Pedestrian state in robot frame with shape ``(N, D)``.
            mask: Validity mask with shape ``(N,)``.

        Returns:
            np.ndarray: Predicted trajectories ``(N, T, 2)`` in robot frame.
        """
        # Baseline predictor must be initialized before prediction
        assert self._baseline_predictor is not None
        steps = max(1, int(self.config.predictive_horizon_steps))
        valid_indices = np.flatnonzero(mask > 0.5)
        if valid_indices.size == 0:
            return np.zeros((state.shape[0], steps, 2), dtype=np.float32)

        valid_state = state[valid_indices]
        observation = {
            "robot": {
                "position": np.array([0.0, 0.0], dtype=np.float32),
                "heading": np.array([0.0], dtype=np.float32),
                "speed": np.array([0.0, 0.0], dtype=np.float32),
            },
            "goal": {
                "current": np.array([1.0, 0.0], dtype=np.float32),
                "next": np.array([1.0, 0.0], dtype=np.float32),
            },
            "pedestrians": {
                "positions": valid_state[:, :2].astype(np.float32),
                "velocities": valid_state[:, 2:4].astype(np.float32),
                "count": np.array([float(valid_indices.size)], dtype=np.float32),
            },
            "map": {},
            "sim": {"time_s": np.array([-1.0], dtype=np.float32)},
        }

        try:
            prediction = self._baseline_predictor.predict(observation)
        except (FloatingPointError, TypeError, ValueError) as exc:
            logger.warning(
                "Baseline predictor failed: {}; using constant-velocity fallback",
                exc,
            )
            return self._constant_velocity_prediction(state, mask)

        future = self._constant_velocity_prediction(state, mask)
        for source_index, trajectory in enumerate(prediction.predictions):
            if source_index >= valid_indices.size:
                break
            mean = np.asarray(trajectory.mean, dtype=np.float32)
            if mean.size == 0:
                continue
            target_index = int(valid_indices[source_index])
            for step_index in range(steps):
                source_step = min(int(step_index * mean.shape[0] / steps), mean.shape[0] - 1)
                future[target_index, step_index, :] = mean[source_step]

        future *= mask[:, None, None].astype(np.float32)
        return future

    def _predictive_uncertainty_std(
        self,
        *,
        state: np.ndarray,
        mask: np.ndarray,
        future: np.ndarray,
    ) -> np.ndarray:
        """Build a heuristic forecast uncertainty envelope around the mean trajectory.

        The current predictive model only emits deterministic mean futures. This helper adds a
        bounded, config-driven uncertainty estimate so benchmark runs can exercise probability-aware
        rollout scoring without retraining the predictor.

        Returns:
            np.ndarray: Standard deviation per agent/time/axis with shape ``(N, T, 2)``.
        """
        base = max(float(self.config.predictive_uncertainty_base_std), 0.0)
        growth = max(float(self.config.predictive_uncertainty_growth_per_step), 0.0)
        speed_scale = max(float(self.config.predictive_uncertainty_speed_scale), 0.0)
        density_scale = max(float(self.config.predictive_uncertainty_density_scale), 0.0)
        steps = future.shape[1]
        valid_count = max(float(np.sum(mask > 0.5)), 0.0)
        density = valid_count / max(float(state.shape[0]), 1.0)
        ped_speed = np.linalg.norm(state[:, 2:4], axis=1, keepdims=True).astype(np.float32)
        time_std = base + growth * np.arange(1, steps + 1, dtype=np.float32).reshape(1, steps, 1)
        agent_std = speed_scale * ped_speed[:, None, :] + density_scale * density
        std = np.broadcast_to(time_std + agent_std, (future.shape[0], steps, 1)).astype(np.float32)
        std = std * mask[:, None, None].astype(np.float32)
        return np.repeat(std, 2, axis=2)

    def _sample_future_trajectories(
        self,
        *,
        state: np.ndarray,
        mask: np.ndarray,
        future: np.ndarray,
    ) -> np.ndarray:
        """Return a batch of future scenarios for risk-aware scoring.

        Returns:
            np.ndarray: Scenario tensor ``(S, N, T, 2)`` where ``S`` is the sample count.
        """
        mode = str(getattr(self.config, "predictive_uncertainty_mode", "deterministic")).strip()
        sample_count = max(1, int(getattr(self.config, "predictive_risk_sample_count", 1)))
        if mode == "deterministic" or sample_count <= 1:
            return future[None, ...]

        std = self._predictive_uncertainty_std(state=state, mask=mask, future=future)
        rng = np.random.default_rng(int(getattr(self.config, "predictive_risk_seed", 7)))
        samples = np.repeat(future[None, ...], sample_count, axis=0).astype(np.float32)
        if sample_count > 1:
            noise = rng.normal(
                loc=0.0,
                scale=std[None, ...],
                size=(sample_count - 1, future.shape[0], future.shape[1], future.shape[2]),
            ).astype(np.float32)
            samples[1:] += noise
        samples *= mask[None, :, None, None].astype(np.float32)
        return samples

    def _aggregate_risk_costs(self, costs: list[float]) -> float:
        """Aggregate scenario costs using the configured risk objective.

        Returns:
            float: Scalar aggregate cost.
        """
        if not costs:
            return float("inf")
        arr = np.asarray(costs, dtype=float)
        objective = str(getattr(self.config, "predictive_risk_objective", "mean")).strip().lower()
        if objective == "cvar":
            alpha = float(
                np.clip(getattr(self.config, "predictive_risk_cvar_alpha", 0.25), 1e-3, 1.0)
            )
            tail_count = max(1, int(np.ceil(arr.size * alpha)))
            return float(np.mean(np.sort(arr)[-tail_count:]))
        return float(np.mean(arr))

    def _score_action_distribution(
        self,
        *,
        observation: dict,
        future_batch: np.ndarray,
        mask: np.ndarray,
        v: float,
        w: float,
        steps: int,
    ) -> float:
        """Score a candidate over multiple future scenarios.

        Returns:
            float: Probability-aware aggregate cost.
        """
        costs = [
            self._score_action(
                observation=observation,
                future_peds=future_sample,
                mask=mask,
                v=v,
                w=w,
                steps=steps,
            )
            for future_sample in future_batch
        ]
        return self._aggregate_risk_costs(costs)

    def _score_sequence_distribution(
        self,
        *,
        observation: dict,
        future_batch: np.ndarray,
        mask: np.ndarray,
        sequence: list[tuple[float, float]],
        steps: int,
    ) -> float:
        """Score an action sequence over multiple future scenarios.

        Returns:
            float: Probability-aware aggregate sequence cost.
        """
        costs = [
            self._score_action_sequence(
                observation=observation,
                future_peds=future_sample,
                mask=mask,
                sequence=sequence,
                steps=steps,
            )
            for future_sample in future_batch
        ]
        return self._aggregate_risk_costs(costs)

    def _min_predicted_distance(
        self,
        *,
        future_peds: np.ndarray,
        mask: np.ndarray,
        steps: int | None = None,
    ) -> float:
        """Return minimum predicted ped distance to robot origin in local frame."""
        if future_peds.size == 0:
            return float("inf")
        valid_idx = np.where(mask > 0.5)[0]
        if valid_idx.size == 0:
            return float("inf")
        t_max = future_peds.shape[1] if steps is None else min(int(steps), future_peds.shape[1])
        if t_max <= 0:
            return float("inf")
        valid = future_peds[valid_idx, :t_max, :]
        dist = np.linalg.norm(valid, axis=2)
        return float(np.min(dist)) if dist.size > 0 else float("inf")

    def _effective_rollout_steps(self, *, future_peds: np.ndarray, mask: np.ndarray) -> int:
        """Select evaluation horizon steps, with optional near-field boosting.

        Returns:
            int: Number of rollout steps for candidate evaluation.
        """
        base_steps = max(1, int(self.config.predictive_horizon_steps))
        max_steps = max(1, int(future_peds.shape[1]))
        if not bool(self.config.predictive_adaptive_horizon_enabled):
            return min(base_steps, max_steps)
        min_pred_dist = self._min_predicted_distance(
            future_peds=future_peds,
            mask=mask,
            steps=min(base_steps, max_steps),
        )
        near_field = float(self.config.predictive_near_field_distance)
        if min_pred_dist <= near_field:
            boosted = base_steps + max(0, int(self.config.predictive_horizon_boost_steps))
            return min(boosted, max_steps)
        return min(base_steps, max_steps)

    def _risk_speed_cap_ratio(
        self,
        *,
        future_peds: np.ndarray,
        mask: np.ndarray,
        min_pred_dist: float | None = None,
    ) -> float:
        """Compute a near-field risk speed cap ratio.

        ``near-field risk`` means predicted pedestrian proximity below
        ``predictive_near_field_distance`` over the short prediction horizon.
        The returned ratio shrinks max candidate speed in dense/conflict states.

        Args:
            future_peds: Predicted pedestrian trajectories in robot frame.
            mask: Agent validity mask.
            min_pred_dist: Optional precomputed minimum predicted distance.
                When provided, the internal ``_min_predicted_distance`` call is
                skipped so callers that already hold this value can avoid
                redundant computation.

        Returns:
            float: Speed cap ratio in ``[0.1, 1.0]``.
        """
        near_field = float(self.config.predictive_near_field_distance)
        if near_field <= 0.0:
            return 1.0
        if min_pred_dist is None:
            min_pred_dist = self._min_predicted_distance(
                future_peds=future_peds,
                mask=mask,
                steps=max(2, min(int(self.config.predictive_horizon_steps), future_peds.shape[1])),
            )
        if not np.isfinite(min_pred_dist):
            return 1.0
        cap = float(self.config.predictive_near_field_speed_cap)
        cap = float(np.clip(cap, 0.1, 1.0))
        if min_pred_dist <= near_field:
            return cap
        if min_pred_dist <= near_field * 1.5:
            return min(1.0, cap + 0.15)
        return 1.0

    def _candidate_set(
        self, *, future_peds: np.ndarray, mask: np.ndarray
    ) -> list[tuple[float, float]]:
        """Build a risk-adaptive candidate command lattice.

        Here, ``lattice`` means a deterministic finite grid of controls formed
        from discrete speed ratios and heading deltas. In near-field risk
        states, the lattice is enriched with extra low-speed/turning options
        and evaluated under a speed cap from ``_risk_speed_cap_ratio``.

        Returns:
            list[tuple[float, float]]: Candidate ``(v, omega)`` commands.
        """
        near_field = float(self.config.predictive_near_field_distance)
        near_horizon = max(2, min(int(self.config.predictive_horizon_steps), future_peds.shape[1]))
        min_pred_dist = self._min_predicted_distance(
            future_peds=future_peds, mask=mask, steps=near_horizon
        )
        cap_ratio = self._risk_speed_cap_ratio(
            future_peds=future_peds, mask=mask, min_pred_dist=min_pred_dist
        )
        base_speed_ratios = [float(v) for v in self.config.predictive_candidate_speeds]
        heading_deltas = [float(v) for v in self.config.predictive_candidate_heading_deltas]
        if np.isfinite(min_pred_dist) and min_pred_dist <= near_field:
            base_speed_ratios.extend(
                float(v) for v in self.config.predictive_near_field_speed_samples
            )
            heading_deltas.extend(
                float(v) for v in self.config.predictive_near_field_heading_deltas
            )

        if bool(self.config.predictive_allow_reverse_candidates):
            reverse_ratios = [float(v) for v in self.config.predictive_reverse_candidate_speeds]
            reverse_allowed = not bool(self.config.predictive_reverse_near_field_only)
            reverse_allowed = reverse_allowed or (
                np.isfinite(min_pred_dist) and min_pred_dist <= near_field * 1.25
            )
            if reverse_allowed:
                base_speed_ratios.extend(reverse_ratios)

        speed_ratios = sorted({float(np.clip(v, -1.0, 1.0)) for v in base_speed_ratios})
        heading_deltas = sorted(set(heading_deltas))
        dt = max(float(self.config.predictive_rollout_dt), self._EPS)
        min_v = (
            -float(self.config.max_linear_speed)
            if bool(self.config.predictive_allow_reverse_candidates)
            else 0.0
        )
        max_v = float(self.config.max_linear_speed) * float(np.clip(cap_ratio, 0.1, 1.0))
        candidates: list[tuple[float, float]] = []
        for ratio in speed_ratios:
            v = float(np.clip(ratio * self.config.max_linear_speed, min_v, max_v))
            for delta in heading_deltas:
                omega = float(
                    np.clip(
                        delta / dt,
                        -self.config.max_angular_speed,
                        self.config.max_angular_speed,
                    )
                )
                candidates.append((v, omega))
        candidates.append((0.0, 0.0))
        # Keep deterministic ordering for stable benchmark outputs.
        return sorted(set(candidates), key=lambda x: (round(x[0], 6), round(x[1], 6)))

    @staticmethod
    def _rollout_robot(
        *,
        v: float,
        w: float,
        dt: float,
        steps: int,
    ) -> np.ndarray:
        """Roll out robot trajectory in its local frame under unicycle dynamics.

        Returns:
            np.ndarray: Trajectory ``(steps, 2)`` in local robot frame.
        """
        if steps < 0:
            raise ValueError("negative dimensions are not allowed")
        v = float(v)
        w = float(w)
        dt = float(dt)

        # Closed-form cumulative unicycle integration that reproduces the legacy
        # sequential scalar recurrence (heading is not wrapped here, so the
        # per-step angles are w*dt*(k+1) for k=0..steps-1). cumsum reorders the
        # float additions, producing last-ULP drift (<1e-15) versus the scalar
        # loop. Per issue #5412 that residual is accepted under atol=1e-12 (no
        # version bump); the scalar loop remains the numeric-parity reference.
        k = np.arange(1, steps + 1, dtype=float)
        angles = w * dt * k
        dx = v * dt * np.cos(angles)
        dy = v * dt * np.sin(angles)
        traj = np.stack([np.cumsum(dx), np.cumsum(dy)], axis=-1)
        return traj

    def _goal_progress(
        self,
        robot_state: dict,
        goal_state: dict,
        v: float,
        w: float,
        *,
        steps: int | None = None,
        robot_traj: np.ndarray | None = None,
    ) -> float:
        """Compute progress toward the goal over the rollout horizon.

        Returns:
            float: Positive value when a candidate reduces distance to goal.
        """
        robot_pos = np.asarray(robot_state.get("position", [0.0, 0.0]), dtype=float)[:2]
        robot_heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        goal = np.asarray(goal_state.get("current", [0.0, 0.0]), dtype=float)[:2]
        initial_dist = float(np.linalg.norm(goal - robot_pos))
        if robot_traj is None:
            dt = max(float(self.config.predictive_rollout_dt), 1e-3)
            steps_val = max(
                1, int(steps if steps is not None else self.config.predictive_horizon_steps)
            )
            robot_traj = self._rollout_robot(v=v, w=w, dt=dt, steps=steps_val)

        cos_h = float(np.cos(robot_heading))
        sin_h = float(np.sin(robot_heading))
        x_world = cos_h * robot_traj[-1, 0] - sin_h * robot_traj[-1, 1]
        y_world = sin_h * robot_traj[-1, 0] + cos_h * robot_traj[-1, 1]
        final_world = robot_pos + np.array([x_world, y_world], dtype=float)
        final_dist = float(np.linalg.norm(goal - final_world))
        return initial_dist - final_dist

    def _collision_cost(
        self,
        *,
        future_peds: np.ndarray,
        mask: np.ndarray,
        v: float,
        w: float,
        steps: int | None = None,
        valid_dists: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Compute collision and near-miss penalties for a candidate action.

        Returns:
            tuple[float, float]: ``(collision_penalty, near_miss_penalty)``.
        """
        steps_val = max(
            1, int(steps if steps is not None else self.config.predictive_horizon_steps)
        )
        radius_margin = float(self.config.predictive_robot_radius) + float(
            self.config.predictive_pedestrian_radius
        )
        speed_margin = float(self.config.predictive_speed_clearance_gain) * abs(float(v))
        safe_dist = float(self.config.predictive_safe_distance) + radius_margin + speed_margin
        near_dist = max(
            float(self.config.predictive_near_distance) + radius_margin + speed_margin, safe_dist
        )

        if valid_dists is not None:
            limit = min(steps_val, future_peds.shape[1], valid_dists.shape[1])
            if limit <= 0 or valid_dists[:, :limit].size == 0:
                return 0.0, 0.0
            collisions = float(np.sum(np.maximum(0.0, safe_dist - valid_dists[:, :limit])))
            near_misses = float(np.sum(np.maximum(0.0, near_dist - valid_dists[:, :limit])))
            return collisions, near_misses

        dt = max(float(self.config.predictive_rollout_dt), 1e-3)
        robot_traj = self._rollout_robot(v=v, w=w, dt=dt, steps=steps_val)
        horizon = min(steps_val, future_peds.shape[1])
        if horizon <= 0:
            return 0.0, 0.0

        valid_idx = np.where(mask > 0.5)[0]
        if valid_idx.size == 0:
            return 0.0, 0.0

        ped = future_peds[valid_idx, :horizon, :]
        if ped.size == 0:
            return 0.0, 0.0

        delta = ped - robot_traj[:horizon].reshape(1, horizon, 2)
        dist = np.linalg.norm(delta, axis=2)
        collisions = float(np.sum(np.maximum(0.0, safe_dist - dist)))
        near_misses = float(np.sum(np.maximum(0.0, near_dist - dist)))
        return collisions, near_misses

    def _min_clearance(
        self,
        *,
        future_peds: np.ndarray,
        mask: np.ndarray,
        v: float,
        w: float,
        steps: int,
        valid_dists: np.ndarray | None = None,
    ) -> float:
        """Compute minimum predicted robot-pedestrian clearance for a candidate.

        Returns:
            float: Minimum center-to-center clearance in meters.
        """
        if valid_dists is not None:
            return float(np.min(valid_dists)) if valid_dists.size > 0 else float("inf")

        dt = max(float(self.config.predictive_rollout_dt), 1e-3)
        robot_traj = self._rollout_robot(v=v, w=w, dt=dt, steps=max(1, int(steps)))
        valid_idx = np.where(mask > 0.5)[0]
        if valid_idx.size == 0:
            return float("inf")
        ped = future_peds[valid_idx, : robot_traj.shape[0], :]
        if ped.size == 0:
            return float("inf")
        delta = ped - robot_traj.reshape(1, robot_traj.shape[0], 2)
        dist = np.linalg.norm(delta, axis=2)
        return float(np.min(dist)) if dist.size > 0 else float("inf")

    def _ttc_penalty(
        self,
        *,
        future_peds: np.ndarray,
        mask: np.ndarray,
        v: float,
        w: float,
        steps: int | None = None,
        valid_dists: np.ndarray | None = None,
    ) -> float:
        """Compute a TTC-style penalty for near-term close approaches.

        Returns:
            float: Penalty that increases for earlier/closer predicted encounters.
        """
        radius_margin = float(self.config.predictive_robot_radius) + float(
            self.config.predictive_pedestrian_radius
        )
        speed_margin = float(self.config.predictive_speed_clearance_gain) * abs(float(v))
        threshold = float(self.config.predictive_ttc_distance) + radius_margin + speed_margin
        if threshold <= 0.0:
            return 0.0
        dt = max(float(self.config.predictive_rollout_dt), 1e-3)
        steps_val = max(
            1, int(steps if steps is not None else self.config.predictive_horizon_steps)
        )

        if valid_dists is not None:
            limit = min(steps_val, future_peds.shape[1], valid_dists.shape[1])
            if limit <= 0 or valid_dists[:, :limit].size == 0:
                return 0.0
            valid_slice = valid_dists[:, :limit]
            shortfall = np.maximum(0.0, threshold - valid_slice)
            time_indices = np.arange(1, limit + 1, dtype=float).reshape(1, limit)
            time_weights = 1.0 / (time_indices * dt + self._EPS)
            penalty = float(np.sum(shortfall * time_weights))
            return penalty

        horizon = min(steps_val, future_peds.shape[1])
        if horizon <= 0:
            return 0.0

        valid_idx = np.where(mask > 0.5)[0]
        if valid_idx.size == 0:
            return 0.0

        robot_traj = self._rollout_robot(v=v, w=w, dt=dt, steps=steps_val)
        ped = future_peds[valid_idx, :horizon, :]
        if ped.size == 0:
            return 0.0

        delta = ped - robot_traj[:horizon].reshape(1, horizon, 2)
        dist = np.linalg.norm(delta, axis=2)
        shortfall = np.maximum(0.0, threshold - dist)
        time_indices = np.arange(1, horizon + 1, dtype=float).reshape(1, horizon)
        time_weights = 1.0 / (time_indices * dt + self._EPS)
        penalty = float(np.sum(shortfall * time_weights))
        return penalty

    def _score_action(
        self,
        *,
        observation: dict,
        future_peds: np.ndarray,
        mask: np.ndarray,
        v: float,
        w: float,
        steps: int,
    ) -> float:
        """Score candidate action by combining goal, safety, smoothness, and occupancy costs.

        Returns:
            float: Scalar cost (lower is better).
        """
        dt = max(float(self.config.predictive_rollout_dt), 1e-3)
        steps_val = max(1, int(steps))
        robot_traj = self._rollout_robot(v=v, w=w, dt=dt, steps=steps_val)

        valid_idx = np.where(mask > 0.5)[0]
        if valid_idx.size > 0:
            ped = future_peds[valid_idx, : robot_traj.shape[0], :]
            if ped.size > 0:
                delta = ped - robot_traj.reshape(1, robot_traj.shape[0], 2)
                valid_dists = np.linalg.norm(delta, axis=2)
            else:
                valid_dists = np.empty((0, robot_traj.shape[0]), dtype=float)
        else:
            valid_dists = np.empty((0, robot_traj.shape[0]), dtype=float)

        robot_state, goal_state, _ped_state = self._socnav_fields(observation)
        goal_progress = self._goal_progress(
            robot_state, goal_state, v, w, steps=steps, robot_traj=robot_traj
        )
        collision_pen, near_pen = self._collision_cost(
            future_peds=future_peds,
            mask=mask,
            v=v,
            w=w,
            steps=steps,
            valid_dists=valid_dists,
        )
        ttc_pen = self._ttc_penalty(
            future_peds=future_peds,
            mask=mask,
            v=v,
            w=w,
            steps=steps,
            valid_dists=valid_dists,
        )
        min_clearance = self._min_clearance(
            future_peds=future_peds,
            mask=mask,
            v=v,
            w=w,
            steps=steps,
            valid_dists=valid_dists,
        )
        progress_risk_shortfall = max(
            0.0, float(self.config.predictive_progress_risk_distance) - float(min_clearance)
        )
        progress_risk_pen = max(0.0, goal_progress) * progress_risk_shortfall
        hard_clearance_shortfall = max(
            0.0, float(self.config.predictive_hard_clearance_distance) - float(min_clearance)
        )

        robot_pos = np.asarray(robot_state.get("position", [0.0, 0.0]), dtype=float)[:2]
        robot_heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        candidate_heading = robot_heading + w * dt
        direction = np.array([np.cos(candidate_heading), np.sin(candidate_heading)], dtype=float)
        _, occ_penalty = self._path_penalty(
            robot_pos=robot_pos,
            direction=direction,
            observation=observation,
            base_distance=max(
                abs(float(v)) * float(self.config.predictive_horizon_steps) * dt, 1e-3
            ),
            num_samples=max(2, int(self.config.predictive_horizon_steps)),
        )

        return (
            -float(self.config.predictive_goal_weight) * goal_progress
            + float(self.config.predictive_collision_weight) * collision_pen
            + float(self.config.predictive_near_miss_weight) * near_pen
            + float(self.config.predictive_progress_risk_weight) * progress_risk_pen
            + float(self.config.predictive_hard_clearance_weight) * hard_clearance_shortfall
            + float(self.config.predictive_velocity_weight) * abs(v)
            + float(self.config.predictive_turn_weight) * abs(w)
            + float(self.config.predictive_ttc_weight) * ttc_pen
            + float(self.config.occupancy_weight) * occ_penalty
        )

    def _rollout_robot_sequence(
        self,
        *,
        sequence: list[tuple[float, float]],
        segment_steps: int,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll out a piecewise-constant action sequence in the robot local frame.

        Returns:
            tuple[np.ndarray, np.ndarray]: Local positions and headings for rollout steps.
        """
        pos = np.zeros(2, dtype=float)
        heading = 0.0
        traj = np.zeros((max(1, len(sequence) * max(1, segment_steps)), 2), dtype=float)
        headings = np.zeros((traj.shape[0],), dtype=float)
        idx = 0
        for v, w in sequence:
            for _ in range(max(1, segment_steps)):
                heading += float(w) * dt
                pos[0] += float(v) * np.cos(heading) * dt
                pos[1] += float(v) * np.sin(heading) * dt
                traj[idx] = pos
                headings[idx] = heading
                idx += 1
        return traj[:idx], headings[:idx]

    def _score_action_sequence(
        self,
        *,
        observation: dict,
        future_peds: np.ndarray,
        mask: np.ndarray,
        sequence: list[tuple[float, float]],
        steps: int,
    ) -> float:
        """Score a short piecewise-constant action sequence; lower is better.

        Returns:
            float: Scalar sequence cost.
        """
        robot_state, goal_state, _ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state.get("position", [0.0, 0.0]), dtype=float)[:2]
        robot_heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        goal = np.asarray(goal_state.get("current", [0.0, 0.0]), dtype=float)[:2]
        dt = max(float(self.config.predictive_rollout_dt), 1e-3)
        segments = max(1, len(sequence))
        segment_steps = max(1, int(np.ceil(max(1, steps) / segments)))
        local_traj, local_headings = self._rollout_robot_sequence(
            sequence=sequence,
            segment_steps=segment_steps,
            dt=dt,
        )
        horizon = min(local_traj.shape[0], int(steps), int(future_peds.shape[1]))
        local_traj = local_traj[:horizon]
        local_headings = local_headings[:horizon]

        radius_margin = float(self.config.predictive_robot_radius) + float(
            self.config.predictive_pedestrian_radius
        )
        min_clearance = float("inf")
        collision_pen = 0.0
        near_pen = 0.0
        ttc_pen = 0.0
        safe_dist = float(self.config.predictive_safe_distance) + radius_margin
        near_dist = max(float(self.config.predictive_near_distance) + radius_margin, safe_dist)
        ttc_threshold = float(self.config.predictive_ttc_distance) + radius_margin

        valid_idx = np.where(mask > 0.5)[0]
        if valid_idx.size > 0:
            ped = future_peds[valid_idx, :horizon, :]
            if ped.size > 0:
                delta = ped - local_traj.reshape(1, horizon, 2)
                dist = np.linalg.norm(delta, axis=2)
                min_clearance = float(np.min(dist)) if dist.size > 0 else float("inf")
                collision_pen = float(np.sum(np.maximum(0.0, safe_dist - dist)))
                near_pen = float(np.sum(np.maximum(0.0, near_dist - dist)))
                time_indices = np.arange(1, horizon + 1, dtype=float).reshape(1, horizon)
                time_weights = 1.0 / (time_indices * dt + self._EPS)
                shortfall = np.maximum(0.0, ttc_threshold - dist)
                ttc_pen = float(np.sum(shortfall * time_weights))

        cos_h = float(np.cos(robot_heading))
        sin_h = float(np.sin(robot_heading))
        final_local = local_traj[-1] if horizon > 0 else np.zeros(2, dtype=float)
        final_world = robot_pos + np.array(
            [
                cos_h * final_local[0] - sin_h * final_local[1],
                sin_h * final_local[0] + cos_h * final_local[1],
            ],
            dtype=float,
        )
        initial_dist = float(np.linalg.norm(goal - robot_pos))
        final_dist = float(np.linalg.norm(goal - final_world))
        goal_progress = initial_dist - final_dist

        direction = final_world - robot_pos
        if np.linalg.norm(direction) <= self._EPS:
            direction = np.array([np.cos(robot_heading), np.sin(robot_heading)], dtype=float)
        _, occ_penalty = self._path_penalty(
            robot_pos=robot_pos,
            direction=direction,
            observation=observation,
            base_distance=max(float(np.linalg.norm(final_world - robot_pos)), 1e-3),
            num_samples=max(2, horizon),
        )

        velocity_pen = float(sum(abs(v) for v, _ in sequence))
        turn_pen = float(sum(abs(w) for _, w in sequence))
        progress_risk_shortfall = max(
            0.0, float(self.config.predictive_progress_risk_distance) - float(min_clearance)
        )
        progress_risk_pen = max(0.0, goal_progress) * progress_risk_shortfall
        hard_clearance_shortfall = max(
            0.0, float(self.config.predictive_hard_clearance_distance) - float(min_clearance)
        )

        goal_heading = float(np.arctan2(goal[1] - robot_pos[1], goal[0] - robot_pos[0]))
        phase_cost = self._sequence_phase_cost(
            sequence=sequence,
            segment_steps=segment_steps,
            horizon=horizon,
            local_headings=local_headings,
            robot_heading=robot_heading,
            goal_heading=goal_heading,
            min_clearance=min_clearance,
            goal_progress=goal_progress,
        )

        return (
            -float(self.config.predictive_goal_weight) * goal_progress
            + float(self.config.predictive_collision_weight) * collision_pen
            + float(self.config.predictive_near_miss_weight) * near_pen
            + float(self.config.predictive_progress_risk_weight) * progress_risk_pen
            + float(self.config.predictive_hard_clearance_weight) * hard_clearance_shortfall
            + float(self.config.predictive_velocity_weight) * velocity_pen
            + float(self.config.predictive_turn_weight) * turn_pen
            + float(self.config.predictive_ttc_weight) * ttc_pen
            + float(self.config.occupancy_weight) * occ_penalty
            + phase_cost
        )

    def _sequence_phase_cost(
        self,
        *,
        sequence: list[tuple[float, float]],
        segment_steps: int,
        horizon: int,
        local_headings: np.ndarray,
        robot_heading: float,
        goal_heading: float,
        min_clearance: float,
        goal_progress: float,
    ) -> float:
        """Phase-logic cost contribution for a scored action sequence.

        Returns:
            float: Additional phase cost (may be negative); zero when phase logic is disabled.
        """
        if not bool(self.config.predictive_phase_logic_enabled):
            return 0.0
        phase_cost = 0.0
        first_v, _first_w = sequence[0]
        first_seg_end = min(max(1, segment_steps), max(horizon, 1)) - 1
        first_heading = robot_heading + (local_headings[first_seg_end] if horizon > 0 else 0.0)
        heading_err = abs(wrap_angle_pi(goal_heading - first_heading))
        phase_cost += float(self.config.predictive_phase_align_weight) * heading_err
        if min_clearance >= float(self.config.predictive_phase_commit_clearance):
            phase_cost -= float(self.config.predictive_phase_commit_weight) * max(0.0, first_v)
        if min_clearance < float(self.config.predictive_phase_yield_clearance):
            phase_cost += float(self.config.predictive_phase_yield_weight) * max(0.0, first_v)
        if min_clearance >= float(
            self.config.predictive_phase_recover_clearance
        ) and goal_progress < float(self.config.predictive_phase_recover_progress):
            phase_cost += float(self.config.predictive_phase_recover_weight)
        return phase_cost

    def _plan_sequence_search(
        self,
        *,
        observation: dict,
        future_batch: np.ndarray,
        mask: np.ndarray,
        steps: int,
    ) -> tuple[float, float]:
        """Run deterministic beam search over short piecewise-constant control sequences.

        Returns:
            tuple[float, float]: First action from the lowest-cost sequence.
        """
        future = future_batch[0]
        candidates = self._candidate_set(future_peds=future, mask=mask)
        base_ranked = sorted(
            (
                (
                    self._score_action_distribution(
                        observation=observation,
                        future_batch=future_batch,
                        mask=mask,
                        v=v,
                        w=w,
                        steps=steps,
                    ),
                    (v, w),
                )
                for v, w in candidates
            ),
            key=lambda item: float(item[0]),
        )
        branch = max(1, int(self.config.predictive_sequence_branch_factor))
        beam_width = max(1, int(self.config.predictive_sequence_beam_width))
        segments = max(1, int(self.config.predictive_sequence_segments))
        stage_candidates = [cand for _score, cand in base_ranked[:branch]]

        beam: list[tuple[float, list[tuple[float, float]]]] = []
        for candidate in stage_candidates:
            seq = [candidate]
            score = self._score_sequence_distribution(
                observation=observation,
                future_batch=future_batch,
                mask=mask,
                sequence=seq,
                steps=steps,
            )
            beam.append((score, seq))
        beam.sort(key=lambda item: float(item[0]))
        beam = beam[:beam_width]

        for _ in range(1, segments):
            expanded: list[tuple[float, list[tuple[float, float]]]] = []
            for _prev_score, seq in beam:
                for candidate in stage_candidates:
                    new_seq = seq + [candidate]
                    score = self._score_sequence_distribution(
                        observation=observation,
                        future_batch=future_batch,
                        mask=mask,
                        sequence=new_seq,
                        steps=steps,
                    )
                    expanded.append((score, new_seq))
            expanded.sort(key=lambda item: float(item[0]))
            beam = expanded[:beam_width]
        return beam[0][1][0] if beam else (0.0, 0.0)

    def _plan_mcts_lite(
        self,
        *,
        observation: dict,
        future_batch: np.ndarray,
        mask: np.ndarray,
        steps: int,
    ) -> tuple[float, float]:
        """Run a bounded MCTS-lite search over short action sequences.

        Returns:
            tuple[float, float]: First action from the best root branch.
        """
        future = future_batch[0]
        candidates = self._candidate_set(future_peds=future, mask=mask)
        branch = max(1, int(self.config.predictive_mcts_branch_factor))
        segments = max(1, int(self.config.predictive_sequence_segments))
        iterations = max(1, int(self.config.predictive_mcts_iterations))
        rollout_count = max(1, int(self.config.predictive_mcts_rollout_count))
        exploration = float(self.config.predictive_mcts_exploration_weight)
        base_ranked = sorted(
            (
                (
                    self._score_action_distribution(
                        observation=observation,
                        future_batch=future_batch,
                        mask=mask,
                        v=v,
                        w=w,
                        steps=steps,
                    ),
                    (v, w),
                )
                for v, w in candidates
            ),
            key=lambda item: float(item[0]),
        )
        stage_candidates = [cand for _score, cand in base_ranked[:branch]]
        if not stage_candidates:
            return (0.0, 0.0)

        rng = np.random.default_rng(int(self.config.predictive_risk_seed) + 17)
        visits: dict[tuple[int, ...], int] = {(): 0}
        value_sum: dict[tuple[int, ...], float] = {(): 0.0}
        untried: dict[tuple[int, ...], list[int]] = {(): list(range(len(stage_candidates)))}

        def _uct(parent: tuple[int, ...], child: tuple[int, ...]) -> float:
            """Compute upper-confidence tree score for one child node.

            Returns:
                float: UCT value, or ``inf`` for an unvisited child.
            """
            child_visits = visits.get(child, 0)
            if child_visits == 0:
                return float("inf")
            parent_visits = max(visits.get(parent, 1), 1)
            mean = value_sum.get(child, 0.0) / child_visits
            bonus = exploration * np.sqrt(np.log(parent_visits + 1.0) / child_visits)
            return float(mean + bonus)

        for _ in range(iterations):
            node: tuple[int, ...] = ()
            path = [node]
            while len(node) < segments:
                choices = untried.setdefault(node, list(range(len(stage_candidates))))
                if choices:
                    idx = choices.pop(0)
                    node = node + (idx,)
                    visits.setdefault(node, 0)
                    value_sum.setdefault(node, 0.0)
                    untried.setdefault(node, list(range(len(stage_candidates))))
                    path.append(node)
                    break
                children = [node + (idx,) for idx in range(len(stage_candidates))]
                node = max(children, key=lambda child: _uct(path[-1], child))
                path.append(node)
            while len(node) < segments:
                idx = int(rng.integers(0, len(stage_candidates)))
                node = node + (idx,)
                path.append(node)

            seq_idx = list(node)
            rollout_values: list[float] = []
            for _roll in range(rollout_count):
                seq = [stage_candidates[idx] for idx in seq_idx]
                rollout_values.append(
                    -self._score_sequence_distribution(
                        observation=observation,
                        future_batch=future_batch,
                        mask=mask,
                        sequence=seq,
                        steps=steps,
                    )
                )
            reward = float(np.mean(rollout_values))
            for state_idx in path:
                visits[state_idx] = visits.get(state_idx, 0) + 1
                value_sum[state_idx] = value_sum.get(state_idx, 0.0) + reward

        root_children = [((idx,), stage_candidates[idx]) for idx in range(len(stage_candidates))]
        best_child, best_action = max(
            root_children,
            key=lambda item: (
                visits.get(item[0], 0),
                value_sum.get(item[0], 0.0) / max(visits.get(item[0], 0), 1),
            ),
        )
        _ = best_child
        return best_action

    def _select_predictive_action(
        self,
        *,
        observation: dict,
        future_batch: np.ndarray,
        mask: np.ndarray,
        steps: int,
    ) -> tuple[tuple[float, float], float]:
        """Select the lowest-cost first action under the configured predictive search mode.

        Returns:
            tuple[tuple[float, float], float]: Best command and its aggregated risk-aware cost.
        """
        if bool(self.config.predictive_mcts_enabled):
            best = self._plan_mcts_lite(
                observation=observation,
                future_batch=future_batch,
                mask=mask,
                steps=steps,
            )
            best_cost = self._score_action_distribution(
                observation=observation,
                future_batch=future_batch,
                mask=mask,
                v=best[0],
                w=best[1],
                steps=steps,
            )
            return best, best_cost

        if bool(self.config.predictive_sequence_search_enabled):
            best = self._plan_sequence_search(
                observation=observation,
                future_batch=future_batch,
                mask=mask,
                steps=steps,
            )
            best_cost = self._score_action_distribution(
                observation=observation,
                future_batch=future_batch,
                mask=mask,
                v=best[0],
                w=best[1],
                steps=steps,
            )
            return best, best_cost

        best = (0.0, 0.0)
        best_cost = float("inf")
        for v, w in self._candidate_set(future_peds=future_batch[0], mask=mask):
            cost = self._score_action_distribution(
                observation=observation,
                future_batch=future_batch,
                mask=mask,
                v=v,
                w=w,
                steps=steps,
            )
            if cost < best_cost:
                best_cost = cost
                best = (v, w)
        return best, best_cost

    def _apply_predictive_progress_escape(
        self,
        *,
        observation: dict,
        robot_heading: float,
        robot_pos: np.ndarray,
        goal: np.ndarray,
        future_batch: np.ndarray,
        mask: np.ndarray,
        steps: int,
        best: tuple[float, float],
    ) -> tuple[tuple[float, float], float]:
        """Inject a minimum forward-progress action when the predicted field is sufficiently clear.

        Returns:
            tuple[tuple[float, float], float]: Updated command and its cost.
        """
        best_cost = self._score_action_distribution(
            observation=observation,
            future_batch=future_batch,
            mask=mask,
            v=best[0],
            w=best[1],
            steps=steps,
        )
        if not bool(self.config.predictive_progress_escape_enabled):
            return best, best_cost

        goal_dist = float(np.linalg.norm(goal - robot_pos))
        if goal_dist <= float(self.config.predictive_progress_escape_distance):
            return best, best_cost

        clearance_gate = float(self.config.predictive_hard_clearance_distance) + float(
            self.config.predictive_progress_escape_clearance_margin
        )
        min_pred_dist = self._min_predicted_distance(
            future_peds=future_batch[0],
            mask=mask,
            steps=min(max(steps, 1), future_batch[0].shape[1]),
        )
        if min_pred_dist < clearance_gate:
            return best, best_cost

        cap_ratio = self._risk_speed_cap_ratio(future_peds=future_batch[0], mask=mask)
        max_v = float(self.config.max_linear_speed) * float(np.clip(cap_ratio, 0.1, 1.0))
        min_v = float(self.config.max_linear_speed) * float(
            np.clip(self.config.predictive_progress_escape_min_speed_ratio, 0.0, 1.0)
        )
        if best[0] >= min_v:
            return best, best_cost

        goal_heading = float(np.arctan2(goal[1] - robot_pos[1], goal[0] - robot_pos[0]))
        heading_err = wrap_angle_pi(goal_heading - robot_heading)
        heading_scale = max(0.2, 1.0 - abs(float(heading_err)) / np.pi)
        forced_v = float(np.clip(min_v * heading_scale, 0.0, max_v))
        forced_w = float(
            np.clip(
                float(self.config.predictive_progress_escape_heading_gain) * float(heading_err),
                -float(self.config.max_angular_speed),
                float(self.config.max_angular_speed),
            )
        )
        forced = (forced_v, forced_w)
        forced_cost = self._score_action_distribution(
            observation=observation,
            future_batch=future_batch,
            mask=mask,
            v=forced[0],
            w=forced[1],
            steps=steps,
        )
        if forced_cost < best_cost:
            return forced, forced_cost
        return best, best_cost

    def plan(self, observation: dict) -> tuple[float, float]:
        """Compute (v, w) via predictive rollout search over learned trajectories.

        Returns:
            tuple[float, float]: Linear and angular command.
        """
        robot_state, goal_state, _ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state.get("position", [0.0, 0.0]), dtype=float)[:2]
        goal = np.asarray(goal_state.get("current", [0.0, 0.0]), dtype=float)[:2]
        if float(np.linalg.norm(goal - robot_pos)) <= float(self.config.goal_tolerance):
            return 0.0, 0.0

        state, mask, _robot_pos, _robot_heading = self._build_model_input(observation)
        future = self._predict_trajectories(state, mask)
        future_batch = self._sample_future_trajectories(state=state, mask=mask, future=future)
        steps = self._effective_rollout_steps(future_peds=future_batch[0], mask=mask)
        best, _best_cost = self._select_predictive_action(
            observation=observation,
            future_batch=future_batch,
            mask=mask,
            steps=steps,
        )
        best, _best_cost = self._apply_predictive_progress_escape(
            observation=observation,
            robot_heading=float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0]),
            robot_pos=robot_pos,
            goal=goal,
            future_batch=future_batch,
            mask=mask,
            steps=steps,
            best=best,
        )
        return best

    def diagnostics(self) -> dict[str, Any]:
        """Return execution diagnostics."""
        return {"planner_type": "PredictionPlannerAdapter"}


def make_prediction_policy(
    config: SocNavPlannerConfig | None = None, *, allow_fallback: bool = False
) -> SocNavPlannerPolicy:
    """
    Convenience constructor for predictive planner policy.

    Set ``allow_fallback=True`` to permit constant-velocity fallback behavior
    when the predictive model checkpoint cannot be loaded.

    Returns:
        SocNavPlannerPolicy: Policy wrapping PredictionPlannerAdapter.
    """
    return SocNavPlannerPolicy(
        adapter=PredictionPlannerAdapter(config=config, allow_fallback=allow_fallback)
    )


class SocNavBenchSamplingAdapter(SamplingPlannerAdapter):
    """
    Adapter that attempts to delegate to the upstream SocNavBench SamplingPlanner.

    Warning:
        This adapter requires the upstream SocNavBench planner by default. Set
        ``allow_fallback=True`` to fall back to the heuristic SamplingPlannerAdapter;
        in fallback mode it is **not benchmark-ready**.
    """

    def __init__(
        self,
        config: SocNavPlannerConfig | None = None,
        socnav_root: Path | None = None,
        planner_factory: Callable[[], Any] | None = None,
        *,
        allow_fallback: bool = False,
    ) -> None:
        """Initialize the adapter with upstream delegation enabled."""

        super().__init__(
            config=config,
            socnav_root=socnav_root,
            planner_factory=planner_factory,
            use_upstream=True,
            allow_fallback=allow_fallback,
        )

    def diagnostics(self) -> dict[str, Any]:
        """Return execution diagnostics."""
        return {"planner_type": "SocNavBenchSamplingAdapter"}
