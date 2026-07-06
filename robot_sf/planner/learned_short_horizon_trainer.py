"""CPU trainer for the issue #4013 learned short-horizon pedestrian predictor.

This produces a *diagnostic-only* trained checkpoint for the small state-based
predictor in :mod:`robot_sf.planner.learned_short_horizon_predictor`. It trains on
a seeded synthetic short-horizon task (pedestrians drifting under a simple
robot-repulsion field) so the smoke lane can load real learned weights instead of
the zero-initialized ``diagnostic_untrained_smoke`` model.

Claim boundary: the synthetic task is a reproducible learnability probe, not real
ETH/UCY pedestrian data. A checkpoint from this trainer is `smoke evidence` that
the predictor trains and loads without fallback -- it is NOT benchmark evidence,
navigation-quality evidence, or a paper/dissertation claim.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_sf.data_ingestion.real_trajectory_contract import load_manifest, run_preflight
from robot_sf.planner.learned_short_horizon_predictor import (
    LearnedShortHorizonPredictorConfig,
    build_predictor_module,
    encode_predictor_features,
    pedestrian_world_state,
    predictor_io_dims,
)

SCHEMA_VERSION = "issue_4013.short_horizon_predictor_training.v1"
EVIDENCE_TIER = "diagnostic-only"
CLAIM_BOUNDARY = (
    "diagnostic learnability probe; synthetic mode uses a seeded robot-repulsion "
    "task, real-trajectory mode requires a checksum-validated BYO manifest; smoke "
    "evidence only, not benchmark or "
    "paper-facing evidence"
)
REAL_TRAJECTORY_EVIDENCE_TIER = "real-trajectory-smoke"


@dataclass(frozen=True)
class ShortHorizonTrainerConfig:
    """Configuration for the diagnostic short-horizon predictor trainer."""

    max_pedestrians: int = 16
    horizon_steps: int = 4
    rollout_dt: float = 0.2
    hidden_dim: int = 64
    device: str = "cpu"
    seed: int = 4013
    num_samples: int = 512
    epochs: int = 400
    learning_rate: float = 1e-3
    repulsion_gain: float = 1.0
    scene_radius: float = 4.0
    training_data_manifest_path: str | None = None
    real_trajectory_file_glob: str = "**/*"
    output_dir: str = "output/models/issue_4013/short_horizon_predictor"

    def predictor_config(
        self, checkpoint_path: str | None = None
    ) -> LearnedShortHorizonPredictorConfig:
        """Return the inference predictor config that matches this trainer.

        Returns:
            LearnedShortHorizonPredictorConfig: Config that loads the trained model.
        """

        return LearnedShortHorizonPredictorConfig(
            checkpoint_path=checkpoint_path,
            device=self.device,
            max_pedestrians=self.max_pedestrians,
            horizon_steps=self.horizon_steps,
            rollout_dt=self.rollout_dt,
            hidden_dim=self.hidden_dim,
            model_type="mlp",
        )


@dataclass
class TrainingResult:
    """Outcome of a diagnostic short-horizon predictor training run."""

    checkpoint_path: Path
    manifest_path: Path
    metrics_path: Path
    initial_loss: float
    final_loss: float
    epochs: int
    num_samples: int
    predictor_config: dict[str, Any] = field(default_factory=dict)

    @property
    def loss_reduction(self) -> float:
        """Return the absolute training-loss reduction.

        Returns:
            float: ``initial_loss - final_loss``.
        """

        return float(self.initial_loss - self.final_loss)


def _repulsion_residual(
    ped_positions_world: np.ndarray,
    robot_pos: np.ndarray,
    *,
    horizon_steps: int,
    dt: float,
    gain: float,
    max_pedestrians: int,
) -> np.ndarray:
    """Compute the target learned residual over constant velocity.

    Pedestrians accelerate away from the robot with a magnitude that decays with
    distance. The residual displacement after ``tau`` seconds is ``0.5 * a * tau**2``.
    The target depends only on the robot-relative position, which is part of the
    predictor feature vector, so the task is genuinely learnable.

    Returns:
        np.ndarray: Flattened target vector of length ``max_pedestrians*horizon*2``.
    """

    target = np.zeros((max_pedestrians, horizon_steps, 2), dtype=float)
    count = min(ped_positions_world.shape[0], max_pedestrians)
    for idx in range(count):
        rel = ped_positions_world[idx] - robot_pos
        dist = float(np.linalg.norm(rel))
        direction = rel / dist if dist > 1e-6 else np.zeros(2, dtype=float)
        accel = gain * direction / (1.0 + dist)
        for step in range(horizon_steps):
            tau = float(step + 1) * dt
            target[idx, step, :] = 0.5 * accel * tau * tau
    return target.reshape(-1)


def generate_training_batch(
    config: ShortHorizonTrainerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a training batch from real trajectories or synthetic data.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature matrix ``[N, in]`` and target
        matrix ``[N, out]``.
    """

    if config.training_data_manifest_path:
        return generate_real_trajectory_training_batch(config)
    return generate_synthetic_training_batch(config)


def generate_synthetic_training_batch(
    config: ShortHorizonTrainerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a seeded synthetic training batch.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature matrix ``[N, in]`` and target matrix ``[N, out]``.
    """

    rng = np.random.default_rng(config.seed)
    input_dim, output_dim = predictor_io_dims(config.predictor_config())
    features = np.zeros((config.num_samples, input_dim), dtype=float)
    targets = np.zeros((config.num_samples, output_dim), dtype=float)
    for sample in range(config.num_samples):
        robot_pos = rng.uniform(-1.0, 1.0, size=2)
        heading = float(rng.uniform(-np.pi, np.pi))
        speed = float(rng.uniform(0.0, 1.0))
        goal = robot_pos + rng.uniform(-3.0, 3.0, size=2)
        count = int(rng.integers(1, config.max_pedestrians + 1))
        ped_pos_world = robot_pos + rng.uniform(
            -config.scene_radius, config.scene_radius, size=(count, 2)
        )
        ped_vel_world = rng.uniform(-1.0, 1.0, size=(count, 2))
        cos_h, sin_h = float(np.cos(heading)), float(np.sin(heading))
        ped_vel_ego = np.empty_like(ped_vel_world)
        ped_vel_ego[:, 0] = cos_h * ped_vel_world[:, 0] + sin_h * ped_vel_world[:, 1]
        ped_vel_ego[:, 1] = -sin_h * ped_vel_world[:, 0] + cos_h * ped_vel_world[:, 1]
        observation = {
            "robot": {
                "position": robot_pos,
                "heading": np.asarray([heading], dtype=float),
                "speed": np.asarray([speed], dtype=float),
            },
            "goal": {"current": goal},
            "pedestrians": {
                "positions": ped_pos_world,
                "velocities": ped_vel_ego,
                "count": np.asarray([float(count)], dtype=float),
            },
        }
        pos_w, vel_w = pedestrian_world_state(observation)
        features[sample] = encode_predictor_features(
            observation, pos_w, vel_w, max_pedestrians=config.max_pedestrians
        )
        targets[sample] = _repulsion_residual(
            pos_w,
            robot_pos,
            horizon_steps=config.horizon_steps,
            dt=config.rollout_dt,
            gain=config.repulsion_gain,
            max_pedestrians=config.max_pedestrians,
        )
    return features, targets


def generate_real_trajectory_training_batch(
    config: ShortHorizonTrainerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate short-horizon residual examples from staged real trajectories.

    The manifest must pass the generic real-trajectory preflight with
    ``availability: validated``. Files are read from the manifest staging
    directory only; no network access or raw-data publication is performed.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature matrix ``[N, in]`` and target
        matrix ``[N, out]``.

    Raises:
        ValueError: manifest is not validated or no trainable windows exist.
    """

    if not config.training_data_manifest_path:
        raise ValueError("training_data_manifest_path is required for real-trajectory training")

    manifest = load_manifest(Path(config.training_data_manifest_path))
    preflight = run_preflight(manifest)
    if not preflight.ok:
        messages = "; ".join(issue.message for issue in preflight.errors)
        raise ValueError(f"real-trajectory manifest failed preflight: {messages}")
    if preflight.availability != "validated":
        raise ValueError(
            "real-trajectory training requires manifest availability 'validated', "
            f"got {preflight.availability!r}"
        )

    conversion = manifest["conversion"]
    staging_dir = Path(os.path.expandvars(manifest["staging"]["staging_dir"])).expanduser()
    files = _trajectory_files(staging_dir, config.real_trajectory_file_glob)
    rows = _read_trajectory_rows(files, conversion)
    features, targets = _examples_from_trajectory_rows(rows, config, conversion)
    if features.shape[0] == 0:
        raise ValueError(
            "real-trajectory manifest is validated but staged files contain no "
            "windows long enough for the configured horizon"
        )
    return features, targets


def _trajectory_files(staging_dir: Path, pattern: str) -> list[Path]:
    """Return candidate CSV/TSV trajectory files under ``staging_dir``."""

    if not staging_dir.is_dir():
        raise ValueError(f"real-trajectory staging_dir does not exist: {staging_dir}")
    paths = [
        path
        for path in sorted(staging_dir.glob(pattern))
        if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".txt"}
    ]
    if not paths:
        raise ValueError(f"no CSV/TSV trajectory files found under {staging_dir}")
    return paths


def _read_trajectory_rows(
    files: list[Path],
    conversion: dict[str, Any],
) -> list[dict[str, Any]]:
    """Read finite trajectory rows using manifest conversion fields.

    Returns:
        list[dict[str, Any]]: Canonical scene/frame/agent/x/y row dictionaries.
    """

    timestamp_field = conversion["timestamp_field"]
    agent_id_field = conversion["agent_id_field"]
    x_field, y_field = conversion["position_fields"][:2]
    scene_field = conversion.get("map_context_field")
    rows: list[dict[str, Any]] = []
    for path in files:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t ") if sample.strip() else csv.excel
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, dialect=dialect)
            required = {timestamp_field, agent_id_field, x_field, y_field}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"trajectory file {path} missing required columns: {sorted(required)}"
                )
            for raw in reader:
                scene = raw.get(scene_field) if scene_field else None
                rows.append(
                    {
                        "scene": scene or path.stem,
                        "frame": float(raw[timestamp_field]),
                        "agent": str(raw[agent_id_field]),
                        "x": float(raw[x_field]),
                        "y": float(raw[y_field]),
                    }
                )
    return rows


def _examples_from_trajectory_rows(
    rows: list[dict[str, Any]],
    config: ShortHorizonTrainerConfig,
    conversion: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert trajectory rows into predictor feature and residual target arrays.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature and residual target matrices.
    """

    input_dim, output_dim = predictor_io_dims(config.predictor_config())
    examples_x: list[np.ndarray] = []
    examples_y: list[np.ndarray] = []
    frame_rate = float(conversion["frame_rate_hz"])
    dt = 1.0 / frame_rate

    scene_frames: dict[str, dict[float, dict[str, np.ndarray]]] = {}
    for row in rows:
        scene = str(row["scene"])
        scene_frames.setdefault(scene, {}).setdefault(float(row["frame"]), {})[
            str(row["agent"])
        ] = np.asarray([float(row["x"]), float(row["y"])], dtype=float)

    for frames in scene_frames.values():
        ordered_frames = sorted(frames)
        frame_index = {frame: idx for idx, frame in enumerate(ordered_frames)}
        for frame in ordered_frames:
            idx = frame_index[frame]
            future_frames = ordered_frames[idx + 1 : idx + 1 + config.horizon_steps]
            if len(future_frames) < config.horizon_steps:
                continue
            current_agents = sorted(frames[frame])[: config.max_pedestrians]
            example = _frame_example(
                frames=frames,
                frame=frame,
                future_frames=future_frames,
                current_agents=current_agents,
                config=config,
                dt=dt,
                output_dim=output_dim,
            )
            if example is None:
                continue
            features, target = example
            examples_x.append(features)
            examples_y.append(target)
            if len(examples_x) >= config.num_samples:
                return np.vstack(examples_x), np.vstack(examples_y)

    if not examples_x:
        return np.zeros((0, input_dim), dtype=float), np.zeros((0, output_dim), dtype=float)
    return np.vstack(examples_x), np.vstack(examples_y)


def _frame_example(
    *,
    frames: dict[float, dict[str, np.ndarray]],
    frame: float,
    future_frames: list[float],
    current_agents: list[str],
    config: ShortHorizonTrainerConfig,
    dt: float,
    output_dim: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build one real-trajectory predictor training example.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: Feature/target pair, or ``None``
            when no current pedestrian has a complete future horizon.
    """

    ped_positions: list[np.ndarray] = []
    ped_velocities: list[np.ndarray] = []
    residuals = np.zeros((config.max_pedestrians, config.horizon_steps, 2), dtype=float)
    for ped_idx, agent in enumerate(current_agents):
        if any(agent not in frames[future_frame] for future_frame in future_frames):
            continue
        position = frames[frame][agent]
        next_position = frames[future_frames[0]][agent]
        velocity = (next_position - position) / dt
        ped_positions.append(position)
        ped_velocities.append(velocity)
        for step, future_frame in enumerate(future_frames):
            tau = float(step + 1) * dt
            expected = position + velocity * tau
            residuals[ped_idx, step, :] = frames[future_frame][agent] - expected
    if not ped_positions:
        return None
    ped_pos_world = np.vstack(ped_positions)
    ped_vel_world = np.vstack(ped_velocities)
    centroid = ped_pos_world.mean(axis=0)
    observation = {
        "robot": {
            "position": centroid,
            "heading": np.asarray([0.0], dtype=float),
            "speed": np.asarray([0.0], dtype=float),
        },
        "goal": {"current": centroid + np.asarray([1.0, 0.0], dtype=float)},
        "pedestrians": {
            "positions": ped_pos_world,
            "velocities": ped_vel_world,
            "count": np.asarray([float(len(ped_positions))], dtype=float),
        },
    }
    pos_w, vel_w = pedestrian_world_state(observation)
    features = encode_predictor_features(
        observation, pos_w, vel_w, max_pedestrians=config.max_pedestrians
    )
    return features, residuals.reshape(output_dim)


def train_short_horizon_predictor(config: ShortHorizonTrainerConfig) -> TrainingResult:
    """Train the diagnostic short-horizon predictor and write artifacts.

    Returns:
        TrainingResult: Paths and loss metrics for the training run.
    """

    torch.manual_seed(config.seed)
    features_np, targets_np = generate_training_batch(config)
    input_dim, output_dim = predictor_io_dims(config.predictor_config())
    module = build_predictor_module(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config.hidden_dim,
        device=config.device,
    )
    features = torch.as_tensor(features_np, dtype=torch.float32, device=config.device)
    targets = torch.as_tensor(targets_np, dtype=torch.float32, device=config.device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(module.parameters(), lr=config.learning_rate)

    module.train()
    with torch.no_grad():
        initial_loss = float(loss_fn(module(features), targets).item())
    final_loss = initial_loss
    for _ in range(config.epochs):
        optimizer.zero_grad()
        loss = loss_fn(module(features), targets)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "short_horizon_predictor.pt"
    manifest_path = output_dir / "training_manifest.json"
    metrics_path = output_dir / "training_metrics.json"

    module.eval()
    state_dict = {key: value.detach().cpu() for key, value in module.state_dict().items()}
    torch.save(
        {
            "state_dict": state_dict,
            "schema_version": SCHEMA_VERSION,
            "evidence_tier": EVIDENCE_TIER,
            "input_dim": input_dim,
            "output_dim": output_dim,
        },
        checkpoint_path,
    )

    predictor_config = asdict(config.predictor_config(str(checkpoint_path)))
    feature_stats = {
        "feature_mean": [float(x) for x in features_np.mean(axis=0)],
        "feature_std": [float(x) for x in features_np.std(axis=0)],
        "note": (
            "The inference predictor consumes raw (unnormalized) features; these "
            "statistics are provenance metadata only, not an applied normalizer."
        ),
    }
    data_source = "real_trajectory_manifest" if config.training_data_manifest_path else "synthetic"
    evidence_tier = (
        REAL_TRAJECTORY_EVIDENCE_TIER if config.training_data_manifest_path else EVIDENCE_TIER
    )
    metrics = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": float(initial_loss - final_loss),
        "epochs": config.epochs,
        "num_samples": config.num_samples,
        "data_source": data_source,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "issue": 4013,
        "evidence_tier": evidence_tier,
        "claim_boundary": CLAIM_BOUNDARY,
        "task": (
            "validated real-trajectory short-horizon residual"
            if config.training_data_manifest_path
            else "seeded synthetic robot-repulsion short-horizon residual"
        ),
        "training_data_manifest_path": config.training_data_manifest_path,
        "trainer_config": asdict(config),
        "predictor_config": predictor_config,
        "architecture": {
            "model_type": "mlp",
            "input_dim": input_dim,
            "hidden_dim": config.hidden_dim,
            "output_dim": output_dim,
            "activation": "tanh",
        },
        "metrics": metrics,
        "feature_stats": feature_stats,
        "checkpoint": checkpoint_path.name,
        "not_full_world_model": True,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        initial_loss=initial_loss,
        final_loss=final_loss,
        epochs=config.epochs,
        num_samples=config.num_samples,
        predictor_config=predictor_config,
    )
