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

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
    "diagnostic synthetic learnability probe; trained on a seeded robot-repulsion "
    "task, not real pedestrian data; smoke evidence only, not benchmark or "
    "paper-facing evidence"
)


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
    metrics = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": float(initial_loss - final_loss),
        "epochs": config.epochs,
        "num_samples": config.num_samples,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "issue": 4013,
        "evidence_tier": EVIDENCE_TIER,
        "claim_boundary": CLAIM_BOUNDARY,
        "task": "seeded synthetic robot-repulsion short-horizon residual",
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
