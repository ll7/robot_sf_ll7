"""CPU smoke training utilities for issue #4010 diffusion policy.

This module intentionally trains only on a tiny deterministic synthetic fixture.
The artifacts prove the checkpoint/normalizer/load contract, not navigation
quality or COLSON reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.planner.diffusion_policy import (
    CLAIM_BOUNDARY,
    EVIDENCE_TIER,
    DiffusionActionSampler,
    RobotPedestrianGraphEncoder,
)

try:  # pragma: no cover - exercised only in environments without torch installed.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


SMOKE_MANIFEST_SCHEMA_VERSION = "diffusion_policy_smoke_manifest.v1"
SMOKE_CONFIG_SCHEMA_VERSION = "diffusion_policy_training_smoke.v1"


@dataclass(frozen=True)
class DiffusionPolicyTrainingSmokeConfig:
    """Configuration for the CPU-only diffusion-policy smoke trainer."""

    schema_version: str = SMOKE_CONFIG_SCHEMA_VERSION
    seed: int = 4010
    training_steps: int = 8
    batch_size: int = 8
    learning_rate: float = 0.001
    max_pedestrians: int = 4
    hidden_dim: int = 64
    action_dim: int = 2
    denoising_steps: int = 4
    num_action_samples: int = 4
    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.0
    output_dir: str = "output/diffusion_policy/issue_4010_smoke"
    artifact_prefix: str = "diffusion_policy_issue_4010_smoke"

    def __post_init__(self) -> None:
        """Validate the intentionally narrow smoke-training contract."""
        if self.schema_version != SMOKE_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported diffusion smoke schema_version: {self.schema_version}")
        if self.training_steps <= 0:
            raise ValueError("training_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.max_pedestrians <= 0:
            raise ValueError("max_pedestrians must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.action_dim != 2:
            raise ValueError("issue #4010 smoke training supports action_dim=2 only")
        if self.denoising_steps <= 0:
            raise ValueError("denoising_steps must be positive")
        if self.num_action_samples <= 0:
            raise ValueError("num_action_samples must be positive")


@dataclass(frozen=True)
class DiffusionPolicySmokeArtifacts:
    """Paths and manifest payload produced by the smoke trainer."""

    checkpoint_path: Path
    normalizer_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def load_smoke_config(path: Path) -> DiffusionPolicyTrainingSmokeConfig:
    """Load a smoke-training YAML config.

    Args:
        path: YAML path containing either the config fields directly or under
            ``diffusion_policy_training_smoke``.

    Returns:
        Validated smoke-training configuration.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Diffusion policy smoke config must be a YAML mapping")
    raw_config = payload.get("diffusion_policy_training_smoke", payload)
    if not isinstance(raw_config, dict):
        raise ValueError("diffusion_policy_training_smoke must be a YAML mapping")
    return DiffusionPolicyTrainingSmokeConfig(**raw_config)


def run_training_smoke(
    config: DiffusionPolicyTrainingSmokeConfig,
    *,
    output_dir: Path | None = None,
) -> DiffusionPolicySmokeArtifacts:
    """Run the CPU smoke trainer and write checkpoint, normalizer, and manifest.

    Args:
        config: Validated smoke-training config.
        output_dir: Optional test override for generated artifacts.

    Returns:
        Paths and parsed manifest for the generated smoke artifacts.
    """
    _require_torch()
    _seed_everything(config.seed)

    artifact_dir = (output_dir or Path(config.output_dir)).expanduser()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    encoder = RobotPedestrianGraphEncoder(
        max_pedestrians=config.max_pedestrians,
        hidden_dim=config.hidden_dim,
    )
    sampler = DiffusionActionSampler(
        condition_dim=encoder.output_dim,
        action_dim=config.action_dim,
        max_linear_speed=config.max_linear_speed,
        max_angular_speed=config.max_angular_speed,
        hidden_dim=config.hidden_dim,
    )
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *sampler.parameters()],
        lr=config.learning_rate,
    )
    loss_fn = nn.MSELoss()

    fixture = _build_synthetic_fixture(config)
    feature_tensor = torch.stack([item[0] for item in fixture])
    mask_tensor = torch.stack([item[1] for item in fixture])
    target_actions = torch.stack([item[2] for item in fixture])
    normalizer = _build_normalizer(feature_tensor, mask_tensor)

    losses: list[float] = []
    generator = torch.Generator(device=torch.device("cpu"))
    generator.manual_seed(config.seed)
    for step in range(config.training_steps):
        indices = torch.arange(config.batch_size) % len(fixture)
        indices = torch.roll(indices, shifts=step)
        features = feature_tensor[indices]
        masks = mask_tensor[indices]
        targets = target_actions[indices]
        conditions = encoder(features, masks)
        noise = torch.randn(targets.shape, generator=generator)
        timestep = torch.full((targets.shape[0], 1), 0.5, dtype=targets.dtype)
        prediction = sampler.net(torch.cat([targets + noise, timestep, conditions], dim=-1))
        loss = loss_fn(prediction, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    checkpoint_path = artifact_dir / f"{config.artifact_prefix}.pt"
    normalizer_path = artifact_dir / f"{config.artifact_prefix}.normalizer.json"
    manifest_path = artifact_dir / f"{config.artifact_prefix}.manifest.json"

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "sampler": sampler.state_dict(),
            "schema_version": SMOKE_MANIFEST_SCHEMA_VERSION,
            "config": asdict(config),
        },
        checkpoint_path,
    )
    normalizer_path.write_text(json.dumps(normalizer, indent=2, sort_keys=True), encoding="utf-8")

    manifest = _build_manifest(
        config=config,
        checkpoint_path=checkpoint_path,
        normalizer_path=normalizer_path,
        losses=losses,
        normalizer=normalizer,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return DiffusionPolicySmokeArtifacts(
        checkpoint_path=checkpoint_path,
        normalizer_path=normalizer_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _require_torch() -> None:
    """Fail closed when smoke training dependencies are unavailable."""
    if torch is None or nn is None:
        raise RuntimeError("Diffusion policy smoke training requires PyTorch.")


def _seed_everything(seed: int) -> None:
    """Seed NumPy and PyTorch for reproducible smoke artifacts."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_synthetic_fixture(
    config: DiffusionPolicyTrainingSmokeConfig,
) -> list[tuple[Any, Any, Any]]:
    """Create deterministic crossing-style observations and bounded target actions.

    Returns:
        Synthetic graph feature, mask, and action-target tuples.
    """
    encoder = RobotPedestrianGraphEncoder(
        max_pedestrians=config.max_pedestrians,
        hidden_dim=config.hidden_dim,
    )
    rng = np.random.default_rng(config.seed)
    fixture: list[tuple[Any, Any, Any]] = []
    for index in range(max(config.batch_size, 6)):
        side = -1.0 if index % 2 else 1.0
        distance = 0.55 + 0.08 * index
        observation = {
            "robot": {
                "position": [0.0, 0.0],
                "velocity": [0.05, 0.0],
                "goal": [2.0, 0.15 * side],
                "heading": 0.0,
                "radius": 0.3,
            },
            "agents": [
                {
                    "position": [distance, 0.18 * side],
                    "velocity": [-0.05, -0.02 * side],
                    "radius": 0.25,
                },
                {
                    "position": [distance + 0.3, -0.22 * side],
                    "velocity": [0.0, 0.03 * side],
                    "radius": 0.25,
                },
            ],
            "dt": 0.1,
        }
        features, mask = encoder.encode_observation(observation)
        linear = np.clip(0.45 + 0.05 * rng.normal(), 0.0, config.max_linear_speed)
        angular = np.clip(0.25 * side, -config.max_angular_speed, config.max_angular_speed)
        target = torch.tensor([linear, angular], dtype=torch.float32)
        fixture.append((features, mask, target))
    return fixture


def _build_normalizer(feature_tensor: Any, mask_tensor: Any) -> dict[str, Any]:
    """Build a small JSON normalizer from valid graph-node features.

    Returns:
        JSON-serializable normalizer payload.
    """
    valid = feature_tensor[mask_tensor]
    mean = valid.mean(dim=0)
    std = valid.std(dim=0, unbiased=False).clamp(min=1e-6)
    return {
        "schema_version": "diffusion_policy_normalizer.v1",
        "feature_order": [
            "rel_or_goal_x",
            "rel_or_goal_y",
            "vx",
            "vy",
            "radius_or_cos_heading",
            "distance_or_sin_heading",
            "role_radius_or_is_pedestrian",
            "is_robot",
        ],
        "mean": [float(value) for value in mean.tolist()],
        "std": [float(value) for value in std.tolist()],
        "sample_count": int(valid.shape[0]),
    }


def _build_manifest(
    *,
    config: DiffusionPolicyTrainingSmokeConfig,
    checkpoint_path: Path,
    normalizer_path: Path,
    losses: list[float],
    normalizer: dict[str, Any],
) -> dict[str, Any]:
    """Build provenance manifest for generated smoke artifacts.

    Returns:
        JSON-serializable smoke artifact manifest.
    """
    config_payload = asdict(config)
    return {
        "schema_version": SMOKE_MANIFEST_SCHEMA_VERSION,
        "issue": 4010,
        "artifact_kind": "diffusion_policy_cpu_training_smoke",
        "evidence_tier": EVIDENCE_TIER,
        "claim_boundary": CLAIM_BOUNDARY,
        "provenance": {
            "git_commit": _git_value(["rev-parse", "HEAD"]),
            "git_dirty": bool(_git_value(["status", "--short"])),
            "config_hash": _stable_hash(config_payload),
            "training_data": "deterministic synthetic crossing fixture; not benchmark evidence",
        },
        "config": config_payload,
        "artifacts": {
            "checkpoint_path": checkpoint_path.name,
            "normalizer_path": normalizer_path.name,
            "checkpoint_format": "encoder_sampler_state_dict",
            "normalizer_schema_version": normalizer["schema_version"],
        },
        "training": {
            "status": "completed",
            "steps": config.training_steps,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "losses": losses,
        },
        "out_of_scope": [
            "no full benchmark campaign",
            "no Slurm or GPU submission",
            "no paper or dissertation claim edits",
        ],
    }


def _stable_hash(payload: dict[str, Any]) -> str:
    """Return stable short hash for JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _git_value(args: list[str]) -> str:
    """Return compact git command output for manifest provenance."""
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    """Run the issue #4010 CPU smoke trainer from a YAML config.

    Returns:
        Process-style exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    artifacts = run_training_smoke(load_smoke_config(args.config), output_dir=args.output_dir)
    sys.stdout.write(json.dumps({"manifest_path": str(artifacts.manifest_path)}, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
