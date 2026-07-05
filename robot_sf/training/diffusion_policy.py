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
    DiffusionPolicyAdapter,
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
DIAGNOSTIC_PACKET_SCHEMA_VERSION = "diffusion_policy_diagnostic_packet.v1"
MULTIMODAL_PROBE_SCHEMA_VERSION = "diffusion_policy_multimodal_probe.v1"
REPRESENTATIVE_ROLLOUT_SCHEMA_VERSION = "diffusion_policy_representative_rollout.v1"


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


def build_diagnostic_packet(
    manifest: dict[str, Any],
    *,
    map_runner_metadata: dict[str, Any] | None = None,
    multimodal_probe: dict[str, Any] | None = None,
    representative_rollout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize issue #4010 smoke/runtime artifacts as diagnostic-only evidence.

    The packet is an integration handoff for the staged implementation lane: it
    records what is now wired, what remains blocked, and why the result is not a
    benchmark or paper-facing claim.

    Returns:
        JSON-serializable diagnostic packet with claim boundary and blockers.
    """
    if manifest.get("schema_version") != SMOKE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Diffusion policy diagnostic packet requires {SMOKE_MANIFEST_SCHEMA_VERSION} manifest"
        )
    artifacts = _validated_smoke_artifacts(manifest)

    metadata = map_runner_metadata or {}
    diffusion_metadata = metadata.get("diffusion_policy", {})
    checkpoint_status = diffusion_metadata.get("checkpoint_status", "unknown")
    normalizer_status = diffusion_metadata.get("normalizer_status", "unknown")
    multimodal_probe_status = _multimodal_probe_status(multimodal_probe)
    representative_rollout_status = _representative_rollout_status(representative_rollout)
    rollout_runtime_loaded = bool(
        (representative_rollout_status["summary"] or {}).get("runtime_loaded_checkpoint")
    )
    runtime_loaded = (
        checkpoint_status == "checkpoint_loaded" and normalizer_status == "loaded"
    ) or rollout_runtime_loaded
    remaining_blockers: list[dict[str, str]] = []
    if not runtime_loaded:
        remaining_blockers.append(
            {
                "id": "checkpoint_backed_map_runner_load",
                "status": "blocked",
                "reason": "map-runner metadata does not prove checkpoint and normalizer loaded",
            }
        )
    if not representative_rollout_status["passed"]:
        remaining_blockers.append(
            {
                "id": "representative_rollout",
                "status": "remaining",
                "reason": str(representative_rollout_status["reason"]),
            }
        )
    if not multimodal_probe_status["passed"]:
        remaining_blockers.append(
            {
                "id": "multimodal_action_probe",
                "status": "remaining",
                "reason": str(multimodal_probe_status["reason"]),
            }
        )
    remaining_blockers.append(
        {
            "id": "paper_grade_benchmark_claim",
            "status": "blocked",
            "reason": "CPU smoke fixture is synthetic and cannot support benchmark or paper claims",
        }
    )

    return {
        "schema_version": DIAGNOSTIC_PACKET_SCHEMA_VERSION,
        "issue": 4010,
        "evidence_tier": EVIDENCE_TIER,
        "claim_boundary": CLAIM_BOUNDARY,
        "new_capability": "checkpoint-backed diffusion-policy diagnostic integration packet",
        "integrated_contracts": {
            "runtime_adapter": "DiffusionPolicyAdapter",
            "map_runner_policy": "diffusion_policy",
            "training_smoke_manifest": SMOKE_MANIFEST_SCHEMA_VERSION,
        },
        "artifact_inputs": {
            "checkpoint_path": artifacts["checkpoint_path"],
            "normalizer_path": artifacts["normalizer_path"],
            "manifest_kind": manifest.get("artifact_kind"),
        },
        "runtime_metadata": {
            "checkpoint_status": checkpoint_status,
            "normalizer_status": normalizer_status,
            "allow_untrained_smoke": diffusion_metadata.get("allow_untrained_smoke"),
        },
        "acceptance_status": {
            "smoke_manifest_present": True,
            "checkpoint_backed_map_runner_load": runtime_loaded,
            "representative_rollout": representative_rollout_status["passed"],
            "multimodal_action_probe": multimodal_probe_status["passed"],
            "benchmark_campaign_run": False,
            "slurm_or_gpu_submission": False,
            "paper_or_dissertation_claim": False,
        },
        "representative_rollout": representative_rollout_status["summary"],
        "multimodal_probe": multimodal_probe_status["summary"],
        "remaining_blockers": remaining_blockers,
    }


def build_multimodal_probe(
    manifest: dict[str, Any],
    *,
    artifact_dir: Path | None = None,
    sample_count: int = 64,
    seed: int = 4010,
    angular_threshold: float = 0.05,
    slow_speed_threshold: float = 0.15,
) -> dict[str, Any]:
    """Classify fixed-conflict diffusion samples as diagnostic-only mode evidence.

    Returns:
        JSON-serializable report for issue #4010 multimodal action probe.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    artifacts = _validated_smoke_artifacts(manifest)
    artifact_base = Path(".") if artifact_dir is None else artifact_dir
    checkpoint_path = _resolve_manifest_artifact(artifact_base, artifacts["checkpoint_path"])
    normalizer_path = _resolve_manifest_artifact(artifact_base, artifacts["normalizer_path"])
    config_payload = manifest.get("config", {})
    if not isinstance(config_payload, dict):
        config_payload = {}
    adapter = DiffusionPolicyAdapter(
        {
            "checkpoint_path": str(checkpoint_path),
            "normalizer_path": str(normalizer_path),
            "seed": seed,
            "deterministic": False,
            "allow_untrained_smoke": False,
            "max_pedestrians": int(config_payload.get("max_pedestrians", 4)),
            "max_linear_speed": float(config_payload.get("max_linear_speed", 1.0)),
            "max_angular_speed": float(config_payload.get("max_angular_speed", 1.0)),
            "denoising_steps": int(config_payload.get("denoising_steps", 4)),
            "num_action_samples": sample_count,
            "diagnostics": {"record_raw_samples": True},
        }
    )
    observation = _fixed_conflict_observation()
    selected = adapter.plan(observation)
    raw_samples = adapter.diagnostics()["diffusion_policy"].get("raw_samples", [])
    labels = [
        _classify_action_mode(sample, angular_threshold, slow_speed_threshold)
        for sample in raw_samples
    ]
    mode_counts = {label: labels.count(label) for label in sorted(set(labels))}
    non_empty_core_modes = [
        label
        for label in ("pass_left", "pass_right", "slow_or_wait")
        if mode_counts.get(label, 0) > 0
    ]
    return {
        "schema_version": MULTIMODAL_PROBE_SCHEMA_VERSION,
        "issue": 4010,
        "evidence_tier": EVIDENCE_TIER,
        "claim_boundary": (
            "diagnostic fixed-conflict action diversity probe only; not rollout, "
            "benchmark, COLSON reproduction, or paper evidence"
        ),
        "scenario_id": "issue_4010_fixed_conflict_probe",
        "sample_count": len(raw_samples),
        "seed": seed,
        "selected_action": [float(selected[0]), float(selected[1])],
        "mode_clusters": [
            {"label": label, "count": int(count)} for label, count in sorted(mode_counts.items())
        ],
        "distinct_core_mode_count": len(non_empty_core_modes),
        "passed": len(non_empty_core_modes) >= 2,
        "out_of_scope": [
            "no representative scenario rollout",
            "no benchmark campaign",
            "no Slurm/GPU submission",
            "no paper/dissertation claim",
        ],
    }


def build_representative_rollout(
    manifest: dict[str, Any],
    *,
    artifact_dir: Path | None = None,
    step_count: int = 12,
    seed: int = 4010,
) -> dict[str, Any]:
    """Run a short checkpoint-backed crossing rollout and record diagnostic provenance.

    Returns:
        JSON-serializable representative rollout report for issue #4010.
    """
    if step_count <= 0:
        raise ValueError("step_count must be positive")
    artifacts = _validated_smoke_artifacts(manifest)
    artifact_base = Path(".") if artifact_dir is None else artifact_dir
    checkpoint_path = _resolve_manifest_artifact(artifact_base, artifacts["checkpoint_path"])
    normalizer_path = _resolve_manifest_artifact(artifact_base, artifacts["normalizer_path"])
    config_payload = manifest.get("config", {})
    if not isinstance(config_payload, dict):
        raise ValueError("Diffusion policy smoke manifest config must be a mapping")

    from robot_sf.benchmark.map_runner import _build_policy  # noqa: PLC0415

    max_linear_speed = float(config_payload.get("max_linear_speed", 1.0))
    max_angular_speed = float(config_payload.get("max_angular_speed", 1.0))
    policy, metadata = _build_policy(
        "diffusion_policy",
        {
            "checkpoint_path": str(checkpoint_path),
            "normalizer_path": str(normalizer_path),
            "seed": seed,
            "deterministic": True,
            "allow_untrained_smoke": False,
            "max_pedestrians": int(config_payload.get("max_pedestrians", 4)),
            "max_linear_speed": max_linear_speed,
            "max_angular_speed": max_angular_speed,
            "denoising_steps": int(config_payload.get("denoising_steps", 4)),
            "num_action_samples": int(config_payload.get("num_action_samples", 4)),
        },
        robot_kinematics="differential_drive",
    )
    reset = getattr(policy, "_planner_reset", None)
    if callable(reset):
        reset(seed=seed)

    dt = 0.1
    robot_x = 0.0
    robot_y = 0.0
    heading = 0.0
    trajectory: list[dict[str, Any]] = []
    finite_command_count = 0
    for step in range(step_count):
        obs = _representative_rollout_observation(step, robot_x, robot_y, heading, dt)
        linear, angular = policy(obs)
        linear = float(linear)
        angular = float(angular)
        if np.isfinite(linear) and np.isfinite(angular):
            finite_command_count += 1
        trajectory.append(
            {
                "step": step,
                "robot_position": [robot_x, robot_y],
                "command": [linear, angular],
            }
        )
        heading += angular * dt
        robot_x += linear * np.cos(heading) * dt
        robot_y += linear * np.sin(heading) * dt

    close = getattr(policy, "_planner_close", None)
    if callable(close):
        close()

    commands_valid = all(
        0.0 <= row["command"][0] <= max_linear_speed and abs(row["command"][1]) <= max_angular_speed
        for row in trajectory
    )
    runtime_loaded = (
        metadata.get("diffusion_policy", {}).get("checkpoint_status") == "checkpoint_loaded"
        and metadata.get("diffusion_policy", {}).get("normalizer_status") == "loaded"
    )
    passed = runtime_loaded and finite_command_count == step_count and commands_valid
    return {
        "schema_version": REPRESENTATIVE_ROLLOUT_SCHEMA_VERSION,
        "issue": 4010,
        "evidence_tier": EVIDENCE_TIER,
        "claim_boundary": (
            "diagnostic representative crossing rollout only; not benchmark campaign, "
            "COLSON reproduction, navigation-quality result, or paper evidence"
        ),
        "scenario_id": "issue_4010_representative_crossing_smoke",
        "scenario_family": "crossing",
        "step_count": step_count,
        "seed": seed,
        "finite_command_count": finite_command_count,
        "commands_within_limits": commands_valid,
        "runtime_loaded_checkpoint": runtime_loaded,
        "final_robot_position": [robot_x, robot_y],
        "trajectory": trajectory,
        "runtime_metadata": metadata.get("diffusion_policy", {}),
        "passed": passed,
        "out_of_scope": [
            "no full benchmark campaign",
            "no Slurm/GPU submission",
            "no baseline performance comparison claim",
            "no paper/dissertation claim",
        ],
    }


def _validated_smoke_artifacts(manifest: dict[str, Any]) -> dict[str, str]:
    """Return required smoke artifact paths or fail closed."""
    if manifest.get("schema_version") != SMOKE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Diffusion policy probe requires {SMOKE_MANIFEST_SCHEMA_VERSION} manifest"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Diffusion policy smoke manifest missing artifacts mapping")
    required_artifacts = ("checkpoint_path", "normalizer_path")
    missing_artifacts = [
        artifact for artifact in required_artifacts if not str(artifacts.get(artifact, "")).strip()
    ]
    if missing_artifacts:
        raise ValueError(
            "Diffusion policy smoke manifest missing required artifacts: "
            + ", ".join(missing_artifacts)
        )
    return {artifact: str(artifacts[artifact]) for artifact in required_artifacts}


def _resolve_manifest_artifact(artifact_dir: Path, artifact_path: str) -> Path:
    """Resolve manifest artifact path relative to its smoke artifact directory.

    Returns:
        Absolute or artifact-directory-relative path to the generated file.
    """
    path = Path(artifact_path).expanduser()
    if path.is_absolute():
        return path
    return artifact_dir.expanduser() / path


def _multimodal_probe_status(probe: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize optional multimodal probe payload for diagnostic packet status.

    Returns:
        Compact status payload consumed by the integration diagnostic packet.
    """
    if probe is None:
        return {
            "passed": False,
            "reason": "candidate diversity has not yet been classified on a fixed conflict case",
            "summary": None,
        }
    if probe.get("schema_version") != MULTIMODAL_PROBE_SCHEMA_VERSION:
        return {
            "passed": False,
            "reason": "multimodal probe schema is not recognized",
            "summary": {"schema_version": probe.get("schema_version"), "passed": False},
        }
    passed = bool(probe.get("passed"))
    return {
        "passed": passed,
        "reason": (
            "fixed-conflict probe did not record at least two non-empty core action modes"
            if not passed
            else "fixed-conflict probe recorded at least two non-empty core action modes"
        ),
        "summary": {
            "schema_version": probe.get("schema_version"),
            "scenario_id": probe.get("scenario_id"),
            "sample_count": probe.get("sample_count"),
            "mode_clusters": probe.get("mode_clusters", []),
            "distinct_core_mode_count": probe.get("distinct_core_mode_count"),
            "passed": passed,
            "claim_boundary": probe.get("claim_boundary"),
        },
    }


def _representative_rollout_status(rollout: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize optional representative rollout payload diagnostic packet status.

    Returns:
        Compact status payload consumed by the integration diagnostic packet.
    """
    if rollout is None:
        return {
            "passed": False,
            "reason": "representative checkpoint-backed scenario rollout not yet recorded",
            "summary": None,
        }
    if rollout.get("schema_version") != REPRESENTATIVE_ROLLOUT_SCHEMA_VERSION:
        return {
            "passed": False,
            "reason": "representative rollout schema not recognized",
            "summary": {"schema_version": rollout.get("schema_version"), "passed": False},
        }
    passed = bool(rollout.get("passed"))
    return {
        "passed": passed,
        "reason": (
            "representative rollout produced finite bounded checkpoint-backed commands"
            if passed
            else "representative rollout did not prove finite bounded checkpoint-backed commands"
        ),
        "summary": {
            "schema_version": rollout.get("schema_version"),
            "scenario_id": rollout.get("scenario_id"),
            "scenario_family": rollout.get("scenario_family"),
            "step_count": rollout.get("step_count"),
            "finite_command_count": rollout.get("finite_command_count"),
            "commands_within_limits": rollout.get("commands_within_limits"),
            "runtime_loaded_checkpoint": rollout.get("runtime_loaded_checkpoint"),
            "passed": passed,
            "claim_boundary": rollout.get("claim_boundary"),
        },
    }


def _fixed_conflict_observation() -> dict[str, Any]:
    """Return deterministic symmetric conflict observation for action diversity probe."""
    return {
        "robot": {
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "goal": [2.0, 0.0],
            "heading": 0.0,
            "radius": 0.3,
        },
        "agents": [
            {"position": [0.85, 0.18], "velocity": [-0.12, -0.02], "radius": 0.25},
            {"position": [0.85, -0.18], "velocity": [-0.12, 0.02], "radius": 0.25},
        ],
        "obstacles": [],
        "dt": 0.1,
    }


def _representative_rollout_observation(
    step: int, robot_x: float, robot_y: float, heading: float, dt: float
) -> dict[str, Any]:
    """Return map-runner style crossing observation for a short diagnostic rollout."""
    ped_a_y = 0.45 - 0.06 * step
    ped_b_y = -0.45 + 0.06 * step
    return {
        "robot": {
            "position": [robot_x, robot_y],
            "velocity": [0.0, 0.0],
            "goal": [2.0, 0.0],
            "heading": [heading],
            "radius": [0.3],
        },
        "pedestrians": {
            "positions": [[0.9, ped_a_y], [1.1, ped_b_y]],
            "velocities": [[0.0, -0.25], [0.0, 0.25]],
            "radii": [0.25, 0.25],
            "count": [2],
        },
        "dt": [dt],
    }


def _classify_action_mode(
    action: list[float] | tuple[float, ...],
    angular_threshold: float,
    slow_speed_threshold: float,
) -> str:
    """Classify one `(linear, angular)` command into diagnostic mode labels.

    Returns:
        Diagnostic mode label for the sampled command.
    """
    linear = float(action[0])
    angular = float(action[1])
    if linear <= slow_speed_threshold:
        return "slow_or_wait"
    if angular >= angular_threshold:
        return "pass_left"
    if angular <= -angular_threshold:
        return "pass_right"
    return "straight_or_uncertain"


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
            "map_runner_load_contract": {
                "allow_untrained_smoke": False,
                "requires_checkpoint_path": True,
                "requires_normalizer_path": True,
                "algo_config_fragment": {
                    "checkpoint_path": checkpoint_path.name,
                    "normalizer_path": normalizer_path.name,
                    "allow_untrained_smoke": False,
                    "deterministic": True,
                    "seed": config.seed,
                    "max_pedestrians": config.max_pedestrians,
                    "denoising_steps": config.denoising_steps,
                    "num_action_samples": config.num_action_samples,
                    "max_linear_speed": config.max_linear_speed,
                    "max_angular_speed": config.max_angular_speed,
                },
            },
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
    parser.add_argument(
        "--write-diagnostic-packet",
        action="store_true",
        help="Also write a diagnostic-only integration packet next to the smoke manifest.",
    )
    parser.add_argument(
        "--write-multimodal-probe",
        action="store_true",
        help="Also write a diagnostic-only fixed-conflict multimodal action probe.",
    )
    parser.add_argument(
        "--write-representative-rollout",
        action="store_true",
        help="Also write a diagnostic-only checkpoint-backed representative rollout.",
    )
    parser.add_argument(
        "--multimodal-samples",
        type=int,
        default=64,
        help="Number of fixed-conflict candidate actions to classify.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=12,
        help="Number of representative rollout steps to execute.",
    )
    args = parser.parse_args(argv)
    artifacts = run_training_smoke(load_smoke_config(args.config), output_dir=args.output_dir)
    payload = {"manifest_path": str(artifacts.manifest_path)}
    multimodal_probe = None
    representative_rollout = None
    if args.write_multimodal_probe:
        probe_path = artifacts.manifest_path.with_suffix(".multimodal_probe.json")
        multimodal_probe = build_multimodal_probe(
            artifacts.manifest,
            artifact_dir=artifacts.manifest_path.parent,
            sample_count=args.multimodal_samples,
        )
        probe_path.write_text(
            json.dumps(multimodal_probe, indent=2, sort_keys=True), encoding="utf-8"
        )
        payload["multimodal_probe_path"] = str(probe_path)
    if args.write_representative_rollout:
        rollout_path = artifacts.manifest_path.with_suffix(".representative_rollout.json")
        representative_rollout = build_representative_rollout(
            artifacts.manifest,
            artifact_dir=artifacts.manifest_path.parent,
            step_count=args.rollout_steps,
        )
        rollout_path.write_text(
            json.dumps(representative_rollout, indent=2, sort_keys=True), encoding="utf-8"
        )
        payload["representative_rollout_path"] = str(rollout_path)
    if args.write_diagnostic_packet:
        packet_path = artifacts.manifest_path.with_suffix(".diagnostic_packet.json")
        packet = build_diagnostic_packet(
            artifacts.manifest,
            multimodal_probe=multimodal_probe,
            representative_rollout=representative_rollout,
        )
        packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
        payload["diagnostic_packet_path"] = str(packet_path)
    sys.stdout.write(json.dumps(payload, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
