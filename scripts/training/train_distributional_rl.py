"""QR-DQN-style diagnostic training entry point for issue #4016.

This entry point intentionally produces smoke/provenance artifacts only. It
exercises the merged distributional-RL primitives on CPU and writes a checkpoint
that the map-runner adapter can load, but it is not benchmark or paper evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from robot_sf.training.discrete_action_lattice import DiscreteUnicycleActionLattice
from robot_sf.training.distributional_rl import (
    QuantileQNetwork,
    build_qr_dqn_targets,
    hard_update_target_network,
    quantile_huber_loss,
    save_quantile_checkpoint,
)
from robot_sf.training.risk_objectives import RISK_OBJECTIVES, score_action_quantiles

LOGGER = logging.getLogger(__name__)

_ALLOWED_ROOT_KEYS = {
    "policy_id",
    "algorithm",
    "scenario_config",
    "total_timesteps",
    "seed",
    "device",
    "num_envs",
    "env_overrides",
    "env_factory_kwargs",
    "observation",
    "action_lattice",
    "critic",
    "risk_selection",
    "dqn",
    "tracking",
    "output_dir",
}

_CLAIM_BOUNDARY = "diagnostic distributional-RL smoke; not benchmark or paper-grade evidence"


@dataclass(frozen=True)
class ObservationConfig:
    """Observation-shape options for the diagnostic smoke trainer."""

    transform: str = "ego"
    flatten_dict: bool = True
    synthetic_observation_dim: int = 8


@dataclass(frozen=True)
class CriticConfig:
    """Quantile critic network parameters."""

    hidden_sizes: tuple[int, ...] = (64, 64)
    num_quantiles: int = 8
    kappa: float = 1.0
    double_q: bool = True
    target_update_interval: int = 32


@dataclass(frozen=True)
class RiskSelectionConfig:
    """Risk-aware action-selection parameters."""

    objective: str = "cvar_lower"
    alpha: float = 0.2
    blend_beta: float = 1.0


@dataclass(frozen=True)
class DQNConfig:
    """Small QR-DQN optimizer/smoke-loop parameters."""

    replay_size: int = 4096
    batch_size: int = 32
    learning_rate: float = 1e-4
    gamma: float = 0.99
    learning_starts: int = 16
    train_freq: int = 1
    gradient_steps: int = 1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    exploration_fraction: float = 0.3
    max_grad_norm: float = 10.0


@dataclass(frozen=True)
class DistributionalRLTrainingConfig:
    """Resolved QR-DQN smoke training configuration."""

    policy_id: str
    algorithm: str
    scenario_config: Path
    total_timesteps: int
    seed: int
    device: str
    num_envs: int
    output_dir: Path
    env_overrides: dict[str, Any] = field(default_factory=dict)
    env_factory_kwargs: dict[str, Any] = field(default_factory=dict)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    action_lattice: DiscreteUnicycleActionLattice = field(
        default_factory=lambda: DiscreteUnicycleActionLattice(
            linear_values=(0.0, 0.5),
            angular_values=(-0.5, 0.0, 0.5),
            max_linear_speed=1.0,
            max_angular_speed=1.0,
        )
    )
    critic: CriticConfig = field(default_factory=CriticConfig)
    risk_selection: RiskSelectionConfig = field(default_factory=RiskSelectionConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    tracking: dict[str, Any] = field(default_factory=dict)


def _require_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return dict(value)


def _resolve_path(value: Any, *, base_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_observation_config(payload: Any) -> ObservationConfig:
    data = _require_mapping(payload, field_name="observation")
    unknown = set(data) - {"transform", "flatten_dict", "synthetic_observation_dim"}
    if unknown:
        raise ValueError(f"unknown observation keys: {sorted(unknown)}")
    dim = int(data.get("synthetic_observation_dim", 8))
    if dim < 1:
        raise ValueError("observation.synthetic_observation_dim must be positive")
    return ObservationConfig(
        transform=str(data.get("transform", "ego")),
        flatten_dict=bool(data.get("flatten_dict", True)),
        synthetic_observation_dim=dim,
    )


def _load_critic_config(payload: Any) -> CriticConfig:
    data = _require_mapping(payload, field_name="critic")
    unknown = set(data) - {
        "hidden_sizes",
        "num_quantiles",
        "kappa",
        "double_q",
        "target_update_interval",
    }
    if unknown:
        raise ValueError(f"unknown critic keys: {sorted(unknown)}")
    hidden_sizes = tuple(int(value) for value in data.get("hidden_sizes", (64, 64)))
    if not hidden_sizes or any(value < 1 for value in hidden_sizes):
        raise ValueError("critic.hidden_sizes must contain positive widths")
    num_quantiles = int(data.get("num_quantiles", 8))
    if num_quantiles < 2:
        raise ValueError("critic.num_quantiles must be at least 2")
    kappa = float(data.get("kappa", 1.0))
    if kappa <= 0.0:
        raise ValueError("critic.kappa must be positive")
    target_update_interval = int(data.get("target_update_interval", 32))
    if target_update_interval < 1:
        raise ValueError("critic.target_update_interval must be positive")
    return CriticConfig(
        hidden_sizes=hidden_sizes,
        num_quantiles=num_quantiles,
        kappa=kappa,
        double_q=bool(data.get("double_q", True)),
        target_update_interval=target_update_interval,
    )


def _load_risk_selection_config(payload: Any) -> RiskSelectionConfig:
    data = _require_mapping(payload, field_name="risk_selection")
    unknown = set(data) - {"objective", "alpha", "blend_beta"}
    if unknown:
        raise ValueError(f"unknown risk_selection keys: {sorted(unknown)}")
    objective = str(data.get("objective", "cvar_lower")).strip()
    if objective not in RISK_OBJECTIVES:
        raise ValueError(f"risk_selection.objective must be one of {RISK_OBJECTIVES}")
    alpha = float(data.get("alpha", 0.2))
    if not 0.0 < alpha <= 1.0:
        raise ValueError("risk_selection.alpha must be in (0, 1]")
    blend_beta = float(data.get("blend_beta", 1.0))
    if not 0.0 <= blend_beta <= 1.0:
        raise ValueError("risk_selection.blend_beta must be in [0, 1]")
    return RiskSelectionConfig(objective=objective, alpha=alpha, blend_beta=blend_beta)


def _load_dqn_config(payload: Any) -> DQNConfig:
    data = _require_mapping(payload, field_name="dqn")
    unknown = set(data) - {
        "replay_size",
        "batch_size",
        "learning_rate",
        "gamma",
        "learning_starts",
        "train_freq",
        "gradient_steps",
        "exploration_initial_eps",
        "exploration_final_eps",
        "exploration_fraction",
        "max_grad_norm",
    }
    if unknown:
        raise ValueError(f"unknown dqn keys: {sorted(unknown)}")
    config = DQNConfig(
        replay_size=int(data.get("replay_size", 4096)),
        batch_size=int(data.get("batch_size", 32)),
        learning_rate=float(data.get("learning_rate", 1e-4)),
        gamma=float(data.get("gamma", 0.99)),
        learning_starts=int(data.get("learning_starts", 16)),
        train_freq=int(data.get("train_freq", 1)),
        gradient_steps=int(data.get("gradient_steps", 1)),
        exploration_initial_eps=float(data.get("exploration_initial_eps", 1.0)),
        exploration_final_eps=float(data.get("exploration_final_eps", 0.05)),
        exploration_fraction=float(data.get("exploration_fraction", 0.3)),
        max_grad_norm=float(data.get("max_grad_norm", 10.0)),
    )
    if config.replay_size < 1 or config.batch_size < 1:
        raise ValueError("dqn replay_size and batch_size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("dqn.learning_rate must be positive")
    if not 0.0 <= config.gamma <= 1.0:
        raise ValueError("dqn.gamma must be in [0, 1]")
    if config.train_freq < 1 or config.gradient_steps < 1:
        raise ValueError("dqn train_freq and gradient_steps must be positive")
    return config


def load_distributional_rl_training_config(
    config_path: str | Path,
) -> DistributionalRLTrainingConfig:
    """Load and validate a QR-DQN smoke training YAML."""

    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("training config root must be a mapping")
    unknown = set(data) - _ALLOWED_ROOT_KEYS
    if unknown:
        raise ValueError(f"unknown training config keys: {sorted(unknown)}")

    algorithm = str(data.get("algorithm", "qr_dqn")).strip()
    if algorithm not in {"qr_dqn", "mean_dqn"}:
        raise ValueError("algorithm must be 'qr_dqn' or 'mean_dqn'")
    total_timesteps = int(data["total_timesteps"])
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be positive")
    num_envs = int(data.get("num_envs", 1))
    if num_envs != 1:
        raise ValueError("diagnostic distributional RL smoke currently requires num_envs=1")

    base_dir = path.parent
    output_dir = _resolve_path(
        data.get("output_dir", "output/models/distributional_rl"), base_dir=base_dir
    )
    lattice = DiscreteUnicycleActionLattice.from_dict(
        _require_mapping(data["action_lattice"], field_name="action_lattice")
    )
    return DistributionalRLTrainingConfig(
        policy_id=str(data["policy_id"]),
        algorithm=algorithm,
        scenario_config=_resolve_path(data["scenario_config"], base_dir=base_dir),
        total_timesteps=total_timesteps,
        seed=int(data.get("seed", 0)),
        device=str(data.get("device", "cpu")),
        num_envs=num_envs,
        output_dir=output_dir,
        env_overrides=_require_mapping(data.get("env_overrides"), field_name="env_overrides"),
        env_factory_kwargs=_require_mapping(
            data.get("env_factory_kwargs"), field_name="env_factory_kwargs"
        ),
        observation=_load_observation_config(data.get("observation")),
        action_lattice=lattice,
        critic=_load_critic_config(data.get("critic")),
        risk_selection=_load_risk_selection_config(data.get("risk_selection")),
        dqn=_load_dqn_config(data.get("dqn")),
        tracking=_require_mapping(data.get("tracking"), field_name="tracking"),
    )


def _config_to_dict(config: DistributionalRLTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["scenario_config"] = str(config.scenario_config)
    payload["output_dir"] = str(config.output_dir)
    payload["action_lattice"] = config.action_lattice.to_dict()
    return payload


def _epsilon_at_step(config: DQNConfig, step: int, total_timesteps: int) -> float:
    decay_steps = max(1, int(total_timesteps * config.exploration_fraction))
    fraction = min(1.0, step / decay_steps)
    return config.exploration_initial_eps + fraction * (
        config.exploration_final_eps - config.exploration_initial_eps
    )


def _synthetic_transition(
    *,
    observation_dim: int,
    action_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = torch.randn(observation_dim, device=device)
    next_obs = torch.randn(observation_dim, device=device)
    action = torch.randint(0, action_count, (1,), device=device).squeeze(0)
    reward = torch.tanh(obs[: min(3, observation_dim)].sum() * 0.2) - 0.03 * action.float()
    done = torch.rand((), device=device) < 0.03
    return obs, action, reward, next_obs, done.to(dtype=torch.float32)


def _sample_replay(
    replay: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = random.sample(replay, k=batch_size)
    observations, actions, rewards, next_observations, dones = zip(*batch, strict=True)
    return (
        torch.stack(observations),
        torch.stack(actions).long(),
        torch.stack(rewards).float(),
        torch.stack(next_observations),
        torch.stack(dones).float(),
    )


def _write_manifest(
    config: DistributionalRLTrainingConfig,
    *,
    checkpoint_path: Path,
    action_lattice_path: Path,
    resolved_config_path: Path,
    trace_path: Path,
    dry_run: bool,
    train_steps: int,
    final_loss: float | None,
) -> Path:
    manifest = {
        "policy_id": config.policy_id,
        "algorithm": config.algorithm,
        "evidence_tier": "smoke",
        "claim_boundary": _CLAIM_BOUNDARY,
        "scenario_config": str(config.scenario_config),
        "seed": config.seed,
        "total_timesteps": config.total_timesteps,
        "train_steps": train_steps,
        "num_quantiles": config.critic.num_quantiles,
        "risk_objective": config.risk_selection.objective,
        "risk_alpha": config.risk_selection.alpha,
        "risk_blend_beta": config.risk_selection.blend_beta,
        "action_count": config.action_lattice.action_count,
        "action_lattice_path": str(action_lattice_path),
        "checkpoint_path": str(checkpoint_path),
        "resolved_config_path": str(resolved_config_path),
        "training_trace_path": str(trace_path),
        "fallback_or_degraded": False,
        "dry_run": dry_run,
        "final_loss": final_loss,
    }
    manifest_path = config.output_dir / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def run_distributional_rl_training(
    config: DistributionalRLTrainingConfig,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a CPU QR-DQN diagnostic smoke loop and write provenance artifacts."""

    if config.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {config.device!r} unavailable")
    device = torch.device(config.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / f"{config.policy_id}.pt"
    action_lattice_path = config.output_dir / "action_lattice.json"
    resolved_config_path = config.output_dir / "resolved_config.yaml"
    trace_path = config.output_dir / "training_trace.jsonl"

    model = QuantileQNetwork(
        observation_dim=config.observation.synthetic_observation_dim,
        action_count=config.action_lattice.action_count,
        num_quantiles=config.critic.num_quantiles,
        hidden_sizes=config.critic.hidden_sizes,
    ).to(device)
    target = QuantileQNetwork(
        observation_dim=config.observation.synthetic_observation_dim,
        action_count=config.action_lattice.action_count,
        num_quantiles=config.critic.num_quantiles,
        hidden_sizes=config.critic.hidden_sizes,
    ).to(device)
    hard_update_target_network(model, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.dqn.learning_rate)

    train_steps = 0
    final_loss: float | None = None
    replay: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    config.action_lattice.to_json_file(action_lattice_path)
    resolved_config_path.write_text(
        yaml.safe_dump(_config_to_dict(config), sort_keys=True), encoding="utf-8"
    )

    with trace_path.open("w", encoding="utf-8") as trace_handle:
        if not dry_run:
            for step in range(config.total_timesteps):
                replay.append(
                    _synthetic_transition(
                        observation_dim=config.observation.synthetic_observation_dim,
                        action_count=config.action_lattice.action_count,
                        device=device,
                    )
                )
                if len(replay) > config.dqn.replay_size:
                    replay.pop(0)
                if step < config.dqn.learning_starts or len(replay) < config.dqn.batch_size:
                    continue
                if step % config.dqn.train_freq != 0:
                    continue
                for _ in range(config.dqn.gradient_steps):
                    observations, actions, rewards, next_observations, dones = _sample_replay(
                        replay, config.dqn.batch_size
                    )
                    quantiles = model(observations)
                    selected_quantiles = quantiles.gather(
                        dim=1,
                        index=actions.view(-1, 1, 1).expand(-1, 1, config.critic.num_quantiles),
                    ).squeeze(1)
                    with torch.no_grad():
                        online_next = model(next_observations)
                        next_scores = score_action_quantiles(
                            online_next,
                            objective=config.risk_selection.objective,
                            alpha=config.risk_selection.alpha,
                            blend_beta=config.risk_selection.blend_beta,
                        )
                        next_actions = next_scores.argmax(dim=-1)
                        target_batch = build_qr_dqn_targets(
                            rewards,
                            dones,
                            target(next_observations),
                            next_actions,
                            gamma=config.dqn.gamma,
                        )
                    loss = quantile_huber_loss(
                        selected_quantiles,
                        target_batch.target_quantiles,
                        kappa=config.critic.kappa,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.dqn.max_grad_norm)
                    optimizer.step()
                    train_steps += 1
                    final_loss = float(loss.detach().cpu())
                    trace_handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "train_step": train_steps,
                                "loss": final_loss,
                                "epsilon": _epsilon_at_step(
                                    config.dqn, step, config.total_timesteps
                                ),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                if step % config.critic.target_update_interval == 0:
                    hard_update_target_network(model, target)

    save_quantile_checkpoint(
        checkpoint_path,
        model=model.cpu(),
        action_lattice=config.action_lattice,
    )
    manifest_path = _write_manifest(
        config,
        checkpoint_path=checkpoint_path,
        action_lattice_path=action_lattice_path,
        resolved_config_path=resolved_config_path,
        trace_path=trace_path,
        dry_run=dry_run,
        train_steps=train_steps,
        final_loss=final_loss,
    )
    LOGGER.info("wrote diagnostic distributional RL artifacts under %s", config.output_dir)
    return {
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_path),
        "action_lattice_path": str(action_lattice_path),
        "resolved_config_path": str(resolved_config_path),
        "training_trace_path": str(trace_path),
        "train_steps": train_steps,
        "dry_run": dry_run,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the QR-DQN smoke trainer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to QR-DQN training YAML")
    parser.add_argument("--dry-run", action="store_true", help="Validate and write artifacts only")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the QR-DQN smoke trainer CLI."""

    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    config = load_distributional_rl_training_config(args.config)
    result = run_distributional_rl_training(config, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
