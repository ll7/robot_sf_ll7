"""PPO-Lagrangian constrained-RL training entry point.

This script adds a small config-first lane around the constrained reward wrapper from issue #4017.
It is intended for CPU smoke and diagnostic training runs only; it does not promote benchmark or
paper-facing safety claims.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from loguru import logger

try:  # pragma: no cover - exercised by runtime smoke validation when dependency is present.
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Stable-Baselines3 is required for constrained-RL PPO training.") from exc

from robot_sf.training.constrained_reward_wrapper import ConstrainedRewardWrapper
from robot_sf.training.safety_constraints import SafetyConstraintSpec
from scripts.training import train_ppo

_TOP_LEVEL_KEYS = {
    "policy_id",
    "algorithm",
    "scenario_config",
    "total_timesteps",
    "seed",
    "num_envs",
    "device",
    "policy",
    "env_overrides",
    "env_factory_kwargs",
    "ppo_hyperparams",
    "safety_constraints",
    "tracking",
    "output_dir",
}
_SAFETY_KEYS = {"schema_version", "enabled", "method", "update_mode", "constraints"}
_CONSTRAINT_KEYS = {
    "name",
    "source_key",
    "budget_per_episode",
    "multiplier_init",
    "multiplier_lr",
    "multiplier_max",
    "normalize_by_episode_steps",
}
_ALLOWED_PPO_HYPERPARAMS = {
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "ent_coef",
    "vf_coef",
    "clip_range",
    "max_grad_norm",
    "target_kl",
}


@dataclass(frozen=True, slots=True)
class SafetyConstraintsConfig:
    """Validated constrained-RL safety-constraint block."""

    schema_version: str = "constrained_rl.v1"
    enabled: bool = False
    method: str = "lagrangian_ppo"
    update_mode: str = "episode"
    constraints: tuple[SafetyConstraintSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class ConstrainedRLConfig:
    """Validated PPO-Lagrangian training config."""

    policy_id: str
    algorithm: str
    scenario_config: Path
    total_timesteps: int
    seed: int
    num_envs: int = 1
    device: str = "cpu"
    policy: str = "MultiInputPolicy"
    env_overrides: dict[str, object] = field(default_factory=dict)
    env_factory_kwargs: dict[str, object] = field(default_factory=dict)
    ppo_hyperparams: dict[str, object] = field(default_factory=dict)
    safety_constraints: SafetyConstraintsConfig = field(default_factory=SafetyConstraintsConfig)
    tracking: dict[str, object] = field(default_factory=dict)
    output_dir: Path = Path("output/models/constrained_rl/issue_4017")

    @property
    def algorithm_manifest_id(self) -> str:
        """Return algorithm identifier used in manifests."""

        if self.safety_constraints.enabled:
            return "ppo_lagrangian"
        return "ppo_unconstrained"


class ConstrainedRLCallback(BaseCallback):
    """Update Lagrange multipliers from terminal wrapper diagnostics."""

    def __init__(self, *, trace_path: Path, constraints_enabled: bool) -> None:
        """Initialize callback that appends compact constraint trace records."""

        super().__init__()
        self._trace_path = trace_path
        self._constraints_enabled = constraints_enabled

    def _on_training_start(self) -> None:
        """Ensure the trace path exists before training begins."""

        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace_path.write_text("", encoding="utf-8")

    def _on_step(self) -> bool:
        """Collect completed episode diagnostics and update wrapper multipliers."""

        if not self._constraints_enabled:
            return True
        infos = self.locals.get("infos", ())
        for env_index, info in enumerate(infos):
            if not isinstance(info, Mapping) or "constraint_episode" not in info:
                continue
            episode = info["constraint_episode"]
            if not isinstance(episode, Mapping):
                continue
            costs = _float_mapping(episode.get("costs", {}))
            episode_steps = int(episode.get("episode_steps") or 0)
            updates = self.training_env.env_method(
                "update_multipliers_from_episode",
                episode_costs=costs,
                episode_steps=episode_steps,
                indices=env_index,
            )
            multipliers_after = updates[0] if updates else {}
            record = {
                "timesteps": int(self.num_timesteps),
                "env_index": env_index,
                "costs": costs,
                "budgets": _float_mapping(episode.get("budgets", {})),
                "violations": _float_mapping(episode.get("violations", {})),
                "multipliers_before_update": _float_mapping(
                    episode.get("multipliers_before_update", {})
                ),
                "multipliers_after_update": _float_mapping(multipliers_after),
                "episode_steps": episode_steps,
            }
            _append_jsonl(self._trace_path, record)
            for name, value in record["violations"].items():
                self.logger.record(f"constraint/{name}_violation", value)
            for name, value in record["multipliers_after_update"].items():
                self.logger.record(f"constraint/{name}_multiplier", value)
        return True


def load_constrained_rl_config(config_path: str | Path) -> ConstrainedRLConfig:
    """Load and validate a constrained-RL YAML config."""

    path = Path(config_path).resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Constrained-RL config must be a YAML mapping.")
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "config")

    scenario_config = _resolve_path(path, raw["scenario_config"])
    output_dir = _resolve_output_dir(
        raw.get("output_dir", "output/models/constrained_rl/issue_4017")
    )
    ppo_hyperparams = dict(_mapping(raw.get("ppo_hyperparams", {}), "ppo_hyperparams"))
    _reject_unknown_keys(ppo_hyperparams, _ALLOWED_PPO_HYPERPARAMS, "ppo_hyperparams")

    config = ConstrainedRLConfig(
        policy_id=str(raw["policy_id"]),
        algorithm=str(raw.get("algorithm", "ppo")),
        scenario_config=scenario_config,
        total_timesteps=_positive_int(raw.get("total_timesteps"), "total_timesteps"),
        seed=int(raw.get("seed", 4017)),
        num_envs=_positive_int(raw.get("num_envs", 1), "num_envs"),
        device=str(raw.get("device", "cpu")),
        policy=str(raw.get("policy", "MultiInputPolicy")),
        env_overrides=dict(_mapping(raw.get("env_overrides", {}), "env_overrides")),
        env_factory_kwargs=dict(_mapping(raw.get("env_factory_kwargs", {}), "env_factory_kwargs")),
        ppo_hyperparams=ppo_hyperparams,
        safety_constraints=_parse_safety_constraints(raw.get("safety_constraints", {})),
        tracking=dict(_mapping(raw.get("tracking", {}), "tracking")),
        output_dir=output_dir,
    )
    if config.algorithm != "ppo":
        raise ValueError("Only algorithm: ppo is supported by this entry point.")
    if config.device != "cpu":
        raise ValueError("Issue #4017 smoke configs must use device: cpu.")
    return config


def train_constrained_rl(
    config: ConstrainedRLConfig,
    *,
    config_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Run or dry-run PPO-Lagrangian training and persist provenance metadata."""

    start_time = time.perf_counter()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "resolved_config.yaml"
    manifest_path = output_dir / "training_manifest.json"
    trace_path = output_dir / "constraint_trace.jsonl"
    checkpoint_path = output_dir / f"{config.policy_id}.zip"

    resolved_payload = _config_to_payload(config)
    _write_yaml(resolved_config_path, resolved_payload)
    trace_path.write_text("", encoding="utf-8")

    if dry_run:
        manifest = _training_manifest(
            config,
            config_path=config_path,
            checkpoint_path=None,
            resolved_config_path=resolved_config_path,
            trace_path=trace_path,
            dry_run=True,
            runtime_seconds=round(time.perf_counter() - start_time, 6),
        )
        _write_json(manifest_path, manifest)
        logger.info("Dry-run wrote constrained-RL manifest to {}", manifest_path)
        return {
            "manifest_path": manifest_path,
            "resolved_config_path": resolved_config_path,
            "trace_path": trace_path,
            "checkpoint_path": None,
        }

    vec_env = _build_vec_env(config)
    try:
        model = PPO(
            config.policy,
            vec_env,
            seed=config.seed,
            device=config.device,
            verbose=0,
            **config.ppo_hyperparams,
        )
        callback = ConstrainedRLCallback(
            trace_path=trace_path,
            constraints_enabled=config.safety_constraints.enabled,
        )
        model.learn(total_timesteps=config.total_timesteps, callback=callback)
        model.save(checkpoint_path)
    finally:
        vec_env.close()

    manifest = _training_manifest(
        config,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        resolved_config_path=resolved_config_path,
        trace_path=trace_path,
        dry_run=False,
        runtime_seconds=round(time.perf_counter() - start_time, 6),
    )
    _write_json(manifest_path, manifest)
    logger.info("Saved constrained-RL checkpoint to {}", checkpoint_path)
    return {
        "manifest_path": manifest_path,
        "resolved_config_path": resolved_config_path,
        "trace_path": trace_path,
        "checkpoint_path": checkpoint_path,
    }


def _build_vec_env(config: ConstrainedRLConfig) -> DummyVecEnv:
    """Build a CPU DummyVecEnv matching the constrained-RL config."""

    scenarios = train_ppo.load_scenarios(config.scenario_config)
    factories = []
    for env_index in range(config.num_envs):
        base_factory = train_ppo._make_training_env(
            config.seed + env_index,
            scenario=None,
            scenario_definitions=scenarios,
            scenario_path=config.scenario_config,
            exclude_scenarios=(),
            suite_name="constrained_rl",
            algorithm_name=config.algorithm_manifest_id,
            env_overrides=config.env_overrides,
            env_factory_kwargs=config.env_factory_kwargs,
            scenario_sampling={},
        )

        def _factory(base_factory=base_factory):
            env = base_factory()
            if config.safety_constraints.enabled:
                return ConstrainedRewardWrapper(env, config.safety_constraints.constraints)
            return env

        factories.append(_factory)
    return DummyVecEnv(factories)


def _parse_safety_constraints(raw: object) -> SafetyConstraintsConfig:
    data = dict(_mapping(raw, "safety_constraints"))
    _reject_unknown_keys(data, _SAFETY_KEYS, "safety_constraints")
    constraints_raw = data.get("constraints", ())
    if constraints_raw is None:
        constraints_raw = ()
    if not isinstance(constraints_raw, Sequence) or isinstance(constraints_raw, (str, bytes)):
        raise ValueError("safety_constraints.constraints must be a list.")
    specs = tuple(_parse_constraint_spec(item, index=i) for i, item in enumerate(constraints_raw))
    enabled = bool(data.get("enabled", False))
    method = str(data.get("method", "lagrangian_ppo"))
    update_mode = str(data.get("update_mode", "episode"))
    if method != "lagrangian_ppo":
        raise ValueError("safety_constraints.method must be lagrangian_ppo.")
    if update_mode != "episode":
        raise ValueError("safety_constraints.update_mode must be episode.")
    if enabled and not specs:
        raise ValueError("Enabled safety_constraints require at least one constraint.")
    if not enabled and specs:
        raise ValueError("Disabled safety_constraints must not define constraints.")
    return SafetyConstraintsConfig(
        schema_version=str(data.get("schema_version", "constrained_rl.v1")),
        enabled=enabled,
        method=method,
        update_mode=update_mode,
        constraints=specs,
    )


def _parse_constraint_spec(raw: object, *, index: int) -> SafetyConstraintSpec:
    data = dict(_mapping(raw, f"safety_constraints.constraints[{index}]"))
    _reject_unknown_keys(data, _CONSTRAINT_KEYS, f"safety_constraints.constraints[{index}]")
    return SafetyConstraintSpec(
        name=str(data["name"]),
        source_key=str(data["source_key"]),
        budget_per_episode=float(data["budget_per_episode"]),
        multiplier_init=float(data.get("multiplier_init", 1.0)),
        multiplier_lr=float(data.get("multiplier_lr", 0.05)),
        multiplier_max=float(data.get("multiplier_max", 50.0)),
        normalize_by_episode_steps=bool(data.get("normalize_by_episode_steps", False)),
    )


def _training_manifest(
    config: ConstrainedRLConfig,
    *,
    config_path: Path | None,
    checkpoint_path: Path | None,
    resolved_config_path: Path,
    trace_path: Path,
    dry_run: bool,
    runtime_seconds: float,
) -> dict[str, object]:
    """Build the reviewable training manifest."""

    return {
        "policy_id": config.policy_id,
        "algorithm": config.algorithm_manifest_id,
        "evidence_tier": "smoke",
        "claim_boundary": (
            "diagnostic constrained-RL training smoke; not benchmark, paper-grade, "
            "or dissertation safety evidence"
        ),
        "dry_run": dry_run,
        "scenario_config": str(config.scenario_config),
        "seed": config.seed,
        "total_timesteps": config.total_timesteps,
        "num_envs": config.num_envs,
        "device": config.device,
        "constraints_enabled": config.safety_constraints.enabled,
        "constraints": [asdict(spec) for spec in config.safety_constraints.constraints],
        "runtime_seconds": runtime_seconds,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "resolved_config_path": str(resolved_config_path),
        "constraint_trace_path": str(trace_path),
        "config_path": str(config_path) if config_path else None,
        "fallback_or_degraded": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _config_to_payload(config: ConstrainedRLConfig) -> dict[str, object]:
    payload = {
        "policy_id": config.policy_id,
        "algorithm": config.algorithm,
        "scenario_config": str(config.scenario_config),
        "total_timesteps": config.total_timesteps,
        "seed": config.seed,
        "num_envs": config.num_envs,
        "device": config.device,
        "policy": config.policy,
        "env_overrides": config.env_overrides,
        "env_factory_kwargs": config.env_factory_kwargs,
        "ppo_hyperparams": config.ppo_hyperparams,
        "safety_constraints": {
            "schema_version": config.safety_constraints.schema_version,
            "enabled": config.safety_constraints.enabled,
            "method": config.safety_constraints.method,
            "update_mode": config.safety_constraints.update_mode,
            "constraints": [asdict(spec) for spec in config.safety_constraints.constraints],
        },
        "tracking": config.tracking,
        "output_dir": str(config.output_dir),
    }
    return payload


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _reject_unknown_keys(data: Mapping[str, object], allowed: set[str], field_name: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unsupported keys: {', '.join(unknown)}")


def _positive_int(value: object, field_name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return result


def _resolve_path(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    candidates = ((config_path.parent / path).resolve(), (Path.cwd() / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_output_dir(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _float_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): float(raw) for key, raw in value.items()}


def _append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to constrained-RL YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and write provenance files without PPO optimization.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Loguru log level for the training entry point.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = _parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level=str(args.log_level).upper())
    config_path = Path(args.config).resolve()
    config = load_constrained_rl_config(config_path)
    outputs = train_constrained_rl(config, config_path=config_path, dry_run=bool(args.dry_run))
    logger.info("Constrained-RL outputs: {}", {key: str(value) for key, value in outputs.items()})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
