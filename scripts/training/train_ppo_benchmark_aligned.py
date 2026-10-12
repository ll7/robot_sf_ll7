"""Guarded launcher for #9628's new PPO family; preflight never simulates or trains.

Training seed sampling is owned by PR #10324. This entrypoint consumes its config
interface when installed and refuses execution while that interface is absent.
Reporting seeds are metadata only; all trainer evaluations use selection seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios
from scripts.training import train_ppo

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "configs/training/ppo/benchmark_aligned_v1.yaml"
BENCHMARK = ROOT / "configs/baselines/ppo_issue_791_eval_aligned_large_capacity_cpu.yaml"


def _integers(values: object, role: str) -> tuple[int, ...]:
    """Read identities strictly without echoing rejected values into logs."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{role} seeds must be a nonempty integer sequence")
    if any(type(value) is not int for value in values):
        raise ValueError(f"{role} seeds must contain integers")
    seeds = tuple(values)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{role} seeds must be unique")
    return seeds


def validate_seed_partitions(
    training_range: object, selection: object, reporting: object
) -> tuple[tuple[int, int], tuple[int, ...], tuple[int, ...]]:
    """Reject overlap and forbidden identities using narrow positive allowlists.

    Only the training-only interval and author-declared dev pools are admitted.
    Consequently retired and sealed identities cannot enter any role; neither
    their constants nor rejected values are exposed in diagnostics.

    Returns:
        Inclusive training range, selection identities, and reporting identities.
    """
    bounds = _integers(training_range, "training range")
    if len(bounds) != 2 or bounds[0] > bounds[1]:
        raise ValueError("training range must be an ordered inclusive pair")
    selected = _integers(selection, "selection")
    reported = _integers(reporting, "reporting")
    if set(selected).intersection(reported):
        raise ValueError("selection and reporting seeds overlap")
    if any(bounds[0] <= seed <= bounds[1] for seed in (*selected, *reported)):
        raise ValueError("training range overlaps selection or reporting seeds")
    if bounds != (100000, 199999):
        raise ValueError("training range must be the approved training-only interval")
    if set(selected) != set(range(1031, 1061)):
        raise ValueError("selection seeds must be exactly the approved development pool")
    if set(reported) != set(range(1101, 1131)):
        raise ValueError("reporting seeds must be exactly the approved development pool")
    return (bounds[0], bounds[1]), selected, reported


def _relative(source: Path, value: object) -> Path:
    """Resolve a path with the same leaf-relative semantics as the PPO loader."""
    return (source.parent / str(value)).resolve()


def _selection_seeds(source: Path, data: Mapping[str, Any]) -> tuple[int, ...]:
    """Read the actual manifest consumed by periodic and final trainer evaluation."""
    evaluation = data["evaluation"]
    if evaluation.get("evaluation_seeds"):
        raise ValueError("bind selection only through the family manifest")
    path = _relative(source, evaluation["evaluation_seed_manifest"])
    manifest = yaml.safe_load(path.read_text())
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != (
        "robot-sf-evaluation-seed-manifest.v1"
    ):
        raise ValueError("selection manifest has an invalid schema")
    return _integers(manifest.get("evaluation_seeds"), "selection")


def seed_policy_path(config_path: Path) -> Path:
    """Locate the marked sampler sidecar beside its stock-loader-compatible leaf.

    Returns:
        Family-specific sidecar path.
    """
    return config_path.with_name(f"{config_path.stem}_seed_policy.yaml")


def load_seed_hook(config_path: Path) -> dict[str, Any]:
    """Load only the externally owned sampler's declared interface fields.

    Returns:
        The three #10324 fields, separate from stock trainer configuration.
    """
    path = seed_policy_path(config_path)
    if not path.is_file():
        raise ValueError("missing #10324 training-seed hook sidecar")
    raw = yaml.safe_load(path.read_text())
    fields = {"training_seed_range", "training_master_seed", "development_evaluation_seeds"}
    if not isinstance(raw, Mapping) or set(raw) != fields | {"schema_version"}:
        raise ValueError("training-seed hook must contain exactly its declared interface fields")
    if raw["schema_version"] != "robot-sf-training-seed-hook.v1":
        raise ValueError("training-seed hook has an invalid schema")
    return {key: raw[key] for key in fields}


def validate_family(config_path: Path = FAMILY) -> dict[str, Any]:
    """Validate resolved training/selection conditions before any environment exists.

    Returns:
        Fully inherited launch configuration bound to the marked #10324 hook.
    """
    source = config_path.resolve()
    stock = train_ppo._load_expert_training_config_mapping(source)
    if (
        stock.get("seeds") != [100000]
        or stock.get("randomize_seeds") is not False
        or stock["scenario_sampling"].get("episode_seed_pool") != [100000]
    ):
        raise ValueError("stock leaf must retain the declared loader-only training seed template")
    # Keep the stock loader authoritative for every normal config key. Sampler
    # fields exist only in the marked sidecar until its interface is integrated.
    train_ppo.load_expert_training_config(source)
    data = {
        **stock,
        **load_seed_hook(source),
        "seeds": [],
        "randomize_seeds": True,
        "scenario_sampling": {**stock["scenario_sampling"], "episode_seed_pool": None},
    }
    _, selected, _ = validate_seed_partitions(
        data.get("training_seed_range"),
        _selection_seeds(source, data),
        data.get("development_evaluation_seeds"),
    )
    _validate_seed_mode(data)
    _validate_schedule(data)
    _validate_plant(data)
    _validate_scenarios(source, data, selected)
    return data


def _validate_seed_mode(data: Mapping[str, Any]) -> None:
    """Ensure the family can only request the externally owned range sampler."""
    if type(data.get("training_master_seed")) is not int or data["training_master_seed"] < 0:
        raise ValueError("training master seed must be a nonnegative integer")
    if data.get("seeds") or data["scenario_sampling"].get("episode_seed_pool"):
        raise ValueError("range-seeded family must not declare a fixed training seed pool")
    if data.get("randomize_seeds") is not True:
        raise ValueError("training must consume the master-seeded range interface")


def _validate_schedule(data: Mapping[str, Any]) -> None:
    """Forbid alternate samplers, auxiliary evaluations and early stopping."""
    evaluation = data["evaluation"]
    if evaluation.get("randomize_seeds") is not False:
        raise ValueError("selection evaluation must use fixed seeds")
    if evaluation.get("hold_out_scenarios") or data.get("scenario_id"):
        raise ValueError("selection must cover the full 48-scenario campaign")
    if evaluation.get("full_policy_analysis_on_new_best") or evaluation.get(
        "full_policy_analysis_videos"
    ):
        raise ValueError("family forbids auxiliary evaluation on new best checkpoints")
    if data.get("resume_from") or data.get("resume_model_id"):
        raise ValueError("range-seeded resume requires PR #10324 stream-state support")
    if any(
        data.get(key)
        for key in ("multi_map_protocol", "domain_randomization", "density_curriculum")
    ):
        raise ValueError("family forbids additional condition-changing protocols")
    if data.get("best_checkpoint_metric") != "success_rate":
        raise ValueError("best-checkpoint metric must remain selection success_rate")
    if data["convergence"]["success_rate"] <= 1.0:
        raise ValueError("family must disable convergence-based early stopping")


def _validate_plant(data: Mapping[str, Any]) -> None:
    """Bind policy limits and reverse semantics to the benchmark plant."""
    benchmark = yaml.safe_load(BENCHMARK.read_text())
    overrides = data["env_overrides"]
    robot = overrides["robot_config"]
    if (
        robot.get("max_linear_speed") != benchmark["v_max"]
        or robot.get("max_angular_speed") != benchmark["omega_max"]
    ):
        raise ValueError("training speed limits differ from the benchmark adapter")
    default_drive = DifferentialDriveSettings()
    if robot.get("allow_backwards") != default_drive.allow_backwards or robot.get(
        "limited_reverse", False
    ):
        raise ValueError("training reverse setting differs from the release plant")
    if overrides.get("ppo_action_semantics") != "velocity_delta":
        raise ValueError("training must use the benchmark velocity-delta interface")


def _validate_scenarios(source: Path, data: Mapping[str, Any], selected: tuple[int, ...]) -> None:
    """Check effective goal, horizon and acceleration in all training/selection cells."""
    evaluation = data["evaluation"]
    overrides = data["env_overrides"]
    default_drive = DifferentialDriveSettings()
    for role, raw_path in (
        ("training", data["scenario_config"]),
        ("selection", evaluation["scenario_config"]),
    ):
        path = _relative(source, raw_path)
        scenarios = load_scenarios(path)
        if len(scenarios) != 48 or len({str(row["name"]) for row in scenarios}) != 48:
            raise ValueError(f"{role} scenarios must cover 48 unique campaign cells")
        if evaluation["evaluation_episodes"] != len(scenarios) * len(selected):
            raise ValueError("selection episodes must cover every scenario-seed pair")
        for scenario in scenarios:
            config = build_robot_config_from_scenario(scenario, scenario_path=path)
            train_ppo._apply_env_overrides(config, overrides)
            if config.sim_config.goal_completion_policy != "goal_zone_entry_v1":
                raise ValueError(f"{role} goal completion differs from the release campaign")
            if (
                config.sim_config.time_per_step_in_secs != 0.1
                or config.sim_config.sim_time_in_secs != (60.0)
            ):
                raise ValueError(f"{role} episode horizon must be 600 steps at 0.1 seconds")
            for key in ("max_linear_accel", "max_linear_decel", "max_angular_accel"):
                if getattr(config.robot_config, key) != getattr(default_drive, key):
                    raise ValueError(f"{role} acceleration limits differ from the release plant")


def sampler_interface_available() -> bool:
    """Check for #10324's consumed dataclass interface without implementing its sampler."""
    return {"training_seed_range", "training_master_seed", "development_evaluation_seeds"}.issubset(
        train_ppo.ExpertTrainingConfig.__dataclass_fields__
    )


def load_launch_config(config_path: Path = FAMILY):
    """Fail closed until the actual PPO loader can bind #10324's sampling interface.

    Returns:
        ExpertTrainingConfig with the supported #10324 dataclass fields bound.
    """
    data = validate_family(config_path)
    if not sampler_interface_available():
        raise RuntimeError("Training disabled: integrate PR #10324's training-seed interface first")
    stock = train_ppo.load_expert_training_config(config_path)
    # Bind #10324's supported dataclass interface without adding unsupported
    # keys to the canonical leaf or implementing any episode sampling here.
    config = replace(
        stock,
        training_seed_range=tuple(data["training_seed_range"]),
        training_master_seed=data["training_master_seed"],
        development_evaluation_seeds=tuple(data["development_evaluation_seeds"]),
        seeds=(),
        randomize_seeds=True,
        scenario_sampling=dict(data["scenario_sampling"]),
    )
    bounds, selected, reported = validate_seed_partitions(
        getattr(config, "training_seed_range", None),
        config.evaluation.evaluation_seeds,
        getattr(config, "development_evaluation_seeds", ()),
    )
    if (
        getattr(config, "training_seed_range", None) != bounds
        or getattr(config, "training_master_seed", None) != data["training_master_seed"]
    ):
        raise ValueError("trainer did not bind the declared training seed policy")
    if (
        config.evaluation.evaluation_seeds != selected
        or getattr(config, "development_evaluation_seeds", ()) != reported
    ):
        raise ValueError("trainer did not bind the declared held-out seed pools")
    return config


def main(argv: Sequence[str] | None = None) -> int:
    """Preflight by default only when requested; never silently downgrade a launch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=FAMILY)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        validate_family(args.config)
        print(
            json.dumps(
                {
                    "conditions": "valid",
                    "sampler_available": sampler_interface_available(),
                    "training_started": False,
                }
            )
        )
        return 0
    config = load_launch_config(args.config)
    train_ppo.run_expert_training(
        config,
        config_path=args.config.resolve(),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
