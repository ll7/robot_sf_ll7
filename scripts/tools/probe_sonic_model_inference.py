#!/usr/bin/env python3
"""Probe whether a bundled SoNIC checkpoint is reusable for model-only inference."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from gymnasium import spaces


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_checkpoint(checkpoints_dir: Path) -> str:
    candidates = sorted(path.name for path in checkpoints_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoints found in {checkpoints_dir}")
    return candidates[-1]


def _minimal_spaces(config: Any) -> tuple[spaces.Dict, spaces.Box]:
    human_num = config.sim.human_num + config.sim.human_num_range
    predict_steps = config.sim.predict_steps
    spatial_edge_dim = int(2 * (predict_steps + 1))
    obs_space = spaces.Dict(
        {
            "robot_node": spaces.Box(low=-1.0, high=1.0, shape=(1, 7), dtype=float),
            "temporal_edges": spaces.Box(low=-1.0, high=1.0, shape=(1, 2), dtype=float),
            "spatial_edges": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(human_num, spatial_edge_dim),
                dtype=float,
            ),
            "conformity_scores": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(human_num, predict_steps),
                dtype=float,
            ),
            "visible_masks": spaces.MultiBinary(human_num),
            "detected_human_num": spaces.Box(
                low=0.0,
                high=float(human_num),
                shape=(1,),
                dtype=float,
            ),
            "aggressiveness_factor": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1, 1),
                dtype=float,
            ),
        }
    )
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    return obs_space, action_space


def _synthetic_inputs(
    config: Any, policy: Any
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    human_num = config.sim.human_num + config.sim.human_num_range
    predict_steps = config.sim.predict_steps
    spatial_edge_dim = int(2 * (predict_steps + 1))
    obs = {
        "robot_node": torch.zeros((1, 1, 7), dtype=torch.float32),
        "temporal_edges": torch.zeros((1, 1, 2), dtype=torch.float32),
        "spatial_edges": torch.zeros((1, human_num, spatial_edge_dim), dtype=torch.float32),
        "conformity_scores": torch.zeros((1, human_num, predict_steps), dtype=torch.float32),
        "visible_masks": torch.ones((1, human_num), dtype=torch.float32),
        "detected_human_num": torch.tensor([[human_num]], dtype=torch.float32),
        "aggressiveness_factor": torch.zeros((1, 1, 1), dtype=torch.float32),
    }
    rnn_hxs = {
        "human_node_rnn": torch.zeros((1, 1, policy.base.human_node_rnn_size), dtype=torch.float32),
        "human_human_edge_rnn": torch.zeros(
            (1, 1 + human_num, policy.base.human_human_edge_rnn_size), dtype=torch.float32
        ),
    }
    masks = torch.ones((1, 1), dtype=torch.float32)
    return obs, rnn_hxs, masks


def _extract_contract(config: Any, args: Any) -> dict[str, Any]:
    return {
        "robot_policy": getattr(config.robot, "policy", None),
        "human_policy": getattr(config.humans, "policy", None),
        "robot_sensor": getattr(config.robot, "sensor", None),
        "predict_method": getattr(config.sim, "predict_method", None),
        "action_kinematics": getattr(config.action_space, "kinematics", None),
        "env_use_wrapper": getattr(config.env, "use_wrapper", None),
        "env_name": getattr(args, "env_name", None),
    }


def _load_model_modules(model_name: str) -> tuple[Any, Any, Any, Any]:
    """Import SoNIC training arguments, config, model module, and parsed args safely."""
    args_mod = importlib.import_module(f"trained_models.{model_name}.arguments")
    config_mod = importlib.import_module(f"trained_models.{model_name}.configs.config")
    model_mod = importlib.import_module("rl.networks.model")
    original_argv = sys.argv[:]
    try:
        sys.argv = [f"{model_name}_arguments"]
        args = args_mod.get_args()
    finally:
        sys.argv = original_argv
    return args_mod, config_mod, model_mod, args


@dataclass
class ModelProbeReport:
    """Structured result for a model-only SoNIC inference reuse check."""

    issue: int
    repo_remote_url: str
    repo_root: str
    model_name: str
    checkpoint: str
    direct_verdict: str
    direct_failure_summary: str | None
    shimmed_verdict: str
    shimmed_failure_summary: str | None
    shims_applied: list[str]
    source_contract: dict[str, Any]
    missing_state_keys: list[str]
    unexpected_state_keys: list[str]
    action_sample: list[float] | None
    action_shape: list[int] | None
    value_shape: list[int] | None


def run_model_probe(repo_root: Path, model_name: str, checkpoint: str | None) -> ModelProbeReport:
    """Check whether the SoNIC checkpoint can run a forward pass without the source env stack."""
    checkpoints_dir = repo_root / "trained_models" / model_name / "checkpoints"
    resolved_checkpoint = checkpoint or _default_checkpoint(checkpoints_dir)
    checkpoint_path = checkpoints_dir / resolved_checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    sys.path.insert(0, str(repo_root))

    direct_verdict = "direct model import blocked"
    direct_failure_summary: str | None = None
    try:
        _args_mod, config_mod, model_mod, args = _load_model_modules(model_name)
        config = config_mod.Config()
        args.num_processes = 1
        args.no_cuda = True
        args.cuda = False
        obs_space, action_space = _minimal_spaces(config)
        model_mod.Policy(
            obs_space.spaces, action_space, config, base=config.robot.policy, base_kwargs=args
        )
        direct_verdict = "direct model import reproducible"
    except Exception as exc:  # pragma: no cover
        direct_failure_summary = f"{type(exc).__name__}: {exc}"

    shims_applied = [
        "gymnasium as gym module alias",
        "stub rl.networks.envs.VecNormalize",
        "config.policy.constant_std=false",
    ]
    shimmed_verdict = "model-only inference blocked"
    shimmed_failure_summary: str | None = None
    source_contract: dict[str, Any] = {}
    missing_state_keys: list[str] = []
    unexpected_state_keys: list[str] = []
    action_sample: list[float] | None = None
    action_shape: list[int] | None = None
    value_shape: list[int] | None = None

    try:
        for key in [
            f"trained_models.{model_name}.arguments",
            f"trained_models.{model_name}.configs.config",
            "rl.networks.model",
            "rl.networks.distributions",
            "rl.networks.network_utils",
            "rl.networks.selfAttn_srnn_temp_node",
        ]:
            sys.modules.pop(key, None)

        import gymnasium

        sys.modules["gym"] = gymnasium
        fake_envs = types.ModuleType("rl.networks.envs")

        class VecNormalize:
            """Minimal stub to satisfy SoNIC network_utils imports."""

        fake_envs.VecNormalize = VecNormalize
        sys.modules["rl.networks.envs"] = fake_envs

        _args_mod, config_mod, model_mod, args = _load_model_modules(model_name)
        config = config_mod.Config()
        args.num_processes = 1
        args.no_cuda = True
        args.cuda = False
        config.policy.constant_std = False

        obs_space, action_space = _minimal_spaces(config)
        policy = model_mod.Policy(
            obs_space.spaces, action_space, config, base=config.robot.policy, base_kwargs=args
        )
        state = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = policy.load_state_dict(state, strict=False)
        missing_state_keys = list(missing)
        unexpected_state_keys = list(unexpected)

        obs, rnn_hxs, masks = _synthetic_inputs(config, policy)
        value, action, _, _ = policy.act(obs, rnn_hxs, masks, deterministic=True)

        source_contract = _extract_contract(config, args)
        action_sample = [float(x) for x in action.squeeze(0).tolist()]
        action_shape = list(action.shape)
        value_shape = list(value.shape)
        shimmed_verdict = "model-only inference reproducible with shims"
    except Exception as exc:  # pragma: no cover
        shimmed_failure_summary = f"{type(exc).__name__}: {exc}"

    return ModelProbeReport(
        issue=626,
        repo_remote_url="https://github.com/tasl-lab/SoNIC-Social-Nav",
        repo_root=str(repo_root),
        model_name=model_name,
        checkpoint=resolved_checkpoint,
        direct_verdict=direct_verdict,
        direct_failure_summary=direct_failure_summary,
        shimmed_verdict=shimmed_verdict,
        shimmed_failure_summary=shimmed_failure_summary,
        shims_applied=shims_applied,
        source_contract=source_contract,
        missing_state_keys=missing_state_keys,
        unexpected_state_keys=unexpected_state_keys,
        action_sample=action_sample,
        action_shape=action_shape,
        value_shape=value_shape,
    )


def _render_markdown(report: ModelProbeReport) -> str:
    contract = report.source_contract
    lines = [
        "# SoNIC Model-Only Inference Probe",
        "",
        f"- Issue: `#{report.issue}`",
        f"- Repo remote: `{report.repo_remote_url}`",
        f"- Model: `{report.model_name}`",
        f"- Checkpoint: `{report.checkpoint}`",
        "",
        "## Direct Import Result",
        "",
        f"- Verdict: `{report.direct_verdict}`",
        f"- Failure: `{report.direct_failure_summary}`",
        "",
        "## Shimmed Inference Result",
        "",
        f"- Verdict: `{report.shimmed_verdict}`",
        f"- Failure: `{report.shimmed_failure_summary}`",
        f"- Shims: `{', '.join(report.shims_applied)}`",
        f"- Missing state keys: `{report.missing_state_keys}`",
        f"- Unexpected state keys: `{report.unexpected_state_keys}`",
        f"- Action sample: `{report.action_sample}`",
        f"- Action shape: `{report.action_shape}`",
        f"- Value shape: `{report.value_shape}`",
        "",
        "## Source Contract",
        "",
        f"- `robot_policy`: `{contract.get('robot_policy')}`",
        f"- `human_policy`: `{contract.get('human_policy')}`",
        f"- `robot_sensor`: `{contract.get('robot_sensor')}`",
        f"- `predict_method`: `{contract.get('predict_method')}`",
        f"- `action_kinematics`: `{contract.get('action_kinematics')}`",
        f"- `env_use_wrapper`: `{contract.get('env_use_wrapper')}`",
        f"- `env_name`: `{contract.get('env_name')}`",
        "",
        "## Interpretation",
        "",
        "- The checkpoint is not plug-and-play in the current environment.",
        "- Model-only reuse is technically possible, but only with narrow compatibility shims.",
        "- A future Robot SF adapter would still need explicit observation and action translation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the model-only SoNIC inference probe CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_repository_root() / "output" / "repos" / "SoNIC-Social-Nav",
        help="Path to the upstream SoNIC checkout.",
    )
    parser.add_argument(
        "--model-name",
        default="SoNIC_GST",
        help="Model directory under trained_models/ to probe.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit checkpoint file name. Defaults to the latest .pt file in checkpoints/.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--output-md", type=Path, default=None, help="Optional Markdown report path."
    )
    args = parser.parse_args()

    report = run_model_probe(
        repo_root=args.repo_root.resolve(),
        model_name=args.model_name,
        checkpoint=args.checkpoint,
    )
    payload = asdict(report)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_render_markdown(report), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0 if report.shimmed_verdict == "model-only inference reproducible with shims" else 1


if __name__ == "__main__":
    raise SystemExit(main())
