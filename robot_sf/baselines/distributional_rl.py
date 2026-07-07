"""Distributional RL planner adapter for QR-DQN smoke checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_sf.training.discrete_action_lattice import DiscreteUnicycleActionLattice
from robot_sf.training.distributional_rl import QuantileQNetwork
from robot_sf.training.risk_objectives import RISK_OBJECTIVES, score_action_quantiles


@dataclass
class DistributionalRLPlannerConfig:
    """Configuration for map-runner QR-DQN diagnostic inference."""

    checkpoint_path: str
    action_lattice_path: str | None = None
    device: str = "cpu"
    obs_mode: str = "dict"
    obs_transform: str = "ego"
    risk_objective: str = "cvar_lower"
    risk_alpha: float = 0.2
    risk_blend_beta: float = 1.0
    deterministic: bool = True
    fallback_to_goal: bool = False
    record_quantile_diagnostics: bool = False
    profile: str = "experimental"


class DistributionalRLPlanner:
    """Load a QR-DQN checkpoint and emit absolute unicycle commands."""

    def __init__(
        self,
        config: DistributionalRLPlannerConfig | dict[str, Any],
        *,
        seed: int | None = None,
    ) -> None:
        """Load a QR-DQN smoke checkpoint for diagnostic inference."""

        self.config = self._parse_config(config)
        self._seed = seed
        self._status = "initializing"
        self._fallback_reason: str | None = None
        self._last_decision: dict[str, Any] | None = None
        self._rng = np.random.default_rng(seed)
        self._model: QuantileQNetwork | None = None
        self._action_lattice: DiscreteUnicycleActionLattice | None = None
        self._load_checkpoint()

    @staticmethod
    def _parse_config(
        config: DistributionalRLPlannerConfig | dict[str, Any],
    ) -> DistributionalRLPlannerConfig:
        if isinstance(config, DistributionalRLPlannerConfig):
            parsed = config
        elif isinstance(config, dict):
            known = {
                field.name for field in DistributionalRLPlannerConfig.__dataclass_fields__.values()
            }
            unknown = set(config) - known
            if unknown:
                raise ValueError(
                    f"unknown distributional RL planner config keys: {sorted(unknown)}"
                )
            parsed = DistributionalRLPlannerConfig(**config)
        else:
            raise TypeError("config must be DistributionalRLPlannerConfig or dict")
        if parsed.risk_objective not in RISK_OBJECTIVES:
            raise ValueError(f"risk_objective must be one of {RISK_OBJECTIVES}")
        if not 0.0 < float(parsed.risk_alpha) <= 1.0:
            raise ValueError("risk_alpha must be in (0, 1]")
        if not 0.0 <= float(parsed.risk_blend_beta) <= 1.0:
            raise ValueError("risk_blend_beta must be in [0, 1]")
        return parsed

    def _load_checkpoint(self) -> None:
        path = Path(self.config.checkpoint_path)
        if not path.exists():
            self._status = "missing"
            self._fallback_reason = "checkpoint_missing"
            if self.config.fallback_to_goal:
                return
            raise FileNotFoundError(f"distributional RL checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=True)
        model_metadata = checkpoint["model_metadata"]
        lattice_payload = checkpoint["action_lattice"]
        if self.config.action_lattice_path:
            lattice = DiscreteUnicycleActionLattice.from_json_file(self.config.action_lattice_path)
        else:
            lattice = DiscreteUnicycleActionLattice.from_dict(lattice_payload)
        action_count = int(model_metadata["action_count"])
        if action_count != lattice.action_count:
            raise ValueError(
                "checkpoint action_count does not match action lattice: "
                f"{action_count} != {lattice.action_count}"
            )
        model = QuantileQNetwork(
            observation_dim=int(model_metadata["observation_dim"]),
            action_count=action_count,
            num_quantiles=int(model_metadata["num_quantiles"]),
            hidden_sizes=_infer_hidden_sizes(checkpoint["model_state_dict"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.config.device)
        model.eval()
        self._model = model
        self._action_lattice = lattice
        self._status = "ok"
        self._fallback_reason = None

    def reset(self, *, seed: int | None = None) -> None:
        """Reset planner state for a new episode."""

        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
        self._last_decision = None

    def close(self) -> None:
        """Release checkpoint state."""

        self._model = None

    def step(self, obs: dict[str, Any]) -> dict[str, float]:
        """Select one absolute unicycle command from the configured risk objective.

        Returns:
            Dictionary with ``v`` and ``omega`` absolute unicycle command values.
        """

        if self._model is None or self._action_lattice is None:
            if self.config.fallback_to_goal:
                self._status = "fallback"
                self._fallback_reason = self._fallback_reason or "model_unavailable"
                return {"v": 0.0, "omega": 0.0}
            raise RuntimeError("distributional RL model unavailable")
        observation = self._flatten_observation(obs, observation_dim=self._model.observation_dim)
        with torch.no_grad():
            tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=self.config.device
            ).view(1, -1)
            quantiles = self._model(tensor)
            scores = score_action_quantiles(
                quantiles,
                objective=self.config.risk_objective,
                alpha=self.config.risk_alpha,
                blend_beta=self.config.risk_blend_beta,
            )
            selected_index = int(scores.argmax(dim=-1).item())
            mean_scores = score_action_quantiles(
                quantiles,
                objective="mean",
                alpha=self.config.risk_alpha,
                blend_beta=self.config.risk_blend_beta,
            )
            cvar_scores = score_action_quantiles(
                quantiles,
                objective="cvar_lower",
                alpha=self.config.risk_alpha,
                blend_beta=self.config.risk_blend_beta,
            )
        command = self._action_lattice.command_at(selected_index)
        self._last_decision = {
            "selected_action_index": selected_index,
            "selected_command": [command.linear_velocity, command.angular_velocity],
            "risk_objective": self.config.risk_objective,
            "risk_alpha": self.config.risk_alpha,
            "selected_score": float(scores[0, selected_index].detach().cpu()),
            "selected_mean_return": float(mean_scores[0, selected_index].detach().cpu()),
            "selected_cvar_return": float(cvar_scores[0, selected_index].detach().cpu()),
            "candidate_count": self._action_lattice.action_count,
        }
        return {"v": float(command.linear_velocity), "omega": float(command.angular_velocity)}

    @staticmethod
    def _flatten_observation(obs: dict[str, Any], *, observation_dim: int) -> np.ndarray:
        values: list[np.ndarray] = []
        for key in sorted(obs):
            try:
                arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                continue
            if arr.size:
                values.append(arr)
        flat = (
            np.concatenate(values).astype(np.float32) if values else np.zeros(0, dtype=np.float32)
        )
        if flat.size < observation_dim:
            flat = np.pad(flat, (0, observation_dim - flat.size))
        return flat[:observation_dim]

    def get_metadata(self) -> dict[str, Any]:
        """Return diagnostic-only adapter metadata."""

        metadata = {
            "algorithm": "distributional_rl",
            "status": self._status,
            "policy_semantics": "qr_dqn_discrete_unicycle_lattice_risk_aware_selector",
            "evidence_tier": "diagnostic-only",
            "claim_boundary": (
                "distributional value/risk-selection diagnostic; not paper-grade safety evidence"
            ),
            "config": asdict(self.config),
            "fallback_or_degraded": self._status != "ok",
            "planner_kinematics": {
                "planner_command_space": "unicycle_vw",
                "planner_output_keys": ["v", "omega"],
                "native_env_action": False,
            },
        }
        if self._fallback_reason is not None:
            metadata["fallback_reason"] = self._fallback_reason
        if self._action_lattice is not None:
            metadata["action_lattice"] = self._action_lattice.to_dict()
        return metadata

    def diagnostics(self) -> dict[str, Any]:
        """Return latest per-step risk-selection diagnostics."""

        payload: dict[str, Any] = {}
        if self._last_decision is not None:
            payload["last_decision"] = dict(self._last_decision)
        return payload


def _infer_hidden_sizes(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    """Infer ``QuantileQNetwork`` hidden sizes from sequential linear weights.

    Returns:
        Hidden layer widths needed to reconstruct the saved MLP.
    """

    linear_layers: list[tuple[int, torch.Tensor]] = []
    for key, value in state_dict.items():
        if not key.startswith("net.") or not key.endswith(".weight"):
            continue
        layer_text = key.split(".")[1]
        if not layer_text.isdigit():
            continue
        linear_layers.append((int(layer_text), value))
    linear_layers.sort(key=lambda item: item[0])
    if not linear_layers:
        raise ValueError("checkpoint missing QuantileQNetwork linear weights")
    return tuple(int(weight.shape[0]) for _, weight in linear_layers[:-1])
