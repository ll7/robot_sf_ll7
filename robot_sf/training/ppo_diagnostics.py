"""Training-time diagnostics for PPO-based feature extractor studies."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch as th
from stable_baselines3 import PPO


def _grad_l2_norm(parameters: Iterable[th.nn.Parameter]) -> float:
    """Compute the L2 norm across all currently populated gradients."""
    squared_norm = 0.0
    has_grad = False
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        has_grad = True
        grad_norm = float(gradient.detach().norm(2).item())
        squared_norm += grad_norm * grad_norm
    if not has_grad:
        return 0.0
    return math.sqrt(squared_norm)


def _module_grad_norm(module: Any) -> float:
    """Compute the gradient norm for a module when present."""
    if module is None:
        return 0.0
    return _grad_l2_norm(module.parameters())


def _summarize_samples(samples: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-mini-batch diagnostics into one PPO-update summary."""
    if not samples:
        return {}

    summary: dict[str, float] = {"mini_batch_count": float(len(samples))}
    metric_names = sorted({name for sample in samples for name in sample})

    for metric_name in metric_names:
        values = [sample[metric_name] for sample in samples if metric_name in sample]
        if not values:
            continue
        summary[f"{metric_name}_mean"] = float(sum(values) / len(values))
        summary[f"{metric_name}_max"] = float(max(values))

    return summary


class DiagnosticPPO(PPO):
    """PPO variant that records gradient and feature statistics per update."""

    def __init__(
        self,
        *args,
        diagnostics_path: str | Path | None = None,
        diagnostics_start_timestep: int = 0,
        **kwargs,
    ) -> None:
        self._diagnostics_path = Path(diagnostics_path) if diagnostics_path is not None else None
        self._diagnostics_start_timestep = int(diagnostics_start_timestep)
        self.last_training_diagnostics: dict[str, float] = {}
        super().__init__(*args, **kwargs)

    def _collect_batch_diagnostics(self, parameters: Iterable[th.nn.Parameter]) -> dict[str, float]:
        policy = self.policy
        mlp_extractor = getattr(policy, "mlp_extractor", None)
        action_net = getattr(policy, "action_net", None)
        value_net = getattr(policy, "value_net", None)

        diagnostics = {
            "grad_norm_total": _grad_l2_norm(parameters),
            "grad_norm_features_extractor": _module_grad_norm(
                getattr(policy, "features_extractor", None)
            ),
            "grad_norm_policy_net": _module_grad_norm(getattr(mlp_extractor, "policy_net", None)),
            "grad_norm_value_net": _module_grad_norm(getattr(mlp_extractor, "value_net", None)),
            "grad_norm_action_head": _module_grad_norm(action_net),
            "grad_norm_value_head": _module_grad_norm(value_net),
        }

        feature_extractor = getattr(policy, "features_extractor", None)
        feature_stats_getter = getattr(feature_extractor, "latest_feature_stats", None)
        if callable(feature_stats_getter):
            for name, value in feature_stats_getter().items():
                diagnostics[f"feature_{name}"] = float(value)

        return diagnostics

    def _append_training_diagnostics(self, payload: dict[str, float]) -> None:
        """Append one training-update diagnostics record as JSONL."""
        if self._diagnostics_path is None:
            return
        self._diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        with self._diagnostics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def train(self) -> None:
        """Run one PPO update while capturing gradient and feature statistics."""
        collected_samples: list[dict[str, float]] = []
        original_clip_grad_norm = th.nn.utils.clip_grad_norm_

        def _patched_clip_grad_norm_(
            parameters: Iterable[th.nn.Parameter],
            max_norm: float,
            *args: Any,
            **kwargs: Any,
        ) -> th.Tensor:
            parameter_list = list(parameters)
            collected_samples.append(self._collect_batch_diagnostics(parameter_list))
            return original_clip_grad_norm(parameter_list, max_norm, *args, **kwargs)

        th.nn.utils.clip_grad_norm_ = _patched_clip_grad_norm_
        try:
            super().train()
        finally:
            th.nn.utils.clip_grad_norm_ = original_clip_grad_norm

        summary = _summarize_samples(collected_samples)
        summary["num_timesteps"] = float(self.num_timesteps)
        summary["n_updates"] = float(self._n_updates)
        self.last_training_diagnostics = summary

        if collected_samples and self.num_timesteps >= self._diagnostics_start_timestep:
            self._append_training_diagnostics(summary)
