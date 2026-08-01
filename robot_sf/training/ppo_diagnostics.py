"""Training-time diagnostics for PPO-based feature extractor studies."""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch as th


def _init_classes() -> dict[str, Any]:  # noqa: C901

    import torch as th  # noqa: PLC0415
    from stable_baselines3 import PPO  # noqa: PLC0415

    def _grad_l2_norm(parameters: Iterable[th.nn.Parameter]) -> float:
        squared_sums = [
            th.sum(parameter.grad.detach() ** 2)
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not squared_sums:
            return 0.0
        return math.sqrt(float(th.sum(th.stack(squared_sums)).item()))

    def _module_grad_norm(module: Any) -> float:
        if module is None:
            return 0.0
        return _grad_l2_norm(module.parameters())

    def _summarize_samples(samples: list[dict[str, float]]) -> dict[str, float]:
        if not samples:
            return {}

        summary: dict[str, float] = {"mini_batch_count": float(len(samples))}
        metric_names = sorted({name for sample in samples for name in sample})

        for metric_name in metric_names:
            values = [sample[metric_name] for sample in samples if metric_name in sample]
            summary[f"{metric_name}_mean"] = float(sum(values) / len(values))
            summary[f"{metric_name}_max"] = float(max(values))

        return summary

    @contextmanager
    def _patched_clip_grad_norm(
        callback: Any,
    ):
        original_clip_grad_norm = th.nn.utils.clip_grad_norm_

        def _patched(
            parameters: Iterable[th.nn.Parameter],
            max_norm: float,
            *args: Any,
            **kwargs: Any,
        ) -> th.Tensor:
            parameter_list = list(parameters)
            callback(parameter_list)
            return original_clip_grad_norm(parameter_list, max_norm, *args, **kwargs)

        th.nn.utils.clip_grad_norm_ = _patched
        try:
            yield
        finally:
            th.nn.utils.clip_grad_norm_ = original_clip_grad_norm

    class DiagnosticPPO(PPO):
        """PPO variant that records gradient and feature statistics per update."""

        def __init__(
            self,
            *args,
            diagnostics_path: str | Path | None = None,
            diagnostics_start_timestep: int = 0,
            **kwargs,
        ) -> None:
            self._diagnostics_path = (
                Path(diagnostics_path) if diagnostics_path is not None else None
            )
            self._diagnostics_start_timestep = int(diagnostics_start_timestep)
            self.last_training_diagnostics: dict[str, float] = {}
            super().__init__(*args, **kwargs)

        def _collect_batch_diagnostics(
            self, parameters: Iterable[th.nn.Parameter]
        ) -> dict[str, float]:
            policy = self.policy
            mlp_extractor = getattr(policy, "mlp_extractor", None)
            action_net = getattr(policy, "action_net", None)
            value_net = getattr(policy, "value_net", None)

            diagnostics = {
                "grad_norm_total": _grad_l2_norm(parameters),
                "grad_norm_features_extractor": _module_grad_norm(
                    getattr(policy, "features_extractor", None)
                ),
                "grad_norm_policy_net": _module_grad_norm(
                    getattr(mlp_extractor, "policy_net", None)
                ),
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
            if self._diagnostics_path is None:
                return
            self._diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            with self._diagnostics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

        def train(self) -> None:
            collected_samples: list[dict[str, float]] = []
            with _patched_clip_grad_norm(
                lambda parameter_list: collected_samples.append(
                    self._collect_batch_diagnostics(parameter_list)
                )
            ):
                super().train()

            summary = _summarize_samples(collected_samples)
            summary["num_timesteps"] = float(self.num_timesteps)
            summary["n_updates"] = float(self._n_updates)
            self.last_training_diagnostics = summary

            if collected_samples and self.num_timesteps >= self._diagnostics_start_timestep:
                self._append_training_diagnostics(summary)

    return {
        "DiagnosticPPO": DiagnosticPPO,
        "_grad_l2_norm": _grad_l2_norm,
        "_module_grad_norm": _module_grad_norm,
        "_patched_clip_grad_norm": _patched_clip_grad_norm,
        "_summarize_samples": _summarize_samples,
    }


_cache: dict[str, Any] | None = None
_LAZY_NAMES = {
    "DiagnosticPPO",
    "_grad_l2_norm",
    "_module_grad_norm",
    "_patched_clip_grad_norm",
    "_summarize_samples",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_NAMES:
        global _cache
        if _cache is None:
            _cache = _init_classes()
            for lazy_name, value in _cache.items():
                value.__qualname__ = lazy_name
                globals()[lazy_name] = value
        return _cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
