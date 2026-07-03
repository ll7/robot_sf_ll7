"""Config-driven Optuna search-space helpers for expert PPO sweeps."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import optuna

_SUPPORTED_SPEC_TYPES = {"float", "int", "categorical"}
_SUPPORTED_SECTIONS = {"ppo_hyperparams", "policy_net_arch"}


@dataclass(frozen=True, slots=True)
class SearchSpaceUpdate:
    """Resolved trial updates ready to apply to an expert PPO config."""

    ppo_hyperparams: dict[str, object]
    policy_net_arch: tuple[int, ...] | None = None


def validate_search_space(search_space: Mapping[str, object]) -> None:
    """Validate the allowlisted Optuna search-space schema."""

    if not isinstance(search_space, Mapping):
        raise ValueError("search_space must be a mapping.")
    unknown_sections = sorted(set(search_space) - _SUPPORTED_SECTIONS)
    if unknown_sections:
        raise ValueError(f"Unsupported search_space section(s): {', '.join(unknown_sections)}")
    ppo_hyperparams = search_space.get("ppo_hyperparams", {})
    if ppo_hyperparams is not None and not isinstance(ppo_hyperparams, Mapping):
        raise ValueError("search_space.ppo_hyperparams must be a mapping.")
    for name, spec in dict(ppo_hyperparams or {}).items():
        _validate_named_spec(f"ppo_hyperparams.{name}", spec)
    if "policy_net_arch" in search_space:
        _validate_named_spec("policy_net_arch", search_space["policy_net_arch"])


def suggest_search_space(
    trial: optuna.Trial,
    search_space: Mapping[str, object],
    *,
    max_batch_size: int | None = None,
) -> SearchSpaceUpdate:
    """Sample an allowlisted search space for one Optuna trial.

    Returns:
        Resolved config updates for the sampled trial.
    """

    validate_search_space(search_space)
    ppo_updates: dict[str, object] = {}
    ppo_hyperparams = search_space.get("ppo_hyperparams", {})
    for name, spec in dict(ppo_hyperparams or {}).items():
        value = _suggest_value(trial, name, spec)
        if name == "batch_size" and max_batch_size is not None:
            value = min(int(value), int(max_batch_size))
        ppo_updates[str(name)] = value

    policy_net_arch = None
    if "policy_net_arch" in search_space:
        policy_net_arch = _suggest_policy_net_arch(trial, search_space["policy_net_arch"])

    return SearchSpaceUpdate(ppo_hyperparams=ppo_updates, policy_net_arch=policy_net_arch)


def apply_search_space_update(config: Any, update: SearchSpaceUpdate) -> Any:
    """Apply a resolved search-space update to an expert PPO config object.

    Returns:
        The mutated config object.
    """

    merged_hyperparams = dict(getattr(config, "ppo_hyperparams", {}) or {})
    merged_hyperparams.update(update.ppo_hyperparams)
    config.ppo_hyperparams = merged_hyperparams
    if update.policy_net_arch is not None:
        config.policy_net_arch = update.policy_net_arch
    return config


def _validate_named_spec(name: str, spec: object) -> None:
    if not isinstance(spec, Mapping):
        raise ValueError(f"search_space.{name} must be a mapping.")
    spec_type = spec.get("type")
    if spec_type not in _SUPPORTED_SPEC_TYPES:
        raise ValueError(
            f"search_space.{name}.type must be one of {sorted(_SUPPORTED_SPEC_TYPES)}, "
            f"got {spec_type!r}."
        )
    if spec_type in {"float", "int"}:
        if "low" not in spec or "high" not in spec:
            raise ValueError(f"search_space.{name} requires low and high.")
        low = spec["low"]
        high = spec["high"]
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise ValueError(f"search_space.{name} low/high must be numeric.")
        if float(low) > float(high):
            raise ValueError(f"search_space.{name} low must be <= high.")
    if spec_type == "categorical":
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"search_space.{name}.values must be a non-empty list.")


def _suggest_value(trial: optuna.Trial, name: str, spec: object) -> object:
    if not isinstance(spec, Mapping):
        raise ValueError(f"search_space.{name} must be a mapping.")
    spec_type = str(spec["type"])
    if spec_type == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    if spec_type == "int":
        step = int(spec.get("step", 1))
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=step,
            log=bool(spec.get("log", False)),
        )
    if spec_type == "categorical":
        return trial.suggest_categorical(name, list(spec["values"]))
    raise ValueError(f"Unsupported search_space.{name}.type: {spec_type!r}")


def _suggest_policy_net_arch(trial: optuna.Trial, spec: object) -> tuple[int, ...]:
    if not isinstance(spec, Mapping):
        raise ValueError("search_space.policy_net_arch must be a mapping.")
    if spec.get("type") != "categorical":
        raise ValueError("search_space.policy_net_arch only supports categorical specs.")
    values = spec.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError("search_space.policy_net_arch.values must be a non-empty list.")
    encoded_choices = []
    decoded: dict[str, tuple[int, ...]] = {}
    for value in values:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("policy_net_arch categorical values must be non-empty int lists.")
        arch = tuple(int(dim) for dim in value)
        encoded = ",".join(str(dim) for dim in arch)
        encoded_choices.append(encoded)
        decoded[encoded] = arch
    selected = trial.suggest_categorical("policy_net_arch", encoded_choices)
    return decoded[str(selected)]
