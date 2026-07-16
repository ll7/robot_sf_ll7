"""Helpers for managing trained policy artifacts."""

from robot_sf.common.seed import _TORCH_213_RUNTIME_GUARD_APPLIED
from robot_sf.models.registry import (
    RegistryIssue,
    find_latest_wandb_model,
    get_registry_entry,
    load_registry,
    resolve_latest_wandb_model,
    resolve_model_path,
    sha256_of_file,
    upsert_registry_entry,
    validate_registry_entry_benchmark_promotion,
)

# Importing ``robot_sf.common.seed`` applies the process-level guard before a
# direct ``stable_baselines3.PPO`` import. Consume the result here so this
# package boundary documents and retains the ordering without running the
# version/optional-module probe a second time.
_ = _TORCH_213_RUNTIME_GUARD_APPLIED

__all__ = [
    "RegistryIssue",
    "find_latest_wandb_model",
    "get_registry_entry",
    "load_registry",
    "resolve_latest_wandb_model",
    "resolve_model_path",
    "sha256_of_file",
    "upsert_registry_entry",
    "validate_registry_entry_benchmark_promotion",
]
