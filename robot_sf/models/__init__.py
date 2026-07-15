"""Helpers for managing trained policy artifacts."""

from robot_sf.common.seed import _configure_torch_213_runtime
from robot_sf.models.registry import (
    RegistryIssue,
    find_latest_wandb_model,
    get_registry_entry,
    load_registry,
    resolve_latest_wandb_model,
    resolve_model_path,
    upsert_registry_entry,
    validate_registry_entry_benchmark_promotion,
)

# Model resolution is commonly followed by a direct ``stable_baselines3.PPO``
# import. On Torch 2.13/Python 3.12+, preload Triton at this boundary so PPO
# deserialization cannot enter the lazy Dynamo path after TensorFlow is loaded.
_configure_torch_213_runtime()

__all__ = [
    "RegistryIssue",
    "find_latest_wandb_model",
    "get_registry_entry",
    "load_registry",
    "resolve_latest_wandb_model",
    "resolve_model_path",
    "upsert_registry_entry",
    "validate_registry_entry_benchmark_promotion",
]
