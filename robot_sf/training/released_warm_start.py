"""Weights-preserving PPO initialization from registry-pinned release artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from gymnasium import spaces

from robot_sf.common.seed import _configure_torch_213_runtime
from robot_sf.models import get_registry_entry, resolve_model_path

# Match the canonical PPO adapter's optimizer-import runtime setup.
_configure_torch_213_runtime()

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv


def released_checkpoint(model_id: str) -> tuple[Path, dict]:
    """Resolve a released artifact and independently verify its pinned digest and size.

    Returns:
        Verified checkpoint path and portable release provenance.
    """
    entry = get_registry_entry(model_id)
    release = entry.get("github_release") or {}
    if not release.get("sha256") or not release.get("url"):
        raise ValueError("Warm start requires a registry-pinned released checkpoint")
    path = Path(resolve_model_path(model_id, allow_download=True))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != release["sha256"] or path.stat().st_size != release["size_bytes"]:
        raise ValueError("Released checkpoint digest/size mismatch")
    return path, {"model_id": model_id, "artifact_path": release["url"], "sha256": digest}


def load_warm_start(
    checkpoint: Path,
    *,
    env: VecEnv | None = None,
    normalization_path: Path | None = None,
    init_seed: int = 100001,
    device: str = "cpu",
    target_action_space: spaces.Box | None = None,
) -> tuple[PPO, VecEnv | None]:
    """Load actor/value weights and optional observation statistics for a fresh PPO run.

    Initialization resets counters and optimizer moments, keeping every policy tensor
    unchanged. Normalization statistics are external to SB3 ZIP checkpoints; callers
    must supply their pinned companion when the source used VecNormalize. The released
    B1002 checkpoint uses raw dict observations and has no such companion.

    Returns:
        The initialized PPO model and its effective (possibly normalized) environment.
    """
    if not 100000 <= init_seed <= 199999:
        raise ValueError("Initialization seed must be in the training-only range")
    if normalization_path is not None:
        if env is None:
            raise ValueError("Normalization restoration requires an environment")
        env = VecNormalize.load(str(normalization_path), env)
        env.training = True
    custom_objects = None
    if target_action_space is not None:
        source = PPO.load(str(checkpoint), device=device)
        if (
            not isinstance(source.action_space, spaces.Box)
            or source.action_space.shape != target_action_space.shape
            or source.action_space.dtype != target_action_space.dtype
        ):
            raise ValueError("Explicit action-bound migration requires matching Box shape/dtype")
        custom_objects = {"action_space": target_action_space}
    model = PPO.load(
        str(checkpoint), env=env, device=device, force_reset=True, custom_objects=custom_objects
    )
    # SB3's setter changes live generators, but save/load restores this field.
    model.seed = init_seed
    model.set_random_seed(init_seed)
    model.num_timesteps = 0
    model._n_updates = 0
    model._episode_num = 0
    model._last_obs = None
    model.policy.optimizer.state.clear()
    return model, env
