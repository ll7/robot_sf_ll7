"""Threaded vectorized environment for parallel environment stepping."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn


def _init_classes() -> dict[str, Any]:
    from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: PLC0415

    class ThreadedVecEnv(DummyVecEnv):
        """Threaded vectorized environment for parallel environment stepping.

        Subclasses DummyVecEnv to execute environment steps in parallel using a
        ThreadPoolExecutor while keeping the single-process execution model.
        """

        def __init__(
            self,
            env_fns: list[Callable[[], Any]],
            max_workers: int | None = None,
        ):
            super().__init__(env_fns)
            if max_workers is None:
                max_workers = min(32, (mp.cpu_count() or 1) + 4)
            self._executor = ThreadPoolExecutor(max_workers=max_workers)

        def step_async(self, actions: np.ndarray) -> None:
            """Tell all environments to step in parallel.

            Args:
                actions: Actions for each environment.
            """
            self.actions = actions

        def step_wait(self) -> VecEnvStepReturn:
            """Wait for steps to complete and gather results.

            Returns:
                VecEnvStepReturn: Tuple of (obs, rewards, dones, infos).
            """
            futures = [
                self._executor.submit(self._step_single_env, env_idx, action)
                for env_idx, action in enumerate(self.actions)
            ]

            for env_idx, future in enumerate(futures):
                obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = (
                    future.result()
                )
                self._save_obs(env_idx, obs)

            return (
                self._obs_from_buf(),
                np.copy(self.buf_rews),
                np.copy(self.buf_dones),
                self._clone_infos(self.buf_infos),
            )

        def _step_single_env(self, env_idx: int, action: Any) -> tuple:
            """Execute step on a single environment.

            Returns:
                tuple: Tuple of (obs, reward, done, info).
            """
            obs, reward, terminated, truncated, info = self.envs[env_idx].step(action)
            done = terminated or truncated
            self.buf_infos[env_idx] = info

            if done:
                info["terminal_observation"] = obs
                obs, reset_info = self.envs[env_idx].reset()
                info.update(reset_info)

            return obs, reward, done, info

        def _clone_infos(self, infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Clone dictionary elements in infos list.

            Returns:
                list[dict[str, Any]]: Cloned list of info dicts.
            """
            return [info.copy() for info in infos]

        def close(self) -> None:
            """Clean up environment resources and thread pool."""
            self._executor.shutdown(wait=True)
            super().close()

    return {"ThreadedVecEnv": ThreadedVecEnv}


_cache: dict[str, Any] | None = None
_LAZY_NAMES = {"ThreadedVecEnv"}


def __getattr__(name: str) -> Any:
    if name in _LAZY_NAMES:
        global _cache
        if _cache is None:
            _cache = _init_classes()
        return _cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
