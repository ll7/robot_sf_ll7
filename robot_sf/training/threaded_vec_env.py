"""In-process threaded vector environment for Stable-Baselines3 rollouts.

``ThreadedVecEnv`` keeps each Gymnasium environment independent while dispatching
reset and step calls concurrently from one process.  It is intended for Robot SF
environments whose hot numerical kernels release the Python global interpreter
lock; it is not a replacement for subprocess isolation.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

if TYPE_CHECKING:
    from collections.abc import Callable

    import gymnasium as gym


class ThreadedVecEnv(DummyVecEnv):
    """Run independent Gymnasium environments concurrently in one process.

    The class implements the Stable-Baselines3 ``VecEnv`` contract by extending
    ``DummyVecEnv``.  Observations, automatic resets, terminal observations, and
    attribute helpers therefore retain the established SB3 behavior.  Unlike
    ``SubprocVecEnv``, environment objects are not serialized between workers.

    Args:
        env_fns: Factories that each create a distinct Gymnasium environment.
        max_workers: Maximum concurrent reset/step calls. Defaults to one worker
            per environment and must be positive.

    Raises:
        ValueError: If ``max_workers`` is not positive.
        RuntimeError: If a reset or second asynchronous step is requested while
            a previous step is still pending.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        *,
        max_workers: int | None = None,
    ) -> None:
        """Initialize independent environments and a bounded in-process worker pool."""
        super().__init__(env_fns)
        resolved_workers = self.num_envs if max_workers is None else int(max_workers)
        if resolved_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._executor = ThreadPoolExecutor(
            max_workers=min(resolved_workers, self.num_envs),
            thread_name_prefix="robot-sf-vec-env",
        )
        self._step_futures: list[Future[Any]] | None = None
        self._closed = False

    def reset(self) -> Any:
        """Reset all environments concurrently.

        Returns:
            Batched observations in the Stable-Baselines3 ``VecEnv`` format.
        """
        self._ensure_no_pending_step("reset")
        futures = [
            self._executor.submit(
                env.reset,
                seed=self._seeds[env_idx],
                **({"options": self._options[env_idx]} if self._options[env_idx] else {}),
            )
            for env_idx, env in enumerate(self.envs)
        ]
        for env_idx, future in enumerate(futures):
            obs, self.reset_infos[env_idx] = future.result()
            self._save_obs(env_idx, obs)
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

    def step_async(self, actions: np.ndarray) -> None:
        """Dispatch one action per environment without waiting for completion."""
        self._ensure_no_pending_step("step_async")
        copied_actions = np.asarray(actions).copy()
        self._step_futures = [
            self._executor.submit(env.step, copied_actions[env_idx])
            for env_idx, env in enumerate(self.envs)
        ]

    def step_wait(self) -> tuple[Any, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Collect concurrent steps and apply Stable-Baselines3 done handling.

        Returns:
            Batched observations, rewards, done flags, and information dictionaries.
        """
        futures = self._step_futures
        if futures is None:
            raise RuntimeError("step_wait called before step_async")
        try:
            transitions = [future.result() for future in futures]
        finally:
            self._step_futures = None

        for env_idx, (obs, reward, terminated, truncated, info) in enumerate(transitions):
            self.buf_rews[env_idx] = reward
            self.buf_dones[env_idx] = terminated or truncated
            self.buf_infos[env_idx] = info
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated
            if self.buf_dones[env_idx]:
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def close(self) -> None:
        """Stop workers and close the owned environments exactly once."""
        if self._closed:
            return
        self._closed = True
        if self._step_futures is not None:
            for future in self._step_futures:
                future.cancel()
            self._step_futures = None
        self._executor.shutdown(wait=True, cancel_futures=True)
        super().close()

    def _ensure_no_pending_step(self, operation: str) -> None:
        """Reject operations that would race an outstanding asynchronous step."""
        if self._closed:
            raise RuntimeError("cannot use a closed ThreadedVecEnv")
        if self._step_futures is not None:
            raise RuntimeError(f"cannot call {operation} while a step is pending")
