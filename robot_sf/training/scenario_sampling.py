"""Scenario sampling utilities for config-first training runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import Env, spaces
from loguru import logger

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import build_robot_config_from_scenario

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


def scenario_id_from_definition(scenario: Mapping[str, Any], *, index: int) -> str:
    """Derive a stable scenario identifier from a scenario definition.

    Returns:
        Stable scenario identifier for logging and sampling.
    """
    scenario_id = str(scenario.get("name") or scenario.get("scenario_id") or "").strip()
    if not scenario_id:
        raise ValueError(f"Scenario at index {index} is missing a name/scenario_id field.")
    return scenario_id


def _spaces_compatible(  # noqa: C901
    base: spaces.Space,
    other: spaces.Space,
    *,
    allow_box_bounds_mismatch: bool,
) -> bool:
    """Return whether two spaces can safely share one vectorized env contract."""
    if type(base) is not type(other):
        return False
    if isinstance(base, spaces.Box):
        if base.shape != other.shape or np.dtype(base.dtype) != np.dtype(other.dtype):
            return False
        if allow_box_bounds_mismatch:
            return True
        return np.array_equal(base.low, other.low) and np.array_equal(base.high, other.high)
    if isinstance(base, spaces.Discrete):
        return base.n == other.n
    if isinstance(base, spaces.MultiDiscrete):
        return np.array_equal(base.nvec, other.nvec)
    if isinstance(base, spaces.MultiBinary):
        return base.n == other.n
    if isinstance(base, spaces.Dict):
        if list(base.spaces.keys()) != list(other.spaces.keys()):
            return False
        return all(
            _spaces_compatible(
                base.spaces[key],
                other.spaces[key],
                allow_box_bounds_mismatch=allow_box_bounds_mismatch,
            )
            for key in base.spaces
        )
    if isinstance(base, spaces.Tuple):
        if len(base.spaces) != len(other.spaces):
            return False
        return all(
            _spaces_compatible(
                base_space,
                other_space,
                allow_box_bounds_mismatch=allow_box_bounds_mismatch,
            )
            for base_space, other_space in zip(base.spaces, other.spaces, strict=False)
        )
    return base == other


@dataclass(slots=True)
class ScenarioSampler:
    """Random or cyclic sampler for scenario definitions."""

    scenarios: Sequence[Mapping[str, Any]]
    include_scenarios: tuple[str, ...] = ()
    exclude_scenarios: tuple[str, ...] = ()
    weights: Mapping[str, float] | None = None
    seed: int | None = None
    strategy: str = "random"
    _scenario_ids: tuple[str, ...] = field(init=False, repr=False)
    _scenario_map: dict[str, Mapping[str, Any]] = field(init=False, repr=False)
    _weights_array: np.ndarray | None = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _cycle_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize scenario filters and sampling state."""
        scenario_ids, scenario_map = self._index_scenarios()
        filtered_ids = self._apply_filters(scenario_ids)
        if not filtered_ids:
            raise ValueError("Scenario sampler has no scenarios after filtering.")
        self._scenario_ids = tuple(filtered_ids)
        self._scenario_map = scenario_map
        self._rng = np.random.default_rng(self.seed)
        self._weights_array = self._build_weight_array()
        if self.strategy not in {"random", "cycle"}:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        """Return the filtered scenario identifiers in stable order."""
        return self._scenario_ids

    def _index_scenarios(self) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
        """Build ordered scenario ids and lookup map.

        Returns:
            Scenario ids in declaration order and an id-to-definition lookup.
        """
        scenario_ids: list[str] = []
        scenario_map: dict[str, Mapping[str, Any]] = {}
        for idx, scenario in enumerate(self.scenarios):
            scenario_id = scenario_id_from_definition(scenario, index=idx)
            scenario_ids.append(scenario_id)
            scenario_map[scenario_id] = scenario
        return scenario_ids, scenario_map

    def _apply_filters(self, scenario_ids: Sequence[str]) -> list[str]:
        """Filter scenarios based on include/exclude rules.

        Returns:
            Filtered scenario ids in declaration order.
        """
        known = {name.lower() for name in scenario_ids}
        include_set = {name.lower() for name in self.include_scenarios}
        exclude_set = {name.lower() for name in self.exclude_scenarios}
        missing_include = include_set - known
        missing_exclude = exclude_set - known
        if missing_include:
            raise ValueError(f"Unknown include_scenarios: {sorted(missing_include)}")
        if missing_exclude:
            raise ValueError(f"Unknown exclude_scenarios: {sorted(missing_exclude)}")
        return [
            scenario_id
            for scenario_id in scenario_ids
            if (not include_set or scenario_id.lower() in include_set)
            and scenario_id.lower() not in exclude_set
        ]

    def _build_weight_array(self) -> np.ndarray | None:
        """Return normalized sampling weights for the filtered scenario set.

        Returns:
            Normalized sampling weights, or None for uniform sampling.
        """
        if not self.weights:
            return None
        weights = np.array(
            [float(self.weights.get(sid, 0.0)) for sid in self._scenario_ids],
            dtype=float,
        )
        if np.any(weights < 0.0):
            raise ValueError("Scenario weights must be non-negative.")
        if np.all(weights == 0.0):
            raise ValueError("Scenario weights must include at least one positive entry.")
        return weights / weights.sum()

    def sample(self) -> tuple[Mapping[str, Any], str]:
        """Return a scenario mapping and its stable identifier."""
        if self.strategy == "cycle":
            idx = self._cycle_index
            self._cycle_index = (self._cycle_index + 1) % len(self._scenario_ids)
        else:
            idx = int(self._rng.choice(len(self._scenario_ids), p=self._weights_array))
        scenario_id = self._scenario_ids[idx]
        return self._scenario_map[scenario_id], scenario_id


class ScenarioSwitchingEnv(Env):
    """Gymnasium environment wrapper that swaps scenarios between episodes."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        scenario_sampler: ScenarioSampler,
        scenario_path: Any,
        env_factory: Callable[..., Env] = make_robot_env,
        config_builder: Callable[[Mapping[str, Any]], object] | None = None,
        env_factory_kwargs: Mapping[str, Any] | None = None,
        suite_name: str = "ppo_imitation",
        algorithm_name: str = "policy",
        switch_per_reset: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize a wrapper that swaps scenarios between episodes."""
        super().__init__()
        self._scenario_sampler = scenario_sampler
        self._scenario_path = scenario_path
        self._env_factory = env_factory
        self._config_builder = config_builder or (
            lambda scenario: build_robot_config_from_scenario(
                scenario,
                scenario_path=self._scenario_path,
            )
        )
        self._env_factory_kwargs = dict(env_factory_kwargs or {})
        self._suite_name = suite_name
        self._algorithm_name = algorithm_name
        self._switch_per_reset = switch_per_reset
        self._rng = np.random.default_rng(seed)
        self._scenario_coverage: dict[str, int] = {}
        self._current_env: Env | None = None
        self._current_scenario_id: str | None = None
        self._has_reset = False
        self._warned_obs_bounds = False

        env, scenario_id = self._build_env(seed=seed)
        self._activate_env(env, scenario_id)
        self.observation_space = self._current_env.observation_space
        self.action_space = self._current_env.action_space
        self.metadata = getattr(self._current_env, "metadata", {})
        self.render_mode = getattr(self._current_env, "render_mode", None)

    @property
    def scenario_coverage(self) -> dict[str, int]:
        """Return scenario activation counts for this worker."""
        return dict(self._scenario_coverage)

    @property
    def scenario_id(self) -> str | None:
        """Return the active scenario identifier."""
        return self._current_scenario_id

    def _build_env(self, *, seed: int | None) -> tuple[Env, str]:
        """Create one environment for the next sampled scenario.

        Returns:
            Newly created environment and active scenario id.
        """
        scenario, scenario_id = self._scenario_sampler.sample()
        env_seed = int(seed) if seed is not None else int(self._rng.integers(0, 2**31 - 1))
        env = self._env_factory(
            config=self._config_builder(scenario),
            seed=env_seed,
            suite_name=self._suite_name,
            scenario_name=scenario_id,
            algorithm_name=self._algorithm_name,
            **self._env_factory_kwargs,
        )
        return env, scenario_id

    def _activate_env(self, env: Env, scenario_id: str) -> None:
        """Register a newly accepted active environment instance."""
        self._scenario_coverage[scenario_id] = self._scenario_coverage.get(scenario_id, 0) + 1
        self._current_env = env
        self._current_scenario_id = scenario_id

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the current scenario environment, optionally switching scenarios.

        Returns:
            Reset observation and info from the active environment.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._current_env is None:
            env, scenario_id = self._build_env(seed=seed)
            self._activate_env(env, scenario_id)
            self.observation_space = self._current_env.observation_space
            self.action_space = self._current_env.action_space
        elif self._switch_per_reset and self._has_reset:
            previous_scenario_id = self._current_scenario_id
            new_env, new_scenario_id = self._build_env(seed=seed)
            obs_strict = _spaces_compatible(
                self.observation_space,
                new_env.observation_space,
                allow_box_bounds_mismatch=False,
            )
            obs_compatible = _spaces_compatible(
                self.observation_space,
                new_env.observation_space,
                allow_box_bounds_mismatch=True,
            )
            action_compatible = _spaces_compatible(
                self.action_space,
                new_env.action_space,
                allow_box_bounds_mismatch=False,
            )
            if not obs_compatible or not action_compatible:
                new_env.close()
                raise ValueError(
                    "Scenario switching produced incompatible observation/action spaces."
                )
            if not obs_strict and not self._warned_obs_bounds:
                logger.warning(
                    "Scenario switching detected observation-space bound mismatches; "
                    "using initial bounds for compatibility. previous={} current={}",
                    previous_scenario_id,
                    new_scenario_id,
                )
                self._warned_obs_bounds = True
            self._current_env.close()
            self._activate_env(new_env, new_scenario_id)

        self._has_reset = True
        return self._current_env.reset(seed=seed, options=options)

    def step(self, action):
        """Step the active scenario environment.

        Returns:
            Gymnasium step tuple from the active environment.
        """
        if self._current_env is None:
            raise RuntimeError("ScenarioSwitchingEnv.step called before reset().")
        return self._current_env.step(action)

    def render(self, *args, **kwargs):  # pragma: no cover - passthrough
        """Render the active environment when available.

        Returns:
            Render result from the active environment, or None when closed.
        """
        if self._current_env is None:
            return None
        return self._current_env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the active environment."""
        if self._current_env is not None:
            self._current_env.close()
            self._current_env = None

    def __getattr__(self, name: str):
        """Forward attribute access to the active environment.

        Returns:
            Attribute value from the active environment.
        """
        env = self.__dict__.get("_current_env")
        if env is not None:
            return getattr(env, name)
        raise AttributeError(name)
