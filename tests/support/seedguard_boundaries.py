"""Test-only interception of episode boundaries, also used by child Python processes."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import inspect
import json
import os
import sys
from functools import wraps
from numbers import Integral
from pathlib import Path

FALLBACK_HELD_OUT_SEEDS = frozenset(range(111, 141)) | frozenset(
    {
        50036,
        50140,
        50331,
        50403,
        50813,
        51339,
        51709,
        51767,
        52094,
        52175,
        52257,
        52671,
        52850,
        52971,
        53020,
        53198,
        53239,
        53636,
        53671,
        53779,
        55022,
        55379,
        55568,
        56170,
        56966,
        57077,
        57113,
        57494,
        57943,
        59019,
    }
)
try:
    from robot_sf.benchmark.seed_bands import HELD_OUT_SEEDS
except ModuleNotFoundError as error:
    if error.name != "robot_sf.benchmark.seed_bands":
        raise
    HELD_OUT_SEEDS = FALLBACK_HELD_OUT_SEEDS
HELD_OUT_SEEDS = frozenset(HELD_OUT_SEEDS)


class HeldoutSeedError(BaseException):
    """Cannot be swallowed by runner ``except Exception`` recovery."""


def check_simulation_seed(seed, *, boundary):
    """Reject before invoking the original simulation function."""
    if os.environ.get("ROBOT_SF_PYTEST_SEED_GUARD") != "1" or seed is None:
        return
    try:
        value = int(seed)
    except (TypeError, ValueError, OverflowError):
        return
    if value not in HELD_OUT_SEEDS:
        return
    audit = os.environ.get("ROBOT_SF_PYTEST_SEED_AUDIT")
    if audit:
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        path = Path(audit.replace("{worker}", worker))
        event = {
            "test": os.environ.get("PYTEST_CURRENT_TEST", "collection/child process"),
            "seed": value,
            "boundary": boundary,
            "worker": worker,
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
    raise HeldoutSeedError(
        f"held-out simulation seed {value} at {boundary}; use dev seeds 1001..1030"
    )


# Qualified name, direct seed argument, config argument, config seed fields.
BOUNDARIES = {
    "robot_sf.gym_env.environment_factory": [("_apply_global_seed", "seed", None, ())],
    "robot_sf.gym_env.env_util": [("global_reset_seed", "seed", None, ())],
    "robot_sf.gym_env.robot_env": [("RobotEnv.reset", "seed", None, ())],
    "robot_sf.gym_env.pedestrian_env": [("PedestrianEnv.reset", "seed", None, ())],
    "robot_sf.gym_env.crowd_sim_env": [
        ("CrowdSimEnv.__init__", "seed", None, ()),
        ("CrowdSimEnv.reset", "seed", None, ()),
    ],
    "robot_sf.sim.backends.dummy_backend": [
        ("DummySimulator.__init__", "seed", None, ()),
        ("DummySimulator.reset_state", None, "self", ("seed",)),
    ],
    "robot_sf.benchmark.runner": [("run_episode", "seed", None, ())],
    "robot_sf.benchmark.map_runner.map_runner_episode": [
        ("_setup_and_run_step_loop", None, "args", ("seed",))
    ],
    "robot_sf.sim.simulator": [
        (
            "_build_pysf_simulation",
            "response_law_seed",
            "config",
            ("route_spawn_seed", "desired_speed_seed", "archetype_seed", "response_law_seed"),
        )
    ],
    "robot_sf.ped_npc.ped_population": [
        (
            "populate_simulation",
            None,
            "spawn_config",
            ("route_spawn_seed", "archetype_seed", "response_law_seed"),
        ),
        ("populate_ped_routes", None, "config", ("route_spawn_seed",)),
    ],
    "pysocialforce.scene": [("PedState.__init__", None, "config", ("desired_speed_seed",))],
    "random": [("seed", "a", None, ())],
    "numpy.random": [("seed", "seed", None, ())],
}


def patch_module(module):  # noqa: C901 - explicit boundary registry dispatch
    """Patch before the importing caller can retain a reference to the function."""
    for qualified, seed_arg, config_arg, fields in BOUNDARIES[module.__name__]:
        owner = module
        parts = qualified.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        original = getattr(owner, parts[-1])
        if getattr(original, "_seedguard_boundary", None):
            continue
        try:
            signature = inspect.signature(original)
        except ValueError:  # NumPy's C legacy seed function has no signature.
            signature = None
        boundary = qualified
        if module.__name__.endswith("environment_factory"):
            boundary = "environment_factory"
        elif module.__name__.endswith("map_runner_episode"):
            boundary = "map_runner.episode"
        elif module.__name__ == "pysocialforce.scene":
            boundary = "pysocialforce.pedestrian_seed"
        elif module.__name__.endswith("simulator"):
            boundary = "simulator"

        def make_wrapper(original, signature, seed_arg, config_arg, fields, boundary, qualified):
            @wraps(original)
            def guarded(*args, **kwargs):
                if signature is None:
                    values = {"seed": args[0] if args else kwargs.get("seed")}
                else:
                    bound = signature.bind_partial(*args, **kwargs)
                    bound.apply_defaults()
                    values = bound.arguments
                if seed_arg:
                    seed = values.get(seed_arg)
                    if seed is None and qualified.endswith(".reset"):
                        seed = getattr(values.get("self"), "applied_seed", None)
                    # Legacy seed also accepts non-scalar RNG state arrays.
                    if module.__name__ not in {"random", "numpy.random"} or isinstance(
                        seed, (Integral, float)
                    ):
                        check_simulation_seed(seed, boundary=boundary)
                config = values.get(config_arg)
                if boundary == "pysocialforce.pedestrian_seed" and (
                    getattr(config, "desired_speed_mean", None) is None or not len(values["state"])
                ):
                    return original(*args, **kwargs)
                for field in fields:
                    check_simulation_seed(
                        getattr(config, field, None),
                        boundary=f"{boundary}.{field}" if boundary == "simulator" else boundary,
                    )
                return original(*args, **kwargs)

            guarded._seedguard_boundary = boundary
            return guarded

        setattr(
            owner,
            parts[-1],
            make_wrapper(original, signature, seed_arg, config_arg, fields, boundary, qualified),
        )


class GuardLoader:
    """Delegate loading unchanged bytes, then wrap the real runtime boundary."""

    def __init__(self, original):
        """Retain the original Python loader."""
        self.original = original

    def __getattr__(self, name):
        """Preserve optional source and resource loader methods."""
        return getattr(self.original, name)

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        patch_module(module)


class GuardFinder(importlib.abc.MetaPathFinder):
    """Intercept only known simulation modules, never static config parsing."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in BOUNDARIES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = GuardLoader(spec.loader)
        return spec


def install():
    """Install once per pytest worker or guarded child interpreter."""
    if os.environ.get("ROBOT_SF_PYTEST_SEED_GUARD") != "1":
        return
    if not any(isinstance(finder, GuardFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, GuardFinder())
    for name in BOUNDARIES:
        if name in sys.modules:
            patch_module(sys.modules[name])
