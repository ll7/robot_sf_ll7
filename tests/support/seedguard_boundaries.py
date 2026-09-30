"""Test-only interception of episode boundaries, also used by child Python processes."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import inspect
import json
import os
import pickle
import sys
from functools import wraps
from hashlib import sha256
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
HELD_OUT_SEEDS = FALLBACK_HELD_OUT_SEEDS
_POLICY_RESOLVED = False
_ACTIVE = False
_CURRENT_ITEM = None
_STATIC_RNG_STATES = {}
_LEGACY_SEEDS = {}
_LEGACY_RESTORE_STATES = {}
_RNG_STATE_SEEDS = {}


def resolve_held_out_seeds():
    """Load policy only after the child establishes its intended source path.

    Eager project imports would bind isolated lineage checks to the editable
    install before their target script inserts the pinned checkout path.
    """
    global HELD_OUT_SEEDS, _POLICY_RESOLVED
    if not _POLICY_RESOLVED:
        try:
            from robot_sf.benchmark.seed_bands import HELD_OUT_SEEDS as canonical
        except ModuleNotFoundError as error:
            if error.name not in {
                "robot_sf",
                "robot_sf.benchmark",
                "robot_sf.benchmark.seed_bands",
            }:
                raise
        else:
            HELD_OUT_SEEDS = frozenset(canonical)
            _POLICY_RESOLVED = True
    return HELD_OUT_SEEDS


class HeldoutSeedError(BaseException):
    """Cannot be swallowed by runner ``except Exception`` recovery."""


def check_simulation_seed(seed, *, boundary):
    """Reject before invoking the original simulation function."""
    if (not _ACTIVE and os.environ.get("ROBOT_SF_PYTEST_SEED_GUARD") != "1") or seed is None:
        return
    try:
        value = int(seed)
    except (TypeError, ValueError, OverflowError):
        return
    if value not in resolve_held_out_seeds():
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
_PATCHES = []

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
        ("run_map_episode", "seed", None, ()),
        ("_setup_and_run_step_loop", None, "args", ("seed",)),
    ],
    "robot_sf.sim.simulator": [
        ("init_simulators", None, None, ()),
        (
            "Simulator.__init__",
            None,
            "config",
            ("route_spawn_seed", "desired_speed_seed", "archetype_seed", "response_law_seed"),
        ),
        ("Simulator.reset_state", None, None, ()),
        ("Simulator.step_once", None, None, ()),
        (
            "_build_pysf_simulation",
            "response_law_seed",
            "config",
            ("route_spawn_seed", "desired_speed_seed", "archetype_seed", "response_law_seed"),
        ),
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


def set_current_item(item):
    """Keep dynamic pytest markers visible without importing pytest in children."""
    global _CURRENT_ITEM
    _CURRENT_ITEM = item


def _static_seed_test():
    """Only explicitly reasoned static tests may inspect standalone seed RNGs."""
    if _CURRENT_ITEM is not None:
        return any(_CURRENT_ITEM.iter_markers("heldout_seed_ok"))
    return os.environ.get("ROBOT_SF_PYTEST_STATIC_SEED_TEST") == "1"


def restore_static_rngs():
    """Keep allowed static policy RNGs from leaking into subsequent episodes."""
    for owner, state in _STATIC_RNG_STATES.values():
        if owner.__name__ == "random":
            owner.setstate(state)
        else:
            owner.set_state(state)
    _STATIC_RNG_STATES.clear()


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
                if _allow_static_rng(module, boundary):
                    return original(*args, **kwargs)
                values = _call_values(signature, args, kwargs)
                if module.__name__ in {"random", "numpy.random"}:
                    return _record_legacy_seed(module, original, values.get(seed_arg), args, kwargs)
                _check_boundary_seeds(
                    module.__name__, qualified, values, seed_arg, config_arg, fields, boundary
                )
                return original(*args, **kwargs)

            guarded._seedguard_boundary = boundary
            return guarded

        wrapper = make_wrapper(
            original, signature, seed_arg, config_arg, fields, boundary, qualified
        )
        setattr(owner, parts[-1], wrapper)
        _PATCHES.append((owner, parts[-1], original, wrapper))
    if module.__name__ in {"random", "numpy.random"}:
        _patch_rng_state_module(module)


def _allow_static_rng(module, boundary):
    """Static exceptions never admit an actual simulation boundary."""
    if not _static_seed_test():
        return False
    if module.__name__ not in {"random", "numpy.random"}:
        raise HeldoutSeedError(f"heldout_seed_ok cannot execute simulation boundary {boundary}")
    if module.__name__ not in _STATIC_RNG_STATES:
        state = module.getstate() if module.__name__ == "random" else module.get_state()
        _STATIC_RNG_STATES[module.__name__] = (module, state)
    return True


def _call_values(signature, args, kwargs):
    """Bind positional and keyword seeds, including NumPy's C seed function."""
    if signature is None:
        return {"seed": args[0] if args else kwargs.get("seed")}
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


def _record_legacy_seed(module, original, seed, args, kwargs):
    """Record standalone RNG state; reject only when a simulation consumes it.

    Static optimizers, seed-policy checks, and data bootstrap tests may seed
    their own arithmetic. A following simulation boundary must still reject an
    effective held-out global stream before population or stepping.
    """
    value = (
        int(seed)
        if isinstance(seed, Integral) or (isinstance(seed, float) and seed.is_integer())
        else None
    )
    saved = None
    if value in FALLBACK_HELD_OUT_SEEDS and module.__name__ not in _LEGACY_RESTORE_STATES:
        saved = module.getstate() if module.__name__ == "random" else module.get_state()
    result = original(*args, **kwargs)
    if saved is not None:
        _LEGACY_RESTORE_STATES[module.__name__] = (module, saved)
    _LEGACY_SEEDS[module.__name__] = value
    return result


def _check_boundary_seeds(module_name, qualified, values, seed_arg, config_arg, fields, boundary):
    """Check explicit episode/config seeds and global streams at physics entry."""
    if seed_arg:
        seed = values.get(seed_arg)
        if seed is None and qualified.endswith(".reset"):
            seed = getattr(values.get("self"), "applied_seed", None)
        check_simulation_seed(seed, boundary=boundary)
    config = values.get(config_arg)
    if boundary == "pysocialforce.pedestrian_seed" and (
        getattr(config, "desired_speed_mean", None) is None or not len(values["state"])
    ):
        return
    for field in fields:
        check_simulation_seed(
            getattr(config, field, None),
            boundary=f"{boundary}.{field}" if boundary == "simulator" else boundary,
        )
    if module_name in {"robot_sf.sim.simulator", "robot_sf.ped_npc.ped_population"}:
        for owner, seed in _LEGACY_SEEDS.items():
            check_simulation_seed(seed, boundary=f"{qualified}.legacy.{owner}")


def _patch_rng_state_module(owner):
    """Track state restoration only after the RNG module itself is imported."""
    getter, setter = (
        ("getstate", "setstate") if owner.__name__ == "random" else ("get_state", "set_state")
    )
    get_original, set_original = getattr(owner, getter), getattr(owner, setter)
    if getattr(get_original, "_seedguard_boundary", None):
        return

    def make_pair(owner, get_original, set_original):
        @wraps(get_original)
        def get_state(*args, **kwargs):
            state = get_original(*args, **kwargs)
            key = sha256(pickle.dumps(state, protocol=5)).digest()
            _RNG_STATE_SEEDS[(owner.__name__, key)] = _LEGACY_SEEDS.get(owner.__name__)
            return state

        @wraps(set_original)
        def set_state(state, *args, **kwargs):
            result = set_original(state, *args, **kwargs)
            key = sha256(pickle.dumps(state, protocol=5)).digest()
            _LEGACY_SEEDS[owner.__name__] = _RNG_STATE_SEEDS.get((owner.__name__, key))
            return result

        get_state._seedguard_boundary = setter
        return get_state, set_state

    get_wrapper, set_wrapper = make_pair(owner, get_original, set_original)
    for name, original, wrapper in (
        (getter, get_original, get_wrapper),
        (setter, set_original, set_wrapper),
    ):
        setattr(owner, name, wrapper)
        _PATCHES.append((owner, name, original, wrapper))


def restore_unused_legacy_seeds():
    """Keep static held-out RNG arithmetic from contaminating later episodes."""
    for name, (owner, state) in _LEGACY_RESTORE_STATES.items():
        if _LEGACY_SEEDS.get(name) in FALLBACK_HELD_OUT_SEEDS:
            if name == "random":
                owner.setstate(state)
            else:
                owner.set_state(state)
    _LEGACY_RESTORE_STATES.clear()
    _RNG_STATE_SEEDS.clear()


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
    global _ACTIVE
    if os.environ.get("ROBOT_SF_PYTEST_SEED_GUARD") != "1":
        return
    _ACTIVE = True
    if not any(isinstance(finder, GuardFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, GuardFinder())
    for name in BOUNDARIES:
        if name in sys.modules:
            patch_module(sys.modules[name])
    _patch_child_processes()
    _patch_multiprocessing()


def _isolated_python_command(command, boundary_path):
    """Bootstrap isolated interpreters while retaining flags and original argv."""
    if (
        not isinstance(command, (list, tuple))
        or not command
        or "python" not in Path(os.fsdecode(command[0])).name
    ):
        return command
    index = 1
    isolated = False
    while index < len(command):
        option = os.fsdecode(command[index])
        if option in {"-c", "-m", "-", "--"} or not option.startswith("-"):
            break
        if option in {"-h", "--help", "-V", "--version"}:
            return command
        isolated |= (
            option.startswith("-")
            and not option.startswith("--")
            and any(flag in option[1:] for flag in "ISE")
        )
        index += 2 if option in {"-W", "-X", "--check-hash-based-pycs"} else 1
    if not isolated or index >= len(command):
        return command
    target = os.fsdecode(command[index])
    tail = [os.fsdecode(arg) for arg in command[index + 1 :]]
    if target == "--":
        if not tail:
            return command
        target, tail = tail[0], tail[1:]
    if target in {"-c", "-m"}:
        if not tail:
            return command
        payload, extra = tail[0], tail[1:]
    else:
        payload, extra = target, tail
    bootstrap = _isolated_bootstrap(target, payload, extra, boundary_path)
    return [*command[:index], "-c", bootstrap]


def _isolated_bootstrap(target, payload, extra, boundary_path):
    """Build a fail-closed bootstrap for script, module, command, or stdin mode."""
    # Restore the original script path before loading support; never add a
    # checkout root or site-packages to an isolated interpreter's search path.
    setup = ""
    if target not in {"-c", "-m", "-"}:
        setup = f"if not getattr(sys.flags, 'safe_path', sys.flags.isolated): sys.path[0] = os.path.dirname(os.path.abspath({payload!r}))\n"
    bootstrap = (
        "import sys, os, importlib.util, runpy\n"
        + setup
        + f"spec = importlib.util.spec_from_file_location('tests.support.seedguard_boundaries', {str(boundary_path)!r})\n"
        "guard = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = guard\n"
        "spec.loader.exec_module(guard)\n"
        "guard.install()\n"
    )
    if target == "-c":
        bootstrap += f"sys.argv = {['-c', *extra]!r}\nexec(compile({payload!r}, '<string>', 'exec'), {{'__name__': '__main__', '__builtins__': __builtins__}})\n"
    elif target == "-m":
        bootstrap += f"sys.argv = {[payload, *extra]!r}\nrunpy.run_module({payload!r}, run_name='__main__', alter_sys=True)\n"
    elif target == "-":
        bootstrap += f"sys.argv = {['-', *extra]!r}\nexec(compile(sys.stdin.read(), '<stdin>', 'exec'), {{'__name__': '__main__', '__builtins__': __builtins__}})\n"
    else:
        bootstrap += (
            f"sys.argv = {[payload, *extra]!r}\nrunpy.run_path({payload!r}, run_name='__main__')\n"
        )
    return bootstrap


_CHILD_CONTEXT_KEYS = (
    "ROBOT_SF_PYTEST_SEED_AUDIT",
    "PYTEST_CURRENT_TEST",
    "PYTEST_XDIST_WORKER",
    "ROBOT_SF_PYTEST_STATIC_SEED_TEST",
)


def _child_context():
    """Snapshot test identity, including absent keys for persistent forkservers."""
    context = {key: os.environ.get(key) for key in _CHILD_CONTEXT_KEYS}
    context["ROBOT_SF_PYTEST_STATIC_SEED_TEST"] = "1" if _static_seed_test() else None
    context["ROBOT_SF_PYTEST_SEED_GUARD"] = "1"
    return context


def _child_environment(supplied=None):
    """Supply protection without changing the parent's interpreter search path."""
    env = dict(os.environ if supplied is None else supplied)
    for key, value in _child_context().items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    support = Path(__file__).resolve().parent
    paths = [str(support / "seedguard_bootstrap"), str(support), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
    return env


def _patch_child_processes():
    """Protect explicit child environments and grandchildren, including -I/-S."""
    import subprocess

    original = subprocess.Popen.__init__
    if getattr(original, "_seedguard_boundary", None):
        return
    signature = inspect.signature(original)
    support = Path(__file__).resolve().parent

    @wraps(original)
    def guarded(self, *args, **kwargs):
        bound = signature.bind_partial(self, *args, **kwargs)
        supplied = bound.arguments.get("env")
        bound.arguments["env"] = _child_environment(supplied)
        bound.arguments["args"] = _isolated_python_command(
            bound.arguments["args"], support / "seedguard_boundaries.py"
        )
        return original(*bound.args, **bound.kwargs)

    guarded._seedguard_boundary = "subprocess.Popen"
    subprocess.Popen.__init__ = guarded
    _PATCHES.append((subprocess.Popen, "__init__", original, guarded))


def _patch_multiprocessing():
    """Protect spawn/forkserver, which bypass Popen, and refresh each child identity."""
    from multiprocessing import spawn

    # subprocess has already cached its C entry point. multiprocessing imports
    # this attribute on every launch, including resource tracker/forkserver.
    if os.name == "posix":
        import _posixsubprocess

        original_exec = _posixsubprocess.fork_exec
        if not getattr(original_exec, "_seedguard_boundary", None):

            @wraps(original_exec)
            def guarded_exec(*args, **kwargs):
                arguments = list(args)
                encoded_env = arguments[5]
                supplied = (
                    None
                    if encoded_env is None
                    else dict(os.fsdecode(row).split("=", 1) for row in encoded_env)
                )
                arguments[5] = [
                    os.fsencode(f"{key}={value}")
                    for key, value in _child_environment(supplied).items()
                ]
                arguments[0] = _isolated_python_command(arguments[0], Path(__file__).resolve())
                return original_exec(*arguments, **kwargs)

            guarded_exec._seedguard_boundary = "multiprocessing.fork_exec"
            _posixsubprocess.fork_exec = guarded_exec
            _PATCHES.append((_posixsubprocess, "fork_exec", original_exec, guarded_exec))

    original_data = spawn.get_preparation_data
    if getattr(original_data, "_seedguard_boundary", None):
        return
    original_prepare = spawn.prepare

    @wraps(original_data)
    def guarded_data(*args, **kwargs):
        data = original_data(*args, **kwargs)
        data["_seedguard_context"] = _child_context()
        return data

    @wraps(original_prepare)
    def guarded_prepare(data):
        for key, value in data.get("_seedguard_context", {}).items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        set_current_item(None)
        return original_prepare(data)

    for name, original, wrapper in (
        ("get_preparation_data", original_data, guarded_data),
        ("prepare", original_prepare, guarded_prepare),
    ):
        wrapper._seedguard_boundary = f"multiprocessing.{name}"
        setattr(spawn, name, wrapper)
        _PATCHES.append((spawn, name, original, wrapper))


def uninstall():
    """Restore original callables when an embedded pytest session ends."""
    global _ACTIVE
    _ACTIVE = False
    restore_static_rngs()
    restore_unused_legacy_seeds()
    set_current_item(None)
    for owner, name, original, wrapper in reversed(_PATCHES):
        if getattr(owner, name) is wrapper:
            setattr(owner, name, original)
    _PATCHES.clear()
    sys.meta_path[:] = [finder for finder in sys.meta_path if not isinstance(finder, GuardFinder)]
