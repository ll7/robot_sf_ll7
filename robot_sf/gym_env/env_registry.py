"""Environment registry for discoverable, documented environment ids.

Provides a registry of the public Gymnasium-style environments shipped by Robot
SF and the ergonomic factory functions that build them. The registry powers the
``robot-sf envs list`` / ``robot-sf envs describe <env-id>`` CLI (issue #5801) so
users can discover the supported public surface without reading source.

Design note
-----------
The registry is intentionally a *declarative catalog* of the public API surface,
not a factory dispatch table: the canonical construction path stays the typed
``make_*/create_*`` functions in :mod:`robot_sf.gym_env.environment_factory`. Each
entry records the stable ``env_id``, the factory symbol, the environment class it
builds, a one-line purpose, default config kind, agent count, and a stability
level. Stability levels follow the policy documented in ``docs/api/stable_public_api.md``:

* ``stable``     — supported public API; breaking changes follow semver + deprecation.
* ``beta``       — usable but may change within a minor release without deprecation.
* ``experimental`` — exploratory; semantics may change at any time.

Registration happens on import of this module so ``list_envs()`` is populated
before the CLI runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

# Stability levels (kept as literals so docs/lint stay stable).
STABLE = "stable"
BETA = "beta"
EXPERIMENTAL = "experimental"

_STABILITY_ORDER = {STABLE: 0, BETA: 1, EXPERIMENTAL: 2}

_REGISTRY: dict[str, EnvEntry] = {}


@dataclass(frozen=True)
class EnvEntry:
    """Declarative description of one public environment id.

    Attributes:
        env_id: Stable, user-facing identifier (used by ``envs describe``).
        display_name: Human-friendly name shown in ``envs list``.
        summary: One-line purpose of the environment.
        factory: Qualified factory symbol name that constructs the env.
        env_class: Python class the factory builds (best-effort, for discovery).
        agent_count: ``"single"``, ``"multi"``, or ``"none"`` (crowd-only).
        default_config: Config dataclass name used when none is supplied.
        stability: One of ``stable``/``beta``/``experimental``.
        notes: Optional extra guidance / caveats for users.
    """

    env_id: str
    display_name: str
    summary: str
    factory: str
    env_class: str
    agent_count: str
    default_config: str
    stability: str = STABLE
    notes: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return the entry as a plain ``dict`` for JSON/list output."""
        return {
            "env_id": self.env_id,
            "display_name": self.display_name,
            "summary": self.summary,
            "factory": self.factory,
            "env_class": self.env_class,
            "agent_count": self.agent_count,
            "default_config": self.default_config,
            "stability": self.stability,
            "notes": self.notes,
        }


def register_env(entry: EnvEntry, *, override: bool = False) -> None:
    """Register a public environment entry under its ``env_id``.

    Args:
        entry: Declarative :class:`EnvEntry` describing the environment.
        override: Replace an existing registration when ``True``.

    Raises:
        KeyError: If ``env_id`` is already registered and ``override`` is False.
        ValueError: If ``env_id`` is empty or ``stability`` is unknown.
    """
    env_id = (entry.env_id or "").strip()
    if not env_id:
        raise ValueError("env_id must be a non-empty string")
    if entry.stability not in _STABILITY_ORDER:
        raise ValueError(
            f"Unknown stability '{entry.stability}' for '{env_id}'; "
            f"expected one of {sorted(_STABILITY_ORDER)}"
        )
    already_registered = env_id in _REGISTRY
    if already_registered and not override:
        raise KeyError(f"env_id '{env_id}' already registered; pass override=True to replace")
    _REGISTRY[env_id] = entry
    if override and already_registered:
        logger.debug("Re-registered environment (override): {}", env_id)


def get_env(env_id: str) -> EnvEntry:
    """Return the registered entry for ``env_id``.

    Args:
        env_id: Environment identifier to resolve.

    Returns:
        The matching :class:`EnvEntry`.

    Raises:
        KeyError: If ``env_id`` is not registered (message lists known ids).
    """
    try:
        return _REGISTRY[env_id]
    except KeyError as exc:  # provide suggestions
        known = ", ".join(sorted(_REGISTRY.keys())) or "<none>"
        raise KeyError(f"Unknown env_id '{env_id}'. Known: {known}") from exc


def list_envs() -> list[EnvEntry]:
    """Return all registered environment entries sorted by stability then id.

    Returns:
        List of :class:`EnvEntry` ordered by stability (stable first) then id.
    """
    return sorted(
        _REGISTRY.values(),
        key=lambda e: (_STABILITY_ORDER.get(e.stability, 99), e.env_id),
    )


def env_ids() -> list[str]:
    """Return sorted registered ``env_id`` strings."""
    return sorted(_REGISTRY.keys())


def describe_env(env_id: str) -> dict[str, str]:
    """Return the plain-dict description for ``env_id`` (raises if unknown)."""
    return get_env(env_id).to_dict()


def _register_default_envs() -> None:
    """Register the public environment catalog shipped by Robot SF."""
    entries = [
        EnvEntry(
            env_id="robot",
            display_name="Robot navigation environment",
            summary="Single robot navigates a crowd using vector/state observations.",
            factory="robot_sf.gym_env.environment_factory.make_robot_env",
            env_class="robot_sf.gym_env.robot_env.RobotEnv",
            agent_count="single",
            default_config="robot_sf.gym_env.unified_config.RobotSimulationConfig",
            stability=STABLE,
            notes="Primary supported training/evaluation environment.",
        ),
        EnvEntry(
            env_id="robot-image",
            display_name="Robot navigation with image observations",
            summary="Single robot navigating a crowd with image/visual observations.",
            factory="robot_sf.gym_env.environment_factory.make_image_robot_env",
            env_class="robot_sf.gym_env.robot_env_with_image.RobotEnvWithImage",
            agent_count="single",
            default_config="robot_sf.gym_env.unified_config.ImageRobotConfig",
            stability=BETA,
            notes="Image pipeline adds Pygame/render dependency at construction time.",
        ),
        EnvEntry(
            env_id="pedestrian",
            display_name="Pedestrian (adversarial) environment",
            summary="Single adversarial pedestrian agent in the crowd (no robot action).",
            factory="robot_sf.gym_env.environment_factory.make_pedestrian_env",
            env_class="robot_sf.gym_env.pedestrian_env.PedestrianEnv",
            agent_count="single",
            default_config="robot_sf.gym_env.unified_config.PedestrianSimulationConfig",
            stability=STABLE,
        ),
        EnvEntry(
            env_id="crowd-sim",
            display_name="Crowd-only simulation environment",
            summary="Robot-free Social Force pedestrian stepping (reset/step only).",
            factory="robot_sf.gym_env.environment_factory.make_crowd_sim_env",
            env_class="robot_sf.gym_env.crowd_sim_env.CrowdSimEnv",
            agent_count="none",
            default_config="robot_sf.gym_env.crowd_sim_env.CrowdSimulationConfig",
            stability=STABLE,
            notes="step(action=None) accepted for Gymnasium compatibility; actions ignored.",
        ),
        EnvEntry(
            env_id="multi-robot",
            display_name="Multi-robot coordination environment",
            summary="Multiple robots navigate and interact in a shared scene.",
            factory="robot_sf.gym_env.environment_factory.make_multi_robot_env",
            env_class="robot_sf.gym_env.multi_robot_env.MultiRobotEnv",
            agent_count="multi",
            default_config="robot_sf.gym_env.unified_config.MultiRobotConfig",
            stability=BETA,
        ),
    ]
    for entry in entries:
        register_env(entry, override=True)


_register_default_envs()
