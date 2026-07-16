"""User-facing ``robot-sf envs`` UX glue (issue #5801).

Thin, beginner-facing surface over the environment registry in
:mod:`robot_sf.gym_env.env_registry`. ``envs list`` shows every registered
public environment id and its stability; ``envs describe <env-id>`` prints the
full declarative description for one entry. The functions here produce
structured payloads so the CLI dispatcher (:mod:`robot_sf.cli`) and tests share
one implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.gym_env.env_registry import (
    BETA,
    EXPERIMENTAL,
    STABLE,
    EnvEntry,
    describe_env,
    get_env,
    list_envs,
)

if TYPE_CHECKING:  # pragma: no cover - import for type annotations only
    import argparse

__all__ = [
    "describe_env_payload",
    "list_envs_payload",
]

# Short label for the stability level shown in friendly output.
_STABILITY_LABEL = {
    STABLE: "stable (supported public API)",
    BETA: "beta (may change within a minor release)",
    EXPERIMENTAL: "experimental (may change at any time)",
}


def list_envs_payload() -> list[dict[str, str]]:
    """Return one description dict per registered environment, ordered by stability.

    Returns:
        list[dict[str, str]]: Per-environment rows in registry display order.
    """
    return [entry.to_dict() for entry in list_envs()]


def describe_env_payload(env_id: str) -> dict[str, str]:
    """Return the description dict for one env id, or raise ``KeyError`` if unknown.

    Args:
        env_id: Environment identifier to describe.

    Returns:
        dict[str, str]: Full declarative description of the environment.

    Raises:
        KeyError: If ``env_id`` is not registered (delegated from the registry).
    """
    # Touch get_env first so the unknown-id error message is populated.
    get_env(env_id)
    return describe_env(env_id)


def _stability_label(entry: EnvEntry) -> str:
    """Return the friendly stability label for an entry."""
    return _STABILITY_LABEL.get(entry.stability, entry.stability)


def _format_list_envs(rows: list[dict[str, str]]) -> None:
    """Print a friendly table for ``envs list``."""
    import sys  # noqa: PLC0415

    if not rows:
        sys.stdout.write("No environments registered.\n")
        return
    sys.stdout.write(f"{len(rows)} environment(s) registered:\n\n")
    for row in rows:
        stability = row["stability"]
        sys.stdout.write(f"- {row['env_id']}  [{stability}]\n")
        sys.stdout.write(f"    name: {row['display_name']}\n")
        sys.stdout.write(f"    summary: {row['summary']}\n")
        sys.stdout.write(f"    factory: {row['factory']}\n")
        sys.stdout.write("\n")


def _format_describe_env(payload: dict[str, str]) -> None:
    """Print a friendly description for ``envs describe <env-id>``."""
    import sys  # noqa: PLC0415

    sys.stdout.write(f"Environment: {payload['env_id']}\n\n")
    sys.stdout.write(f"  name: {payload['display_name']}\n")
    sys.stdout.write(f"  summary: {payload['summary']}\n")
    sys.stdout.write(f"  stability: {payload['stability']}\n")
    sys.stdout.write(f"  factory: {payload['factory']}\n")
    sys.stdout.write(f"  env class: {payload['env_class']}\n")
    sys.stdout.write(f"  agent count: {payload['agent_count']}\n")
    sys.stdout.write(f"  default config: {payload['default_config']}\n")
    if payload.get("notes"):
        sys.stdout.write(f"  notes: {payload['notes']}\n")


def _handle_envs_list(args: argparse.Namespace) -> int:
    """Dispatch ``robot-sf envs list`` (mirrors ``_handle_models`` in cli.py).

    Returns:
        int: Always 0 (listing never fails for valid registrations).
    """
    import json  # noqa: PLC0415
    import sys  # noqa: PLC0415

    rows = list_envs_payload()
    if args.format == "json":
        sys.stdout.write(json.dumps(rows, indent=2) + "\n")
    else:
        _format_list_envs(rows)
    return 0


def _handle_envs_describe(args: argparse.Namespace) -> int:
    """Dispatch ``robot-sf envs describe <env-id>``.

    Returns:
        int: 0 on success, 2 when the env id is unknown.
    """
    import json  # noqa: PLC0415
    import sys  # noqa: PLC0415

    try:
        payload = describe_env_payload(args.env_id)
    except KeyError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    if args.format == "json":
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        _format_describe_env(payload)
    return 0
