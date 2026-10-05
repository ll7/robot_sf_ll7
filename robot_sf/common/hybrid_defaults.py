"""Source-bound typed defaults for released hybrid and observation contracts.

The registry is separate from algorithm mappings: no compatibility keys enter
canonical effective-config hashes. Registered source and dependency bytes must
match before legacy fill-in is allowed. Explicit constructor/YAML values win.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator  # noqa: TC003 - public runtime type hints
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache, wraps
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
_ACTIVE_POLICY: ContextVar[dict[str, str] | None] = ContextVar(
    "hybrid_default_policy", default=None
)


@cache
def legacy_default_registry() -> dict[str, Any]:
    """Return the explicitly reviewed source registry shipped with this revision."""
    return json.loads(Path(__file__).with_name("legacy_hybrid_defaults.json").read_text())[
        "sources"
    ]


def source_default_policy(source: str | Path | None) -> dict[str, str]:
    """Select typed defaults by exact registered source identity, never its name.

    Returns:
        The default set and registered source, when recognized.
    """
    current = {"default_set": "current"}
    if source is None:
        return current
    path = Path(os.path.abspath(source))
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        return current
    entry = legacy_default_registry().get(relative)
    if entry is None:
        return current
    identities = {relative: entry["sha256"], **entry.get("dependencies", {})}
    for name, expected in identities.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Legacy default source identity changed: {name}")
    return {"default_set": "legacy-0.0.8", "registered_source": relative}


def active_default_policy() -> dict[str, str]:
    """Return a provenance copy of the currently selected typed default policy."""
    return dict(_ACTIVE_POLICY.get() or {"default_set": "current"})


def has_active_default_policy() -> bool:
    """Tell scenario builders whether an episode already selected its source identity.

    Returns:
        Whether a source policy is already scoped.
    """
    return _ACTIVE_POLICY.get() is not None


def current_switch_default() -> bool:
    """Fill an omitted switch without adding fields to the typed config dump.

    Returns:
        True for current inputs and false for verified legacy inputs.
    """
    return active_default_policy()["default_set"] == "current"


@contextmanager
def defaults_for_source(source: str | Path | None) -> Iterator[dict[str, str]]:
    """Scope constructor fill-in to a verified source, restoring it on every exit."""
    policy = source_default_policy(source)
    token = _ACTIVE_POLICY.set(policy)
    try:
        yield policy
    finally:
        _ACTIVE_POLICY.reset(token)


def episode_default_policy(
    function: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Carry source identity through episode builders and record it outside config hashes.

    Returns:
        A wrapped episode runner preserving its public signature.
    """

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        source = kwargs.get("algo_config_path")
        if source is None and not kwargs.get("algo_config"):
            source = kwargs.get("scenario_path")
        with defaults_for_source(source) as policy:
            logger.debug("Typed hybrid defaults: {}", policy)
            record = function(*args, **kwargs)
            metadata = dict(record.get("algorithm_metadata") or {})
            metadata["hybrid_default_policy"] = dict(policy)
            record["algorithm_metadata"] = metadata
            return record

    return wrapped
