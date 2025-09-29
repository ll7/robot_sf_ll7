"""Identity and hashing utilities for benchmark artifacts (T035).

Provides stable JSON serialization and deterministic hashing used to derive
episode identity strings and manifest hashes. Hash stability is critical for
resume logic: the same scenario parameters + seed must yield the same episode
id across runs.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def canonical_dumps(obj: Any) -> str:
    """Serialize `obj` to a canonical JSON string.

    Ensures:
    - UTF-8 safe characters
    - Sorted keys for mapping objects
    - No whitespace differences (separators set tightly)
    - Recursively processes nested dict/list structures
    """

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(obj: Any, *, algo: str = "sha256") -> str:
    """Return hex digest of canonical serialization with chosen algorithm."""

    data = canonical_dumps(obj).encode("utf-8")
    h = hashlib.new(algo)
    h.update(data)
    return h.hexdigest()


def episode_identity_components(
    scenario_params: Mapping[str, Any],
    seed: int,
    extra: Mapping[str, Any] | None = None,
) -> dict:
    """Collect canonical identity components for an episode.

    Parameters
    ----------
    scenario_params : Mapping[str, Any]
        Raw scenario parameters (must be JSON-serializable). Keys are sorted by
        canonical_dumps.
    seed : int
        Episode initial seed.
    extra : Mapping[str, Any] | None
        Optional additional identity-affecting values (e.g., algorithm version).
    """

    base = {"scenario": scenario_params, "seed": seed}
    if extra:
        base["extra"] = extra
    return base


def make_episode_id(scenario_params: Mapping[str, Any], seed: int, prefix: str = "ep") -> str:
    """Generate deterministic episode id string.

    Format: ``{prefix}_{first12hex}`` where `first12hex` are the first 12 hex
    characters of the sha256 digest of canonical identity components.
    """

    comp = episode_identity_components(scenario_params, seed)
    digest = stable_hash(comp)
    return f"{prefix}_{digest[:12]}"


__all__ = [
    "canonical_dumps",
    "episode_identity_components",
    "make_episode_id",
    "stable_hash",
]
