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
    from pathlib import Path


def canonical_dumps(obj: Any) -> str:
    """Serialize `obj` to a canonical JSON string.

    Ensures:
    - UTF-8 safe characters
    - Sorted keys for mapping objects
    - No whitespace differences (separators set tightly)
    - Recursively processes nested dict/list structures

    Returns:
        Canonical JSON string representation.
    """

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(obj: Any, *, algo: str = "sha256") -> str:
    """Return hex digest of canonical serialization with chosen algorithm.

    Returns:
        Hexadecimal digest string of the canonical representation.
    """

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

    Returns
    -------
    dict
        Dictionary containing scenario, seed, and optional extra components.
    """

    base = {"scenario": scenario_params, "seed": seed}
    if extra:
        base["extra"] = extra
    return base


def make_episode_id(scenario_params: Mapping[str, Any], seed: int, prefix: str = "ep") -> str:
    """Generate deterministic episode id string.

    Format: ``{prefix}_{first12hex}`` where `first12hex` are the first 12 hex
    characters of the sha256 digest of canonical identity components.

    Returns:
        Episode identifier string in format 'prefix_hash'.
    """

    comp = episode_identity_components(scenario_params, seed)
    digest = stable_hash(comp)
    return f"{prefix}_{digest[:12]}"


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from a file path.

    Parameters
    ----------
    path : Path
        Path to a JSON file.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def sha256_file(path: Path) -> str:
    """Return hex digest of file contents using SHA-256.

    Parameters
    ----------
    path : Path
        Path to file to hash.

    Returns
    -------
    str
        Hexadecimal digest string.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a single JSONL file into a list of records.

    Parameters
    ----------
    path : Path
        Path to a JSONL file (one JSON object per line).

    Returns
    -------
    list[dict[str, Any]]
        Parsed records.
    """

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


__all__ = [
    "canonical_dumps",
    "episode_identity_components",
    "load_json",
    "make_episode_id",
    "read_jsonl",
    "sha256_file",
    "stable_hash",
]
