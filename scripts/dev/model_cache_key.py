#!/usr/bin/env python3
"""Derive the exact-repeat model-cache key from registry-pinned digests.

The determinism-gate and exact-repeat-model-preflight jobs both need one stable
cache key that changes only when the pinned model assets change. This helper is
the single importable implementation of the former inline workflow snippet:
it resolves the config's required model IDs through the registry and hashes
their pinned SHA-256 digests in deterministic order.

Exit codes:
- 0: key printed (machine mode) or derived successfully
- 1: the model registry or preflight path failed (fail closed)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml

from robot_sf.models.preflight import required_model_ids_for_config
from robot_sf.models.registry import get_registry_entry


def derive_model_cache_key(config_path: str | Path) -> str:
    """Return the 16-hex model-cache key for *config_path*.

    Resolves the config's required model IDs, reads each registry entry's pinned
    ``github_release.sha256`` digest, and hashes the ``|``-joined digests in
    registry order. Fails closed when a required model has no pinned digest.
    """
    cfg: Any = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    ids = required_model_ids_for_config(cfg)
    shas: list[str] = []
    for model_id in ids:
        entry = get_registry_entry(model_id)
        digest = entry.get("github_release", {}).get("sha256", "")
        if not digest:
            raise ValueError(f"model {model_id} has no pinned github_release.sha256 digest")
        shas.append(str(digest))
    return hashlib.sha256("|".join(shas).encode()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    """Print the cache key, optionally with a prefix label for workflow steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the PPO algorithm config YAML")
    parser.add_argument(
        "--label",
        default="key",
        help='Output label when not in machine mode (default: "key")',
    )
    parser.add_argument(
        "--machine",
        action="store_true",
        help="Print only the key, for direct use in GitHub Actions outputs",
    )
    args = parser.parse_args(argv)

    try:
        key = derive_model_cache_key(args.config)
    except Exception as exc:  # noqa: BLE001 - fail closed with a readable error
        print(f"ERROR: model-cache key derivation failed: {exc}", file=sys.stderr)
        return 1

    if args.machine:
        print(key)
    else:
        print(f"{args.label}={key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
