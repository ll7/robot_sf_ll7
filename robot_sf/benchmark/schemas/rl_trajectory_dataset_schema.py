"""Schema helpers for RL trajectory dataset manifests."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.benchmark.rl_trajectory_dataset import (
    RL_TRAJECTORY_DATASET_MANIFEST_SCHEMA_VERSION,
    validate_rl_trajectory_manifest_semantics,
)

SCHEMA_PATH = Path(__file__).with_name("rl_trajectory_dataset_manifest.schema.v1.json")


@lru_cache(maxsize=1)
def load_rl_trajectory_dataset_manifest_schema() -> dict[str, Any]:
    """Load the RL trajectory dataset manifest JSON Schema.

    Returns:
        Manifest schema dictionary.
    """
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _manifest_validator() -> Draft202012Validator:
    return Draft202012Validator(load_rl_trajectory_dataset_manifest_schema())


def validate_rl_trajectory_dataset_manifest(manifest: dict[str, Any]) -> None:
    """Validate manifest JSON Schema and split leakage semantics."""
    if manifest.get("schema_version") != RL_TRAJECTORY_DATASET_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "expected manifest schema_version "
            f"{RL_TRAJECTORY_DATASET_MANIFEST_SCHEMA_VERSION!r}, "
            f"got {manifest.get('schema_version')!r}"
        )
    _manifest_validator().validate(manifest)
    validate_rl_trajectory_manifest_semantics(manifest)


__all__ = [
    "SCHEMA_PATH",
    "load_rl_trajectory_dataset_manifest_schema",
    "validate_rl_trajectory_dataset_manifest",
]
