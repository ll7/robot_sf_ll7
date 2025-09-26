"""Schema validation utilities for SNQI related JSON outputs.

This module intentionally avoids external dependencies (e.g. jsonschema)
to keep runtime lightweight inside benchmark scripts. Validation is
structural and type‑focused with optional numeric finiteness checks.

Three high‑level kinds are currently supported:
  - optimization   (output of scripts/snqi_weight_optimization.py)
  - recompute      (output of scripts/recompute_snqi_weights.py)
  - sensitivity    (output directory JSONs from scripts/snqi_sensitivity_analysis.py)

Each output must embed a top‑level `_metadata` object with at least:
  schema_version: int (currently 1)
  generated_at: ISO8601 datetime string
  git_commit: str
  seed: int | None
  provenance: dict (opaque – minimally checked)

Public entry points:
  validate_snqi(obj: dict, kind: str) -> None
  assert_all_finite(obj: Any, path: str = "$") -> None

If validation fails a ValueError is raised with a concise explanation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

EXPECTED_SCHEMA_VERSION = 1


@dataclass
class _FieldSpec:
    name: str
    required: bool = True


def _expect_keys(d: Mapping[str, Any], specs: Iterable[_FieldSpec], ctx: str) -> None:
    for spec in specs:
        if spec.required and spec.name not in d:
            raise ValueError(f"Missing required key '{spec.name}' in {ctx}")


def _is_number(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def assert_all_finite(obj: Any, path: str = "$") -> None:
    """Recursively assert all numeric leaves are finite (no NaN/inf).

    Non‑numeric values are ignored. Raises ValueError on first failure.
    """
    import math

    if _is_number(obj):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"Non‑finite numeric value at {path}: {obj}")
        return
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            assert_all_finite(v, f"{path}.{k}")
    elif isinstance(obj, list | tuple):
        for i, v in enumerate(obj):
            assert_all_finite(v, f"{path}[{i}]")


def _validate_metadata(meta: Mapping[str, Any]) -> None:
    _expect_keys(
        meta,
        [
            _FieldSpec("schema_version"),
            _FieldSpec("generated_at"),
            _FieldSpec("git_commit"),
            _FieldSpec("seed"),
            _FieldSpec("provenance"),
        ],
        "_metadata",
    )
    version = meta.get("schema_version")
    if version != EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {version}; expected {EXPECTED_SCHEMA_VERSION}",
        )
    if not isinstance(meta.get("provenance"), Mapping):  # minimal structural check
        raise ValueError("_metadata.provenance must be a mapping")


def _validate_optimization(obj: Mapping[str, Any]) -> None:
    # Required top‑level keys (besides _metadata)
    required = ["recommended"]
    for key in required:
        if key not in obj:
            raise ValueError(f"Optimization output missing '{key}'")
    rec = obj["recommended"]
    if not isinstance(rec, Mapping) or "weights" not in rec:
        raise ValueError("'recommended' must be a mapping containing 'weights'")
    if not isinstance(rec["weights"], Mapping):
        raise ValueError("'recommended.weights' must be a mapping")
    # Light numeric sanity check
    for w_name, val in rec["weights"].items():
        if not _is_number(val):  # pragma: no cover - defensive
            raise ValueError(f"Weight '{w_name}' is not numeric: {val}")


def _validate_recompute(obj: Mapping[str, Any]) -> None:
    if "recommended_weights" not in obj:
        raise ValueError("Recompute output missing 'recommended_weights'")
    weights = obj["recommended_weights"]
    if not isinstance(weights, Mapping):
        raise ValueError("'recommended_weights' must be a mapping")
    for w_name, val in weights.items():
        if not _is_number(val):
            raise ValueError(f"Weight '{w_name}' is not numeric: {val}")


def _validate_sensitivity(obj: Mapping[str, Any]) -> None:
    # Expect at least one analysis block
    expected_any = ["weight_sweep", "pairwise", "ablation", "normalization"]
    if not any(k in obj for k in expected_any):
        raise ValueError("Sensitivity output missing analysis sections")
    # Nothing more structural for now; detailed shapes are large.


def validate_snqi(obj: Mapping[str, Any], kind: str, *, check_finite: bool = True) -> None:
    """Validate an SNQI JSON object.

    Args:
        obj: Parsed JSON object (dict)
        kind: One of 'optimization', 'recompute', 'sensitivity'
        check_finite: If True, enforce numeric finiteness across entire structure
    Raises:
        ValueError: On validation failure.
    """
    if not isinstance(obj, Mapping):
        raise ValueError("Root JSON value must be an object")
    if "_metadata" not in obj:
        raise ValueError("Missing top‑level '_metadata'")
    _validate_metadata(obj["_metadata"])

    dispatch = {
        "optimization": _validate_optimization,
        "recompute": _validate_recompute,
        "sensitivity": _validate_sensitivity,
    }
    if kind not in dispatch:
        raise ValueError(f"Unknown validation kind '{kind}'")
    dispatch[kind](obj)

    if check_finite:
        assert_all_finite(obj)


__all__ = ["EXPECTED_SCHEMA_VERSION", "assert_all_finite", "validate_snqi"]
