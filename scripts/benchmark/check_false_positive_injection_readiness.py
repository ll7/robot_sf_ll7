#!/usr/bin/env python3
"""Fail-closed readiness check for a false-positive actor-injection replay condition.

Validates that a replay-condition spec (YAML or JSON) carries the injected-actor
inputs and provenance fields required before a false-positive actor-injection
replay can run.  This is the readiness slice for issue #3300; it does not run a
replay campaign, change sensor semantics, or make any benchmark/safety claim.

Exit codes:

- ``0``  ``ready`` — inputs and provenance are valid.
- ``0``  ``not_available`` — no false-positive actors requested (accepted-unavailable).
- ``3``  ``blocked`` — malformed inputs or missing provenance (actionable blocker).

Usage::

    uv run python scripts/benchmark/check_false_positive_injection_readiness.py \\
        tests/fixtures/benchmark/false_positive_actor_injection/replay_condition_ready.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import yaml

from robot_sf.benchmark.false_positive_injection_readiness import (
    STATUS_BLOCKED,
    FalsePositiveInjectionReadiness,
    check_false_positive_injection_readiness,
)

#: Exit code returned when the condition fails closed (blocked).
BLOCKED_EXIT_CODE = 3


def _load_spec(path: pathlib.Path) -> dict[str, Any]:
    """Load a replay-condition spec from a YAML or JSON file.

    A top-level ``false_positive_injection`` mapping is unwrapped when present so
    the spec can live beside other condition configuration.

    Returns:
        The replay-condition mapping.

    Raises:
        ValueError: When the file cannot be read/parsed or does not parse to a
            mapping.  Callers convert this into a ``blocked`` verdict so loader
            failures honor the same fail-closed contract as malformed specs.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"failed to load spec file {path}: {exc}") from exc
    if isinstance(data, dict) and isinstance(data.get("false_positive_injection"), dict):
        data = data["false_positive_injection"]
    if not isinstance(data, dict):
        raise ValueError(f"spec file {path} must contain a mapping, got {type(data).__name__}")
    return data


def main(argv: list[str] | None = None) -> int:
    """Check one replay-condition spec and print its readiness verdict.

    Returns:
        Process exit code (``BLOCKED_EXIT_CODE`` when blocked, else ``0``).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed readiness check for false-positive actor-injection replay inputs "
            "and provenance (issue #3300). Does not run a replay or make benchmark claims."
        )
    )
    parser.add_argument(
        "spec_path",
        type=str,
        help="Path to a YAML/JSON replay-condition spec.",
    )
    args = parser.parse_args(argv)

    spec_path = pathlib.Path(args.spec_path)
    if not spec_path.exists():
        parser.error(f"spec file not found: {spec_path}")

    try:
        spec = _load_spec(spec_path)
    except ValueError as exc:
        # Loader failures must fail closed like any other blocked spec rather than
        # escaping as a traceback, so the documented exit-code 3 contract holds.
        readiness = FalsePositiveInjectionReadiness(status=STATUS_BLOCKED, blockers=[str(exc)])
        print(json.dumps(readiness.to_dict(), indent=2, sort_keys=True))
        return BLOCKED_EXIT_CODE

    readiness = check_false_positive_injection_readiness(spec)
    print(json.dumps(readiness.to_dict(), indent=2, sort_keys=True))

    if readiness.status == STATUS_BLOCKED:
        return BLOCKED_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
