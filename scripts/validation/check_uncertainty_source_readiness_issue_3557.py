#!/usr/bin/env python3
"""Print issue #3557 uncertainty-source episode-run readiness inventory."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_READINESS_MODULE_PATH = _REPO_ROOT / "robot_sf" / "benchmark" / "uncertainty_source_readiness.py"
_READINESS_SPEC = importlib.util.spec_from_file_location(
    "_issue_3557_uncertainty_source_readiness", _READINESS_MODULE_PATH
)
if _READINESS_SPEC is None or _READINESS_SPEC.loader is None:
    raise ImportError(f"cannot load {_READINESS_MODULE_PATH}")
_READINESS_MODULE = importlib.util.module_from_spec(_READINESS_SPEC)
sys.modules[_READINESS_SPEC.name] = _READINESS_MODULE
_READINESS_SPEC.loader.exec_module(_READINESS_MODULE)

CLASS_PROBABILITY_AMBIGUITY = _READINESS_MODULE.CLASS_PROBABILITY_AMBIGUITY
COVARIANCE_INFLATION = _READINESS_MODULE.COVARIANCE_INFLATION
DEFAULT_SOURCE_SPECS = _READINESS_MODULE.DEFAULT_SOURCE_SPECS
EXISTENCE_DEGRADATION = _READINESS_MODULE.EXISTENCE_DEGRADATION
MISSING_CONDITION_BUILDER = _READINESS_MODULE.MISSING_CONDITION_BUILDER
MISSING_SURROGATE_OUTPUT = _READINESS_MODULE.MISSING_SURROGATE_OUTPUT
SOURCE_READY = _READINESS_MODULE.SOURCE_READY
TRACKING_NOISE = _READINESS_MODULE.TRACKING_NOISE
VISIBILITY_OCCLUSION = _READINESS_MODULE.VISIBILITY_OCCLUSION
SourceReadinessSpec = _READINESS_MODULE.SourceReadinessSpec
inspect_uncertainty_source_readiness = _READINESS_MODULE.inspect_uncertainty_source_readiness


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect condition-builder, scenario-hook, and surrogate-output readiness "
            "for issue #3557 uncertainty-source episode runs."
        )
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--human", action="store_true", help="emit compact human-readable text")
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="exit 1 when any uncertainty source is missing a readiness prerequisite",
    )
    return parser.parse_args(argv)


def _render_human(payload: dict[str, object]) -> str:
    """Render a compact human-readable inventory."""

    lines = [
        f"issue #{payload['issue']} uncertainty-source readiness",
        f"schema: {payload['schema_version']}",
        f"ready sources: {', '.join(payload['ready_sources']) or '(none)'}",
        f"blocked sources: {', '.join(payload['blocked_sources']) or '(none)'}",
        "",
    ]
    for row in payload["sources"]:
        lines.append(f"- {row['source']}: {row['status']}")
        lines.append(f"  condition_builder: {row['condition_builder']['evidence']}")
        lines.append(f"  scenario_hook: {row['scenario_hook']['evidence']}")
        lines.append(
            f"  expected_surrogate_outputs: {row['expected_surrogate_outputs']['evidence']}"
        )
    lines.append("")
    lines.append(str(payload["claim_boundary"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the readiness inventory.

    Returns:
        Exit code 1 only when ``--fail-on-blocked`` is set and a source is blocked.
    """

    args = _parse_args(argv)
    payload = inspect_uncertainty_source_readiness().as_dict()
    if args.human:
        print(_render_human(payload))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if args.fail_on_blocked and not payload["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
