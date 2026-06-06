"""Classify whether a trace pair contains nonzero mechanism-relevant signal."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_MECHANISM_KEY_FRAGMENTS = (
    "mechanism",
    "ammv",
    "static_recenter",
    "topology",
    "corridor_subgoal",
    "recovery",
)
_ACTIVATION_KEY_FRAGMENTS = ("activation", "activated")
_OUTCOME_KEYS = ("outcome", "terminal_outcome", "termination_reason", "status")
_NUMERIC_EPS = 1e-9


def _load_payload(path: Path) -> Any:
    """Load a JSON or JSONL trace payload."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Trace input is empty: {path}")
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        if len(records) == 1:
            return records[0]
        return records
    return json.loads(text)


def _frames(payload: Any) -> list[dict[str, Any]]:
    """Return simulation-trace frames when present."""
    if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
        return [frame for frame in payload["frames"] if isinstance(frame, dict)]
    if isinstance(payload, list):
        return [frame for frame in payload if isinstance(frame, dict)]
    return []


def _normalize_scalar(value: Any) -> Any:
    """Normalize scalar values for stable recursive comparison."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def _canonical(value: Any) -> Any:
    """Return a JSON-like canonical form for recursive equality checks."""
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return _normalize_scalar(value)


def _values_equal(left: Any, right: Any) -> bool:
    """Return whether two values are equal within numeric tolerance."""
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=_NUMERIC_EPS)
    return _canonical(left) == _canonical(right)


def _path_contains(path: str, fragments: tuple[str, ...]) -> bool:
    """Return true if any lowercase fragment is present in a dotted path."""
    lowered = path.lower()
    return any(fragment in lowered for fragment in fragments)


def _iter_paths(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Return recursive paths and values from a nested JSON-like object."""
    paths = [(prefix, value)] if prefix else []
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_iter_paths(item, child_prefix))
        return paths
    if isinstance(value, list):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            paths.extend(_iter_paths(item, child_prefix))
        return paths
    return paths


def _selected_actions(frames: list[dict[str, Any]]) -> list[Any]:
    """Extract planner selected actions from trace frames."""
    actions: list[Any] = []
    for frame in frames:
        planner = frame.get("planner")
        if isinstance(planner, dict):
            actions.append(planner.get("selected_action"))
    return actions


def _trajectory_points(frames: list[dict[str, Any]]) -> list[Any]:
    """Extract robot and pedestrian position samples from trace frames."""
    points: list[Any] = []
    for frame in frames:
        robot = frame.get("robot")
        pedestrians = frame.get("pedestrians")
        points.append(
            {
                "robot_position": robot.get("position") if isinstance(robot, dict) else None,
                "pedestrians": [
                    ped.get("position") if isinstance(ped, dict) else None for ped in pedestrians
                ]
                if isinstance(pedestrians, list)
                else None,
            }
        )
    return points


def _leaf_delta_for_fragments(
    baseline: Any,
    intervention: Any,
    fragments: tuple[str, ...],
) -> bool:
    """Return true when any matching leaf path differs across payloads."""
    left = {path: value for path, value in _iter_paths(baseline) if _path_contains(path, fragments)}
    right = {
        path: value for path, value in _iter_paths(intervention) if _path_contains(path, fragments)
    }
    for path in sorted(set(left) | set(right)):
        if not _values_equal(left.get(path), right.get(path)):
            return True
    return False


def _outcome_delta(baseline: Any, intervention: Any) -> bool:
    """Return true when common outcome/status surfaces differ."""
    if not isinstance(baseline, dict) or not isinstance(intervention, dict):
        return False
    for key in _OUTCOME_KEYS:
        if key in baseline or key in intervention:
            if not _values_equal(baseline.get(key), intervention.get(key)):
                return True
    return False


def classify_mechanism_signal(baseline: Any, intervention: Any) -> dict[str, Any]:
    """Classify nonzero mechanism-signal fields for a baseline/intervention trace pair."""
    baseline_frames = _frames(baseline)
    intervention_frames = _frames(intervention)

    trajectory_delta = not _values_equal(
        _trajectory_points(baseline_frames),
        _trajectory_points(intervention_frames),
    )
    command_delta = not _values_equal(
        _selected_actions(baseline_frames),
        _selected_actions(intervention_frames),
    )
    mechanism_delta = _leaf_delta_for_fragments(
        baseline,
        intervention,
        _MECHANISM_KEY_FRAGMENTS,
    )
    activation_delta = _leaf_delta_for_fragments(
        baseline,
        intervention,
        _ACTIVATION_KEY_FRAGMENTS,
    )
    outcome_delta = _outcome_delta(baseline, intervention)

    if mechanism_delta or activation_delta:
        classification = "mechanism_difference_candidate"
    elif trajectory_delta or command_delta or outcome_delta:
        classification = "qualitative_illustration"
    else:
        classification = "rendering_sanity"

    return {
        "mechanism_signal": {
            "schema_version": "mechanism_signal_check.v1",
            "trajectory_delta_nonzero": trajectory_delta,
            "command_delta_nonzero": command_delta,
            "mechanism_field_delta_nonzero": mechanism_delta,
            "activation_delta_nonzero": activation_delta,
            "outcome_delta_nonzero": outcome_delta,
            "classification": classification,
            "claim_boundary": (
                "Routing guard only; nonzero signal does not establish planner superiority, "
                "transfer, benchmark success, or paper-grade mechanism proof."
            ),
        }
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--intervention-trace", type=Path, required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the JSON classification payload.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the mechanism-signal checker."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = classify_mechanism_signal(
        _load_payload(args.baseline_trace),
        _load_payload(args.intervention_trace),
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
