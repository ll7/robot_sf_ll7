#!/usr/bin/env python3
"""AMV timeout trace analyzer for issue #2308.
Computes time-series metrics and classifies timeout driver.
"""

import json
import sys
from pathlib import Path
from typing import Any


def _numeric_list(value: Any) -> list[float] | None:
    """Normalize a JSON value to a numeric list when possible."""
    if not isinstance(value, list):
        return None
    normalized = []
    for item in value:
        if not isinstance(item, int | float) or isinstance(item, bool):
            return None
        normalized.append(float(item))
    return normalized


def _recorded_bool(record: dict, *keys: str) -> bool | None:
    """Return the first explicit boolean value recorded under any key."""
    for key in keys:
        value = record.get(key)
        if isinstance(value, bool):
            return value
    return None


def _scoring_slowed_progress(record: dict, out: dict) -> bool | None:
    """Conservatively infer whether actuation-aware scoring slowed route progress."""
    recorded = _recorded_bool(
        record,
        "whether_actuation_aware_scoring_slowed_progress",
        "actuation_aware_scoring_slowed_progress",
    )
    if recorded is not None:
        return recorded
    progress = out["progress_over_time"]
    speeds = out["command_speed_profile"]
    if not progress or not speeds:
        return None
    stalled = max(progress) - min(progress) < 1e-3
    conservative_commands = max(speeds) < 0.05
    return stalled and conservative_commands


def analyze_amv(record: dict) -> dict:
    """Extract AMV timeout fields and classify the local timeout driver."""
    out = {
        "progress_over_time": None,
        "clipping_over_time": None,
        "saturation_over_time": None,
        "command_speed_profile": None,
        "timeout_driver": None,
        "whether_actuation_aware_scoring_slowed_progress": None,
    }
    out["progress_over_time"] = _numeric_list(record.get("progress"))
    out["clipping_over_time"] = _numeric_list(record.get("clipping"))
    out["saturation_over_time"] = _numeric_list(record.get("saturation"))
    out["command_speed_profile"] = _numeric_list(record.get("command_speeds"))
    # simple heuristics
    try:
        if out["command_speed_profile"] and max(out["command_speed_profile"]) < 0.05:
            out["timeout_driver"] = "commands_too_conservative"
        elif (
            out["progress_over_time"]
            and max(out["progress_over_time"]) - min(out["progress_over_time"]) < 1e-3
        ):
            out["timeout_driver"] = "route_progress_stalled"
        else:
            out["timeout_driver"] = "other_or_unclassified"
    except Exception:
        out["timeout_driver"] = "unknown"
    out["whether_actuation_aware_scoring_slowed_progress"] = _scoring_slowed_progress(record, out)
    return out


def main() -> None:
    """Run the AMV timeout analyzer CLI."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(2)
    res = []
    with p.open() as f:
        for i, line in enumerate(f):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out = analyze_amv(rec)
            out["_row_index"] = i
            res.append(out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
