#!/usr/bin/env python3
"""AMV timeout trace analyzer for issue #2308.
Computes time-series metrics and classifies timeout driver.
"""

import json
import sys
from pathlib import Path


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
    prog = record.get("progress")
    out["progress_over_time"] = prog if isinstance(prog, list) else None
    out["clipping_over_time"] = record.get("clipping")
    out["saturation_over_time"] = record.get("saturation")
    out["command_speed_profile"] = record.get("command_speeds")
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
