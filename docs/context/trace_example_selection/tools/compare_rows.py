"""Compare two serialized episode rows field by field (reconstruction contract).

    python compare_rows.py <episodes.jsonl A> <episodes.jsonl B>

Prints every flattened field whose value differs and exits 1 when a field outside the
allowed set differs. Allowed (legitimately run-specific): timestamps.start,
timestamps.end, timing.steps_per_second, wall_time_sec.
"""

from __future__ import annotations

import json
import sys
from typing import Any

ALLOWED = {"timestamps.start", "timestamps.end", "timing.steps_per_second", "wall_time_sec"}


def flat(d: Any, p: str = "") -> dict[str, Any]:
    """Flatten nested mappings and lists into dotted paths.

    Returns:
        Path to scalar mapping.
    """
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k in sorted(d):
            out.update(flat(d[k], f"{p}.{k}" if p else str(k)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flat(v, f"{p}[{i}]"))
        if not d:
            out[p] = "[]"
    else:
        out[p] = d
    return out


def main() -> int:
    """Compare and report.

    Returns:
        0 when only allowed fields differ, else 1.
    """
    a = flat(json.loads(open(sys.argv[1], encoding="utf-8").readline()))
    b = flat(json.loads(open(sys.argv[2], encoding="utf-8").readline()))
    diff = sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))
    bad = [k for k in diff if k not in ALLOWED]
    print(
        json.dumps(
            {"fields_a": len(a), "fields_b": len(b), "differing": diff, "outside_allowed": bad},
            sort_keys=True,
        )
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
