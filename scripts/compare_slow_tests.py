#!/usr/bin/env python
"""Compare before/after slow test JSON captures (T023).

Usage:
  python scripts/compare_slow_tests.py --before progress/slow_tests_pre.json \
      --after progress/slow_tests_post.json

The JSON files may be either list-of-samples (simple form from collect_slow_tests.py)
or an object with a "samples" list.

Outputs a markdown summary to stdout including:
  * Count of tests whose duration increased/decreased
  * Top regressions (largest positive delta)
  * Top improvements (largest negative delta)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    """TODO docstring. Document this class."""

    test_identifier: str
    duration_seconds: float


def load_any(path: Path) -> list[Sample]:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.

    Returns:
        TODO docstring.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    data = raw["samples"] if isinstance(raw, dict) and "samples" in raw else raw
    out: list[Sample] = []
    for entry in data:
        try:
            out.append(
                Sample(
                    test_identifier=entry["test_identifier"],
                    duration_seconds=float(entry["duration_seconds"]),
                ),
            )
        except KeyError:
            continue
    return out


def index_by(samples: list[Sample]) -> dict[str, float]:
    """TODO docstring. Document this function.

    Args:
        samples: TODO docstring.

    Returns:
        TODO docstring.
    """
    return {s.test_identifier: s.duration_seconds for s in samples}


def main(argv=None) -> int:
    """TODO docstring. Document this function.

    Args:
        argv: TODO docstring.

    Returns:
        TODO docstring.
    """
    p = argparse.ArgumentParser(description="Compare slow test duration captures")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    args = p.parse_args(argv)
    before = index_by(load_any(Path(args.before)))
    after = index_by(load_any(Path(args.after)))
    deltas = []
    for test_id, new_dur in after.items():
        old_dur = before.get(test_id)
        if old_dur is None:
            continue
        deltas.append((test_id, new_dur - old_dur))
    deltas.sort(key=lambda kv: abs(kv[1]), reverse=True)
    regressions = [d for d in deltas if d[1] > 0]
    improvements = [d for d in deltas if d[1] < 0]
    lines = ["# Slow Test Duration Comparison", "", f"Compared {len(deltas)} common tests"]
    if regressions:
        lines.append("## Top Regressions")
        for tid, delta in regressions[:10]:
            lines.append(f"- {tid}: +{delta:.3f}s")
    if improvements:
        lines.append("\n## Top Improvements")
        for tid, delta in improvements[:10]:
            lines.append(f"- {tid}: {delta:.3f}s")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
