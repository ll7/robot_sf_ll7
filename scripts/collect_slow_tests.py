"""Utility: Parse `pytest --durations=N` output from stdin into structured JSON.

Usage:
    pytest --durations=25 -q | python scripts/collect_slow_tests.py > progress/slow_tests_raw.json

The script reads all stdin, searches for lines matching a pytest durations table entry and outputs a JSON array
with objects: {"test_identifier": str, "duration_seconds": float}.

Heuristic pattern examples (pytest output lines):
    0.45s call     tests/path/test_module.py::test_case
    12.34s setup   tests/benchmark_full/test_big.py::test_scenario

We ignore phase (call/setup/teardown) and keep the max duration per test_identifier.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

LINE_RE = re.compile(r"^(?P<seconds>\d+\.\d+)s\s+(?P<phase>call|setup|teardown)\s+(?P<nodeid>.+)$")


def parse(lines: list[str]) -> list[dict[str, object]]:
    """Parse pytest duration lines.

    Collapses multiple phases per test keeping the max duration.
    """
    durations: dict[str, float] = defaultdict(float)
    for line in lines:
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        secs = float(m.group("seconds"))
        nodeid = m.group("nodeid").strip()
        if secs > durations[nodeid]:
            durations[nodeid] = secs
    timestamp = datetime.now(timezone.utc).isoformat()
    return [
        {"test_identifier": k, "duration_seconds": v, "timestamp": timestamp}
        for k, v in sorted(durations.items(), key=lambda kv: kv[1], reverse=True)
    ]


def main() -> None:
    content = sys.stdin.read().splitlines()
    data = parse(content)
    json.dump(data, sys.stdout, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
