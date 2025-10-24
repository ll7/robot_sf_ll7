"""Parse `pytest --durations=N` output into structured JSON.

Feature task T002: Initially the repository only supported parsing from stdin.
We extend the helper to optionally accept ``--input <file>`` while preserving
the original streaming usage. Output schema (list of objects):

```
{
    "test_identifier": str,
    "duration_seconds": float,
    "timestamp": ISO8601 UTC string
}
```

Usage examples:
    pytest --durations=25 -q | python scripts/collect_slow_tests.py > progress/slow_tests_pre.json
    python scripts/collect_slow_tests.py --input pytest_durations.log > progress/slow_tests_pre.json

Notes:
* We collapse multiple phases (setup/call/teardown) keeping the max duration
    per test node id in line with the feature spec requirement.
* Intentionally minimal: any further classification (soft/hard breach) is
    handled in higher-level reporting helpers under ``tests/perf_utils``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import (
    UTC,  # type: ignore[attr-defined]
    datetime,
)

LINE_RE = re.compile(
    r"^(?P<seconds>\d+(?:\.\d+)?)s\s+(?P<phase>call|setup|teardown)\s+(?P<nodeid>.+)$",
)


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
        durations[nodeid] = max(durations[nodeid], secs)
    timestamp = datetime.now(UTC).isoformat()
    return [
        {"test_identifier": k, "duration_seconds": v, "timestamp": timestamp}
        for k, v in sorted(durations.items(), key=lambda kv: kv[1], reverse=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse pytest --durations output to JSON")
    parser.add_argument(
        "--input",
        help="Optional path to a file containing pytest output; otherwise read stdin",
    )
    args = parser.parse_args()

    if args.input:
        with open(args.input, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    else:
        lines = sys.stdin.read().splitlines()
    data = parse(lines)
    json.dump(data, sys.stdout, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
