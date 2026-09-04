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
import math
import re
import sys
from datetime import (
    UTC,  # type: ignore[attr-defined]
    datetime,
)
from pathlib import Path

LINE_RE = re.compile(
    r"^(?P<seconds>\d+(?:\.\d+)?)s\s+(?P<phase>call|setup|teardown)\s+(?P<nodeid>.+)$",
)


class SlowTestCollectionError(ValueError):
    """Raised when a pytest duration input cannot produce a valid capture."""


def parse(lines: list[str]) -> list[dict[str, object]]:
    """Parse pytest duration lines.

    Collapses multiple phases per test keeping the max duration. Non-duration
    pytest output remains ignored, while a duration line that would produce an
    invalid JSON sample fails closed.
    """
    durations: dict[str, float] = {}
    for line_number, line in enumerate(lines, start=1):
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        secs = float(m.group("seconds"))
        nodeid = m.group("nodeid").strip()
        if not nodeid:
            raise SlowTestCollectionError(
                f"line {line_number} has an empty test_identifier; expected a pytest node id",
            )
        if not math.isfinite(secs) or secs < 0:
            raise SlowTestCollectionError(
                f"line {line_number} field 'duration_seconds' must be a finite, "
                "non-negative number",
            )
        durations[nodeid] = max(durations.get(nodeid, 0.0), secs)
    timestamp = datetime.now(UTC).isoformat()
    return [
        {"test_identifier": k, "duration_seconds": v, "timestamp": timestamp}
        for k, v in sorted(durations.items(), key=lambda kv: kv[1], reverse=True)
    ]


def _read_lines(input_path: str | None) -> list[str]:
    """Read pytest output from a file or stdin with a bounded error contract."""
    try:
        if input_path:
            return Path(input_path).read_text(encoding="utf-8").splitlines()
        return sys.stdin.read().splitlines()
    except (OSError, UnicodeError) as exc:
        source = f"input file '{input_path}'" if input_path else "stdin"
        raise SlowTestCollectionError(f"unable to read {source}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the slow tests collector.

    Parses pytest duration output and writes structured JSON to stdout.
    Reads from stdin by default or from a specified input file.
    """
    parser = argparse.ArgumentParser(description="Parse pytest --durations output to JSON")
    parser.add_argument(
        "--input",
        help="Optional path to a file containing pytest output; otherwise read stdin",
    )
    args = parser.parse_args(argv)
    try:
        data = parse(_read_lines(args.input))
    except SlowTestCollectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    json.dump(data, sys.stdout, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
