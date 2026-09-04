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
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Sample:
    """A single slow-test timing sample.

    Attributes:
        test_identifier: Stable identifier of the test (e.g. ``module::test_name``).
        duration_seconds: Measured duration of the test in seconds.
    """

    test_identifier: str
    duration_seconds: float


class SlowTestCaptureError(ValueError):
    """Raised when a slow-test capture violates the JSON input contract."""


def _invalid(path: Path, message: str) -> SlowTestCaptureError:
    """Build a path-qualified validation error for a capture."""
    return SlowTestCaptureError(f"capture '{path}': {message}")


def _read_json(path: Path) -> Any:
    """Read and decode one capture, converting boundary failures to clean errors."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise _invalid(path, f"unable to read file: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise _invalid(path, f"invalid JSON at line {exc.lineno}, column {exc.colno}") from exc


def _validated_duration(path: Path, index: int, value: Any) -> float:
    """Validate and normalize a JSON duration without accepting booleans or strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(
            path,
            f"sample {index} field 'duration_seconds' must be a finite, "
            "non-negative number (booleans and strings are not accepted)",
        )
    try:
        duration = float(value)
    except (OverflowError, ValueError) as exc:
        raise _invalid(
            path,
            f"sample {index} field 'duration_seconds' must be a finite, non-negative number",
        ) from exc
    if not math.isfinite(duration) or duration < 0:
        raise _invalid(
            path,
            f"sample {index} field 'duration_seconds' must be a finite, non-negative number",
        )
    return duration


def _sample_rows(path: Path, raw: Any) -> list[Any]:
    """Validate the accepted top-level shapes and return their sample rows."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if "samples" not in raw:
            raise _invalid(path, "top level must be a list or contain a 'samples' list")
        data = raw["samples"]
        if not isinstance(data, list):
            raise _invalid(path, "field 'samples' must be a list")
        return data
    raise _invalid(path, "top level must be a list or an object containing a 'samples' list")


def _validated_sample(path: Path, index: int, entry: Any) -> Sample:
    """Validate one row and construct its normalized sample."""
    if not isinstance(entry, dict):
        raise _invalid(path, f"sample {index} must be an object")
    if "test_identifier" not in entry:
        raise _invalid(path, f"sample {index} is missing required field 'test_identifier'")
    identifier = entry["test_identifier"]
    if not isinstance(identifier, str) or not identifier.strip():
        raise _invalid(
            path,
            f"sample {index} field 'test_identifier' must be a non-empty string",
        )
    if "duration_seconds" not in entry:
        raise _invalid(path, f"sample {index} is missing required field 'duration_seconds'")
    return Sample(
        test_identifier=identifier,
        duration_seconds=_validated_duration(path, index, entry["duration_seconds"]),
    )


def load_any(path: Path) -> list[Sample]:
    """Load slow-test timing samples from a JSON capture.

    Accepts either a simple list of sample objects or an object containing a
    ``"samples"`` list. Every row is validated before it is returned; malformed
    rows and duplicate identifiers are rejected instead of being skipped or
    overwritten.

    Args:
        path: Path to the JSON capture file.

    Returns:
        Parsed :class:`Sample` instances in file order.
    """
    data = _sample_rows(path, _read_json(path))
    out: list[Sample] = []
    seen: dict[str, int] = {}
    for index, entry in enumerate(data):
        sample = _validated_sample(path, index, entry)
        if sample.test_identifier in seen:
            raise _invalid(
                path,
                f"sample {index} has duplicate test_identifier {sample.test_identifier!r} "
                f"from sample {seen[sample.test_identifier]}; duplicate identifiers are rejected",
            )
        seen[sample.test_identifier] = index
        out.append(sample)
    return out


def index_by(samples: list[Sample]) -> dict[str, float]:
    """Index timing samples by their test identifier.

    Args:
        samples: Timing samples to index.

    Returns:
        Mapping of test identifier to duration in seconds.
    """
    indexed: dict[str, float] = {}
    for sample in samples:
        if sample.test_identifier in indexed:
            raise SlowTestCaptureError(
                f"duplicate test_identifier {sample.test_identifier!r}; "
                "duplicate identifiers are rejected",
            )
        indexed[sample.test_identifier] = sample.duration_seconds
    return indexed


def main(argv=None) -> int:
    """Compare two slow-test captures and print a Markdown summary.

    Loads the before/after captures, computes per-test duration deltas over the
    common tests, and prints top regressions and improvements sorted by absolute
    delta.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code; ``0`` on success.
    """
    p = argparse.ArgumentParser(description="Compare slow test duration captures")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    args = p.parse_args(argv)
    try:
        before = index_by(load_any(Path(args.before)))
        after = index_by(load_any(Path(args.after)))
    except SlowTestCaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
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
