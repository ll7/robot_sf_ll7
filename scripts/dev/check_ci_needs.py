#!/usr/bin/env python3
"""Evaluate the CI aggregate job's required-needs results.

The final ``ci`` job in ``.github/workflows/ci.yml`` aggregates every required
job's result. This helper replaces the former handwritten ``if``-per-dependency
block with a declarative manifest: it receives a JSON job-result map and the
GitHub event name, applies the event-specific coverage rules, and exits 1 when
any required job did not finish successfully.

Event-specific rules preserved from the workflow:

- ``coverage-gate`` is required only for non-pull_request events;
- ``changed-coverage-gate`` is required only for pull_request / merge_group;
- all other required jobs are always required.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

REQUIRED_JOBS = (
    "fast-feedback",
    "compat-matrix",
    "fast-pysf-compat",
    "wheel-smoke-install",
    "smoke-artifacts",
    "scenario-validation",
    "xdist-scratch-isolation",
    "examples-smoke",
    "notebooks-smoke",
    "determinism-gate",
    "exact-repeat-model-preflight",
)
EVENT_REQUIRED = {
    "coverage-gate": ("push", "schedule", "workflow_dispatch"),
    "changed-coverage-gate": ("pull_request", "merge_group"),
}


def evaluate_needs(results: dict[str, Any], event_name: str) -> list[str]:
    """Return the failing job names for *results* under *event_name*.

    A job fails when it is required for this event and its result is not
    ``success``. Missing keys are reported as failures.
    """
    failures: list[str] = []
    for job in REQUIRED_JOBS:
        if results.get(job) != "success":
            failures.append(job)
    for job, events in EVENT_REQUIRED.items():
        if event_name not in events:
            continue
        if results.get(job) != "success":
            failures.append(job)
    return failures


def main(argv: list[str] | None = None) -> int:
    """Read the job-result map and exit 1 with the failing jobs on stderr."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        required=True,
        help='JSON object mapping job names to results, e.g. {"fast-feedback": "success"}',
    )
    parser.add_argument("--event-name", required=True, help="GitHub event name (e.g. pull_request)")
    args = parser.parse_args(argv)

    try:
        results = json.loads(args.results)
    except json.JSONDecodeError as exc:
        print(f"ERROR: --results is not valid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(results, dict):
        print("ERROR: --results must be a JSON object", file=sys.stderr)
        return 2

    failures = evaluate_needs(results, args.event_name)
    for job in failures:
        print(f"{job} finished with {results.get(job, 'missing')}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
