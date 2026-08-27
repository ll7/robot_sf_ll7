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
CHANGED_COVERAGE_EVENTS = ("pull_request", "merge_group")


def normalize_needs(raw_results: dict[str, Any]) -> dict[str, Any]:
    """Normalize GitHub's ``toJSON(needs)`` object to a job-result mapping.

    GitHub serializes each dependency as an object containing ``result`` and
    ``outputs``. Accept the already-flat mapping too so the helper remains easy
    to call and test outside Actions.
    """
    return {
        job: payload.get("result") if isinstance(payload, dict) else payload
        for job, payload in raw_results.items()
    }


def evaluate_needs(
    results: dict[str, Any], event_name: str, *, treat_cancelled_as_superseded: bool = False
) -> list[str]:
    """Return the failing job names for *results* under *event_name*.

    A job fails when it is required for this event and its result is not
    ``success``. Missing keys are reported as failures.

    ``treat_cancelled_as_superseded`` exempts ``cancelled`` results from the
    failure set. A dependency becomes ``cancelled`` only when the workflow run
    itself was cancelled (latest-main-wins supersession or manual cancel), never
    from an ordinary job failure, so the aggregate ``ci`` job must not turn a
    superseded run red.
    """
    failures: list[str] = []
    for job in REQUIRED_JOBS:
        result = results.get(job)
        if result == "cancelled" and treat_cancelled_as_superseded:
            continue
        if result != "success":
            failures.append(job)
    # Preserve the former workflow's fail-closed expression exactly:
    # coverage-gate applies to every event except pull_request, including
    # merge_group and any future/unknown event name.
    coverage_gate = results.get("coverage-gate")
    if (
        event_name != "pull_request"
        and not (coverage_gate == "cancelled" and treat_cancelled_as_superseded)
        and coverage_gate != "success"
    ):
        failures.append("coverage-gate")
    changed_coverage = results.get("changed-coverage-gate")
    if event_name in CHANGED_COVERAGE_EVENTS and (
        not (changed_coverage == "cancelled" and treat_cancelled_as_superseded)
        and changed_coverage != "success"
    ):
        failures.append("changed-coverage-gate")
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
    parser.add_argument(
        "--treat-cancelled-as-superseded",
        action="store_true",
        help=(
            "Treat cancelled dependencies as a superseded run instead of a "
            "failure (latest-main-wins cancels in-progress runs)."
        ),
    )
    args = parser.parse_args(argv)

    try:
        results = json.loads(args.results)
    except json.JSONDecodeError as exc:
        print(f"ERROR: --results is not valid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(results, dict):
        print("ERROR: --results must be a JSON object", file=sys.stderr)
        return 2

    results = normalize_needs(results)
    failures = evaluate_needs(
        results, args.event_name, treat_cancelled_as_superseded=args.treat_cancelled_as_superseded
    )
    for job in failures:
        print(f"{job} finished with {results.get(job, 'missing')}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
