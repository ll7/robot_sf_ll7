#!/usr/bin/env python3
"""Report whether ``main`` CI is green, based on the latest COMPLETED run.

Motivation (issue #5385). Three separate main-red incidents in 36h
(2026-07-11/12) shared one mechanism: merges kept landing while main CI was
red, and the already-failing required check masked the NEW breakage each merge
introduced, so recovery cost grew with every merge that landed inside the red
window. The cure is a merge hold: gates may review a PR while main is red but
must not merge (except the unbreak-main fix itself) until main is green again.

This helper is the deterministic green/red signal that hold consults. The one
rule that matters — learned the hard way when the escalation guard stayed
silent on 2026-07-11 — is that an IN-PROGRESS run must never count as evidence
either way: only the most recent *completed* run decides. The fetch uses
``--status completed`` so in-progress runs are excluded at the API, and the
pure decision function filters defensively on top so the rule is unit-tested.

Exit code: 0 == green (latest completed CI run on main concluded ``success``),
1 == not green (red, or no completed run to judge from). Prints the run id and
conclusion it decided from. The ``--json`` flag emits the machine-readable
main-signal schema (``main_ci_is_green.v1``) with the same green/red/stale
classification the gate contract consumes, so the gate need not parse the
human line.

The gate's default fetch is deliberately one small bounded window (a single
``gh run list --limit`` call, default 5 runs, 30s timeout): the merge hold must
stay cheap and quick, and a cancellation-saturated window fails closed to
``stale`` rather than blocking on a slower search.  :func:`fetch_run_window`
is the paginated REST reader for callers that need the decisive verdict behind
a cancelled-run flood; :mod:`main_ci_incident_reconcile` uses it.  This module
owns both fetch paths so there is exactly one pagination implementation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple
from urllib.parse import quote, urlencode

from scripts.dev._gh_rest import parse_json, run_gh_api

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_WORKFLOW = "CI"

# Bounded pagination budget for the decisive-run REST search.  GitHub's
# latest-main-wins concurrency can fill whole raw pages with ``cancelled``
# runs, so a raw ``--limit`` does not identify a sufficient evidence window.
# The reader examines at most ``max_pages`` full pages before reporting the
# window as exhausted; hitting that bound means "no verdict found", never red.
REST_PAGE_SIZE = 100
DEFAULT_MAX_PAGES = 10
REST_RUN_TIMEOUT = 90


def _gh(args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a GitHub CLI command (mirrors scripts/dev/compact_ci_snapshot.py)."""
    try:
        return subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=["gh", *args], returncode=124, stdout="", stderr="gh timed out"
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            args=["gh", *args],
            returncode=127,
            stdout="",
            stderr=f"gh not executable: {exc}",
        )


class MainCiRunFetchError(RuntimeError):
    """Raised when the bounded paginated main-CI run window cannot be read."""


class MainCiRunWindow(NamedTuple):
    """Result of a bounded paginated main-CI run read.

    ``window_exhausted`` is True when the page budget ran out before
    ``stop_after_decisive`` decisive runs (or a complete short page) were seen.
    Callers must fail closed on an exhausted window instead of treating the
    partial read as complete history.
    """

    runs: list[dict[str, Any]]
    window_exhausted: bool


def _default_rest_runner(
    path: str,
    payload: object | None = None,
    *,
    method: str | None = None,
    extra_args: list[str] | None = None,
) -> Any:
    """Run one REST request through the shared JSON-stdin transport."""
    return run_gh_api(
        path,
        payload,
        method=method,
        extra_args=extra_args,
        timeout=REST_RUN_TIMEOUT,
        timeout_context="main-CI run window was not verified",
    )


def _rest_json(
    path: str,
    *,
    runner: Callable[..., Any],
    operation: str,
) -> Any:
    """Call a REST endpoint and fail closed on transport or JSON errors."""
    result = runner(path, None, method=None, extra_args=None)
    data, error = parse_json(result, what=operation)
    if error:
        raise MainCiRunFetchError(error)
    return data


def resolve_workflow_selector(
    *,
    repo: str,
    workflow: str,
    max_pages: int,
    runner: Callable[..., Any],
) -> str:
    """Resolve a workflow display name to a stable REST workflow selector."""
    selector = workflow.strip()
    if not selector:
        raise MainCiRunFetchError("workflow must not be empty")
    if selector.isdecimal() or selector.lower().endswith((".yml", ".yaml")):
        return selector

    rows: list[Mapping[str, Any]] = []
    for page in range(1, max_pages + 1):
        endpoint = (
            f"repos/{quote(repo, safe='/')}/actions/workflows?per_page={REST_PAGE_SIZE}&page={page}"
        )
        payload = _rest_json(
            endpoint,
            runner=runner,
            operation="Actions workflow inventory",
        )
        if not isinstance(payload, Mapping):
            raise MainCiRunFetchError("Actions workflow inventory returned a non-object payload")
        page_rows = payload.get("workflows")
        if not isinstance(page_rows, list) or any(
            not isinstance(row, Mapping) for row in page_rows
        ):
            raise MainCiRunFetchError("Actions workflow inventory returned malformed rows")
        rows.extend(row for row in page_rows if isinstance(row, Mapping))
        if len(page_rows) < REST_PAGE_SIZE:
            break
    else:
        raise MainCiRunFetchError(
            f"Actions workflow inventory exceeded the {max_pages}-page budget; "
            "refusing an ambiguous workflow selector"
        )

    matches = [
        row
        for row in rows
        if row.get("name") == selector
        or row.get("path") == selector
        or str(row.get("path") or "").rsplit("/", 1)[-1] == selector
    ]
    if not matches:
        raise MainCiRunFetchError(f"workflow {workflow!r} was not found")
    if len(matches) > 1:
        raise MainCiRunFetchError(f"workflow {workflow!r} resolved to multiple workflows")
    workflow_id = _positive_int(matches[0].get("id"), field="workflow id")
    return str(workflow_id)


def _positive_int(value: Any, *, field: str) -> int:
    """Return a positive integer, rejecting booleans and malformed IDs."""
    if isinstance(value, bool):
        raise MainCiRunFetchError(f"{field} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise MainCiRunFetchError(f"{field} is not an integer: {value!r}") from exc
    if number < 1:
        raise MainCiRunFetchError(f"{field} must be positive")
    return number


def normalize_actions_run(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    """Normalize one Actions REST run to the existing classifier schema."""
    run_id = _positive_int(row.get("id"), field=f"Actions run row {index} id")
    status = row.get("status")
    if not isinstance(status, str) or not status:
        raise MainCiRunFetchError(f"Actions run row {index} has no usable status")
    conclusion = row.get("conclusion")
    if conclusion is not None and not isinstance(conclusion, str):
        raise MainCiRunFetchError(f"Actions run row {index} has a malformed conclusion")
    created_at = row.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise MainCiRunFetchError(f"Actions run row {index} has no usable created_at")
    head_sha = row.get("head_sha")
    if head_sha is not None and not isinstance(head_sha, str):
        raise MainCiRunFetchError(f"Actions run row {index} has a malformed head_sha")
    return {
        "databaseId": run_id,
        "status": status,
        "conclusion": conclusion,
        "headSha": head_sha,
        "createdAt": created_at,
    }


def count_decisive_runs(runs: Sequence[Any]) -> int:
    """Count completed runs carrying a decisive (green/red) verdict."""
    decisive = 0
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise MainCiRunFetchError(f"run window row {index} is malformed")
        if str(run.get("status") or "") != "completed":
            continue
        if classify(run.get("conclusion")) in {"green", "red"}:
            decisive += 1
    return decisive


def fetch_run_window(
    repo: str = DEFAULT_REPO,
    workflow: str = DEFAULT_WORKFLOW,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    stop_after_decisive: int = 1,
    runner: Callable[..., Any] | None = None,
) -> MainCiRunWindow:
    """Read a bounded paginated main-CI run window until decisive runs are seen.

    The raw ``gh run list --limit`` window used by the merge-hold gate can be
    entirely ``cancelled`` during high-throughput windows, hiding the decisive
    green/red verdict behind the flood.  This reader pages full REST pages and
    stops as soon as ``stop_after_decisive`` completed green/red runs are
    visible, or when a complete short page proves history is exhausted.

    Fail-closed contract: when ``max_pages`` is reached first, the returned
    window has ``window_exhausted=True`` and the caller must not read that
    partial window as a verdict.
    """
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    if stop_after_decisive <= 0:
        raise ValueError("stop_after_decisive must be positive")
    rest_runner = runner or _default_rest_runner
    selector = resolve_workflow_selector(
        repo=repo,
        workflow=workflow,
        max_pages=max_pages,
        runner=rest_runner,
    )
    endpoint_base = (
        f"repos/{quote(repo, safe='/')}/actions/workflows/{quote(selector, safe='')}/runs"
        f"?{urlencode({'branch': 'main'})}"
    )
    runs: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        endpoint = f"{endpoint_base}&per_page={REST_PAGE_SIZE}&page={page}"
        payload = _rest_json(
            endpoint,
            runner=rest_runner,
            operation=f"main-CI runs page {page}",
        )
        if not isinstance(payload, Mapping):
            raise MainCiRunFetchError(f"main-CI runs page {page} returned a non-object payload")
        page_rows = payload.get("workflow_runs")
        if not isinstance(page_rows, list) or any(
            not isinstance(row, Mapping) for row in page_rows
        ):
            raise MainCiRunFetchError(f"main-CI runs page {page} returned malformed rows")
        runs.extend(
            normalize_actions_run(row, index=index)
            for index, row in enumerate(page_rows, start=len(runs))
            if isinstance(row, Mapping)
        )
        if count_decisive_runs(runs) >= stop_after_decisive:
            return MainCiRunWindow(runs=runs, window_exhausted=False)
        if len(page_rows) < REST_PAGE_SIZE:
            return MainCiRunWindow(runs=runs, window_exhausted=False)
    return MainCiRunWindow(runs=runs, window_exhausted=True)


# A completed run's ``conclusion`` is DECISIVE only when it actually evaluated
# main's code: ``success`` (green) or ``failure`` (red). Every other value a
# completed run can report — ``cancelled`` (the common case: a run SUPERSEDED by
# a newer push), ``skipped``, ``neutral``, ``timed_out``, ``startup_failure``,
# ``action_required`` or unknown/``None`` — is STALE: the checks rendered no
# verdict on main and must be skipped exactly like an in-progress run. Treating
# ``cancelled`` as red (the pre-2026-07-13 bug) froze every merge during
# high-throughput windows, because rapid merges constantly supersede each other
# into ``cancelled`` runs — the exact freeze that stranded ~8 gate-vetted PRs.
GREEN_CONCLUSIONS = frozenset({"success"})
RED_CONCLUSIONS = frozenset({"failure"})


def classify(conclusion: str | None) -> str:
    """Bucket a completed run's ``conclusion`` into ``green`` / ``red`` / ``stale``."""
    c = str(conclusion) if conclusion is not None else ""
    if c in GREEN_CONCLUSIONS:
        return "green"
    if c in RED_CONCLUSIONS:
        return "red"
    return "stale"


def latest_decisive_run(runs: list[Any]) -> dict[str, Any] | None:
    """Return the newest COMPLETED run with a DECISIVE (green/red) conclusion.

    Skips in-progress runs AND stale completed runs (``cancelled`` etc.),
    neither of which is a verdict on main. Sorts by ``createdAt`` defensively.
    """
    completed = [
        run for run in runs if isinstance(run, dict) and str(run.get("status")) == "completed"
    ]
    completed.sort(key=lambda r: str(r.get("createdAt", "")), reverse=True)
    for run in completed:
        if classify(run.get("conclusion")) in ("green", "red"):
            return run
    return None


def latest_completed_run(runs: list[Any]) -> dict[str, Any] | None:
    """Deprecated alias for :func:`latest_decisive_run` (import stability)."""
    return latest_decisive_run(runs)


def decide(runs: list[Any]) -> tuple[bool, dict[str, Any] | None]:
    """(is_green, deciding_run). Green iff the latest DECISIVE run succeeded.

    Fails closed: if no completed run rendered a decisive verdict (only
    in-progress / stale runs in the window), returns ``(False, None)`` — the
    hold stays, but as a "needs a fresh run" hold, not "main regressed".
    """
    run = latest_decisive_run(runs)
    if run is None:
        return False, None
    return classify(run.get("conclusion")) == "green", run


# The main-signal schema consumed by the red-main merge-hold gate (issue #5571).
# ``status`` is one of green / red / stale, matching :func:`classify`; a stale
# verdict (no decisive completed run in the window) still fails closed to not
# green but is reported distinctly so the gate can hold for a *fresh run* rather
# than treat it as a main regression.
SIGNAL_SCHEMA_VERSION = "main_ci_is_green.v1"


def build_signal(
    is_green: bool,
    run: dict[str, Any] | None,
    repo: str = DEFAULT_REPO,
    workflow: str = DEFAULT_WORKFLOW,
) -> dict[str, Any]:
    """Build the machine-readable main-signal payload (the ``--json`` contract)."""
    if run is None:
        status = "stale"
        deciding_run = None
    else:
        status = classify(run.get("conclusion"))
        deciding_run = {
            "databaseId": run.get("databaseId"),
            "conclusion": run.get("conclusion"),
            "status": run.get("status"),
            "headSha": run.get("headSha"),
            "createdAt": run.get("createdAt"),
        }
    return {
        "schema_version": SIGNAL_SCHEMA_VERSION,
        "is_green": is_green,
        "status": status,
        "repo": repo,
        "workflow": workflow,
        "deciding_run": deciding_run,
    }


def fetch_runs(
    repo: str = DEFAULT_REPO, workflow: str = DEFAULT_WORKFLOW, limit: int = 5
) -> list[dict[str, Any]]:
    """Fetch recent completed main CI runs in one bounded ``gh run list`` call.

    ``--status completed`` is load-bearing.  This is the fast merge-hold path:
    a single window of ``limit`` runs (default 5) with the 30s CLI timeout, so
    the gate cannot become slow or expensive.  A cancellation-saturated window
    fails closed to ``stale``; use :func:`fetch_run_window` when the decisive
    verdict behind such a flood is required.
    """
    proc = _gh(
        [
            "run",
            "list",
            "--repo",
            repo,
            "--branch",
            "main",
            "--workflow",
            workflow,
            "--status",
            "completed",
            "--limit",
            str(limit),
            "--json",
            "databaseId,status,conclusion,headSha,createdAt",
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh run list failed: {proc.stderr.strip() or proc.returncode}")
    data = json.loads(proc.stdout or "[]")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected JSON response type: {type(data).__name__}")
    return data


def main() -> int:
    """CLI entry: exit 0 if main CI is green, 1 otherwise.

    Default and ``--quiet`` modes print the human-readable line (and use the
    exit code) so the gate's existing human fallback keeps working. ``--json``
    emits the machine-readable main-signal schema (issue #5571) and still exits
    0/1, so the gate contract can be satisfied without parsing text.
    """
    ap = argparse.ArgumentParser(description="Is main CI green (latest completed run)?")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--quiet", action="store_true", help="suppress the human line")
    ap.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="emit the machine-readable main-signal JSON (green/red/stale schema)",
    )
    args = ap.parse_args()

    try:
        runs = fetch_runs(args.repo, args.workflow, args.limit)
    except (RuntimeError, json.JSONDecodeError) as exc:
        # Fail closed: an unreadable signal is treated as NOT green so a merge
        # hold errs toward holding, never toward merging on unknown state.
        if args.as_json:
            # A stale schema_version + is_green=false lets the gate fail closed
            # without a traceback; the human UNKNOWN line still goes to stderr.
            print(
                json.dumps(
                    {
                        "schema_version": SIGNAL_SCHEMA_VERSION,
                        "is_green": False,
                        "status": "stale",
                        "repo": args.repo,
                        "workflow": args.workflow,
                        "deciding_run": None,
                        "error": f"fetch failed: {exc}",
                    }
                )
            )
        if not args.quiet:
            print(f"main CI status UNKNOWN ({exc}) -> treated as not-green", file=sys.stderr)
        return 1

    is_green, run = decide(runs)
    if args.as_json:
        print(json.dumps(build_signal(is_green, run, args.repo, args.workflow)))
    elif not args.quiet:
        if run is None:
            print("main CI: no completed run found -> not green")
        else:
            print(
                f"main CI: {run.get('conclusion')} "
                f"(run {run.get('databaseId')}, {str(run.get('headSha'))[:9]}) "
                f"-> {'GREEN' if is_green else 'NOT GREEN'}"
            )
    return 0 if is_green else 1


if __name__ == "__main__":
    raise SystemExit(main())
