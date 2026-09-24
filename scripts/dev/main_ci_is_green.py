#!/usr/bin/env python3
"""Report whether ``main`` CI is green for the exact current main commit.

Motivation (issue #5385). Three separate main-red incidents in 36h
(2026-07-11/12) shared one mechanism: merges kept landing while main CI was
red, and the already-failing required check masked the NEW breakage each merge
introduced, so recovery cost grew with every merge that landed inside the red
window. The cure is a merge hold: gates may review a PR while main is red but
must not merge (except the unbreak-main fix itself) until main is green again.

This helper is the deterministic green/red signal that hold consults. The one
rule that matters — learned the hard way when the escalation guard stayed
silent on 2026-07-11 — is that an IN-PROGRESS run must never count as evidence
either way: only a completed run with an actual matrix verdict decides. A
completed run for an older main SHA cannot certify the current head. The fetch
keeps in-progress runs in its bounded window, classifies status locally, and
requires the main branch head to remain stable while the window is read;
manual dispatch candidates also require compatibility-matrix admission proof.

Exit code: 0 == green (a verified completed CI run on the current main SHA
concluded ``success``), 1 == not green (red, stale, or no same-head run to
judge from). Prints the run id and conclusion it decided from. The ``--json`` flag emits the machine-readable
main-signal schema (``main_ci_is_green.v1``) with the same green/red/stale
classification the gate contract consumes, so the gate need not parse the
human line.

The gate's default fetch is deliberately one small bounded window (a single
unfiltered ``gh run list --limit`` call, default 5 runs, 30s timeout): the merge
hold must stay cheap and quick, and a cancellation-saturated window fails
closed to ``stale`` rather than blocking on a slower search. Status and
conclusion are classified locally because server-side completed-only queries
have intermittently returned an older window than the unfiltered query.
:func:`fetch_run_window`
is the paginated REST reader for callers that need the decisive verdict behind
a cancelled-run flood; :mod:`main_ci_incident_reconcile` uses it.  This module
owns both fetch paths so there is exactly one pagination implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import quote, urlencode

if __package__ in {None, ""}:
    # Direct execution must prefer this checkout over ambient source roots.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dev._gh_rest import parse_json, run_gh_api

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_WORKFLOW = "CI"
DEFAULT_WORKFLOW_FILE = ".github/workflows/ci.yml"

# Bounded pagination budget for the decisive-run REST search.  GitHub's
# latest-main-wins concurrency can fill whole raw pages with ``cancelled``
# runs, so a raw ``--limit`` does not identify a sufficient evidence window.
# The reader examines at most ``max_pages`` full pages before reporting the
# window as exhausted; hitting that bound means "no verdict found", never red.
REST_PAGE_SIZE = 100
DEFAULT_MAX_PAGES = 10
REST_RUN_TIMEOUT = 90
DISPATCH_GATE_SCHEMA_VERSION = "main_ci_dispatch_gate.v1"
DISPATCH_ACTIVE_STATUSES = frozenset({"queued", "in_progress", "requested", "pending", "waiting"})
DISPATCH_MATRIX_JOB_NAME = "compat-matrix"


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


def fetch_main_head_sha(
    repo: str = DEFAULT_REPO,
    *,
    runner: Callable[..., Any] | None = None,
) -> str:
    """Read the current ``main`` branch SHA, rejecting malformed ref evidence."""
    endpoint = f"repos/{quote(repo, safe='/')}/git/ref/heads/main"
    payload = _rest_json(
        endpoint,
        runner=runner or _default_rest_runner,
        operation="main branch ref",
    )
    if not isinstance(payload, Mapping) or payload.get("ref") != "refs/heads/main":
        raise MainCiRunFetchError("main branch ref returned malformed or mismatched evidence")
    obj = payload.get("object")
    if not isinstance(obj, Mapping):
        raise MainCiRunFetchError("main branch ref has no commit object")
    sha = obj.get("sha")
    if (
        not isinstance(sha, str)
        or len(sha) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in sha)
    ):
        raise MainCiRunFetchError("main branch ref has no usable commit SHA")
    return sha.lower()


def _normalize_expected_head_sha(value: str | None) -> str | None:
    if value is None:
        return None
    expected_head = value.strip().lower()
    if len(expected_head) != 40 or any(
        character not in "0123456789abcdef" for character in expected_head
    ):
        raise MainCiRunFetchError("expected main head is not a usable commit SHA")
    return expected_head


def _read_stable_main_ci_signal(
    repo: str, workflow: str, limit: int
) -> tuple[bool, dict[str, Any] | None]:
    """Read a bounded run window bracketed by a stable main ref."""
    main_head_before = fetch_main_head_sha(repo)
    runs = fetch_runs(repo, workflow, limit)
    main_head_after = fetch_main_head_sha(repo)
    if main_head_after != main_head_before:
        raise MainCiRunFetchError("main advanced while its CI signal was being read")
    return decide_verified_main_ci_signal(
        runs,
        repo=repo,
        expected_head_sha=main_head_before,
    )


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
    event = row.get("event")
    if event is not None and not isinstance(event, str):
        raise MainCiRunFetchError(f"Actions run row {index} has a malformed event")
    display_title = row.get("display_title")
    if display_title is not None and not isinstance(display_title, str):
        raise MainCiRunFetchError(f"Actions run row {index} has a malformed display_title")
    return {
        "databaseId": run_id,
        "status": status,
        "conclusion": conclusion,
        "headSha": head_sha,
        "createdAt": created_at,
        "event": event,
        "displayTitle": display_title,
    }


def _validate_dispatch_window_inputs(
    target_branch: str,
    target_sha: str | None,
    current_run_id: int | None,
    max_pages: int,
) -> tuple[str, str, int | None]:
    """Validate branch/window selectors and normalize exact-head identity."""
    branch = target_branch.strip() if isinstance(target_branch, str) else ""
    if (
        not branch
        or branch != target_branch
        or branch.startswith("refs/")
        or any(character.isspace() for character in branch)
    ):
        raise MainCiRunFetchError(
            "dispatch ownership target_branch must be a non-empty branch name, not a full ref"
        )
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    if (target_sha is None) != (current_run_id is None):
        raise MainCiRunFetchError(
            "dispatch ownership target_sha and current_run_id must be supplied together"
        )
    target = target_sha.strip().lower() if isinstance(target_sha, str) else ""
    if target_sha is not None and not target:
        raise MainCiRunFetchError("dispatch ownership target_sha must not be empty")
    current_id = (
        _positive_int(current_run_id, field="current workflow run id")
        if current_run_id is not None
        else None
    )
    return branch, target, current_id


def _is_older_active_exact_head(
    runs: Sequence[Mapping[str, Any]], *, target_sha: str, current_run_id: int
) -> bool:
    """Whether a page already proves this run is a follower, not the owner."""
    return any(
        str(run.get("headSha") or "").lower() == target_sha
        and str(run.get("status") or "").lower() in DISPATCH_ACTIVE_STATUSES
        and _positive_int(run.get("databaseId"), field="active run id") < current_run_id
        for run in runs
    )


def _attach_retry_matrix_admission(
    runs: list[dict[str, Any]],
    *,
    repo: str,
    target_sha: str,
    runner: Callable[..., Any],
    max_pages: int,
) -> None:
    """Attach job-API admission evidence to exact-head explicit retry failures."""
    for run in runs:
        is_retry_failure = (
            str(run.get("headSha") or "").lower() == target_sha
            and str(run.get("status") or "").lower() == "completed"
            and str(run.get("event") or "") == "workflow_dispatch"
            and str(run.get("displayTitle") or "").endswith("retry_failed=true")
            and classify(run.get("conclusion")) == "red"
        )
        if not is_retry_failure:
            continue
        run["fullMatrixAdmitted"] = fetch_dispatch_retry_matrix_admitted(
            repo=repo,
            run_id=_positive_int(run.get("databaseId"), field="retry workflow run id"),
            runner=runner,
            max_pages=max_pages,
        )


def fetch_dispatch_run_window(
    repo: str = DEFAULT_REPO,
    workflow: str = DEFAULT_WORKFLOW,
    *,
    target_branch: str = "main",
    target_sha: str | None = None,
    current_run_id: int | None = None,
    max_pages: int = DEFAULT_MAX_PAGES,
    runner: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Read a bounded run window, stopping early only for an older active owner."""
    branch, target, current_id = _validate_dispatch_window_inputs(
        target_branch, target_sha, current_run_id, max_pages
    )
    rest_runner = runner or _default_rest_runner
    selector = resolve_workflow_selector(
        repo=repo,
        workflow=workflow,
        max_pages=max_pages,
        runner=rest_runner,
    )
    endpoint_base = (
        f"repos/{quote(repo, safe='/')}/actions/workflows/{quote(selector, safe='')}/runs"
        f"?{urlencode({'branch': branch, **({'head_sha': target} if target else {})})}"
    )
    runs: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        endpoint = f"{endpoint_base}&per_page={REST_PAGE_SIZE}&page={page}"
        payload = _rest_json(
            endpoint,
            runner=rest_runner,
            operation=f"recent CI dispatch ownership window page {page}",
        )
        if not isinstance(payload, Mapping):
            raise MainCiRunFetchError(
                f"CI dispatch ownership window page {page} returned a non-object payload"
            )
        rows = payload.get("workflow_runs")
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise MainCiRunFetchError(
                f"CI dispatch ownership window page {page} returned malformed rows"
            )
        page_runs = [
            normalize_actions_run(row, index=index)
            for index, row in enumerate(rows, start=len(runs))
            if isinstance(row, Mapping)
        ]
        runs.extend(page_runs)
        if (
            target
            and current_id is not None
            and _is_older_active_exact_head(page_runs, target_sha=target, current_run_id=current_id)
        ):
            # An older exact-head active run is sufficient to make this run
            # wait. No more pages or retry-receipt queries can change that.
            return runs
        if len(rows) < REST_PAGE_SIZE:
            break
    else:
        raise MainCiRunFetchError(
            f"CI dispatch ownership run window reached the {max_pages}-page budget "
            "without exhaustion or an older same-head active run; refusing a partial window"
        )

    if target and current_id is not None:
        _attach_retry_matrix_admission(
            runs,
            repo=repo,
            target_sha=target,
            runner=rest_runner,
            max_pages=max_pages,
        )
    return runs


def _fetch_retry_jobs_page(
    payload: Any,
    *,
    run_id: int,
    page: int,
) -> list[Mapping[str, Any]]:
    """Validate one Actions jobs response and return its job rows."""
    if not isinstance(payload, Mapping):
        raise MainCiRunFetchError(
            f"retry workflow run {run_id} jobs page {page} returned a non-object payload"
        )
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or any(not isinstance(job, Mapping) for job in jobs):
        raise MainCiRunFetchError(
            f"retry workflow run {run_id} jobs page {page} returned malformed rows"
        )
    for index, job in enumerate(jobs):
        if not isinstance(job.get("name"), str) or not job.get("name"):
            raise MainCiRunFetchError(
                f"retry workflow run {run_id} job row {index} has no usable name"
            )
        if not isinstance(job.get("status"), str) or not job.get("status"):
            raise MainCiRunFetchError(
                f"retry workflow run {run_id} job row {index} has no usable status"
            )
        conclusion = job.get("conclusion")
        if conclusion is not None and not isinstance(conclusion, str):
            raise MainCiRunFetchError(
                f"retry workflow run {run_id} job row {index} has a malformed conclusion"
            )
    return jobs


def _compatibility_page_admission(jobs: Sequence[Mapping[str, Any]], *, run_id: int) -> bool | None:
    """Return page evidence (or None if this page has no compatibility jobs)."""
    matrix_jobs = [
        job
        for job in jobs
        if str(job.get("name") or "") == DISPATCH_MATRIX_JOB_NAME
        or str(job.get("name") or "").startswith(f"{DISPATCH_MATRIX_JOB_NAME} (")
    ]
    if not matrix_jobs:
        return None
    for job in matrix_jobs:
        status = str(job.get("status") or "").lower()
        conclusion = job.get("conclusion")
        if status == "completed" and not isinstance(conclusion, str):
            raise MainCiRunFetchError(
                f"retry workflow run {run_id} compatibility job has no conclusion"
            )
        if status not in {"queued", "in_progress", "completed"}:
            raise MainCiRunFetchError(
                f"retry workflow run {run_id} compatibility job has unknown status"
            )
    return any(
        str(job.get("status") or "").lower() != "completed" or job.get("conclusion") != "skipped"
        for job in matrix_jobs
    )


def fetch_dispatch_retry_matrix_admitted(
    *,
    repo: str,
    run_id: int,
    runner: Callable[..., Any],
    max_pages: int = DEFAULT_MAX_PAGES,
) -> bool:
    """Prove compatibility-job admission or explicit skips; reject missing evidence."""
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    checked_run_id = _positive_int(run_id, field="retry workflow run id")
    endpoint_base = (
        f"repos/{quote(repo, safe='/')}/actions/runs/{checked_run_id}/jobs"
        f"?{urlencode({'per_page': REST_PAGE_SIZE})}"
    )
    recognized_matrix_rows = False
    for page in range(1, max_pages + 1):
        payload = _rest_json(
            f"{endpoint_base}&page={page}",
            runner=runner,
            operation=f"retry workflow run {checked_run_id} jobs page {page}",
        )
        jobs = _fetch_retry_jobs_page(payload, run_id=checked_run_id, page=page)
        page_admission = _compatibility_page_admission(jobs, run_id=checked_run_id)
        if page_admission is True:
            return True
        if page_admission is False:
            recognized_matrix_rows = True
        if len(jobs) < REST_PAGE_SIZE:
            if recognized_matrix_rows:
                return False
            raise MainCiRunFetchError(
                f"retry workflow run {checked_run_id} jobs exhausted with no recognized "
                f"{DISPATCH_MATRIX_JOB_NAME} job; refusing to infer matrix admission"
            )
    if not recognized_matrix_rows:
        raise MainCiRunFetchError(
            f"retry workflow run {checked_run_id} jobs reached the {max_pages}-page budget "
            f"with no recognized {DISPATCH_MATRIX_JOB_NAME} job; refusing to infer matrix admission"
        )
    raise MainCiRunFetchError(
        f"retry workflow run {checked_run_id} jobs exceeded the {max_pages}-page budget; "
        "refusing to treat a partial all-skipped window as complete"
    )


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


def decide_verified_main_ci_signal(
    runs: list[Any],
    *,
    repo: str = DEFAULT_REPO,
    matrix_admission_lookup: Callable[[int], bool] | None = None,
    expected_head_sha: str | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    """Select a decisive run, requiring matrix proof for manual dispatch runs.

    The raw run history is supplied by the bounded ``gh run list`` query. When
    ``expected_head_sha`` is supplied, only evidence for that exact current
    main commit may decide. A successful or failed ``workflow_dispatch`` is
    decisive only when its compatibility matrix was admitted. Explicitly
    all-skipped gate-only dispatches are ignored, but older evidence can decide
    only for that same exact head. Unreadable or ambiguous job evidence raises
    so callers fail closed.
    """
    expected_head = _normalize_expected_head_sha(expected_head_sha)

    if matrix_admission_lookup is None:

        def fetch_matrix_for_signal(run_id: int) -> bool:
            return fetch_dispatch_retry_matrix_admitted(
                repo=repo,
                run_id=run_id,
                runner=_default_rest_runner,
            )

        matrix_admission_lookup = fetch_matrix_for_signal

    completed = [
        run for run in runs if isinstance(run, dict) and str(run.get("status")) == "completed"
    ]
    completed.sort(key=lambda run: str(run.get("createdAt", "")), reverse=True)
    gate_only_head: str | None = None

    for run in completed:
        if classify(run.get("conclusion")) not in {"green", "red"}:
            continue
        head_sha = run.get("headSha")
        normalized_head = head_sha.strip().lower() if isinstance(head_sha, str) else ""
        if expected_head is not None and normalized_head != expected_head:
            return False, None
        if gate_only_head is not None and normalized_head != gate_only_head:
            return False, None

        if str(run.get("event") or "") == "workflow_dispatch":
            if not normalized_head:
                return False, None
            run_id = _positive_int(run.get("databaseId"), field="workflow run id")
            if not matrix_admission_lookup(run_id):
                gate_only_head = normalized_head
                continue
            run["fullMatrixAdmitted"] = True

        return classify(run.get("conclusion")) == "green", run

    return False, None


def dispatch_decision(
    target_sha: str,
    runs: Sequence[Mapping[str, Any]],
    *,
    retry_failed: bool = False,
    retry_receipt_seen: bool = False,
) -> dict[str, Any]:
    """Choose an idempotent watcher action for one main commit.

    A same-head queued or in-progress run is observed, never replaced. A
    completed success is also observed. A completed failure may be retried at
    most once when the caller explicitly supplies the retry intent and no
    durable retry receipt exists. The function is deliberately side-effect
    free so scheduled watchers can record the decision before dispatching.
    """
    target = target_sha.strip().lower()
    if not target:
        raise ValueError("target_sha must not be empty")
    exact = [
        run
        for run in runs
        if isinstance(run, Mapping) and str(run.get("headSha") or "").lower() == target
    ]
    active = [
        run
        for run in exact
        if str(run.get("status") or "").lower() in {"queued", "in_progress", "requested"}
    ]
    if active:
        return {"action": "observe", "reason": "same_head_run_active", "head_sha": target}
    successful = [
        run
        for run in exact
        if str(run.get("status") or "").lower() == "completed"
        and classify(run.get("conclusion")) == "green"
    ]
    if successful:
        return {"action": "observe", "reason": "same_head_success", "head_sha": target}
    failed = [
        run
        for run in exact
        if str(run.get("status") or "").lower() == "completed"
        and classify(run.get("conclusion")) == "red"
    ]
    if failed and retry_failed and not retry_receipt_seen:
        return {"action": "dispatch", "reason": "explicit_failed_retry", "head_sha": target}
    if failed:
        return {"action": "observe", "reason": "same_head_failure", "head_sha": target}
    return {"action": "dispatch", "reason": "no_same_head_decisive_run", "head_sha": target}


def dispatch_gate_decision(
    target_sha: str,
    current_run_id: int,
    runs: Sequence[Mapping[str, Any]],
    *,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Elect one exact-head run to launch the full CI matrix.

    GitHub permits only one pending run in a shared concurrency group and
    replaces that pending run even when ``cancel-in-progress`` is false.  The
    workflow therefore gives manual dispatches unique concurrency identities
    and uses this deterministic gate to keep only the oldest active exact-head
    run as the full-matrix owner.  Followers wait for that owner rather than
    reporting an unproved success.
    """
    target = target_sha.strip().lower()
    if not target:
        raise ValueError("target_sha must not be empty")
    current_id = _positive_int(current_run_id, field="current workflow run id")
    exact = [
        run
        for run in runs
        if isinstance(run, Mapping) and str(run.get("headSha") or "").lower() == target
    ]
    active_ids = {current_id}
    for index, run in enumerate(exact):
        if str(run.get("status") or "").lower() not in DISPATCH_ACTIVE_STATUSES:
            continue
        active_ids.add(_positive_int(run.get("databaseId"), field=f"active run row {index} id"))
    owner_id = min(active_ids)
    if owner_id != current_id:
        return {
            "action": "wait",
            "reason": "older_same_head_run_active",
            "head_sha": target,
            "owner_run_id": owner_id,
            "current_run_id": current_id,
        }

    completed = [
        run
        for run in exact
        if str(run.get("status") or "").lower() == "completed"
        and _positive_int(run.get("databaseId"), field="completed run id") != current_id
    ]
    retry_receipt_seen = any(
        str(run.get("event") or "") == "workflow_dispatch"
        and str(run.get("displayTitle") or "").endswith("retry_failed=true")
        and classify(run.get("conclusion")) in {"green", "red"}
        and run.get("fullMatrixAdmitted") is True
        for run in completed
    )
    policy = dispatch_decision(
        target,
        completed,
        retry_failed=retry_failed,
        retry_receipt_seen=retry_receipt_seen,
    )
    if policy["action"] == "dispatch":
        return {
            "action": "run_full_ci",
            "reason": policy["reason"],
            "head_sha": target,
            "owner_run_id": current_id,
            "current_run_id": current_id,
            "retry_receipt_seen": retry_receipt_seen,
        }
    reason = str(policy["reason"])
    action = "observe_success" if reason == "same_head_success" else "observe_failure"
    return {
        "action": action,
        "reason": reason,
        "head_sha": target,
        "owner_run_id": current_id,
        "current_run_id": current_id,
        "retry_receipt_seen": retry_receipt_seen,
    }


def wait_for_dispatch_gate(  # noqa: PLR0913 - explicit polling/test seams are intentional.
    target_sha: str,
    current_run_id: int,
    *,
    repo: str = DEFAULT_REPO,
    workflow: str = DEFAULT_WORKFLOW,
    target_branch: str = "main",
    target_ref_type: str | None = None,
    retry_failed: bool = False,
    poll_seconds: float = 30.0,
    max_wait_seconds: float = 3000.0,
    fetcher: Callable[[], list[dict[str, Any]]] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait for ownership or an exact-head verdict within the selected branch."""
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    if max_wait_seconds < 0:
        raise ValueError("max_wait_seconds must not be negative")
    if target_ref_type != "branch":
        raise MainCiRunFetchError(
            "dispatch ownership supports branch refs only; refusing a tag or missing ref type"
        )
    read_runs = fetcher or (
        lambda: fetch_dispatch_run_window(
            repo,
            workflow,
            target_branch=target_branch,
            target_sha=target_sha,
            current_run_id=current_run_id,
        )
    )
    deadline = time.monotonic() + max_wait_seconds
    while True:
        decision = dispatch_gate_decision(
            target_sha,
            current_run_id,
            read_runs(),
            retry_failed=retry_failed,
        )
        if decision["action"] != "wait":
            return decision
        if time.monotonic() >= deadline:
            return {
                **decision,
                "action": "timeout",
                "reason": "older_same_head_run_did_not_finish_within_budget",
            }
        sleeper(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def _write_dispatch_gate_output(path: str | None, *, run_full_ci: bool) -> None:
    """Append the gate output using the GitHub Actions output-file contract."""
    if not path:
        return
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(f"run_full_ci={'true' if run_full_ci else 'false'}\n")


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
    """Fetch recent main CI runs in one bounded, unfiltered ``gh run list`` call.

    This is the fast merge-hold path: a single window of ``limit`` runs
    (default 5) with the 30s CLI timeout, so the gate cannot become slow or
    expensive. Status and conclusion are filtered locally. A saturated window
    without an admissible current-head verdict fails closed to ``stale``; use
    :func:`fetch_run_window` when the decisive verdict behind such a flood is
    required. The default ``CI`` display name is mapped to its workflow file
    path to avoid ambiguous name matches.
    """
    workflow_selector = DEFAULT_WORKFLOW_FILE if workflow == DEFAULT_WORKFLOW else workflow
    proc = _gh(
        [
            "run",
            "list",
            "--repo",
            repo,
            "--branch",
            "main",
            "--workflow",
            workflow_selector,
            "--limit",
            str(limit),
            "--json",
            "databaseId,status,conclusion,headSha,createdAt,event",
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh run list failed: {proc.stderr.strip() or proc.returncode}")
    data = json.loads(proc.stdout or "[]")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected JSON response type: {type(data).__name__}")
    for index, run in enumerate(data):
        if not isinstance(run, Mapping):
            raise RuntimeError(f"Unexpected run row type at index {index}: {type(run).__name__}")
        if not isinstance(run.get("event"), str) or not run["event"]:
            raise RuntimeError(f"Run row {index} has no usable event")
        if not isinstance(run.get("headSha"), str) or not run["headSha"]:
            raise RuntimeError(f"Run row {index} has no usable head SHA")
        if not isinstance(run.get("createdAt"), str) or not run["createdAt"]:
            raise RuntimeError(f"Run row {index} has no usable createdAt")
    return data


def main(argv: list[str] | None = None) -> int:
    """CLI entry: exit 0 if main CI is green, 1 otherwise.

    Default and ``--quiet`` modes print the human-readable line (and use the
    exit code) so the gate's existing human fallback keeps working. ``--json``
    emits the machine-readable main-signal schema (issue #5571) and still exits
    0/1, so the gate contract can be satisfied without parsing text.
    """
    ap = argparse.ArgumentParser(description="Is main CI green (latest verified completed run)?")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--quiet", action="store_true", help="suppress the human line")
    ap.add_argument(
        "--dispatch-gate",
        action="store_true",
        help="elect or observe the one exact-head manual run allowed to launch full CI",
    )
    ap.add_argument("--target-sha", default=os.environ.get("GITHUB_SHA"))
    ap.add_argument("--target-branch", default=os.environ.get("GITHUB_REF_NAME"))
    ap.add_argument("--target-ref-type", default=os.environ.get("GITHUB_REF_TYPE"))
    ap.add_argument("--current-run-id", type=int, default=os.environ.get("GITHUB_RUN_ID"))
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--max-wait-seconds", type=float, default=3000.0)
    ap.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    ap.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="emit the machine-readable main-signal JSON (green/red/stale schema)",
    )
    args = ap.parse_args(argv)

    if args.dispatch_gate:
        if (
            not args.target_sha
            or args.current_run_id is None
            or not args.target_branch
            or not args.target_ref_type
        ):
            ap.error(
                "--dispatch-gate requires --target-sha, --target-branch, "
                "--target-ref-type, and --current-run-id"
            )
        try:
            decision = wait_for_dispatch_gate(
                args.target_sha,
                args.current_run_id,
                repo=args.repo,
                workflow=args.workflow,
                target_branch=args.target_branch,
                target_ref_type=args.target_ref_type,
                retry_failed=args.retry_failed,
                poll_seconds=args.poll_seconds,
                max_wait_seconds=args.max_wait_seconds,
            )
        except (MainCiRunFetchError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _write_dispatch_gate_output(args.github_output, run_full_ci=False)
            print(
                json.dumps(
                    {
                        "schema_version": DISPATCH_GATE_SCHEMA_VERSION,
                        "action": "error",
                        "reason": str(exc),
                        "head_sha": str(args.target_sha).lower(),
                        "current_run_id": args.current_run_id,
                    }
                )
            )
            return 1
        run_full_ci = decision["action"] == "run_full_ci"
        _write_dispatch_gate_output(args.github_output, run_full_ci=run_full_ci)
        print(json.dumps({"schema_version": DISPATCH_GATE_SCHEMA_VERSION, **decision}))
        return 0 if decision["action"] in {"run_full_ci", "observe_success"} else 1

    try:
        is_green, run = _read_stable_main_ci_signal(args.repo, args.workflow, args.limit)
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
