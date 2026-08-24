"""Deterministic terminal-label reconciliation for closed or merged items.

After an issue or pull request reaches a verified terminal state, active
dispatch and review labels from a controlled namespace become stale: they make
queue snapshots report false dispatch candidates and distort WIP reporting.
This planner derives an exact before/after label plan for a declared terminal
class and executes it with compare-and-swap semantics.

The controlled namespace is derived from ``docs/ai/label-taxonomy.md``: active
execution and review labels must not survive a verified terminal transition,
while type/resource/evidence/provenance/priority labels and the terminal marker
itself are preserved. ``decision-required`` is retained unless the terminal
class explicitly resolves the decision (a merge or a ``ruled`` closure with a
recorded ruling).

The operation is exact-item scoped and idempotent. ``--report`` mode performs
no GitHub mutation and lists every proposed change; ``--apply`` mode re-reads
the live item state immediately before mutating each label, aborts on reopen or
concurrent label drift, and keeps manual labels outside the controlled
namespace untouched. The planner never closes issues, merges PRs, or creates
labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from scripts.dev._gh_rest import gh_api_get, parse_json
from scripts.dev.gh_pr_label_rest import add_label, remove_label

SCHEMA = "terminal_label_reconcile.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"

# Terminal classes supported by the planner.
TERMINAL_CLASSES = frozenset(
    {
        "completed",
        "not_planned",
        "duplicate",
        "pr_merged",
        "pr_closed_unmerged",
        "terminal_unverified",
        "reopened",
    }
)

# Active execution/review labels that must not survive a verified terminal
# transition (derived from docs/ai/label-taxonomy.md execution-state family).
ACTIVE_LABELS = frozenset(
    {
        "state:ready",
        "state:running",
        "state:working",
        "state:review",
        "needs-review",
        "agent-ready",
        "merge-ready",
        "dependency:has-blockers",
        "dependency:blocks-others",
        "bounty:in-progress",
    }
)

# ``blocked:*`` prefixed labels are active dependency holds and are reconciled
# except for the terminal-level ``blocked:needs-maintainer`` when the terminal
# state already resolved the ruling.
BLOCKED_PREFIX = "blocked:"
BLOCKED_MAINTAINER = "blocked:needs-maintainer"


def _terminal_removals(terminal_class: str, current: set[str]) -> set[str]:
    """Return the deterministic removal policy for one verified terminal class.

    Active execution/review labels are always removed; ``decision-required`` is
    removed only when the terminal transition resolves it (completed, merged,
    duplicate); ``blocked:*`` holds are removed when resolved by the terminal
    class. ``reopened`` keeps active labels and only clears the terminal marker.
    """
    if terminal_class == "reopened":
        return {"state:done"} if "state:done" in current else set()
    removals = {label for label in current if label in ACTIVE_LABELS}
    for label in current:
        if label.startswith(BLOCKED_PREFIX):
            if terminal_class == "duplicate":
                removals.add(label)
            elif label == BLOCKED_MAINTAINER and terminal_class in {"completed", "pr_merged"}:
                removals.add(label)
    if "decision-required" in current and terminal_class in {
        "completed",
        "pr_merged",
        "duplicate",
    }:
        removals.add("decision-required")
    return removals


def plan_for_terminal(
    terminal_class: str,
    current_labels: list[str],
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    """Return the deterministic add/remove label plan for one declared terminal class.

    ``reason`` is the GitHub state reason when available (``completed``,
    ``not_planned``, ``duplicate``, ``reopened``). Non-controlled labels are
    always preserved; unknown labels are preserved by default.
    """
    current = set(current_labels)
    add: list[str] = []
    if terminal_class == "terminal_unverified":
        # No receipt: leave active labels; never add the terminal marker.
        if "state:done" in current and reason == "completed":
            current.discard("state:done")
        return {
            "terminal_class": terminal_class,
            "reason": reason,
            "add": [],
            "remove": sorted({"state:done"} & current),
            "preserved": sorted(current),
        }
    if terminal_class not in {"reopened"}:
        add = ["state:done"]
    removals = _terminal_removals(terminal_class, current)
    preserved = sorted(current - removals)
    return {
        "terminal_class": terminal_class,
        "reason": reason,
        "add": add,
        "remove": sorted(removals),
        "preserved": preserved,
    }


def fetch_item_state(number: int, *, repo: str = DEFAULT_REPO) -> dict[str, Any]:
    """Read the live issue/PR row and normalized labels via REST.

    Returns a payload with ``number``, ``state``, ``reason``, ``labels``, and an
    ``ok`` flag; ``ok=False`` carries an error string.
    """
    result = gh_api_get(f"repos/{repo}/issues/{number}", timeout=30)
    payload, error = parse_json(result, what=f"issue #{number} read")
    if error:
        return {"ok": False, "error": error}
    if not isinstance(payload, dict):
        return {"ok": False, "error": f"issue #{number} response was not an object"}
    raw_labels = payload.get("labels")
    names = (
        [
            entry["name"]
            for entry in raw_labels
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        ]
        if isinstance(raw_labels, list)
        else []
    )
    return {
        "ok": True,
        "number": number,
        "state": str(payload.get("state") or "").lower(),
        "reason": str(payload.get("state_reason") or "").lower() or None,
        "labels": names,
        "html_url": str(payload.get("html_url") or ""),
    }


def _apply_label_change(
    number: int,
    *,
    repo: str,
    label: str,
    action: str,
    applied: dict[str, Any],
) -> dict[str, Any] | None:
    """Apply one compare-and-swap label mutation.

    Returns an abort payload (state reopened / read failure) when the live state
    is no longer terminal, ``None`` when the mutation was attempted.
    """
    current = fetch_item_state(number, repo=repo)
    if not current["ok"]:
        applied["failures"].append({"label": label, "error": current["error"]})
        return {
            "ok": False,
            "error": f"state re-read failed before {action} '{label}'",
            "applied_changes": applied,
        }
    if current["state"] not in {"closed", "merged"}:
        return {
            "ok": False,
            "error": f"item reopened (state={current['state']}); plan aborted",
            "applied_changes": applied,
        }
    if action == "remove":
        if label not in current["labels"]:
            applied["remove"].append({"label": label, "skipped": True})
            return None
        result = remove_label(number, label, repo=repo)
        if result.get("status") != "ok":
            applied["failures"].append({"label": label, "error": result.get("error")})
            return None
        applied["remove"].append({"label": label, "skipped": False})
        return None
    if label in current["labels"]:
        applied["add"].append({"label": label, "skipped": True})
        return None
    result = add_label(number, label, repo=repo)
    if result.get("status") != "ok":
        applied["failures"].append({"label": label, "error": result.get("error")})
        return None
    applied["add"].append({"label": label, "skipped": False})
    return None


def _apply_plan(
    number: int,
    terminal_class: str,
    plan: dict[str, Any],
    *,
    repo: str,
    reason: str | None,
) -> dict[str, Any]:
    """Apply a label plan with compare-and-swap state checks."""
    applied: dict[str, Any] = {"add": [], "remove": [], "failures": []}
    for label in plan["remove"]:
        abort = _apply_label_change(
            number, repo=repo, label=label, action="remove", applied=applied
        )
        if abort is not None:
            return {
                "schema": SCHEMA,
                "number": number,
                "terminal_class": terminal_class,
                "ok": False,
                "applied": True,
                "error": abort["error"],
                "applied_changes": abort["applied_changes"],
            }
    for label in plan["add"]:
        abort = _apply_label_change(number, repo=repo, label=label, action="add", applied=applied)
        if abort is not None:
            return {
                "schema": SCHEMA,
                "number": number,
                "terminal_class": terminal_class,
                "ok": False,
                "applied": True,
                "error": abort["error"],
                "applied_changes": abort["applied_changes"],
            }
    failures = applied["failures"]
    final_state = fetch_item_state(number, repo=repo)
    return {
        "schema": SCHEMA,
        "number": number,
        "terminal_class": terminal_class,
        "ok": not failures,
        "applied": True,
        "failures": failures,
        "applied_changes": applied,
        "final_labels": sorted(final_state["labels"]) if final_state["ok"] else None,
        "reason": reason,
    }


def reconcile_item(
    number: int,
    terminal_class: str,
    *,
    repo: str = DEFAULT_REPO,
    reason: str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Compute (and optionally apply) the terminal-label plan for one item.

    In apply mode the live state is re-read immediately before each label
    mutation; a reopen or a concurrent label change aborts with a structured
    error. Manual labels outside the controlled namespace are never touched.
    """
    live = fetch_item_state(number, repo=repo)
    if not live["ok"]:
        return {"schema": SCHEMA, "number": number, "ok": False, "error": live["error"]}
    effective_reason = reason or live.get("reason")
    plan = plan_for_terminal(terminal_class, live["labels"], reason=effective_reason)
    if not apply:
        return {
            "schema": SCHEMA,
            "number": number,
            "terminal_class": terminal_class,
            "ok": True,
            "applied": False,
            "before": sorted(live["labels"]),
            "add": plan["add"],
            "remove": plan["remove"],
            "after": sorted((set(live["labels"]) - set(plan["remove"])) | set(plan["add"])),
            "preserved": plan["preserved"],
            "reason": effective_reason,
        }
    return _apply_plan(
        number,
        terminal_class,
        plan,
        repo=repo,
        reason=effective_reason,
    )


def build_report(
    items: list[tuple[int, str]],
    *,
    repo: str = DEFAULT_REPO,
    apply: bool = False,
) -> dict[str, Any]:
    """Reconcile a list of ``(number, terminal_class)`` pairs."""
    rows = [
        reconcile_item(number, terminal_class, repo=repo, apply=apply)
        for number, terminal_class in items
    ]
    failed = [row for row in rows if not row.get("ok")]
    return {
        "schema": SCHEMA,
        "ok": not failed,
        "applied": apply,
        "item_count": len(rows),
        "failed_count": len(failed),
        "items": rows,
    }


def _parse_int_list(values: list[str]) -> list[tuple[int, str]]:
    """Parse ``NUM=TERMINAL_CLASS`` pairs from ``--item`` arguments."""
    parsed: list[tuple[int, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--item must be NUM=TERMINAL_CLASS, got {value!r}")
        number_text, terminal_class = value.split("=", 1)
        number = int(number_text)
        if number < 1:
            raise ValueError(f"item number must be positive, got {number}")
        if terminal_class not in TERMINAL_CLASSES:
            raise ValueError(
                f"unknown terminal class {terminal_class!r}; expected one of "
                + ", ".join(sorted(TERMINAL_CLASSES))
            )
        parsed.append((number, terminal_class))
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Run the planner CLI in report-only or apply mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--item",
        action="append",
        dest="items",
        required=True,
        help="NUM=TERMINAL_CLASS pair; repeatable (e.g. 42=completed).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the plan with compare-and-swap; default is report-only.",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    args = parser.parse_args(argv)

    try:
        items = _parse_int_list(args.items)
    except ValueError as exc:
        print(json.dumps({"schema": SCHEMA, "ok": False, "error": str(exc)}, sort_keys=True))
        return 2

    try:
        report = build_report(items, repo=args.repo, apply=args.apply)
    except (OSError, ValueError) as exc:
        print(json.dumps({"schema": SCHEMA, "ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
