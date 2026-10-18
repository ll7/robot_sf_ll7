#!/usr/bin/env python3
"""Fail closed when a live ``state:*`` label is missing from the shared taxonomy.

An unclassified ``state:*`` label makes every consumer that reasons about issue
state refuse the issue with a reason that describes a code defect rather than the
issue's actual state.  This check enumerates the live label inventory and fails
when any ``state:*`` label is absent from
:data:`scripts.dev.issue_state_taxonomy.KNOWN_STATE_LABELS`.

The only tolerated absences are the explicitly named entries in
:data:`DELIBERATELY_UNCLASSIFIED_STATE_LABELS`.  There is no wildcard: a
mistyped label must still fail closed.

This is a read-only advisory check.  It never writes labels, and it is not wired
into required CI, because it depends on the GitHub label API.  It supports
``--labels-file`` so the unit test is hermetic and does not need network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    # Resolve this checkout before any other editable installation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dev import issue_state_taxonomy as taxonomy
from scripts.dev._gh_rest import run_gh_api

SCHEMA = "state_label_inventory.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"


def classify_labels(labels: list[str]) -> dict[str, Any]:
    """Classify one label inventory against the shared taxonomy.

    Returns unclassified labels split into ``unclassified`` (a genuine gap that
    must fail closed) and ``deliberately_unclassified`` (named exclusions). An
    empty ``unclassified`` list means the inventory is fully covered.
    """
    state_labels = sorted({label for label in labels if label.startswith(taxonomy.STATE_PREFIX)})
    excluded = taxonomy.DELIBERATELY_UNCLASSIFIED_STATE_LABELS
    unclassified: list[str] = []
    deliberately: list[str] = []
    for label in state_labels:
        if label in taxonomy.KNOWN_STATE_LABELS:
            continue
        if label in excluded:
            deliberately.append(label)
            continue
        unclassified.append(label)
    return {
        "schema": SCHEMA,
        "state_labels": state_labels,
        "unclassified": unclassified,
        "deliberately_unclassified": deliberately,
        "deliberate_exclusions": sorted(excluded),
        "ok": not unclassified,
    }


def _fetch_live_labels(repo: str) -> list[str]:
    """Read the live repository label inventory through the GitHub API."""
    result = run_gh_api(f"repos/{repo}/labels", extra_args=["--paginate", "--jq", ".[].name"])
    if result.returncode != 0:
        raise RuntimeError(f"label inventory read failed (exit {result.returncode})")
    text = result.stdout
    return [line.strip().strip('"') for line in text.splitlines() if line.strip()]


def _read_labels_file(path: str) -> list[str]:
    """Read a label inventory from a local file, one label per line or JSON list."""
    raw = Path(path).read_text(encoding="utf-8")
    stripped = raw.strip()
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        return [str(item) for item in parsed]
    return [
        line.strip() for line in stripped.splitlines() if line.strip() and not line.startswith("#")
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the inventory check and emit one JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--labels-file",
        help="Read the label inventory from a local file instead of the GitHub API.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args(argv)

    source = "labels_file" if args.labels_file else "github_api"
    try:
        labels = (
            _read_labels_file(args.labels_file)
            if args.labels_file
            else _fetch_live_labels(args.repo)
        )
    except (OSError, ValueError, RuntimeError) as exc:
        result = {
            "schema": SCHEMA,
            "source": source,
            "repo": args.repo,
            "ok": False,
            "error": str(exc),
        }
        print(json.dumps(result, sort_keys=True) if args.json else f"FAIL: {exc}")
        return 2

    result = classify_labels(labels)
    result["source"] = source
    result["repo"] = args.repo

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print(
            f"OK: {len(result['state_labels'])} live state:* label(s) classified; "
            f"{len(result['deliberately_unclassified'])} deliberate exclusion(s)."
        )
    else:
        print("FAIL: unclassified state:* label(s): " + ", ".join(result["unclassified"]))
        print("  Add each to STATE_QUALIFIER_LABELS or EXECUTION_STATE_LABELS in")
        print("  scripts/dev/issue_state_taxonomy.py, or name it in")
        print("  DELIBERATELY_UNCLASSIFIED_STATE_LABELS if the gap is deliberate.")

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
