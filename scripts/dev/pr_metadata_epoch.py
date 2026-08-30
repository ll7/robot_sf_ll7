#!/usr/bin/env python3
"""Compute and verify deterministic PR metadata epochs (issue #7649).

A metadata epoch binds the dimensions that must be stable before a pull
request enters expensive integration: exact head/base identity, normalized
title and body digest, linked-issue and closing-reference sets, admission
labels, reviewer state, and approval requirements. The epoch is report-only:
it never authorizes merge, labels, comments, or any GitHub write.

Usage:
    gh pr view 8010 --json number,title,body,headRefOid,baseRefOid,labels,\
closingIssuesReferences,reviewDecision,state \
      | python scripts/dev/pr_metadata_epoch.py compute --json

    python scripts/dev/pr_metadata_epoch.py verify --previous epoch.json \
      --current pr.json --json

Exit codes for ``verify``: 0 epoch stable, 1 material change, 2 usage/input
error. Cosmetic-only body differences are normalized away and never change
the epoch (see ``scripts/dev/pr_metadata.py`` normalization rule).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from scripts.dev.pr_metadata import (
    PrMetadataEpochInputs,
    build_pr_metadata_epoch,
    diff_epochs,
)

_ISSUE_REF_RE = re.compile(r"(?<!\w)#(\d+)\b")


def _extract_issue_refs(pr: dict) -> tuple[list[int], list[int]]:
    closing: list[int] = []
    for ref in pr.get("closingIssuesReferences") or []:
        number = ref.get("number")
        if isinstance(number, int):
            closing.append(number)
    body_refs = [int(match.group(1)) for match in _ISSUE_REF_RE.finditer(pr.get("body") or "")]
    linked = sorted({n for n in (*body_refs, *closing) if n > 0})
    return linked, sorted(set(closing))


def _requested_reviewers(pr: dict) -> list[str]:
    reviewers: list[str] = []
    for source in ("requestedReviewers", "reviewRequests"):
        for entry in pr.get(source) or []:
            login = entry.get("login") or entry.get("name")
            if login:
                reviewers.append(str(login))
    return reviewers


def _sha(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("sha") or "")
    return str(value or "")


def compute_epoch(pr: dict, repository: str) -> dict:
    """Build an epoch record from a ``gh pr view``- or REST pulls-shaped JSON object."""
    linked, closing = _extract_issue_refs(pr)
    return build_pr_metadata_epoch(
        PrMetadataEpochInputs(
            pr_number=int(pr.get("number") or 0),
            repository=repository,
            head_sha=_sha(pr.get("headRefOid") or pr.get("head")),
            base_sha=_sha(pr.get("baseRefOid") or pr.get("base")),
            title=pr.get("title") or "",
            body=pr.get("body") or "",
            linked_issues=linked,
            closing_references=closing,
            labels=[str(label["name"]) for label in (pr.get("labels") or []) if label.get("name")],
            requested_reviewers=_requested_reviewers(pr),
            review_decision=pr.get("reviewDecision") or "",
            domain_approval_required=bool(pr.get("domainApprovalRequired")),
        )
    )


def _load_json(path: str | None) -> dict:
    if path == "-" or path is None:
        return json.load(sys.stdin)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    compute = sub.add_parser("compute", help="Compute an epoch record from PR JSON.")
    compute.add_argument("--input", "-i", help="gh pr view JSON file ('-' or stdin by default)")
    compute.add_argument("--repository", default="ll7/robot_sf_ll7")
    compute.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    verify = sub.add_parser("verify", help="Verify a recorded epoch against fresh PR JSON.")
    verify.add_argument("--previous", "-p", required=True, help="Recorded epoch JSON file")
    verify.add_argument("--current", "-c", required=True, help="Fresh PR JSON file ('-' for stdin)")
    verify.add_argument("--repository", default="ll7/robot_sf_ll7")
    verify.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "compute":
        pr = _load_json(args.input)
        epoch = compute_epoch(pr, args.repository)
        if args.json:
            print(json.dumps(epoch, indent=2, ensure_ascii=False))
        else:
            print(f"epoch digest: {epoch['digest']}")
            print(f"head: {epoch['head_sha']}  base: {epoch['base_sha']}")
            print(f"title: {epoch['title_normalized']}")
        return 0

    previous = _load_json(args.previous)
    if previous.get("schema") != "pr_metadata_epoch.v1":
        raise SystemExit(f"previous file is not pr_metadata_epoch.v1: {previous.get('schema')} (2)")
    current_pr = _load_json(args.current)
    current = compute_epoch(current_pr, args.repository)
    changes = diff_epochs(previous, current)
    stable = current["digest"] == previous.get("digest")
    report = {
        "status": "stable" if stable else "changed",
        "previous_digest": previous.get("digest"),
        "current_digest": current["digest"],
        "changed_fields": changes,
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"status: {report['status']}")
        for change in changes:
            print(f"  {change['dimension']}: {change['before']} -> {change['after']}")
    return 0 if stable else 1


if __name__ == "__main__":
    try:
        sys.exit(_main())
    except SystemExit as exc:
        if isinstance(exc.code, int):
            raise
        sys.stderr.write(str(exc) + "\n")
        sys.exit(2)
