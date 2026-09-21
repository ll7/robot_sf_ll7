#!/usr/bin/env python3
"""Semantics-preserving contract repair for mechanically incomplete issues.

The preparation lane stalls on issues whose intent is complete but whose
contract defect is mechanical: an unambiguous heading alias for a missing
canonical section. This helper emits a versioned repair packet
(:func:`scripts.dev.issue_implementability.build_repair_packet`) and, in
apply mode, writes the renamed body through a compare-and-swap exact-read
cycle, re-reads the issue, and reruns canonical admission check-only.

The helper never adds ``state:ready``, never acquires a claim, never touches
labels, and never invents content: a packet is refused when acceptance or
verification prose would have to be synthesized, reinterpreted, or decided.
Readiness and claiming stay on the existing readiness-gate and admission
owners.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.dev import goal_issue_admission, issue_implementability
from scripts.dev._gh_rest import parse_json, run_gh_api

SCHEMA = "issue_contract_repair_receipt.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_REMOTE = "origin"
DEFAULT_SOURCE_REF = "origin/main"


def _read_body_file(path: str) -> str:
    """Read one local body file for offline packet planning."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read body file {path}: {exc}") from exc


def _exact_read_body(number: int, *, repo: str) -> dict[str, Any]:
    """Exact-read one live issue body for compare-and-swap repair."""
    try:
        live = issue_implementability.fetch_live_issue(number, repo=repo)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"live issue read failed before repair: {exc}") from exc
    body = live.get("body")
    if not isinstance(body, str):
        raise RuntimeError("live issue body was not a string")
    return {"body": body, "labels": live.get("labels", [])}


def _patch_body(number: int, *, repo: str, expected_sha256: str, body: str) -> None:
    """Write one repaired body after verifying the live digest still matches."""
    result = run_gh_api(f"repos/{repo}/issues/{number}")
    payload, error = parse_json(result, what=f"issue {number} pre-write read")
    if error or not isinstance(payload, dict):
        raise RuntimeError(f"pre-write re-read failed: {error or 'non-object payload'}")
    live_body = payload.get("body")
    if not isinstance(live_body, str):
        raise RuntimeError("pre-write issue body was not a string")
    if issue_implementability.inspect_contract(live_body)["body_sha256"] != expected_sha256:
        raise RuntimeError("issue drifted between plan and write; refusing to apply")
    write = run_gh_api(f"repos/{repo}/issues/{number}", {"body": body}, method="PATCH")
    if write.returncode != 0:
        detail = (write.stderr or write.stdout or "").strip() or "body PATCH failed"
        raise RuntimeError(detail)


def _label_snapshot(raw: Any) -> list[str] | None:
    """Normalize live labels to a read-only snapshot; None when malformed."""
    if not isinstance(raw, list):
        return None
    names: list[str] = []
    for value in raw:
        if isinstance(value, str):
            name = value.strip()
        elif isinstance(value, dict) and isinstance(value.get("name"), str):
            name = str(value["name"]).strip()
        else:
            return None
        if name:
            names.append(name)
    return sorted(set(names))


def plan_for_body(body: str, *, labels: list[str] | None = None) -> dict[str, Any]:
    """Build a repair packet for one body without performing any I/O."""
    return issue_implementability.build_repair_packet(body, labels=labels)


def apply_for_issue(
    number: int,
    *,
    repo: str = DEFAULT_REPO,
    remote: str = DEFAULT_REMOTE,
    source_ref: str = DEFAULT_SOURCE_REF,
) -> dict[str, Any]:
    """Repair one live issue body, then verify it and rerun admission."""
    first = _exact_read_body(number, repo=repo)
    packet = issue_implementability.build_repair_packet(
        first["body"], labels=_label_snapshot(first.get("labels"))
    )
    if not packet.get("repairable"):
        return {
            "schema": SCHEMA,
            "issue": number,
            "repo": repo,
            "applied": False,
            "reason": packet.get("reason", "repair refused"),
            "packet": packet,
            "admission": None,
        }
    repaired = issue_implementability.apply_repair_packet(first["body"], packet)
    _patch_body(number, repo=repo, expected_sha256=str(packet["body_sha256"]), body=repaired)
    second = _exact_read_body(number, repo=repo)
    if second["body"] != repaired:
        raise RuntimeError("post-write readback differs from the repaired body")
    admission = goal_issue_admission.admit_issue(
        number,
        repo=repo,
        remote=remote,
        source_ref=source_ref,
        check_only=True,
    )
    return {
        "schema": SCHEMA,
        "issue": number,
        "repo": repo,
        "applied": True,
        "reason": "",
        "packet": packet,
        "post_body_sha256": packet.get("expected_body_sha256"),
        "admission": {
            "ok": admission.get("ok"),
            "outcome": admission.get("outcome"),
            "classification": (admission.get("preflight") or {}).get("classification"),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository as OWNER/REPO.")
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="Git remote for admission.")
    parser.add_argument(
        "--source-ref", default=DEFAULT_SOURCE_REF, help="Source ref for admission."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="print a repair packet without writing")
    plan.add_argument("--body-file", default=None, help="Offline body file to plan.")
    plan.add_argument("--issue", type=int, default=None, help="Live issue to plan.")
    apply = subparsers.add_parser("apply", help="repair one live issue with CAS checks")
    apply.add_argument("--issue", type=int, required=True, help="Live issue to repair.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run repair planning or CAS-guarded application."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            if args.body_file is not None:
                packet = plan_for_body(_read_body_file(args.body_file))
            elif args.issue is not None:
                packet = plan_for_body(_exact_read_body(args.issue, repo=args.repo)["body"])
            else:
                print("repair plan needs --body-file or --issue", file=sys.stderr)
                return 2
            print(json.dumps(packet, indent=2, sort_keys=True))
            return 0
        receipt = apply_for_issue(
            args.issue, repo=args.repo, remote=args.remote, source_ref=args.source_ref
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        # Success means the repair was applied and verified; the embedded
        # admission outcome may still name other lanes (labels, readiness).
        return 0 if receipt.get("applied") else 1
    except (RuntimeError, ValueError) as exc:
        print(f"issue contract repair blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
