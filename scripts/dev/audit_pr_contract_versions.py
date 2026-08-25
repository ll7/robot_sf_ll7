"""Inventory PR contract versions across open pull requests.

Classifies each open PR as ``v2_valid``, ``v2_invalid``, ``v1_compatibility``,
or an explicit unavailable state using the canonical ``pr_contract_v2`` parser.
The tool is read-only: it never edits a PR body, exposes tokens, or infers
scientific validity.  A deterministic JSON report is available for offline
fixtures, and ``--check`` fails closed on malformed v2 markers.

Part A of issue #7892 (pr-contract.v2 live-migration slice).  v1 removal and
retirement behavior are explicitly out of scope.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.dev.pr_contract_v2 import parse_pr_contract_v2

REPORT_SCHEMA = "pr_contract_version_inventory.v1"
RETIREMENT_POLICY = "compatibility"  # v1-compatible PRs are reported, not failed


def _classify_pr(record: dict[str, Any]) -> dict[str, Any]:
    """Classify one PR metadata record against the v2 contract parser."""
    number = record.get("number")
    body = record.get("body")
    if body is None:
        return {
            "number": number,
            "url": record.get("url"),
            "author": record.get("author"),
            "draft": bool(record.get("isDraft") or record.get("draft")),
            "head_sha": record.get("headRefOid") or record.get("headSha"),
            "contract_class": "body_missing",
            "contract_status": "absent",
            "errors": [],
            "message": "PR body unavailable; cannot classify.",
        }
    result = parse_pr_contract_v2(body, source=f"pr-{number}")
    if result.status == "ok":
        contract_class = "v2_valid"
    elif result.status == "malformed":
        contract_class = "v2_invalid"
    else:
        contract_class = "v1_compatibility"
    return {
        "number": number,
        "url": record.get("url"),
        "author": record.get("author"),
        "draft": bool(record.get("isDraft") or record.get("draft")),
        "head_sha": record.get("headRefOid") or record.get("headSha"),
        "contract_class": contract_class,
        "contract_status": result.status,
        "errors": list(result.errors),
        "message": result.message,
    }


def _fetch_open_prs(repo: str) -> list[dict[str, Any]]:
    """Fetch open PR metadata through ``gh`` (REST-compatible JSON fields)."""
    command = [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--json",
        "number,title,url,author,isDraft,headRefOid,body",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"gh pr list failed (exit {proc.returncode}): {proc.stderr.strip()}")
    try:
        records = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gh pr list returned invalid JSON: {exc}") from exc
    if not isinstance(records, list):
        raise RuntimeError("gh pr list returned a non-list payload")
    return records


def _load_fixture(path: str) -> list[dict[str, Any]]:
    """Load a PR metadata fixture from JSON (offline test path)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("prs"), list):
        return payload["prs"]
    raise ValueError("fixture must be a list of PR records or an object with a 'prs' list")


def build_inventory(records: list[dict[str, Any]], *, source: str = "fixture") -> dict[str, Any]:
    """Build the versioned inventory report from PR metadata records."""
    rows = [_classify_pr(record) for record in records]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["contract_class"]] = counts.get(row["contract_class"], 0) + 1
    return {
        "schema": REPORT_SCHEMA,
        "source": source,
        "counts": counts,
        "retirement_policy": RETIREMENT_POLICY,
        "prs": rows,
    }


def _check_inventory(report: dict[str, Any]) -> tuple[int, list[str]]:
    """Return (exit_code, problems).  Malformed v2 markers fail closed."""
    problems: list[str] = []
    for row in report["prs"]:
        if row["contract_class"] == "v2_invalid":
            problems.append(
                f"PR {row['number']} has a malformed pr-contract:v2 marker: "
                + "; ".join(row["errors"])
            )
    return (1 if problems else 0, problems)


def main(argv: list[str] | None = None) -> int:
    """Run the inventory CLI and return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="owner/repo to inventory through gh")
    parser.add_argument("--fixture", help="path to a JSON fixture of PR metadata records (offline)")
    parser.add_argument("--check", action="store_true", help="fail closed on malformed v2 markers")
    parser.add_argument("--output", help="write the JSON report to this path")
    args = parser.parse_args(argv)

    if args.repo and args.fixture:
        parser.error("--repo and --fixture are mutually exclusive")
    if not args.repo and not args.fixture:
        parser.error("one of --repo or --fixture is required")

    if args.repo:
        records = _fetch_open_prs(args.repo)
        report = build_inventory(records, source=args.repo)
    else:
        records = _load_fixture(args.fixture)
        report = build_inventory(records, source=f"fixture:{args.fixture}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    exit_code, problems = _check_inventory(report)
    if args.check and problems:
        print(json.dumps(report, indent=2, sort_keys=True))
        for problem in problems:
            print(f"check failed: {problem}", file=sys.stderr)
        return exit_code
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
