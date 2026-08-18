#!/usr/bin/env python3
"""Validate exact-head SHA carriers embedded in pull-request metadata."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

FULL_SHA = re.compile(r"[0-9a-fA-F]{40}")
_CARRIER_PATTERNS = (
    ("gate-verdict", re.compile(r"(?im)\bgate-verdict\s*:[^\n]*?\b([0-9a-f]{40})\b")),
    ("base-policy", re.compile(r"(?im)\bbase-policy\s*:[^\n]*?\b([0-9a-f]{40})\b")),
    ("Exact head", re.compile(r"(?im)\bexact\s+head\s*:\s*[^\n]*?\b([0-9a-f]{40})\b")),
)


@dataclass(frozen=True)
class ShaCarrier:
    """A recognized provenance label and its full SHA-1 value."""

    label: str
    sha: str


def extract_sha_carriers(body: str) -> list[ShaCarrier]:
    """Extract full SHA carriers from the supported metadata labels."""

    carriers: list[ShaCarrier] = []
    for label, pattern in _CARRIER_PATTERNS:
        carriers.extend(
            ShaCarrier(label=label, sha=match.group(1)) for match in pattern.finditer(body)
        )
    return carriers


def git_object_type(sha: str, *, repo_root: Path) -> str | None:
    """Return the local Git object type, or ``None`` when the SHA is unresolved."""

    try:
        result = subprocess.run(
            ["git", "cat-file", "-t", sha],
            cwd=repo_root,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    object_type = result.stdout.strip()
    return object_type or None


def validate_sha_carriers(
    body: str,
    *,
    live_head_sha: str,
    repo_root: Path,
) -> list[str]:
    """Return fail-closed errors for fabricated exact-head metadata.

    A full carrier is truthful for this narrow guard when it is the live PR
    head or resolves to an object in the checked-out repository. The second
    allowance preserves valid historical references while rejecting values
    that never existed anywhere locally and are not the live head.
    """

    carriers = extract_sha_carriers(body)
    if not carriers:
        return []
    if not FULL_SHA.fullmatch(live_head_sha):
        return [f"live PR head is not a full SHA: {live_head_sha!r}"]

    live_lower = live_head_sha.lower()
    errors: list[str] = []
    for carrier in carriers:
        if carrier.sha.lower() == live_lower:
            continue
        if git_object_type(carrier.sha, repo_root=repo_root) is not None:
            continue
        errors.append(
            f"{carrier.label} carrier SHA {carrier.sha} is neither the live PR head "
            f"{live_head_sha} nor a local Git object"
        )
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--body-file", type=Path)
    source.add_argument(
        "--github-event-path",
        type=Path,
        help="pull_request event JSON containing the live body and head SHA",
    )
    parser.add_argument("--head-sha", help="live pull-request head SHA (required with --body-file)")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate a body file and emit one deterministic result."""

    args = _build_parser().parse_args(argv)
    if args.body_file is not None:
        if args.head_sha is None:
            _build_parser().error("--head-sha is required with --body-file")
        try:
            body = args.body_file.read_text(encoding="utf-8")
        except OSError as exc:
            payload = {"status": "error", "error": f"could not read body file: {exc}"}
            print(json.dumps(payload, sort_keys=True) if args.json else payload["error"])
            return 2
        live_head_sha = args.head_sha
    else:
        assert args.github_event_path is not None
        try:
            event = json.loads(args.github_event_path.read_text(encoding="utf-8"))
            pull_request = event["pull_request"]
            body = pull_request.get("body") or ""
            live_head_sha = pull_request["head"]["sha"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            payload = {"status": "error", "error": f"malformed GitHub event: {exc}"}
            print(json.dumps(payload, sort_keys=True) if args.json else payload["error"])
            return 2

    carriers = extract_sha_carriers(body)
    errors = validate_sha_carriers(
        body,
        live_head_sha=live_head_sha,
        repo_root=args.repo_root,
    )
    payload = {
        "status": "pass" if not errors else "blocked",
        "live_head_sha": live_head_sha,
        "carriers": [asdict(carrier) for carrier in carriers],
        "errors": errors,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif errors:
        print("PR body provenance validation failed:")
        for error in errors:
            print(f"- {error}")
    else:
        print(f"PR body provenance validation passed ({len(carriers)} carrier(s)).")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
