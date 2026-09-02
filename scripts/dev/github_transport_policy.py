#!/usr/bin/env python3
"""Declare and audit the transport contract for GitHub workflow helpers.

The repository has two independent GitHub transport concerns:

* native ``gh`` reads can fail because a CLI asks for a retired GraphQL field;
* REST writes and reads must remain fail-closed for authentication, permission,
  repository-resolution, malformed-response, and verification failures.

This module is the single policy owner for those decisions.  Runtime helpers
import the marker constants or the classifier, while shell helpers invoke the
small ``check`` or ``classify`` commands.  The ``audit`` command discovers
every ``gh_*.py`` and ``gh_*.sh`` helper and rejects a new file until it has a
registered contract, a policy reference, and a focused smoke-test path.

Examples:

    python scripts/dev/github_transport_policy.py show
    python scripts/dev/github_transport_policy.py audit --json
    python scripts/dev/github_transport_policy.py check --helper scripts/dev/gh_comment.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POLICY_SCHEMA = "github_transport_policy.v1"
PROJECT_CARDS_ERROR_MARKER = "repository.issue.projectCards"

# GraphQL-path failures mean the native CLI route is unavailable, not that the
# requested issue is invalid.  The specific fail-closed markers are checked
# first, so a permission failure containing the word GraphQL never falls back.
FALLBACK_ELIGIBLE_MARKERS = (PROJECT_CARDS_ERROR_MARKER, "graphql:")
FAIL_CLOSED_ERROR_MARKERS = (
    "bad credentials",
    "requires authentication",
    "authentication required",
    "resource not accessible by integration",
    "forbidden",
    "permission denied",
    "could not resolve to a repository",
    "could not resolve to an issue",
    "repository not found",
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_HELPER_GLOBS = ("gh_*.py", "gh_*.sh")


@dataclass(frozen=True, slots=True)
class TransportContract:
    """Machine-readable transport expectations for one GitHub helper."""

    helper: str
    purpose: str
    allowed_transports: tuple[str, ...]
    fallback_markers: tuple[str, ...]
    fail_closed_markers: tuple[str, ...]
    smoke_test: str
    help_command: str = "--help"
    policy_reference: str = "github_transport_policy"

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        return {
            "helper": self.helper,
            "purpose": self.purpose,
            "allowed_transports": list(self.allowed_transports),
            "fallback_markers": list(self.fallback_markers),
            "fail_closed_markers": list(self.fail_closed_markers),
            "smoke_test": self.smoke_test,
            "help_command": self.help_command,
            "policy_reference": self.policy_reference,
        }


def _rest_contract(helper: str, purpose: str, smoke_test: str) -> TransportContract:
    """Build the common pure-REST contract for one helper."""
    return TransportContract(
        helper=helper,
        purpose=purpose,
        allowed_transports=("rest",),
        fallback_markers=(),
        fail_closed_markers=FAIL_CLOSED_ERROR_MARKERS,
        smoke_test=smoke_test,
    )


TRANSPORT_CONTRACTS: dict[str, TransportContract] = {
    "gh_comment.sh": _rest_contract(
        "gh_comment.sh",
        "publish issue and pull-request conversation comments",
        "tests/test_ci_script_contract.py",
    ),
    "gh_issue_rest.py": TransportContract(
        helper="gh_issue_rest.py",
        purpose="read complete issue threads with a bounded native-to-REST fallback",
        allowed_transports=("native_gh", "rest_fallback"),
        fallback_markers=FALLBACK_ELIGIBLE_MARKERS,
        fail_closed_markers=FAIL_CLOSED_ERROR_MARKERS,
        smoke_test="tests/dev/test_gh_issue_rest.py",
    ),
    "gh_issue_view.sh": TransportContract(
        helper="gh_issue_view.sh",
        purpose="provide the issue-view compatibility entry point",
        allowed_transports=("native_gh", "rest_fallback"),
        fallback_markers=FALLBACK_ELIGIBLE_MARKERS,
        fail_closed_markers=FAIL_CLOSED_ERROR_MARKERS,
        smoke_test="tests/test_ci_script_contract.py",
    ),
    "gh_pr_body_rest.py": _rest_contract(
        "gh_pr_body_rest.py",
        "update and verify pull-request title and body metadata",
        "tests/dev/test_gh_pr_body_rest.py",
    ),
    "gh_pr_comments_rest.py": _rest_contract(
        "gh_pr_comments_rest.py",
        "read pull-request conversation comments",
        "tests/dev/test_gh_pr_comments_rest.py",
    ),
    "gh_pr_label_rest.py": _rest_contract(
        "gh_pr_label_rest.py",
        "read, add, or remove and verify issue or pull-request labels",
        "tests/dev/test_gh_pr_label_rest.py",
    ),
    "gh_pr_merge.sh": TransportContract(
        helper="gh_pr_merge.sh",
        purpose=(
            "perform an exact-head native merge with worktree-conflict or "
            "GraphQL-quota REST fallback"
        ),
        allowed_transports=("native_gh", "rest_fallback"),
        # This is a transport-level marker.  gh_pr_merge.sh applies the
        # narrower rate-limit/quota predicate before entering this fallback.
        fallback_markers=("already used by worktree", "graphql:"),
        fail_closed_markers=FAIL_CLOSED_ERROR_MARKERS,
        smoke_test="tests/test_ci_script_contract.py",
    ),
    "gh_pr_review_rest.py": _rest_contract(
        "gh_pr_review_rest.py",
        "publish an exact-head pull-request review",
        "tests/dev/test_gh_pr_review_rest.py",
    ),
}


def registered_helpers() -> tuple[str, ...]:
    """Return registered helper names in deterministic order."""
    return tuple(sorted(TRANSPORT_CONTRACTS))


def get_transport_contract(helper: str | Path) -> TransportContract:
    """Return the contract identified by a helper path or basename."""
    name = Path(helper).name
    try:
        return TRANSPORT_CONTRACTS[name]
    except KeyError as exc:
        known = ", ".join(registered_helpers())
        raise KeyError(f"unregistered GitHub helper {name!r}; known helpers: {known}") from exc


def classify_error(helper: str | Path, error: str) -> dict[str, Any]:
    """Classify one transport error as fallback-eligible or fail-closed.

    Fail-closed markers deliberately take precedence over fallback markers.  A
    missing, empty, or unrecognized error is never eligible for fallback.
    """
    contract = get_transport_contract(helper)
    normalized = str(error or "").casefold()
    fail_matches = tuple(
        marker for marker in contract.fail_closed_markers if marker.casefold() in normalized
    )
    fallback_matches = tuple(
        marker for marker in contract.fallback_markers if marker.casefold() in normalized
    )
    if fail_matches:
        decision = "fail_closed"
        reason = "fail_closed_marker"
    elif fallback_matches:
        decision = "fallback"
        reason = "fallback_marker"
    else:
        decision = "fail_closed"
        reason = "unrecognized_error"
    return {
        "schema": POLICY_SCHEMA,
        "helper": contract.helper,
        "decision": decision,
        "reason": reason,
        "matched_fail_closed_markers": list(fail_matches),
        "matched_fallback_markers": list(fallback_matches),
    }


def is_fallback_eligible(error: str, *, helper: str = "gh_issue_rest.py") -> bool:
    """Return whether *error* may safely use the helper's fallback route."""
    return classify_error(helper, error)["decision"] == "fallback"


def _discovered_helpers(scripts_dir: Path) -> list[str]:
    """Find issue, pull-request, and comment helpers in one scripts directory."""
    discovered: set[str] = set()
    for pattern in _HELPER_GLOBS:
        discovered.update(path.name for path in scripts_dir.glob(pattern) if path.is_file())
    return sorted(discovered)


def _helper_path(root: Path, helper: str) -> Path:
    """Resolve a helper name or path against the repository root."""
    candidate = Path(helper)
    if candidate.is_absolute():
        return candidate
    if candidate.parts[:2] == ("scripts", "dev"):
        return root / candidate
    return root / "scripts" / "dev" / candidate.name


def check_helper(helper: str | Path, *, root: Path = _REPOSITORY_ROOT) -> dict[str, Any]:
    """Validate registration, source reference, and smoke-test metadata."""
    root = root.resolve()
    path = _helper_path(root, str(helper))
    name = path.name
    findings: list[dict[str, str]] = []
    contract = TRANSPORT_CONTRACTS.get(name)
    if contract is None:
        findings.append(
            {
                "kind": "missing_registration",
                "helper": name,
                "detail": "helper is not present in TRANSPORT_CONTRACTS",
            }
        )
        return {
            "schema": POLICY_SCHEMA,
            "status": "error",
            "helper": name,
            "path": str(path),
            "findings": findings,
        }

    if not path.is_file():
        findings.append(
            {"kind": "missing_file", "helper": name, "detail": f"file does not exist: {path}"}
        )
    else:
        source = path.read_text(encoding="utf-8")
        if contract.policy_reference not in source:
            findings.append(
                {
                    "kind": "missing_policy_reference",
                    "helper": name,
                    "detail": f"source does not reference {contract.policy_reference}",
                }
            )

    smoke_path = root / contract.smoke_test
    if not smoke_path.is_file():
        findings.append(
            {
                "kind": "missing_smoke_test",
                "helper": name,
                "detail": f"smoke test does not exist: {contract.smoke_test}",
            }
        )
    if not contract.help_command:
        findings.append(
            {"kind": "missing_help_contract", "helper": name, "detail": "help command is empty"}
        )

    return {
        "schema": POLICY_SCHEMA,
        "status": "ok" if not findings else "error",
        "helper": name,
        "path": str(path),
        "contract": contract.as_dict(),
        "findings": findings,
    }


def audit_helpers(*, root: Path = _REPOSITORY_ROOT) -> dict[str, Any]:
    """Audit all discovered ``gh_*`` helpers against the canonical registry."""
    root = root.resolve()
    scripts_dir = root / "scripts" / "dev"
    discovered = _discovered_helpers(scripts_dir) if scripts_dir.is_dir() else []
    checks = [check_helper(name, root=root) for name in discovered]
    findings = [finding for check in checks for finding in check.get("findings", [])]
    missing_files = sorted(set(registered_helpers()) - set(discovered))
    findings.extend(
        {
            "kind": "registered_file_missing",
            "helper": name,
            "detail": "registered helper is not discovered in scripts/dev",
        }
        for name in missing_files
    )
    return {
        "schema": POLICY_SCHEMA,
        "status": "ok" if not findings else "error",
        "root": str(root),
        "scripts_dir": str(scripts_dir),
        "discovered_helpers": discovered,
        "registered_helpers": list(registered_helpers()),
        "checks": checks,
        "findings": findings,
    }


def _print_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    """Print either stable JSON or a concise human-readable result."""
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    status = payload.get("status", "ok")
    print(f"{payload.get('schema', POLICY_SCHEMA)}: {status}")
    if payload.get("helper"):
        print(f"helper: {payload['helper']}")
    if payload.get("decision"):
        print(f"decision: {payload['decision']} ({payload.get('reason', '')})")
    findings = payload.get("findings", [])
    for finding in findings:
        print(f"- {finding.get('kind')}: {finding.get('detail')}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show, classify, and audit the canonical GitHub helper transport policy."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show", help="show all registered helper contracts")
    show.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    audit = subparsers.add_parser("audit", help="audit every discovered gh_* helper")
    audit.add_argument("--root", type=Path, default=_REPOSITORY_ROOT)
    audit.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    check = subparsers.add_parser(
        "check", help="check one helper registration and source reference"
    )
    check.add_argument("--helper", required=True, help="helper path or basename")
    check.add_argument("--root", type=Path, default=_REPOSITORY_ROOT)
    check.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    classify = subparsers.add_parser("classify", help="classify one helper error")
    classify.add_argument("--helper", required=True, help="helper path or basename")
    classify.add_argument("--error", required=True, help="transport error text")
    classify.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the policy CLI and return a fail-closed process status."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "show":
            payload = {
                "schema": POLICY_SCHEMA,
                "status": "ok",
                "contracts": [TRANSPORT_CONTRACTS[name].as_dict() for name in registered_helpers()],
            }
            _print_payload(payload, as_json=args.json)
            return 0
        if args.command == "audit":
            payload = audit_helpers(root=args.root)
            _print_payload(payload, as_json=args.json)
            return 0 if payload["status"] == "ok" else 1
        if args.command == "check":
            payload = check_helper(args.helper, root=args.root)
            _print_payload(payload, as_json=args.json)
            return 0 if payload["status"] == "ok" else 1
        payload = classify_error(args.helper, args.error)
    except (KeyError, OSError, UnicodeError) as exc:
        payload = {
            "schema": POLICY_SCHEMA,
            "status": "error",
            "error": str(exc),
        }
        _print_payload(payload, as_json=args.json)
        return 2

    _print_payload(payload, as_json=args.json)
    return 0 if payload["decision"] == "fallback" else 1


if __name__ == "__main__":
    sys.exit(main())
