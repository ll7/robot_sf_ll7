"""Narrow PR-body claims checks backed by readiness and residual ownership (#9852)."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Mapping

import yaml

READINESS_CLAIM_TAG = "[readiness-claim]"
CLOSE_WHILE_DEFERRED_TAG = "[close-while-deferred]"
_GLOBAL_PASS = re.compile(
    r"\b(?:full(?:\s+test)?\s+suite\s+(?:has\s+)?(?:passed|passes|is\s+green)"
    r"|all\s+tests\s+(?:have\s+)?(?:pass|passed|are\s+green))\b",
    re.IGNORECASE,
)
_INDEPENDENT_ALL_LANE = re.compile(r"(?:--lane\s+all\b|ROBOT_SF_TEST_LANE=all\b)")
_NEGATED_RUN = re.compile(r"\b(?:did\s+not\s+run|not\s+run|wasn['’]t\s+run|skipped)\b", re.I)
_V2_BLOCK = re.compile(r"<!--\s*pr-contract:v2\s*\n(.*?)\n\s*-->", re.DOTALL)
_DECLARED_CLOSE = re.compile(r"(?im)^\s*[-*]\s*(?:closes|fixes|resolves)\s+#(\d+)\b")
_RESIDUAL_WORDING = re.compile(r"\b(?:not\s+done|deferred|pending|unresolved)\b", re.I)
_NO_RESIDUAL = re.compile(
    r"\b(?:no\s+deferred(?:\s+implementation)?\s+work|nothing\s+remains|none)\b",
    re.I,
)
_PROSE_OWNER = re.compile(
    r"\b(?:remains?\s+under|owned\s+by|tracked\s+in|deferred\s+to)\s+#(\d+)\b",
    re.I,
)


def _section(body: str, heading: str) -> str:
    match = re.search(rf"(?ims)^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s+|\Z)", body)
    return match.group(1) if match else ""


def check_readiness_claims(body: str, receipt_text: str | None) -> list[str]:
    """Reject a global pass claim attributed to core-only readiness evidence.

    A separately named all-lane test command remains outside this narrow guard.
    In hosted CI, where the local ignored receipt is unavailable, a global
    claim needs an explicitly named independent all-lane command.
    """
    if not _GLOBAL_PASS.search(body):
        return []
    if any(
        _GLOBAL_PASS.search(line)
        and _INDEPENDENT_ALL_LANE.search(line)
        and not _NEGATED_RUN.search(line)
        for line in body.splitlines()
    ):
        return []
    if receipt_text is None:
        return [
            f"BLOCKER: {READINESS_CLAIM_TAG} A full-suite/all-tests pass claim has no "
            "readiness lane receipt or explicitly named independent all-lane command. "
            "State the lanes that ran or provide separate all-lane test evidence."
        ]
    if "Readiness lane coverage summary" not in receipt_text:
        return [f"BLOCKER: {READINESS_CLAIM_TAG} Readiness lane receipt has no coverage summary."]
    latest = receipt_text.rsplit("Readiness lane coverage summary", 1)[-1]
    core_ran = re.search(r"(?m)^\s*core lane:\s*ran\s*$", latest) is not None
    optional_ran = re.search(r"(?m)^\s*optional lane:\s*ran\s*$", latest) is not None
    optional_skipped = re.search(r"(?m)^\s*optional lane:\s*skipped\b", latest) is not None
    extended_ran = re.search(r"(?m)^\s*extended lane:\s*ran\b", latest) is not None
    extended_skipped = re.search(r"(?m)^\s*extended lane:\s*NOT RUN\s*$", latest) is not None
    if (
        not core_ran
        or not (optional_ran or optional_skipped)
        or not (extended_ran or extended_skipped)
    ):
        return [f"BLOCKER: {READINESS_CLAIM_TAG} Readiness lane receipt is incomplete."]
    if optional_skipped and extended_skipped:
        return [
            f"BLOCKER: {READINESS_CLAIM_TAG} PR body claims full-suite/all-tests success "
            "from a readiness receipt where only the core lane ran. Scope the claim to core."
        ]
    return []


def _v2_deferred(body: str) -> tuple[set[int], set[int], bool]:
    match = _V2_BLOCK.search(body)
    if not match:
        return set(), set(), False
    try:
        payload = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return set(), set(), False
    if not isinstance(payload, Mapping):
        return set(), set(), False
    linked = payload.get("linked_issues")
    deferred = payload.get("deferred_work")
    if not isinstance(linked, Mapping) or not isinstance(deferred, Mapping):
        return set(), set(), False
    closes = linked.get("closes")
    issues = deferred.get("issues")
    close_ids = (
        {value for value in closes if type(value) is int and value > 0}
        if isinstance(closes, list)
        else set()
    )
    owner_ids = (
        {value for value in issues if type(value) is int and value > 0}
        if isinstance(issues, list)
        else set()
    )
    residual = deferred.get("status") not in (None, "none") or bool(owner_ids)
    return close_ids, owner_ids, residual


def _issue_is_open(issue: int, repo: str) -> bool | None:
    try:
        result = subprocess.run(
            ["gh", "issue", "view", str(issue), "--json", "state", "--repo", repo],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return None
        state = json.loads(result.stdout).get("state")
        return state == "OPEN" if state in ("OPEN", "CLOSED") else None
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        return None


def check_close_while_deferred(
    body: str,
    repo: str,
    *,
    issue_is_open: Callable[[int, str], bool | None] = _issue_is_open,
) -> list[str]:
    """Flag a closing PR whose residual work lacks a different open owner."""
    linked_section = _section(body, "Linked Issues")
    markdown_closes = {int(value) for value in _DECLARED_CLOSE.findall(linked_section)}
    v2_closes, owners, v2_residual = _v2_deferred(body)
    closes = markdown_closes | v2_closes
    if not closes:
        return []
    residual_section = _section(body, "Follow-Up / Residual Scope")
    prose_residual = bool(_RESIDUAL_WORDING.search(residual_section)) and not bool(
        _NO_RESIDUAL.search(residual_section)
    )
    if not (v2_residual or prose_residual):
        return []
    if not owners:
        owners = {int(issue) for issue in _PROSE_OWNER.findall(residual_section)}
    blockers: list[str] = []
    if not owners:
        blockers.append(
            f"BLOCKER: {CLOSE_WHILE_DEFERRED_TAG} PR closes issue(s) "
            f"{', '.join(f'#{issue}' for issue in sorted(closes))} while residual work has "
            "no open issue owner. Use Refs or assign the residual to a different open issue."
        )
    for issue in sorted(owners):
        if issue in closes:
            blockers.append(
                f"BLOCKER: {CLOSE_WHILE_DEFERRED_TAG} PR closes #{issue} while deferred "
                f"work is still owned by #{issue}. Use Refs or a different open owner."
            )
        elif issue_is_open(issue, repo) is not True:
            blockers.append(
                f"BLOCKER: {CLOSE_WHILE_DEFERRED_TAG} Deferred owner #{issue} is not "
                "verified open; a closing PR needs a surviving open owner."
            )
    return blockers
