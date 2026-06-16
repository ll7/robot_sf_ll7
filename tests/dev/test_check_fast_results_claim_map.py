"""Tests for the fast-results claim-map queue validator."""

from __future__ import annotations

import json

from scripts.dev.check_fast_results_claim_map import main, validate_claim_map


def test_validate_claim_map_accepts_executable_priority_tables(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The happy path should require owner issues, statuses, and gates."""
    claim_map = tmp_path / "claim_map.md"
    claim_map.write_text(
        """# Claim Map

### p0_now -- No blocking gates

| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|
| Resolve table conflict | #2612 | ready | Artifact: context note update | Diff proof | Missing until #2612 closes |

### p1_after_gate -- Gated

| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|
| Suite freeze | #2910 | blocked | Artifact: freeze note | #2911 accepted | Missing until gate closes |

### parked_blocked -- Parked

| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|
| External row | #2397 | do-not-claim | Artifact: unavailable note | External runtime available | Not available |
""",
        encoding="utf-8",
    )

    report = validate_claim_map(claim_map)

    assert report["ok"] is True
    assert report["sections"]["p0_now"]["row_count"] == 1


def test_validate_claim_map_rejects_p0_without_exactly_one_owner_issue(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """p0 rows must name exactly one executable owner issue."""
    claim_map = tmp_path / "claim_map.md"
    claim_map.write_text(
        """# Claim Map

### p0_now -- No blocking gates

| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|
| Missing owner | #1 and #2 | ready | Artifact: context note update | Diff proof | Missing until close |

### p1_after_gate -- Gated

| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|
| Gate | #3 | blocked | Artifact: gate note | Gate proof | Missing until close |

### parked_blocked -- Parked

| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|
| Parked | #4 | do-not-claim | Artifact: park note | Gate proof | Missing until close |
""",
        encoding="utf-8",
    )

    report = validate_claim_map(claim_map)

    assert report["ok"] is False
    assert any(error["field"] == "owner issue" for error in report["errors"])


def test_validate_claim_map_accepts_empty_well_formed_priority_tables(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Empty but well-formed priority tables should be valid."""
    claim_map = tmp_path / "claim_map.md"
    claim_map.write_text(
        """# Claim Map

### p0_now -- No blocking gates

| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|

### p1_after_gate -- Gated

| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|

### parked_blocked -- Parked

| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|
""",
        encoding="utf-8",
    )

    report = validate_claim_map(claim_map)

    assert report["ok"] is True
    assert report["sections"]["p0_now"]["row_count"] == 0
    assert report["sections"]["p1_after_gate"]["row_count"] == 0
    assert report["sections"]["parked_blocked"]["row_count"] == 0


def test_validate_claim_map_rejects_completed_local_output_only_evidence(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Completed rows must not point only at worktree-local output."""
    claim_map = tmp_path / "claim_map.md"
    claim_map.write_text(
        """# Claim Map

### p0_now -- No blocking gates

| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|
| Done badly | #2612 | completed | Artifact: result note | Diff proof | output/run/report.json |

### p1_after_gate -- Gated

| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|
| Gate | #3 | blocked | Artifact: gate note | Gate proof | Missing until close |

### parked_blocked -- Parked

| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|
| Parked | #4 | do-not-claim | Artifact: park note | Gate proof | Missing until close |
""",
        encoding="utf-8",
    )

    report = validate_claim_map(claim_map)

    assert report["ok"] is False
    assert any(error["field"] == "durable evidence" for error in report["errors"])


def test_main_emits_json_report(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    """CLI JSON mode should be stable for CI consumption."""
    claim_map = tmp_path / "claim_map.md"
    claim_map.write_text("# Claim Map\n", encoding="utf-8")

    rc = main(["--claim-map", str(claim_map), "--json"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "fast_results_claim_map_check.v1"
    assert payload["ok"] is False
