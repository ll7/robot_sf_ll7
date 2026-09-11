"""Focused tests for pre-submission SLURM array mapping verifier (#8854)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts.validation.verify_slurm_array_mapping import (
    EXIT_BLOCKED,
    EXIT_MALFORMED,
    EXIT_VERIFIED,
    EXPECTED_LEDGER_SCHEMA,
    LAUNCHER_SCHEMA,
    RECEIPT_SCHEMA,
    main,
    verify_array_mapping,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_ledger(
    count: int = 10,
    *,
    campaign_id: str = "campaign-smoke",
    duplicate_row: bool = False,
    empty: bool = False,
    schema: str = EXPECTED_LEDGER_SCHEMA,
) -> dict[str, Any]:
    """Create test expected-row ledger payload."""
    if empty:
        return {"schema_version": schema, "campaign_id": campaign_id, "rows": []}

    rows = []
    for i in range(count):
        row_id = f"row_{i:03d}"
        rows.append(
            {
                "row_id": row_id,
                "planner": "orca",
                "scenario": f"scene_{i % 3}",
                "seed": 100 + i,
                "replicate": 0,
                "output_path": f"output/{row_id}.json",
            }
        )
    if duplicate_row and rows:
        rows.append(dict(rows[0]))

    return {
        "schema_version": schema,
        "campaign_id": campaign_id,
        "source_tree_sha256": "a" * 64,
        "rows": rows,
    }


def _make_launcher(
    *,
    job_type: str = "array",
    campaign_id: str = "campaign-smoke",
    schema: str = LAUNCHER_SCHEMA,
    array_spec: dict[str, Any] | None = None,
    mapping_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create test SLURM array launcher payload."""
    launcher: dict[str, Any] = {
        "schema_version": schema,
        "campaign_id": campaign_id,
        "source_tree_sha256": "a" * 64,
        "job_type": job_type,
        "resource_request": {
            "partition": "compute",
            "time_limit": "01:00:00",
            "cpus_per_task": 1,
            "mem_mb": 2048,
        },
        "mapping": mapping_override or {},
    }
    if job_type == "array":
        default_spec: dict[str, Any] = {
            "min_index": 0,
            "max_index": 9,
            "step": 1,
            "chunk_size": 1,
            "task_offset": 0,
            "zero_based": True,
            "concurrency": 4,
        }
        if array_spec:
            default_spec.update(array_spec)
        launcher["array_spec"] = default_spec
    return launcher


def test_valid_exact_mapping(tmp_path: Path) -> None:
    """Verify clean 1-to-1 exact array mapping."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(10)
    launcher = _make_launcher(array_spec={"min_index": 0, "max_index": 9, "chunk_size": 1})

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    res = verify_array_mapping(ledger, launcher, ledger_path=ledger_p, launcher_path=launcher_p)
    assert res["status"] == "verified"
    assert res["schema_version"] == RECEIPT_SCHEMA
    assert res["expected_row_count"] == 10
    assert res["scheduled_task_count"] == 10
    assert res["scheduled_row_count"] == 10
    assert res["reason_codes"] == []
    assert res["first_error"] is None
    assert len(res["scheduled_positions"]) == 10


def test_chunked_rows_with_non_divisible_tail(tmp_path: Path) -> None:
    """Verify chunked tasks with non-divisible tail boundary."""
    # 10 rows with chunk_size 3 -> tasks 0, 1, 2, 3 (4 tasks: 3+3+3+1 rows)
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(10)
    launcher = _make_launcher(array_spec={"min_index": 0, "max_index": 3, "chunk_size": 3})

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    res = verify_array_mapping(ledger, launcher, ledger_path=ledger_p, launcher_path=launcher_p)
    assert res["status"] == "verified"
    assert res["mapping_summary"]["has_partial_tail"] is True
    assert res["mapping_summary"]["tail_size"] == 1
    assert res["scheduled_row_count"] == 10
    assert res["scheduled_task_count"] == 4


def test_scalar_job_mapping(tmp_path: Path) -> None:
    """Verify scalar job covering all expected rows."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(5)
    launcher = _make_launcher(job_type="scalar")

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    res = verify_array_mapping(ledger, launcher, ledger_path=ledger_p, launcher_path=launcher_p)
    assert res["status"] == "verified"
    assert res["mapping_summary"]["job_type"] == "scalar"
    assert res["scheduled_row_count"] == 5
    assert res["scheduled_task_count"] == 1


def test_off_by_one_bounds(tmp_path: Path) -> None:
    """Detect off-by-one array bounds (e.g. 11 tasks for 10 rows)."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(10)
    launcher = _make_launcher(array_spec={"min_index": 0, "max_index": 10, "chunk_size": 1})

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    res = verify_array_mapping(ledger, launcher, ledger_path=ledger_p, launcher_path=launcher_p)
    assert res["status"] == "blocked"
    assert "off_by_one_bounds" in res["reason_codes"]


def test_zero_one_based_confusion(tmp_path: Path) -> None:
    """Detect zero/one-based confusion where min_index=1 without task_offset."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(10)
    launcher = _make_launcher(
        array_spec={"min_index": 1, "max_index": 10, "chunk_size": 1, "zero_based": True}
    )

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    res = verify_array_mapping(ledger, launcher, ledger_path=ledger_p, launcher_path=launcher_p)
    assert res["status"] == "blocked"
    assert "zero_one_based_confusion" in res["reason_codes"]


def test_gap_in_scheduled_rows(tmp_path: Path) -> None:
    """Detect gap in scheduled rows when step skips tasks."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(10)
    # step=2 over 0..8 gives only 5 tasks, leaving 5 unmapped rows
    launcher = _make_launcher(
        array_spec={"min_index": 0, "max_index": 8, "step": 2, "chunk_size": 1}
    )

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    res = verify_array_mapping(ledger, launcher, ledger_path=ledger_p, launcher_path=launcher_p)
    assert res["status"] == "blocked"
    assert "gap_in_scheduled_rows" in res["reason_codes"]
    assert "unmapped_expected_rows" in res["reason_codes"]


def test_output_path_collision(tmp_path: Path) -> None:
    """Detect collision when output_pattern collapses distinct rows to same path."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(5)
    launcher = _make_launcher(
        array_spec={"min_index": 0, "max_index": 4, "chunk_size": 1},
        mapping_override={"output_pattern": "output/constant_name.json"},
    )

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    res = verify_array_mapping(ledger, launcher, ledger_path=ledger_p, launcher_path=launcher_p)
    assert res["status"] == "blocked"
    assert "output_path_collision" in res["reason_codes"]


def test_shard_reorder_detection(tmp_path: Path) -> None:
    """Detect scrambled manifest ordering relative to canonical ledger sequence."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(5)
    # reverse order in manifest_order
    scrambled = [r["row_id"] for r in reversed(ledger["rows"])]
    launcher = _make_launcher(
        array_spec={"min_index": 0, "max_index": 4, "chunk_size": 1},
        mapping_override={"manifest_order": scrambled},
    )

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    res = verify_array_mapping(ledger, launcher, ledger_path=ledger_p, launcher_path=launcher_p)
    assert res["status"] == "blocked"
    assert "shard_reorder" in res["reason_codes"]


def test_retry_overlap_detection(tmp_path: Path) -> None:
    """Detect retry offset that overlaps with active task range."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(10)
    launcher = _make_launcher(
        array_spec={"min_index": 0, "max_index": 9, "chunk_size": 1},
        mapping_override={"retry_mode": "retry_missing", "retry_offset": 5},
    )

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    res = verify_array_mapping(ledger, launcher, ledger_path=ledger_p, launcher_path=launcher_p)
    assert res["status"] == "blocked"
    assert "retry_overlap" in res["reason_codes"]


def test_resume_conflict_detection(tmp_path: Path) -> None:
    """Detect resume conflict when preserved row is not in expected ledger."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(5)
    launcher = _make_launcher(
        array_spec={"min_index": 0, "max_index": 4, "chunk_size": 1},
        mapping_override={
            "resume_mode": "skip_existing",
            "existing_row_ids": ["non_existent_row_999"],
        },
    )

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    res = verify_array_mapping(ledger, launcher, ledger_path=ledger_p, launcher_path=launcher_p)
    assert res["status"] == "blocked"
    assert "resume_conflict" in res["reason_codes"]


def test_integer_overflow_fails_closed(tmp_path: Path) -> None:
    """Detect integer overflow in array_spec parameters."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(5)
    launcher = _make_launcher(
        array_spec={"min_index": 0, "max_index": 999_999_999_999, "chunk_size": 1}
    )

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    res = verify_array_mapping(ledger, launcher, ledger_path=ledger_p, launcher_path=launcher_p)
    assert res["status"] == "malformed"
    assert "integer_overflow" in res["reason_codes"]


def test_invalid_concurrency_fails_closed(tmp_path: Path) -> None:
    """Detect invalid concurrency <= 0."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(5)
    launcher = _make_launcher(
        array_spec={"min_index": 0, "max_index": 4, "chunk_size": 1, "concurrency": 0}
    )

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    res = verify_array_mapping(ledger, launcher, ledger_path=ledger_p, launcher_path=launcher_p)
    assert res["status"] == "blocked"
    assert "invalid_concurrency" in res["reason_codes"]


def test_campaign_identity_mismatch(tmp_path: Path) -> None:
    """Detect campaign_id mismatch between ledger and launcher."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(5, campaign_id="campaign-A")
    launcher = _make_launcher(
        campaign_id="campaign-B",
        array_spec={"min_index": 0, "max_index": 4, "chunk_size": 1},
    )

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    res = verify_array_mapping(ledger, launcher, ledger_path=ledger_p, launcher_path=launcher_p)
    assert res["status"] == "blocked"
    assert "identity_mismatch" in res["reason_codes"]


def test_empty_ledger_fails_closed(tmp_path: Path) -> None:
    """Detect empty ledger with zero rows."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(empty=True)
    launcher = _make_launcher(array_spec={"min_index": 0, "max_index": 0})

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    res = verify_array_mapping(ledger, launcher, ledger_path=ledger_p, launcher_path=launcher_p)
    assert res["status"] == "blocked"
    assert "empty_ledger" in res["reason_codes"]


def test_duplicate_expected_rows(tmp_path: Path) -> None:
    """Detect duplicate row_id within the expected ledger itself."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(5, duplicate_row=True)
    launcher = _make_launcher(array_spec={"min_index": 0, "max_index": 5})

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    res = verify_array_mapping(ledger, launcher, ledger_path=ledger_p, launcher_path=launcher_p)
    assert res["status"] == "blocked"
    assert "duplicate_expected_rows" in res["reason_codes"]


def test_cli_verified_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI main returns 0 on verified mapping and emits valid JSON."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(6)
    launcher = _make_launcher(array_spec={"min_index": 0, "max_index": 2, "chunk_size": 2})

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    code = main(["--check", "--ledger", str(ledger_p), "--launcher", str(launcher_p)])
    assert code == EXIT_VERIFIED
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "verified"
    assert payload["expected_row_count"] == 6


def test_cli_blocked_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI main returns 2 on mapping error."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(6)
    launcher = _make_launcher(array_spec={"min_index": 0, "max_index": 5, "chunk_size": 2})

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    code = main(["--check", "--ledger", str(ledger_p), "--launcher", str(launcher_p)])
    assert code == EXIT_BLOCKED
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "blocked"


def test_cli_malformed_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI main returns 3 on missing file or malformed JSON."""
    ledger_p = tmp_path / "missing.json"
    launcher_p = tmp_path / "launcher.json"
    launcher_p.write_text("{}", encoding="utf-8")

    code = main(["--check", "--ledger", str(ledger_p), "--launcher", str(launcher_p)])
    assert code == EXIT_MALFORMED
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "malformed"
    assert "ledger_missing_or_unreadable" in payload["reason_codes"]


def test_cli_csv_format_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI emits CSV output when --format csv is requested."""
    ledger_p = tmp_path / "ledger.json"
    launcher_p = tmp_path / "launcher.json"
    ledger = _make_ledger(4)
    launcher = _make_launcher(array_spec={"min_index": 0, "max_index": 3, "chunk_size": 1})

    ledger_p.write_text(json.dumps(ledger), encoding="utf-8")
    launcher_p.write_text(json.dumps(launcher), encoding="utf-8")

    code = main(
        [
            "--check",
            "--ledger",
            str(ledger_p),
            "--launcher",
            str(launcher_p),
            "--format",
            "csv",
        ]
    )
    assert code == EXIT_VERIFIED
    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert lines[0] == "task_id,intra_task_index,scheduled_row_index,row_id,output_path"
    assert len(lines) == 5  # header + 4 rows
