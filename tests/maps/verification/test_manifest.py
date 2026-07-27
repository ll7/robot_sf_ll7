"""Contract tests locking the output behavior of the manifest serializers.

These tests pin the *current* parsed-JSON structure emitted by
``robot_sf.maps.verification.manifest.write_manifest`` and
``write_jsonl_manifest`` so that a silent schema, enum-serialization,
timestamp, or line-layout regression is caught directly at the output boundary
rather than only by a downstream consumer.

They assert parsed JSON structures (never raw formatting/whitespace) and confine
all filesystem effects to pytest's ``tmp_path``. They are intentionally
behavior-locking: if the observed serializer output changes, these tests fail so
the change is reviewed before it reaches downstream tooling.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from robot_sf.maps.verification.context import (
    FactoryType,
    VerificationResult,
    VerificationRunSummary,
    VerificationStatus,
)
from robot_sf.maps.verification.manifest import write_jsonl_manifest, write_manifest

if TYPE_CHECKING:
    from pathlib import Path

# Deterministic constants used to build invariant-valid fixtures. Fixed values
# keep the assertions reproducible rather than dependent on datetime.now().
RUN_ID = "run-abc-123"
GIT_SHA = "0123456789abcdef0123456789abcdef01234567"


def _ts(hour: int, minute: int = 0, second: int = 0) -> datetime:
    """Build a fixed UTC-aware timestamp for deterministic fixtures.

    Timezone-aware datetimes are used so the emitted ISO-8601 strings carry an
    explicit offset; the serializers round-trip them via ``.isoformat()``.
    """
    return datetime(2026, 7, 26, hour, minute, second, tzinfo=UTC)


STARTED_AT = _ts(12, 0, 0)
FINISHED_AT = _ts(12, 0, 5)

# Exact key contracts emitted by the current serializers. Asserting the full key
# set catches both accidental additions and removals at the output boundary.
MANIFEST_TOP_KEYS = {"run_id", "git_sha", "started_at", "finished_at", "summary", "results"}
SUMMARY_KEYS = {"total_maps", "passed", "failed", "warned", "slow_maps"}
RESULT_KEYS = {
    "map_id",
    "status",
    "rule_ids",
    "duration_ms",
    "factory_used",
    "message",
    "timestamp",
}
JSONL_METADATA_KEYS = {
    "type",
    "run_id",
    "git_sha",
    "started_at",
    "finished_at",
    "total_maps",
    "passed",
    "failed",
    "warned",
}
JSONL_RESULT_KEYS = {
    "type",
    "run_id",
    "map_id",
    "status",
    "rule_ids",
    "duration_ms",
    "factory_used",
    "message",
    "timestamp",
}


def _make_result(
    *,
    map_id: str,
    status: VerificationStatus,
    factory_used: FactoryType = FactoryType.ROBOT,
    rule_ids: list[str] | None = None,
    duration_ms: float = 10.0,
    message: str = "diagnostic",
    timestamp: datetime | None = None,
) -> VerificationResult:
    """Build an invariant-valid :class:`VerificationResult`.

    Non-PASS statuses require non-empty ``rule_ids`` (enforced by
    ``VerificationResult.__post_init__``); this helper supplies a default rule id
    for them so fixtures stay valid by construction.
    """
    if rule_ids is None:
        rule_ids = [] if status == VerificationStatus.PASS else ["R001"]
    return VerificationResult(
        map_id=map_id,
        status=status,
        rule_ids=rule_ids,
        duration_ms=duration_ms,
        factory_used=factory_used,
        message=message,
        timestamp=timestamp if timestamp is not None else STARTED_AT,
    )


def _make_summary(
    *,
    results: list[VerificationResult],
    finished_at: datetime | None = FINISHED_AT,
    slow_maps: list[str] | None = None,
    git_sha: str | None = GIT_SHA,
) -> VerificationRunSummary:
    """Build an invariant-valid :class:`VerificationRunSummary`.

    Aggregate counts and ``total_maps`` are derived from ``results`` so the
    ``passed + failed + warned == total_maps`` invariant holds by construction.
    ``finished_at`` defaults to a timestamp at or after ``STARTED_AT``.
    """
    passed = sum(1 for r in results if r.status == VerificationStatus.PASS)
    failed = sum(1 for r in results if r.status == VerificationStatus.FAIL)
    warned = sum(1 for r in results if r.status == VerificationStatus.WARN)
    return VerificationRunSummary(
        run_id=RUN_ID,
        git_sha=git_sha,
        total_maps=len(results),
        passed=passed,
        failed=failed,
        warned=warned,
        slow_maps=slow_maps if slow_maps is not None else [],
        artifact_path=None,
        started_at=STARTED_AT,
        finished_at=finished_at,
        results=results,
    )


def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into a list of per-line dicts; fail on any bad line."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


# ---------------------------------------------------------------------------
# write_manifest() contract
# ---------------------------------------------------------------------------


def test_write_manifest_creates_missing_parent_directories(tmp_path: Path) -> None:
    """write_manifest should create deeply nested missing parent directories."""
    output_path = tmp_path / "nested" / "deeper" / "manifest.json"
    assert not output_path.parent.exists()

    write_manifest(_make_summary(results=[]), output_path)

    assert output_path.is_file()
    # The created file must contain valid JSON with the expected run id.
    assert json.loads(output_path.read_text())["run_id"] == RUN_ID


def test_write_manifest_emits_documented_top_level_and_summary(tmp_path: Path) -> None:
    """Top-level keys and the aggregate summary block must match the contract."""
    results = [
        _make_result(map_id="map_a", status=VerificationStatus.FAIL, message="boom"),
        _make_result(map_id="map_b", status=VerificationStatus.PASS, message="ok"),
    ]
    summary = _make_summary(results=results, slow_maps=["map_a"])

    output_path = tmp_path / "manifest.json"
    write_manifest(summary, output_path)

    data = json.loads(output_path.read_text())
    assert set(data) == MANIFEST_TOP_KEYS
    assert data["run_id"] == RUN_ID
    assert data["git_sha"] == GIT_SHA

    summary_block = data["summary"]
    assert set(summary_block) == SUMMARY_KEYS
    assert summary_block == {
        "total_maps": 2,
        "passed": 1,
        "failed": 1,
        "warned": 0,
        "slow_maps": ["map_a"],
    }


def test_write_manifest_per_map_result_fields(tmp_path: Path) -> None:
    """Each emitted result dict must carry exactly the documented fields."""
    results = [
        _make_result(
            map_id="map_a",
            status=VerificationStatus.WARN,
            factory_used=FactoryType.PEDESTRIAN,
            rule_ids=["R009"],
            duration_ms=42.5,
            message="watch out",
            timestamp=_ts(12, 0, 3),
        ),
    ]
    summary = _make_summary(results=results)

    output_path = tmp_path / "manifest.json"
    write_manifest(summary, output_path)

    data = json.loads(output_path.read_text())
    assert len(data["results"]) == 1
    result = data["results"][0]
    assert set(result) == RESULT_KEYS
    assert result["map_id"] == "map_a"
    assert result["rule_ids"] == ["R009"]
    assert result["duration_ms"] == 42.5
    assert result["message"] == "watch out"


@pytest.mark.parametrize(
    ("status", "factory", "rule_ids"),
    [
        (VerificationStatus.PASS, FactoryType.ROBOT, []),
        (VerificationStatus.FAIL, FactoryType.ROBOT, ["E001"]),
        (VerificationStatus.WARN, FactoryType.PEDESTRIAN, ["W001"]),
    ],
)
def test_write_manifest_serializes_enums_and_datetimes(
    tmp_path: Path,
    status: VerificationStatus,
    factory: FactoryType,
    rule_ids: list[str],
) -> None:
    """Status/factory enums serialize as their public string values and
    datetimes serialize as ISO-8601 strings that round-trip exactly."""
    result = _make_result(
        map_id="m",
        status=status,
        factory_used=factory,
        rule_ids=rule_ids,
        timestamp=_ts(12, 0, 1),
    )
    summary = _make_summary(results=[result])

    output_path = tmp_path / "manifest.json"
    write_manifest(summary, output_path)

    data = json.loads(output_path.read_text())
    emitted = data["results"][0]
    assert emitted["status"] == status.value
    assert isinstance(emitted["status"], str)
    assert emitted["factory_used"] == factory.value
    assert isinstance(emitted["factory_used"], str)

    # ISO-8601 datetime strings round-trip back to the original datetimes.
    assert data["started_at"] == STARTED_AT.isoformat()
    assert datetime.fromisoformat(data["started_at"]) == STARTED_AT
    assert datetime.fromisoformat(data["finished_at"]) == FINISHED_AT
    assert datetime.fromisoformat(emitted["timestamp"]) == _ts(12, 0, 1)


def test_write_manifest_missing_finished_at_is_json_null(tmp_path: Path) -> None:
    """A missing finished_at (and git_sha) must serialize as JSON null."""
    summary = _make_summary(results=[], finished_at=None, git_sha=None)

    output_path = tmp_path / "manifest.json"
    write_manifest(summary, output_path)

    data = json.loads(output_path.read_text())
    assert data["finished_at"] is None
    assert data["git_sha"] is None
    # started_at remains a present, ISO-formatted string when finished_at is null.
    assert data["started_at"] == STARTED_AT.isoformat()


def test_write_manifest_empty_results_is_empty_array(tmp_path: Path) -> None:
    """An empty result list must emit an empty results array."""
    summary = _make_summary(results=[])

    output_path = tmp_path / "manifest.json"
    write_manifest(summary, output_path)

    data = json.loads(output_path.read_text())
    assert data["results"] == []


# ---------------------------------------------------------------------------
# write_jsonl_manifest() contract
# ---------------------------------------------------------------------------


def test_write_jsonl_manifest_creates_missing_parent_directories(tmp_path: Path) -> None:
    """write_jsonl_manifest should create deeply nested missing parent dirs."""
    output_path = tmp_path / "nested" / "deeper" / "manifest.jsonl"
    assert not output_path.parent.exists()

    write_jsonl_manifest(_make_summary(results=[]), output_path)

    assert output_path.is_file()
    first_line = output_path.read_text().splitlines()[0]
    assert json.loads(first_line)["run_id"] == RUN_ID


def test_write_jsonl_manifest_one_metadata_then_one_result_per_line(tmp_path: Path) -> None:
    """Exactly one run_metadata line precedes exactly one map_result per result,
    and every line is valid standalone JSON."""
    results = [
        _make_result(map_id="map_a", status=VerificationStatus.FAIL, message="boom"),
        _make_result(map_id="map_b", status=VerificationStatus.PASS, message="ok"),
    ]
    summary = _make_summary(results=results)

    output_path = tmp_path / "manifest.jsonl"
    write_jsonl_manifest(summary, output_path)

    lines = [line for line in output_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1 + len(results)
    parsed = [json.loads(line) for line in lines]  # every line parses standalone

    assert parsed[0]["type"] == "run_metadata"
    assert [row["type"] for row in parsed[1:]] == ["map_result", "map_result"]


def test_write_jsonl_manifest_metadata_line_fields(tmp_path: Path) -> None:
    """The run_metadata line must carry exactly the documented header fields."""
    results = [_make_result(map_id="map_a", status=VerificationStatus.FAIL)]
    summary = _make_summary(results=results, slow_maps=["map_a"])

    output_path = tmp_path / "manifest.jsonl"
    write_jsonl_manifest(summary, output_path)

    metadata = _read_jsonl(output_path)[0]
    assert set(metadata) == JSONL_METADATA_KEYS
    assert metadata["run_id"] == RUN_ID
    assert metadata["git_sha"] == GIT_SHA
    assert metadata["total_maps"] == 1
    assert metadata["passed"] == 0
    assert metadata["failed"] == 1
    assert metadata["warned"] == 0


def test_write_jsonl_manifest_carries_run_id_and_result_fields(tmp_path: Path) -> None:
    """Each map_result line carries run_id and exactly the documented fields."""
    results = [
        _make_result(
            map_id="map_a",
            status=VerificationStatus.FAIL,
            factory_used=FactoryType.ROBOT,
            rule_ids=["E001"],
            duration_ms=7.25,
            message="nope",
            timestamp=_ts(12, 0, 2),
        ),
        _make_result(
            map_id="map_b",
            status=VerificationStatus.PASS,
            factory_used=FactoryType.PEDESTRIAN,
            message="ok",
        ),
    ]
    summary = _make_summary(results=results)

    output_path = tmp_path / "manifest.jsonl"
    write_jsonl_manifest(summary, output_path)

    result_rows = [row for row in _read_jsonl(output_path) if row["type"] == "map_result"]
    assert len(result_rows) == len(results)
    for row, result in zip(result_rows, results, strict=True):
        assert set(row) == JSONL_RESULT_KEYS
        assert row["run_id"] == RUN_ID  # run_id carried into each record
        assert row["map_id"] == result.map_id
        assert row["status"] == result.status.value
        assert row["factory_used"] == result.factory_used.value
        assert row["rule_ids"] == result.rule_ids
        assert row["duration_ms"] == result.duration_ms
        assert datetime.fromisoformat(row["timestamp"]) == result.timestamp


def test_write_jsonl_manifest_missing_finished_at_is_json_null(tmp_path: Path) -> None:
    """A missing finished_at (and git_sha) must serialize as JSON null in JSONL."""
    summary = _make_summary(results=[], finished_at=None, git_sha=None)

    output_path = tmp_path / "manifest.jsonl"
    write_jsonl_manifest(summary, output_path)

    metadata = _read_jsonl(output_path)[0]
    assert metadata["finished_at"] is None
    assert metadata["git_sha"] is None
    # started_at remains a present, ISO-formatted string when finished_at is null.
    assert metadata["started_at"] == STARTED_AT.isoformat()


def test_write_jsonl_manifest_empty_results_single_metadata_line(tmp_path: Path) -> None:
    """An empty result list must emit exactly one metadata line and nothing else."""
    summary = _make_summary(results=[])

    output_path = tmp_path / "manifest.jsonl"
    write_jsonl_manifest(summary, output_path)

    rows = _read_jsonl(output_path)
    assert len(rows) == 1
    assert rows[0]["type"] == "run_metadata"
