"""Focused contract tests for the fail-closed output-capacity preflight (#8841)."""

from __future__ import annotations

import contextlib
import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from scripts.tools import check_output_capacity_preflight as tool

FIX = Path(__file__).parent / "fixtures" / "output_capacity_preflight"
OK, EXCEEDED, UNKNOWN = FIX / "ok", FIX / "exceeded", FIX / "unknown"
PRIVATE_PACKET = "/scratch/private-researcher-42/capacity"
PRIVATE_HOST = "gpu-node-17.cluster.invalid"


def _run(argv: list[str]) -> tuple[int, str]:
    out = StringIO()
    with contextlib.redirect_stdout(out):
        code = tool.main(argv)
    return code, out.getvalue()


def _cli(packet: Path, storage: Path, fmt: str = "json") -> list[str]:
    return [
        "--check",
        "--packet",
        str(packet),
        "--storage-projection",
        str(storage),
        "--format",
        fmt,
    ]


def _load(directory: Path) -> tuple[int, dict[str, Any]]:
    code, output = _run(_cli(directory / "packet.json", directory / "storage_capability.json"))
    return code, json.loads(output)


def _packet() -> dict[str, Any]:
    return json.loads((OK / "packet.json").read_text(encoding="utf-8"))


def _projection() -> dict[str, Any]:
    return json.loads((OK / "storage_capability.json").read_text(encoding="utf-8"))


def _component(payload: dict[str, Any], component_id: str) -> dict[str, Any]:
    return next(item for item in payload["components"] if item["component_id"] == component_id)


def _run_case(
    packet: dict[str, Any], projection: dict[str, Any], tmp_path: Path
) -> tuple[int, dict[str, Any]]:
    packet_path = tmp_path / "packet.json"
    storage_path = tmp_path / "storage_capability.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    storage_path.write_text(json.dumps(projection), encoding="utf-8")
    code, output = _run(_cli(packet_path, storage_path))
    return code, json.loads(output)


def _codes(payload: dict[str, Any]) -> set[str]:
    return {issue["code"] for issue in payload["issues"]}


def test_ok_fixture_passes_with_provenance_class_separation_and_bounds():
    """Exact pass keeps every class/kind separate and binds empirical provenance."""
    code, payload = _load(OK)
    assert code == 0 and payload["status"] == tool.STATUS_OK and payload["issues"] == []
    assert payload["check_only"] is True and payload["schema"] == tool.REPORT_SCHEMA
    estimate = payload["estimate"]
    assert estimate["row_count"] == 100 and estimate["row_scaling"] == "bounded"
    assert set(estimate["by_storage_class"]) == set(tool.STORAGE_CLASSES)
    assert set(estimate["by_output_kind"]) == set(tool.OUTPUT_KINDS) - {"other"}
    assert estimate["destination"]["storage_classes"] == sorted(tool.DESTINATION_CLASSES)
    assert estimate["destination"]["bytes"]["upper"] == 139500
    assert estimate["totals"]["peak_bytes"]["upper"] == 159000
    for bounds in (estimate["totals"]["bytes"], estimate["totals"]["files"]):
        assert bounds["lower"] <= bounds["expected"] <= bounds["upper"]
    rows = _component(estimate, "rows-task")
    assert rows["basis"] == "empirical" and rows["source_identity"] == "smoke-0001"
    assert payload["transfer"]["deadline_fit"] == "fits"
    assert payload["storage"]["source"]["fits_bytes"] is True
    assert payload["storage"]["destination"]["fits_inodes"] is True
    assert len(payload["packet_sha256"]) == 64 and len(payload["storage_projection_sha256"]) == 64


def test_exceeded_fixture_reports_each_conservative_storage_miss():
    """A bounded packet with oversized outputs exceeds source and destination capacity."""
    code, payload = _load(EXCEEDED)
    assert code == 2 and payload["status"] == tool.STATUS_EXCEEDED
    assert _codes(payload) == {
        "source_bytes_exceeded",
        "source_inodes_exceeded",
        "destination_bytes_exceeded",
        "destination_inodes_exceeded",
    }
    assert payload["storage"]["source"]["fits_bytes"] is False
    assert payload["storage"]["source"]["fits_inodes"] is False
    assert payload["storage"]["destination"]["fits_bytes"] is False
    assert payload["storage"]["destination"]["fits_inodes"] is False


def test_unknown_fixture_never_passes_on_unbounded_output_or_missing_probes():
    """Unbounded traces plus unavailable destination, rate, and deadline stay unknown."""
    code, payload = _load(UNKNOWN)
    assert code == 2 and payload["status"] == tool.STATUS_UNKNOWN
    assert {
        "uncertainty_unbounded",
        "capacity_unavailable",
        "access_deadline_unavailable",
        "transfer_rate_unavailable",
        "transfer_rate_uncertainty_unknown",
    } <= _codes(payload)
    assert payload["storage"]["destination"]["fits_bytes"] is None
    assert payload["transfer"]["deadline_fit"] == "unknown"
    assert payload["transfer"]["duration_seconds"]["upper"] is None


def test_exact_boundary_passes_and_one_unit_over_exceeds(tmp_path):
    """Conservative upper plus margin must fit exactly; one byte or inode less blocks."""
    code, baseline = _load(OK)
    assert code == 0
    projection = _projection()
    source, destination = baseline["storage"]["source"], baseline["storage"]["destination"]
    projection["source"] = {
        "free_bytes": source["required_bytes"]["upper"],
        "free_inodes": source["required_inodes"]["upper"],
        "retention_class": "scratch_ephemeral",
    }
    projection["destination"] = {
        "free_bytes": destination["required_bytes"]["upper"],
        "free_inodes": destination["required_inodes"]["upper"],
        "retention_class": "durable_90d",
    }
    projection["safety_margin"] = {"bytes": 0, "inodes": 0, "time_seconds": 0}
    code, payload = _run_case(_packet(), projection, tmp_path)
    assert code == 0 and payload["status"] == tool.STATUS_OK
    projection["destination"]["free_bytes"] = destination["required_bytes"]["upper"] - 1
    code, payload = _run_case(_packet(), projection, tmp_path)
    assert code == 2 and _codes(payload) == {"destination_bytes_exceeded"}


def test_many_small_files_exceed_destination_inodes(tmp_path):
    """File-per-row pressure alone can fail the inode budget."""
    packet, projection = _packet(), _projection()
    _component(packet, "rows-task")["files"] = {"per_row": [2, 2, 5], "fixed": [0, 0, 0]}
    projection["destination"]["free_inodes"] = 550
    code, payload = _run_case(packet, projection, tmp_path)
    assert code == 2 and _codes(payload) == {"destination_inodes_exceeded"}
    assert payload["storage"]["destination"]["required_inodes"]["upper"] == 517


def test_sparse_large_checkpoints_exceed_destination_bytes_keep_inodes(tmp_path):
    """Sparse checkpoint volume can exhaust bytes while staying inside the inode budget."""
    packet, projection = _packet(), _projection()
    _component(packet, "checkpoints-sparse")["bytes"] = {
        "per_row": [0, 0, 0],
        "fixed": [1000000, 2000000, 3000000],
    }
    code, payload = _run_case(packet, projection, tmp_path)
    assert code == 2
    assert "destination_bytes_exceeded" in _codes(payload)
    assert "destination_inodes_exceeded" not in _codes(payload)


def test_compression_expansion_exceeds_source_peak_only(tmp_path):
    """Temporary compression workspace is peak-only and never part of the transfer volume."""
    packet, projection = _packet(), _projection()
    _component(packet, "compression-work")["peak_bytes"] = {
        "per_row": [0, 0, 0],
        "fixed": [1000000, 20000000, 50000000],
    }
    code, payload = _run_case(packet, projection, tmp_path)
    assert code == 2 and _codes(payload) == {"source_bytes_exceeded"}
    assert payload["storage"]["destination"]["fits_bytes"] is True
    assert payload["estimate"]["destination"]["bytes"]["upper"] == 139500


def test_transfer_deadline_miss_uses_conservative_rate_and_reports_uncertainty(tmp_path):
    """A conservative lower-bound rate that misses the access window blocks the packet."""
    packet, projection = _packet(), _projection()
    projection["transfer"]["rate_bytes_per_second"] = [100, 200, 400]
    projection["access_deadline_utc"] = packet["transfer"]["task_completion_latest_utc"]
    code, payload = _run_case(packet, projection, tmp_path)
    assert code == 2 and _codes(payload) == {"transfer_deadline_missed"}
    assert payload["transfer"]["window_seconds"] == 0.0
    assert payload["transfer"]["duration_seconds"] == {
        "lower": 273.75,
        "expected": 597.5,
        "upper": 1395.0,
    }


def test_unknown_transfer_rate_uncertainty_blocks_pass(tmp_path):
    """Bounded rate numbers with unknown rate uncertainty stay explicitly unknown."""
    packet, projection = _packet(), _projection()
    projection["transfer"]["rate_uncertainty"] = "unknown"
    code, payload = _run_case(packet, projection, tmp_path)
    assert code == 2 and _codes(payload) == {"transfer_rate_uncertainty_unknown"}
    assert payload["transfer"]["deadline_fit"] == "unknown"


@pytest.mark.parametrize(
    ("mutate", "code", "expected"),
    [
        (
            lambda packet, projection: packet.update(row_scaling="unbounded"),
            "row_scaling_unbounded",
            tool.STATUS_UNKNOWN,
        ),
        (
            lambda packet, projection: _component(packet, "slurm-logs").update(
                uncertainty="unbounded"
            ),
            "uncertainty_unbounded",
            tool.STATUS_UNKNOWN,
        ),
        (
            lambda packet, projection: _component(packet, "rows-task").update(
                compatibility="incompatible"
            ),
            "incompatible_source_identity",
            tool.STATUS_UNKNOWN,
        ),
        (
            lambda packet, projection: projection["transfer"].update(
                rate_uncertainty="unavailable"
            ),
            "transfer_rate_uncertainty_unknown",
            tool.STATUS_UNKNOWN,
        ),
    ],
)
def test_unbounded_or_unknown_inputs_cannot_pass(tmp_path, mutate, code, expected):
    """Each unbounded or unknown input fails closed with status capacity_unknown."""
    packet, projection = _packet(), _projection()
    mutate(packet, projection)
    result, payload = _run_case(packet, projection, tmp_path)
    assert result == 2 and payload["status"] == expected
    assert code in _codes(payload)


def test_malformed_wrong_schema_and_unknown_fields_fail_closed(tmp_path):
    """Unreadable input exits 3; wrong schema or unknown fields stay unknown, not passing."""
    code, _ = _run(_cli(tmp_path / "missing.json", OK / "storage_capability.json"))
    assert code == 3
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[1, 2, 3]", encoding="utf-8")
    code, _ = _run(_cli(malformed, OK / "storage_capability.json"))
    assert code == 3
    packet = _packet()
    packet["schema"] = "wrong.v9"
    packet["extra"] = 1
    code, payload = _run_case(packet, _projection(), tmp_path)
    assert code == 2 and payload["status"] == tool.STATUS_UNKNOWN
    assert {"invalid_schema", "unknown_field"} <= _codes(payload)
    with pytest.raises(SystemExit):
        tool.main(_cli(OK / "packet.json", OK / "storage_capability.json")[1:])


def test_private_values_are_never_echoed(tmp_path):
    """Private paths and hostnames are rejected and sanitized out of the report."""
    packet = _packet()
    packet["packet_id"] = PRIVATE_PACKET
    _component(packet, "rows-task")["source_identity"] = PRIVATE_HOST
    code, payload = _run_case(packet, _projection(), tmp_path)
    assert code == 2 and "unsanitized_input" in _codes(payload)
    rendered = json.dumps(payload)
    assert PRIVATE_PACKET not in rendered and PRIVATE_HOST not in rendered


def test_render_is_byte_stable_and_text_is_compact(capsys):
    """Repeated renders are byte-stable; text output is a short human explanation."""
    payload = tool.build_report(_packet(), _projection())
    first, second = tool.render_report_json(payload), tool.render_report_json(payload)
    assert first == second
    assert tool.render_report_text(payload).startswith("capacity_ok:")
    code, output = _run(_cli(OK / "packet.json", OK / "storage_capability.json", fmt="text"))
    assert code == 0 and "deadline=fits" in output and len(output.splitlines()) <= 7
    assert capsys.readouterr().err == ""


def test_check_only_does_not_mutate_inputs(tmp_path):
    """A full preflight leaves both input documents byte-identical."""
    packet_path, storage_path = OK / "packet.json", OK / "storage_capability.json"
    before = (packet_path.read_bytes(), storage_path.read_bytes())
    _load(OK)
    assert (packet_path.read_bytes(), storage_path.read_bytes()) == before
