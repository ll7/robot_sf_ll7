"""CLI coverage for the SNQI weight-set provenance inventory."""

from __future__ import annotations

import json
import sys

import pytest
from loguru import logger

from robot_sf.benchmark.snqi import cli as snqi_cli
from robot_sf.benchmark.snqi import weights_inventory


def test_weights_inventory_alias_reports_conflict_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The explicit weight-inventory alias reports #3723 conflicts fail-closed."""
    monkeypatch.setattr(sys, "argv", ["robot_sf_snqi", "weights-inventory", "--json"])

    assert snqi_cli.main() == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["has_blocking_conflict"] is True
    assert {record["name"] for record in payload["records"]} == {
        "code_default",
        "model_canonical_v1",
        "camera_ready_v1",
        "camera_ready_v2",
        "camera_ready_v3",
    }
    assert all(len(record["content_sha256"]) == 64 for record in payload["records"])
    assert any(
        conflict["kind"] == "canonical_direction_conflict"
        and conflict["severity"] == "error"
        and set(conflict["sources"]) == {"code_default", "model_canonical_v1"}
        for conflict in payload["conflicts"]
    )


def test_weights_inventory_json_is_stdout_only_with_retained_stdout_sink(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A retained Loguru stdout sink must not corrupt the machine-readable report."""
    sink_id = logger.add(lambda message: print(message, end=""), level="ERROR")
    try:
        monkeypatch.setattr(sys, "argv", ["robot_sf_snqi", "inventory", "--json"])

        assert snqi_cli.main() == 2

        captured = capsys.readouterr()
    finally:
        logger.remove(sink_id)

    payload = json.loads(captured.out)
    assert payload["has_blocking_conflict"] is True
    assert "SNQI weight-set provenance preflight failed" in captured.err


def test_weights_inventory_json_runtime_failure_is_stderr_only_with_retained_stdout_sink(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unexpected JSON-mode failures stay off stdout even with a retained sink."""

    def _fail_inventory() -> object:
        raise RuntimeError("inventory unavailable")

    monkeypatch.setattr(weights_inventory, "build_inventory_report", _fail_inventory)
    sink_id = logger.add(lambda message: print(message, end=""), level="ERROR")
    try:
        monkeypatch.setattr(sys, "argv", ["robot_sf_snqi", "inventory", "--json"])

        assert snqi_cli.main() == 3

        captured = capsys.readouterr()
    finally:
        logger.remove(sink_id)

    assert captured.out == ""
    diagnostic = json.loads(captured.err)
    assert diagnostic["event"] == "snqi_cli_failed"
    assert diagnostic["stage"] == "weights_inventory"
    assert diagnostic["exception_type"] == "RuntimeError"
    assert diagnostic["error"] == "inventory unavailable"
