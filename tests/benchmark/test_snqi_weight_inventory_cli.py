"""CLI coverage for the SNQI weight-set provenance inventory."""

from __future__ import annotations

import json
import sys

import pytest

from robot_sf.benchmark.snqi import cli as snqi_cli


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
