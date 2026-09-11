"""Contract tests for the compute-window schema/reader inventory (#8861)."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from scripts.tools import compute_window_schema_reader_inventory as inventory

if TYPE_CHECKING:
    from pathlib import Path


def _write(root: Path, relative: str, payload: object) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _role(root: Path, name: str, fmt: str, *, adapter: bool = False) -> dict:
    version = f"{name}.v1"
    schema_units_display = {"distance": {"unit": "m", "display": "Distance (m)"}}
    units_display = {"distance": {"unit": "m", "display": "Distance (m)"}}
    output = _write(root, f"outputs/{name}.json", {"schema_version": version, "rows": []})
    source = _write(root, f"readers/{name}.py", f"def read_{name}(): pass\n")
    return {
        "role": name,
        "path": output.relative_to(root).as_posix(),
        "format": fmt,
        "schema": {
            "name": f"{name}.schema",
            "version": version,
            "units_display": schema_units_display,
        },
        "reader": {
            "symbol": f"read_{name}",
            "source_path": source.relative_to(root).as_posix(),
            "source_sha256": _sha(source),
            "available": True,
        },
        "dependencies": [],
        "compatibility": (
            {"mode": "declared_adapter", "adapter_symbol": "migrate_v0_to_v1"}
            if adapter
            else {"mode": "native"}
        ),
        "units_display": units_display,
        "source_material": [
            {"path": source.relative_to(root).as_posix(), "sha256": _sha(source), "kind": "reader"}
        ],
        "check_command": f"uv run python scripts/tools/check_{name}.py --check",
    }


def test_all_representative_formats_are_deterministic(tmp_path: Path) -> None:
    roles = [
        _role(tmp_path, "result", "json"),
        _role(tmp_path, "episodes", "jsonl"),
        _role(tmp_path, "table", "parquet-like"),
        _role(tmp_path, "trace", "trace"),
        _role(tmp_path, "snapshot", "snapshot"),
        _role(tmp_path, "migrated", "migrated", adapter=True),
    ]
    (tmp_path / "outputs/episodes.json").write_text(
        '{"schema_version": "episodes.v1", "episode_id": "e1"}\n', encoding="utf-8"
    )
    packet = {"schema_version": inventory.SCHEMA, "issue": 8861, "roles": list(reversed(roles))}
    first = inventory.build_inventory(packet, tmp_path)
    second = inventory.build_inventory(packet, tmp_path)
    assert first == second and first["ok"]
    assert [role["role"] for role in first["roles"]] == sorted(role["role"] for role in roles)
    assert next(role for role in first["roles"] if role["role"] == "migrated")["status"] == (
        "readable_with_declared_adapter"
    )


@pytest.mark.parametrize(
    ("mutation", "status", "code"),
    [
        ("unversioned", "unversioned", "unversioned"),
        ("wrong_schema_version", "conflict", "reader_schema_mismatch"),
        ("missing_reader", "reader_unavailable", "reader_unavailable"),
        ("missing_symbol", "conflict", "reader_schema_mismatch"),
        ("wrong_digest", "conflict", "reader_schema_mismatch"),
        ("missing_dependency", "conflict", "optional_dependency_gap"),
        ("unit_drift", "conflict", "unit_display_drift"),
        ("absolute_path", "conflict", "hidden_absolute_path"),
        ("missing_check_command", "conflict", "missing_field"),
        ("missing_role", "conflict", "ambiguous_role"),
    ],
)
def test_fail_closed_states_are_explicit(  # noqa: C901
    tmp_path: Path, mutation: str, status: str, code: str
) -> None:
    role = _role(tmp_path, "fixture", "json")
    if mutation == "unversioned":
        role["schema"]["version"] = ""
    elif mutation == "wrong_schema_version":
        _write(tmp_path, "outputs/fixture.json", {"schema_version": "fixture.v0", "rows": []})
    elif mutation == "missing_reader":
        role["reader"]["available"] = False
    elif mutation == "missing_symbol":
        role["reader"]["symbol"] = "not_in_source"
    elif mutation == "wrong_digest":
        role["reader"]["source_sha256"] = "0" * 64
    elif mutation == "missing_dependency":
        role["dependencies"] = [{"name": "pyarrow", "required": True, "available": False}]
    elif mutation == "unit_drift":
        role["units_display"]["distance"]["unit"] = "km"
    elif mutation == "absolute_path":
        role["path"] = "/tmp/fixture.json"
    elif mutation == "missing_check_command":
        del role["check_command"]
    elif mutation == "missing_role":
        role["role"] = ""
    report = inventory.build_inventory(
        {"schema_version": inventory.SCHEMA, "issue": 8861, "roles": [role]}, tmp_path
    )
    assert report["ok"] is False
    assert report["roles"][0]["status"] == status
    assert code in {error["code"] for error in report["errors"]}


def test_cli_json_and_table_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    role = _role(tmp_path, "fixture", "json")
    packet = _write(
        tmp_path,
        "packet.json",
        {"schema_version": inventory.SCHEMA, "issue": 8861, "roles": [role]},
    )
    assert inventory.main(["--packet", str(packet), "--check"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    assert inventory.main(["--packet", str(packet), "--format", "table"]) == 0
    assert "role | format | schema | status" in capsys.readouterr().out


def test_cli_table_is_fail_closed_for_malformed_role(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    packet = _write(
        tmp_path,
        "packet.json",
        {"schema_version": inventory.SCHEMA, "issue": 8861, "roles": [{"role": "broken"}]},
    )
    assert inventory.main(["--packet", str(packet), "--format", "table"]) == 2
    assert "broken | ? | ?@? | conflict" in capsys.readouterr().out
