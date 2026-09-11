"""Contract tests for the compute-window schema/reader inventory (#8861)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.tools import compute_window_schema_reader_inventory as inventory


def _write(root: Path, relative: str, payload: object) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packet(*roles: dict) -> dict:
    return {"schema_version": inventory.SCHEMA, "issue": 8861, "roles": list(roles)}


def _role(root: Path, name: str, fmt: str, *, adapter: bool = False) -> dict:
    version = f"{name}.v1"
    units_display = {"distance": {"unit": "m", "display": "Distance (m)"}}
    schema_units_display = json.loads(json.dumps(units_display))
    output = _write(root, f"outputs/{name}.json", {"schema_version": version, "rows": []})
    source = _write(
        root,
        f"readers/{name}.py",
        f'{name.upper().replace("-", "_")}_SCHEMA_VERSION = "{version}"\ndef read_{name}(path): return path\n',
    )
    source_path = source.relative_to(root).as_posix()
    source_sha = _sha(source)
    source_ref = {"path": source_path, "sha256": source_sha}
    source_kinds = ["schema", "reader"] + (["adapter"] if adapter else [])
    compatibility = {"mode": "declared_adapter" if adapter else "native"}
    if adapter:
        compatibility.update(
            adapter_symbol=f"read_{name}",
            adapter_source_path=source_path,
            adapter_source_sha256=source_sha,
        )
    return {
        "role": name,
        "path": output.relative_to(root).as_posix(),
        "format": fmt,
        "schema": {
            "name": name,
            "version": version,
            "units_display": schema_units_display,
            "source_path": source_path,
            "source_sha256": source_sha,
        },
        "reader": {
            "symbol": f"read_{name}",
            "source_path": source_path,
            "source_sha256": source_sha,
            "available": True,
            "execution": "python_path",
        },
        "dependencies": [],
        "compatibility": compatibility,
        "units_display": units_display,
        "source_material": [{**source_ref, "kind": kind} for kind in source_kinds],
        "check_command": f"uv run python scripts/tools/check_{name}.py --check",
    }


def test_all_representative_formats_are_deterministic(tmp_path: Path) -> None:
    specs = "result:json episodes:jsonl table:parquet-like trace:trace snapshot:snapshot migrated:migrated".split()
    roles = [
        _role(tmp_path, *spec.split(":"), adapter=spec.startswith("migrated")) for spec in specs
    ]
    _write(tmp_path, "outputs/episodes.json", {"schema_version": "episodes.v1", "episode_id": "e1"})
    packet = _packet(*reversed(roles))
    first = inventory.build_inventory(packet, tmp_path)
    second = inventory.build_inventory(packet, tmp_path)
    assert first == second and first["ok"]
    assert [role["role"] for role in first["roles"]] == sorted(role["role"] for role in roles)
    assert (
        next(role for role in first["roles"] if role["role"] == "migrated")["status"]
        == "readable_with_declared_adapter"
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
        ("missing_source_material", "schema_only", "missing_source_material"),
        ("stale_adapter", "conflict", "stale_adapter"),
        ("absolute_path", "conflict", "hidden_absolute_path"),
        ("missing_check_command", "conflict", "missing_field"),
        ("missing_role", "conflict", "ambiguous_role"),
    ],
)
def test_fail_closed_states_are_explicit(
    tmp_path: Path, mutation: str, status: str, code: str
) -> None:
    role = _role(tmp_path, "fixture", "json")
    mutations = {
        "unversioned": lambda: role["schema"].update(version=""),
        "wrong_schema_version": lambda: _write(
            tmp_path, "outputs/fixture.json", {"schema_version": "fixture.v0", "rows": []}
        ),
        "missing_reader": lambda: role["reader"].update(available=False),
        "missing_symbol": lambda: role["reader"].update(symbol="not_in_source"),
        "wrong_digest": lambda: role["reader"].update(source_sha256="0" * 64),
        "missing_dependency": lambda: role.update(
            dependencies=[{"name": "pyarrow", "required": True, "available": False}]
        ),
        "unit_drift": lambda: role["units_display"]["distance"].update(unit="km"),
        "missing_source_material": lambda: role.update(source_material=[]),
        "stale_adapter": lambda: role.update(compatibility={"mode": "declared_adapter"}),
        "absolute_path": lambda: role.update(path="/tmp/fixture.json"),
        "missing_check_command": lambda: role.pop("check_command"),
        "missing_role": lambda: role.update(role=""),
    }
    mutations[mutation]()
    report = inventory.build_inventory(_packet(role), tmp_path)
    assert report["roles"][0]["status"] == status
    assert code in {error["code"] for error in report["errors"]}


def test_cli_json_and_table_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    role = _role(tmp_path, "fixture", "json")
    packet = _write(tmp_path, "packet.json", _packet(role))
    assert inventory.main(["--packet", str(packet), "--check"]) == 0
    first = capsys.readouterr().out
    assert inventory.main(["--packet", str(packet), "--check"]) == 0
    assert capsys.readouterr().out == first
    assert inventory.main(["--packet", str(packet), "--format", "table"]) == 0
    assert "role | format | schema | status" in capsys.readouterr().out


def test_cli_table_is_fail_closed_for_malformed_role(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    packet = _write(tmp_path, "packet.json", _packet({"role": "broken"}))
    assert inventory.main(["--packet", str(packet), "--format", "table"]) == 2
    assert "broken | ? | ?@? | conflict" in capsys.readouterr().out


def test_checked_in_active_role_manifest_is_readable() -> None:
    root = Path(__file__).resolve().parents[2]
    packet = inventory.load_packet(
        root / "tests/fixtures/compute_window_schema_reader_inventory/active_roles.json"
    )
    report = inventory.build_inventory(packet, root)
    assert {role["format"] for role in report["roles"]} == inventory.FORMATS
    statuses = {role["role"]: role["status"] for role in report["roles"]}
    assert (
        statuses["simulation_trace_export"]
        == statuses["research_yield_snapshot"]
        == "readable_verified"
    )
    assert set(statuses.values()) == {"readable_verified", "reader_unavailable"}


@pytest.mark.parametrize(
    "reader_body, expected_message",
    [
        ("def read_fixture(path): raise ValueError('reject')\n", "reader rejected"),
        ("import time\ndef read_fixture(path): time.sleep(3)\n", "reader timed out"),
    ],
)
def test_reader_hook_fail_closed(tmp_path: Path, reader_body: str, expected_message: str) -> None:
    role = _role(tmp_path, "fixture", "json")
    source = _write(
        tmp_path,
        "readers/reject.py",
        f'FIXTURE_SCHEMA_VERSION = "fixture.v1"\n{reader_body}',
    )
    source_path, source_sha = source.relative_to(tmp_path).as_posix(), _sha(source)
    role["schema"].update(source_path=source_path, source_sha256=source_sha)
    role["reader"].update(source_path=source_path, source_sha256=source_sha)
    role["source_material"] = [
        {"kind": kind, "path": source_path, "sha256": source_sha} for kind in ("schema", "reader")
    ]
    report = inventory.build_inventory(_packet(role), tmp_path)
    assert any(
        error["code"] == "reader_schema_mismatch" and expected_message in error["message"]
        for error in report["errors"]
    )


def test_schema_binding_rejects_unrelated_digest_valid_source(tmp_path: Path) -> None:
    role = _role(tmp_path, "fixture", "json")
    unrelated = _write(tmp_path, "readers/unrelated.py", 'OTHER_SCHEMA_VERSION = "fixture.v1"\n')
    path, digest = unrelated.relative_to(tmp_path).as_posix(), _sha(unrelated)
    role["source_material"][0] = {"kind": "schema", "path": path, "sha256": digest}
    report = inventory.build_inventory(_packet(role), tmp_path)
    assert "source_material_mismatch" in {error["code"] for error in report["errors"]}
    role["schema"].update(source_path=path, source_sha256=digest)
    assert any(
        error["code"] == "schema_source_mismatch"
        for error in inventory.build_inventory(_packet(role), tmp_path)["errors"]
    )


def test_duplicate_roles_are_conflicts(tmp_path: Path) -> None:
    role = _role(tmp_path, "fixture", "json")
    report = inventory.build_inventory(_packet(role, role), tmp_path)
    assert sum(error["code"] == "ambiguous_role" for error in report["errors"]) == 1
