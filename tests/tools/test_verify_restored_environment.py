"""Tests for verify_restored_environment (#8827)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.tools.verify_restored_environment import main, verify_restoration

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "restored_environment"


@pytest.fixture
def base_fixture(tmp_path: Path) -> tuple[Path, dict]:
    man_path = FIXTURES_DIR / "canonical_manifest.json"
    env_path = FIXTURES_DIR / "canonical_env.json"
    dest_man = tmp_path / "manifest.json"
    dest_env = tmp_path / "canonical_env.json"
    data = json.loads(man_path.read_text(encoding="utf-8"))
    dest_man.write_text(json.dumps(data), encoding="utf-8")
    dest_env.write_text(env_path.read_text(encoding="utf-8"), encoding="utf-8")
    return dest_man, data


def test_canonical_fixtures_standalone(tmp_path: Path) -> None:
    man = FIXTURES_DIR / "canonical_manifest.json"
    root = tmp_path / "root"
    rep = verify_restoration(man, None, root)
    assert rep["verdict"] == "pass"
    assert rep["label"] == "restoration_smoke"
    assert rep["scientific_reproduction"] is False
    assert rep["capabilities"]["unavailable"] == ["carla"]
    assert "row_inspection" in rep["capabilities"]["supported"]
    assert all(rep["verified_identities"].values())


def test_cli_invocation_and_formats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    man = FIXTURES_DIR / "canonical_manifest.json"
    root = tmp_path / "root_cli"
    code_json = main(["--check", "--manifest", str(man), "--root", str(root), "--format", "json"])
    assert code_json == 0
    out_json = capsys.readouterr().out
    assert json.loads(out_json)["verdict"] == "pass"

    code_text = main(["--check", "--manifest", str(man), "--root", str(root), "--format", "text"])
    assert code_text == 0
    assert "Verdict: PASS" in capsys.readouterr().out


def _apply_case_mutation(data: dict, case_name: str) -> None:
    if case_name == "missing_file":
        data["files"].append({"path": "data/missing.txt", "sha256": "abc", "size_bytes": 10})
    elif case_name == "changed_digest":
        data["files"][0]["content"] = '{"id": "r1", "val": 999}\n{"id": "r2", "val": 20}\n'
    elif case_name == "stale_source_path":
        data["files"][1]["content"] = json.dumps({"seed": 42, "path": "/home/user/work/out"})
        data["files"][1]["size_bytes"] = len(data["files"][1]["content"])
        data["files"][1]["sha256"] = None
    elif case_name == "unsupported_schema":
        data["schema"] = "artifact_transfer_manifest.v999"
    elif case_name == "private_path_leak":
        data["files"][1]["content"] = json.dumps({"seed": 42, "path": "/root/.ssh/id_rsa"})
        data["files"][1]["size_bytes"] = len(data["files"][1]["content"])
        data["files"][1]["sha256"] = None
    elif case_name == "source_host_access_attempt":
        data["files"][1]["content"] = json.dumps({"seed": 42, "host": "ssh://cluster-login/data"})
        data["files"][1]["size_bytes"] = len(data["files"][1]["content"])
        data["files"][1]["sha256"] = None


@pytest.mark.parametrize(
    ("case_name", "expected_verdict", "expected_reason"),
    [
        ("complete_restore", "pass", None),
        ("missing_file", "fail", "missing_file"),
        ("changed_digest", "fail", "checksum_mismatch"),
        ("stale_source_path", "fail", "stale_source_path"),
        ("unavailable_optional_dependency", "pass", None),
        ("unsupported_schema", "fail", "unsupported_schema"),
        ("private_path_leak", "blocked", "private_path_leak"),
        ("source_host_access_attempt", "blocked", "source_host_access_attempt"),
    ],
)
def test_all_scenarios(
    tmp_path: Path,
    base_fixture: tuple[Path, dict],
    case_name: str,
    expected_verdict: str,
    expected_reason: str | None,
) -> None:
    _, base_data = base_fixture
    data = copy.deepcopy(base_data)
    _apply_case_mutation(data, case_name)
    m_path = tmp_path / f"{case_name}.json"
    m_path.write_text(json.dumps(data), encoding="utf-8")
    rep = verify_restoration(m_path, None, tmp_path / f"root_{case_name}")
    assert rep["verdict"] == expected_verdict
    if expected_reason:
        assert any(expected_reason in r for r in rep["reasons"])


def test_symlink_rejection(tmp_path: Path, base_fixture: tuple[Path, dict]) -> None:
    dest_man, _ = base_fixture
    root = tmp_path / "root_sym"
    root.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "dummy.txt"
    target.write_text("hello")
    link = root / "link.txt"
    link.symlink_to(target)
    rep = verify_restoration(dest_man, None, root)
    assert rep["verdict"] == "blocked"
    assert "symlink_forbidden" in rep["reasons"]


def test_missing_manifest_and_env(tmp_path: Path) -> None:
    rep_missing_m = verify_restoration(tmp_path / "nonexistent.json", None, tmp_path / "root")
    assert rep_missing_m["verdict"] == "blocked"
    assert "manifest_not_found" in rep_missing_m["reasons"]

    man_no_env = tmp_path / "no_env.json"
    man_no_env.write_text(
        json.dumps(
            {
                "schema": "artifact_transfer_manifest.v1",
                "source_commit": "e01434909462b682b571746a8572bd1abd53286b",
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    rep_no_env = verify_restoration(man_no_env, None, tmp_path / "root_no_env")
    assert rep_no_env["verdict"] == "blocked"
    assert "missing_environment_manifest" in rep_no_env["reasons"]
