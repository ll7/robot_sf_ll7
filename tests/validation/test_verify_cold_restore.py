"""Focused contract tests for the issue #8895 cold-restore verifier."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from scripts.validation import verify_cold_restore as tool

CAPSULE = (
    Path(__file__).resolve().parents[2]
    / "tests/validation/fixtures/cold_restore/synthetic_capsule.json"
)


def _mutate_declaration(monkeypatch: pytest.MonkeyPatch, mutation: Any) -> None:
    declaration = json.loads(CAPSULE.read_text(encoding="utf-8"))
    mutation(declaration)
    original = tool._read_json

    def load(path: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return (
            declaration if path.resolve() == CAPSULE.resolve() else original(path, *args, **kwargs)
        )

    monkeypatch.setattr(tool, "_read_json", load)


def _restore(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "restore"
    result = tool.restore_capsule(CAPSULE, root, offline=True)
    assert result["result"]["status"] == "verified", result
    return root, result


def test_synthetic_restore_and_verify_are_deterministic(tmp_path: Path) -> None:
    first_root, first = _restore(tmp_path / "first")
    second_root, second = _restore(tmp_path / "second")
    assert tool.normalize_receipt(first) == tool.normalize_receipt(second)
    assert first["receipt_digest"] == second["receipt_digest"]
    assert first["restored"]["member_count"] == 7
    assert first["restored"]["transferred_bytes"] == first["restored"]["byte_count"]
    assert first["validation"]["consumer"] == {
        "name": "episode_schema_reader",
        "rows": 2,
        "status": "verified",
    }
    assert (
        first_root.is_dir() and second_root.is_dir() and (first_root / tool.RECEIPT_NAME).is_file()
    )
    assert tool.verify_destination(first_root)["result"]["status"] == "verified"
    assert str(tmp_path) not in json.dumps(first)


def test_cli_restore_emits_json_and_keeps_destination(tmp_path: Path, capsys: Any) -> None:
    destination = tmp_path / "cli-restore"
    code = tool.main(
        [
            "restore",
            "--capsule",
            str(CAPSULE),
            "--destination",
            str(destination),
            "--offline",
            "--json",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert code == 0 and output["result"]["status"] == "verified" and destination.is_dir()


def test_known_cache_variables_are_isolated_and_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ambient = tmp_path / "ambient-cache"
    monkeypatch.setenv("HF_HOME", str(ambient))
    root, result = _restore(tmp_path / "isolated")
    assert result["cache"] == {
        "ambient_reads": 0,
        "status": "isolated",
        "variables": list(tool.CACHE_ENV_VARS),
    }
    assert os.environ["HF_HOME"] == str(ambient) and root.is_dir()


@pytest.mark.parametrize(
    ("mutation", "code", "member"),
    [
        (lambda d: d.update({"schema": "cold_restore_capsule.v2"}), "unsupported_schema", None),
        (lambda d: d["artifact"].update({"digest": "0" * 64}), "stale_pointer", tool.MANIFEST_NAME),
        (
            lambda d: d["artifact"].update({"version": "9.9.9"}),
            "artifact_version_mismatch",
            tool.MANIFEST_NAME,
        ),
        (
            lambda d: d.update({"fixture_path": "../../../../../../etc/passwd"}),
            "path_escape",
            "fixture_path",
        ),
        (lambda d: d["cache_policy"].update({"ambient_mode": "allow"}), "ambient_cache_hit", None),
        (
            lambda d: d["environment"].update({"allowed_os": ["never"]}),
            "unsupported_environment",
            None,
        ),
    ],
)
def test_declaration_failures_are_stable_and_do_not_create_a_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: Any, code: str, member: str | None
) -> None:
    _mutate_declaration(monkeypatch, mutation)
    root = tmp_path / "rejected"
    result = tool.restore_capsule(CAPSULE, root, offline=True)
    assert result["result"] == {"failure": {"code": code, "member": member}, "status": "failed"}
    assert not root.exists()


def test_real_mode_is_explicitly_unavailable(tmp_path: Path) -> None:
    result = tool.restore_capsule(CAPSULE, tmp_path / "real", offline=True, mode="real")
    assert result["result"] == {
        "status": "unavailable",
        "failure": {"code": "real_capsule_unavailable", "member": None},
    }


@pytest.mark.parametrize(
    ("change", "code", "member"),
    [
        (lambda root: (root / "artifact/report.json").unlink(), "missing_member", "report.json"),
        (
            lambda root: (root / tool.MANIFEST_NAME).unlink(),
            "restore_incomplete",
            tool.MANIFEST_NAME,
        ),
        (
            lambda root: (root / "artifact/config.json").write_text("tampered\n", encoding="utf-8"),
            "checksum_mismatch",
            "config.json",
        ),
    ],
)
def test_verify_failures_identify_the_affected_member(
    tmp_path: Path, change: Any, code: str, member: str
) -> None:
    root, _ = _restore(tmp_path)
    before = (root / tool.RECEIPT_NAME).read_bytes()
    change(root)
    result = tool.verify_destination(root)
    assert result["result"] == {"status": "failed", "failure": {"code": code, "member": member}}
    assert (root / tool.RECEIPT_NAME).read_bytes() == before
