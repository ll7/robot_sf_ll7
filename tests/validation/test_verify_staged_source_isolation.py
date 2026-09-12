"""Tests for verify_staged_source_isolation.py (issue #8858).

Verifies fail-closed detection of untracked source, dirty files, sibling worktree
imports, stale editable installations, user-site packages, .pth injection, and
unexpected native extension origins.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.validation.verify_staged_source_isolation import (
    RECEIPT_SCHEMA,
    evaluate_source_isolation,
    load_packet,
    main,
    sanitize_path,
)


@pytest.fixture()
def clean_source(tmp_path: Path) -> Path:
    """Fixture creating an isolated clean staged source directory."""
    src = tmp_path / "staged_repo"
    src.mkdir()
    (src / "robot_sf").mkdir()
    (src / "robot_sf" / "__init__.py").write_text('__version__ = "0.1.0"\n', encoding="utf-8")
    return src


def _base_probe(clean_source: Path, **overrides: Any) -> dict[str, Any]:
    probe: dict[str, Any] = {
        "sys_path": [str(clean_source)],
        "user_site_enabled": False,
        "user_site_dir": None,
        "pth_files": [],
        "distributions": [],
        "first_party_imports": {
            "robot_sf": {
                "imported": True,
                "file": str(clean_source / "robot_sf" / "__init__.py"),
                "path": [str(clean_source / "robot_sf")],
                "native_extensions": [],
            }
        },
    }
    probe.update(overrides)
    return probe


def _eval(src: Path, probe: dict[str, Any], **kw: Any) -> tuple[dict[str, Any], list[Any]]:
    return evaluate_source_isolation(
        source_root=src,
        declared_companions=kw.get("companions"),
        first_party_packages=kw.get("pkgs", ["robot_sf"]),
        command_tokens=None,
        probe_data=probe,
        git_state=kw.get("git_state"),
    )


def test_clean_staged_source(clean_source: Path) -> None:
    """A clean staged source without untracked files or leakage must pass."""
    git_state = {"dirty": False, "untracked_files": [], "modified_files": []}
    receipt, problems = _eval(clean_source, _base_probe(clean_source), git_state=git_state)
    assert receipt["schema_version"] == RECEIPT_SCHEMA
    assert receipt["status"] == "passed"
    assert not problems


def test_sibling_checkout_detected(clean_source: Path, tmp_path: Path) -> None:
    """Importing from a sibling worktree must fail closed."""
    sibling = tmp_path / "repo.worktrees" / "sibling" / "robot_sf"
    sibling.mkdir(parents=True)
    (sibling / "__init__.py").write_text("# sibling\n", encoding="utf-8")
    probe = _base_probe(
        clean_source,
        sys_path=[str(clean_source), str(sibling.parent)],
        first_party_imports={
            "robot_sf": {
                "imported": True,
                "file": str(sibling / "__init__.py"),
                "path": [str(sibling)],
                "native_extensions": [],
            }
        },
    )
    receipt, problems = _eval(clean_source, probe)
    assert receipt["status"] == "blocked"
    assert "sibling_worktree_import" in receipt["problem_codes"]
    assert any("<sibling_worktree>" in p.sanitized_path for p in problems)


def test_stale_editable_installation(clean_source: Path, tmp_path: Path) -> None:
    """An editable distribution pointing outside staged source must fail closed."""
    ext_dir = tmp_path / "other_checkout"
    ext_dir.mkdir()
    probe = _base_probe(
        clean_source,
        distributions=[
            {
                "name": "robot-sf",
                "version": "0.1.0",
                "direct_url": {
                    "url": f"file://{ext_dir.resolve()}",
                    "dir_info": {"editable": True},
                },
            }
        ],
    )
    receipt, _ = _eval(clean_source, probe)
    assert receipt["status"] == "blocked"
    assert "stale_editable_installation" in receipt["problem_codes"]


def test_user_site_leakage(clean_source: Path, tmp_path: Path) -> None:
    """User-site presence in sys.path must fail closed."""
    user_site = tmp_path / "user_site" / "lib" / "python3.13" / "site-packages"
    user_site.mkdir(parents=True)
    probe = _base_probe(
        clean_source,
        sys_path=[str(clean_source), str(user_site)],
        user_site_enabled=True,
        user_site_dir=str(user_site),
    )
    receipt, _ = _eval(clean_source, probe)
    assert receipt["status"] == "blocked"
    assert "user_site_leakage" in receipt["problem_codes"]


def test_pth_injection(clean_source: Path, tmp_path: Path) -> None:
    """A .pth file injecting arbitrary paths must fail closed."""
    injected = tmp_path / "injected_dir"
    injected.mkdir()
    probe = _base_probe(
        clean_source,
        pth_files=[{"file": str(clean_source / "vendor.pth"), "lines": [str(injected)]}],
    )
    receipt, _ = _eval(clean_source, probe)
    assert receipt["status"] == "blocked"
    assert "pth_injection" in receipt["problem_codes"]


def test_generated_drift_and_untracked(clean_source: Path) -> None:
    """Untracked source files and modified working tree files must fail closed."""
    git_state = {
        "dirty": True,
        "untracked_files": ["robot_sf/new_untracked_helper.py"],
        "modified_files": ["robot_sf/__init__.py"],
    }
    receipt, _ = _eval(clean_source, _base_probe(clean_source), git_state=git_state)
    assert receipt["status"] == "blocked"
    assert "untracked_source" in receipt["problem_codes"]
    assert "dirty_generated_files" in receipt["problem_codes"]


def test_unexpected_native_extension_origin(clean_source: Path, tmp_path: Path) -> None:
    """Native extension resolving outside staged source must fail closed."""
    leak_so = tmp_path / "ambient" / "fast_math.so"
    leak_so.parent.mkdir()
    leak_so.write_text("binary", encoding="utf-8")
    probe = _base_probe(
        clean_source,
        first_party_imports={
            "robot_sf": {
                "imported": True,
                "file": str(clean_source / "robot_sf" / "__init__.py"),
                "path": [str(clean_source / "robot_sf")],
                "native_extensions": [str(leak_so)],
            }
        },
    )
    receipt, _ = _eval(clean_source, probe)
    assert receipt["status"] == "blocked"
    assert "unexpected_native_extension_origin" in receipt["problem_codes"]


def test_valid_declared_companion(clean_source: Path, tmp_path: Path) -> None:
    """Imports from an explicitly declared companion must be permitted."""
    companion_root = tmp_path / "companion_prebuilt"
    companion_pkg = companion_root / "robot_sf_companion"
    companion_pkg.mkdir(parents=True)
    (companion_pkg / "__init__.py").write_text("# companion\n", encoding="utf-8")
    probe = _base_probe(
        clean_source,
        sys_path=[str(clean_source), str(companion_root)],
        first_party_imports={
            "robot_sf": {
                "imported": True,
                "file": str(clean_source / "robot_sf" / "__init__.py"),
                "path": [str(clean_source / "robot_sf")],
                "native_extensions": [],
            },
            "robot_sf_companion": {
                "imported": True,
                "file": str(companion_pkg / "__init__.py"),
                "path": [str(companion_pkg)],
                "native_extensions": [],
            },
        },
    )
    receipt, problems = _eval(
        clean_source,
        probe,
        companions={"prebuilt": companion_root},
        pkgs=["robot_sf", "robot_sf_companion"],
    )
    assert receipt["status"] == "passed"
    assert not problems
    companion_record = receipt["first_party_imports"]["robot_sf_companion"]
    assert companion_record["path_class"] == "declared_companion"


def test_sanitization_no_private_raw_paths(clean_source: Path, tmp_path: Path) -> None:
    """Sanitized strings must never expose raw machine or user paths."""
    ext = tmp_path / "secret_work" / "private_user" / "file.py"
    ext.parent.mkdir(parents=True)
    ext.touch()
    _, sanitized = sanitize_path(ext, source_root=clean_source)
    assert str(tmp_path) not in sanitized
    assert "private_user" not in sanitized
    assert sanitized.startswith("<external>/")


def test_load_packet(tmp_path: Path, clean_source: Path) -> None:
    """Test packet parsing and schema validation."""
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(
        json.dumps(
            {
                "schema_version": "staged_source_isolation_packet.v1",
                "source_root": str(clean_source),
            }
        ),
        encoding="utf-8",
    )
    assert load_packet(packet_path)["schema_version"] == "staged_source_isolation_packet.v1"
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to read packet file"):
        load_packet(bad_path)


def test_cli_main_with_packet(
    tmp_path: Path,
    clean_source: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test CLI exit code and output formatting."""
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(
        json.dumps(
            {
                "schema_version": "staged_source_isolation_packet.v1",
                "source_root": str(clean_source),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.validation.verify_staged_source_isolation.inspect_git_state",
        lambda _: {"dirty": False, "untracked_files": [], "modified_files": []},
    )
    monkeypatch.setattr(
        "scripts.validation.verify_staged_source_isolation.run_environment_probe",
        lambda **_: _base_probe(clean_source),
    )
    assert main(["--packet", str(packet_path), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["schema_version"] == RECEIPT_SCHEMA
