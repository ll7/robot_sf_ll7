"""Tests for source-host artifact prune eligibility guard (#8846)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools.check_prune_eligibility import (
    EXIT_BLOCKED,
    EXIT_ELIGIBLE,
    check_prune_eligibility,
    main,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "prune_eligibility"


def _setup_base_case(tmp_path: Path):
    s_dir = tmp_path / "source"
    s_dir.mkdir()
    log = s_dir / "job_evidence.log"
    log.write_text("hello\n", encoding="utf-8")
    m_path, d_path = tmp_path / "m.json", tmp_path / "d.json"
    m_path.write_text(
        (FIXTURES_DIR / "source_manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    d_path.write_text(
        (FIXTURES_DIR / "destination_receipt.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    return m_path, d_path, s_dir, log


def test_canonical_fixtures_standalone():
    """Verify the checked-in canonical fixtures pass eligibility."""
    report = check_prune_eligibility(
        FIXTURES_DIR / "source_manifest.json", FIXTURES_DIR / "destination_receipt.json"
    )
    assert report["status"] == "eligible"
    assert report["summary"]["eligible_verified"] == 1
    assert len(report["deletion_plan"]) == 1


def test_fixture_complete_custody(tmp_path: Path):
    """Complete verified custody produces eligible status and deterministic deletion plan."""
    m_path, d_path, s_dir, _log = _setup_base_case(tmp_path)
    report = check_prune_eligibility(m_path, d_path, source_root=s_dir)
    assert report["status"] == "eligible"
    assert report["summary"]["eligible_verified"] == 1
    assert report["deletion_plan"][0]["path"] == "job_evidence.log"


@pytest.mark.parametrize(
    ("kind", "expected_key"),
    [
        ("consumer", "blocked_active_consumer"),
        ("partial", "blocked_missing_destination"),
        ("changed_source", "blocked_checksum"),
        ("changed_dest", "blocked_checksum"),
        ("mutable_uri", "blocked_missing_destination"),
        ("active_writer", "blocked_active_consumer"),
        ("unknown_owner", "blocked_unknown_owner"),
    ],
)
def test_blocking_scenarios(tmp_path: Path, kind: str, expected_key: str):
    """Verify each failure condition blocks eligibility fail-closed."""
    m_path, d_path, s_dir, log = _setup_base_case(tmp_path)
    c_path = None
    if kind == "consumer":
        c_path = tmp_path / "c.json"
        c_path.write_text(
            json.dumps(
                {
                    "consumers": [
                        {
                            "id": "w",
                            "state": "active",
                            "refs": ["job_evidence.log"],
                            "points_to_destination": False,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    elif kind == "partial":
        receipt = json.loads(d_path.read_text(encoding="utf-8"))
        receipt.update({"partial_transfer": True, "status": "blocked"})
        receipt["files"][0]["state"] = "pending"
        d_path.write_text(json.dumps(receipt), encoding="utf-8")
    elif kind == "changed_source":
        log.write_text("tampered\n", encoding="utf-8")
    elif kind == "changed_dest":
        receipt = json.loads(d_path.read_text(encoding="utf-8"))
        receipt["files"][0]["sha256"] = "0" * 64
        d_path.write_text(json.dumps(receipt), encoding="utf-8")
    elif kind == "mutable_uri":
        receipt = json.loads(d_path.read_text(encoding="utf-8"))
        receipt["destination_locator"] = "s3://b/art:latest"
        d_path.write_text(json.dumps(receipt), encoding="utf-8")
    elif kind == "active_writer":
        manifest = json.loads(m_path.read_text(encoding="utf-8"))
        manifest["writers"] = [{"writer_id": "jw", "state": "active"}]
        m_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif kind == "unknown_owner":
        manifest = json.loads(m_path.read_text(encoding="utf-8"))
        manifest["owner"], manifest["inventory"][0]["owner"] = None, None
        m_path.write_text(json.dumps(manifest), encoding="utf-8")

    rep = check_prune_eligibility(m_path, d_path, source_root=s_dir, consumers_path=c_path)
    assert rep["status"] == "blocked"
    assert rep["summary"][expected_key] == 1
    assert len(rep["deletion_plan"]) == 0


def test_fixture_unknown_file(tmp_path: Path):
    """Untracked file on source root is classified not_managed and excluded from plan."""
    m_path, d_path, s_dir, _log = _setup_base_case(tmp_path)
    (s_dir / "untracked.tmp").write_text("temp data", encoding="utf-8")
    report = check_prune_eligibility(m_path, d_path, source_root=s_dir)
    assert report["summary"]["not_managed"] == 1
    assert not any(item["path"] == "untracked.tmp" for item in report["deletion_plan"])


def test_fixture_already_pruned(tmp_path: Path):
    """Idempotent handling when file is already pruned on disk."""
    m_path, d_path, s_dir, log = _setup_base_case(tmp_path)
    log.unlink()
    report = check_prune_eligibility(m_path, d_path, source_root=s_dir)
    assert report["status"] == "eligible"
    assert report["summary"]["eligible_verified"] == 1
    assert len(report["deletion_plan"]) == 0


@pytest.mark.parametrize(
    ("disposition", "expected_status", "expected_class"),
    [
        ("tracked-manifest", "retain_required", "retain_required"),
        ("handoff-needed", "review_required", "eligible_after_review"),
        ("disposable", "eligible", "eligible_verified"),
    ],
)
def test_triage_dispositions_integration(
    tmp_path: Path, disposition: str, expected_status: str, expected_class: str
):
    """Consumes #8255/#8425 dispositions accurately."""
    m_path, d_path, s_dir, _log = _setup_base_case(tmp_path)
    disp_path = tmp_path / "disp.json"
    disp_path.write_text(
        json.dumps({"dispositions": {"job_evidence.log": disposition}}), encoding="utf-8"
    )
    report = check_prune_eligibility(m_path, d_path, source_root=s_dir, dispositions_path=disp_path)
    assert report["status"] == expected_status
    assert report["summary"][expected_class] == 1


def test_check_mode_zero_side_effects(tmp_path: Path):
    """Ordinary check mode produces plan but performs zero file deletions."""
    m_path, d_path, s_dir, log = _setup_base_case(tmp_path)
    report = check_prune_eligibility(m_path, d_path, source_root=s_dir, apply=False)
    assert report["mode"] == "check"
    assert report["status"] == "eligible"
    assert log.exists()


def test_apply_cas_deletion(tmp_path: Path):
    """Explicit apply route validates CAS before unlinking."""
    m_path, d_path, s_dir, log = _setup_base_case(tmp_path)
    report = check_prune_eligibility(m_path, d_path, source_root=s_dir, apply=True)
    assert report["mode"] == "apply"
    assert report["apply"]["success"] is True
    assert report["apply"]["deleted_count"] == 1
    assert not log.exists()


def test_cli_exit_codes_and_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """CLI exit codes and output formatting."""
    m_path, d_path, _s_dir, _log = _setup_base_case(tmp_path)
    assert (
        main(["--check", "--source-manifest", str(m_path), "--destination-receipt", str(d_path)])
        == EXIT_ELIGIBLE
    )
    assert json.loads(capsys.readouterr().out)["status"] == "eligible"

    receipt = json.loads(d_path.read_text(encoding="utf-8"))
    receipt["durability_class"] = "local_scratch"
    d_blocked = tmp_path / "dest_blocked.json"
    d_blocked.write_text(json.dumps(receipt), encoding="utf-8")

    assert (
        main(
            [
                "--check",
                "--source-manifest",
                str(m_path),
                "--destination-receipt",
                str(d_blocked),
                "--format",
                "summary",
            ]
        )
        == EXIT_BLOCKED
    )
    assert "prune eligibility: blocked" in capsys.readouterr().out
