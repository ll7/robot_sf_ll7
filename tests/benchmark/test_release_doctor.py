"""Tests for full benchmark-release doctor admission checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from robot_sf.benchmark import release_doctor
from robot_sf.benchmark.release_doctor import ReleaseDoctorCheck
from robot_sf.cli import main as robot_sf_main


def test_manifest_doctor_confirms_s30_h600_cardinality() -> None:
    """The current 14-arm predecessor resolves to exactly 20,160 cells."""
    check, manifest, cfg = release_doctor._manifest_check(
        Path(
            "configs/benchmarks/releases/"
            "paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml"
        ),
        20160,
    )
    assert check.status == "pass"
    assert "20160-cell" in check.summary
    assert manifest is not None
    assert cfg is not None


def test_cluster_check_requires_admission_and_exact_source(tmp_path: Path) -> None:
    """A launch packet cannot admit a different public source commit."""
    packet = tmp_path / "launch.json"
    packet.write_text(
        json.dumps(
            {
                "admission": {"status": "admitted"},
                "dispatchable": True,
                "identity": {"public_source_commit": "a" * 40},
            }
        ),
        encoding="utf-8",
    )
    assert release_doctor._cluster_check(packet, "a" * 40).status == "pass"
    rejected = release_doctor._cluster_check(packet, "b" * 40)
    assert rejected.status == "fail"
    assert "source SHA" in rejected.summary


def test_cluster_check_rejects_preparation_only_packet(tmp_path: Path) -> None:
    """An admitted preparation packet is not sufficient for real dispatch."""
    packet = tmp_path / "launch.yaml"
    packet.write_text(
        "\n".join(
            [
                "admission:",
                "  status: admitted",
                "dispatchable: false",
                "identity:",
                f"  public_source_commit: {'a' * 40}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rejected = release_doctor._cluster_check(packet, "a" * 40)
    assert rejected.status == "fail"
    assert "not dispatchable" in rejected.summary


def test_doctor_report_is_credential_free(monkeypatch, tmp_path: Path) -> None:
    """Collected output exposes only sanitized check summaries."""
    passed = ReleaseDoctorCheck("fixture", "pass", "safe")
    monkeypatch.setattr(
        release_doctor, "_manifest_check", lambda *args: (passed, object(), object())
    )
    monkeypatch.setattr(release_doctor, "_git_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_ci_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_tag_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_release_identity_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_checkpoint_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_cluster_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_disk_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_zenodo_check", lambda *args, **kwargs: [passed])
    monkeypatch.setattr(release_doctor, "_dissertation_check", lambda *args: passed)
    report = release_doctor.collect_release_doctor_report(
        repo=tmp_path,
        manifest_path=tmp_path / "manifest.yaml",
        expected_release_sha="a" * 40,
        expected_base_sha="b" * 40,
        tag="paper-matrix-v2-h600-s30-2026-08-aaaaaaaaaaaa",
        checkpoint_receipt=None,
        private_launch_packet=None,
        dissertation=None,
        token_file=tmp_path / "super-secret-token",
    )
    encoded = json.dumps(report)
    assert report["status"] == "pass"
    assert "super-secret-token" not in encoded


def test_zenodo_hook_check_does_not_surface_hook_config(monkeypatch, tmp_path: Path) -> None:
    """Hook inspection reports active state without echoing private configuration."""
    secret = "private-receiver-token"
    hook_payload = [
        {
            "active": True,
            "config": {"url": f"https://zenodo.org/hooks/receiver?token={secret}"},
        }
    ]
    monkeypatch.setattr(
        release_doctor,
        "_run",
        lambda *args: subprocess.CompletedProcess([], 0, json.dumps(hook_payload), ""),
    )
    monkeypatch.setattr(release_doctor, "read_token_file", lambda path: "token")

    class Session:
        """Successful auth probe."""

        def get(self, *args, **kwargs):
            """Return a successful requests-like response."""

            class Response:
                def raise_for_status(self) -> None:
                    """Accept the probe."""

            return Response()

    monkeypatch.setattr(release_doctor, "build_session", lambda path: Session())
    checks = release_doctor._zenodo_check(
        tmp_path,
        tmp_path / "token",
        require_hook_disabled=True,
    )
    encoded = json.dumps([check.summary for check in checks])
    assert checks[-1].status == "fail"
    assert secret not in encoded


def test_top_level_cli_registers_release_doctor(capsys) -> None:
    """The public CLI exposes the requested release doctor command."""
    try:
        robot_sf_main(["release", "doctor", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    assert "--expected-release-sha" in output
    assert "--expected-base-sha" in output
