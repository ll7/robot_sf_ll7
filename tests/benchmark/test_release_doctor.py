"""Tests for full benchmark-release doctor admission checks."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from robot_sf import cli as robot_sf_cli
from robot_sf import release_cli
from robot_sf.benchmark import release_doctor
from robot_sf.benchmark.camera_ready import _config as camera_ready_config
from robot_sf.benchmark.camera_ready import _run_state as camera_ready_run_state
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


def test_doctor_anchors_manifest_and_git_checks_to_explicit_release_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reviewed tooling may inspect a different untouched ``--repo`` checkout."""
    tooling_root = tmp_path / "tooling-worktree"
    release_root = tmp_path / "frozen-release-worktree"
    tooling_root.mkdir()
    release_root.mkdir()
    manifest_path = release_root / "configs" / "release.yaml"
    manifest_path.parent.mkdir()
    manifest_path.write_text("manifest", encoding="utf-8")
    campaign_path = release_root / "configs" / "campaign.yaml"
    campaign_path.write_text("campaign", encoding="utf-8")
    manifest = SimpleNamespace(
        canonical_campaign_config_path=campaign_path,
        schema_version="benchmark-release-manifest.v0.1",
        expected_episode_cells=1,
    )
    cfg = SimpleNamespace(planners=[SimpleNamespace(enabled=True)])
    roots: list[Path] = []
    monkeypatch.setattr(
        release_doctor,
        "load_release_manifest",
        lambda path, **kwargs: roots.append(kwargs["repository_root"]) or manifest,
    )
    monkeypatch.setattr(
        release_doctor,
        "load_campaign_config",
        lambda path, **kwargs: roots.append(kwargs["repository_root"]) or cfg,
    )
    monkeypatch.setattr(
        release_doctor,
        "validate_release_manifest",
        lambda value, **kwargs: (
            roots.append(kwargs["repository_root"]) or {"problems": [], "status": "valid"}
        ),
    )
    monkeypatch.setattr(
        release_doctor,
        "_load_campaign_scenarios",
        lambda value, **kwargs: roots.append(kwargs["repository_root"]) or [{"id": "one"}],
    )
    monkeypatch.setattr(release_doctor, "_resolved_seed_inventory", lambda _: [1])
    check, _, _ = release_doctor._manifest_check(
        manifest_path,
        1,
        repository_root=release_root,
    )
    assert check.status == "pass", check.summary
    assert roots == [release_root, release_root, release_root, release_root]

    run_cwds: list[Path] = []

    def fake_run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        run_cwds.append(cwd)
        return subprocess.CompletedProcess(
            command, 0, "a" * 40 + "\n" if command[1] == "rev-parse" else "", ""
        )

    monkeypatch.setattr(release_doctor, "_run", fake_run)
    git_check = release_doctor._git_check(release_root, "a" * 40)
    assert git_check.status == "pass", git_check.summary
    assert run_cwds == [release_root, release_root]


def test_campaign_asset_paths_use_explicit_release_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A tooling import must not make relative packet paths resolve in its own checkout."""
    tooling_root = tmp_path / "tooling-worktree"
    release_root = tmp_path / "frozen-release-worktree"
    tooling_matrix = tooling_root / "configs" / "scenarios" / "matrix.yaml"
    release_matrix = release_root / "configs" / "scenarios" / "matrix.yaml"
    tooling_seed_sets = tooling_root / "configs" / "benchmarks" / "seed_sets.yaml"
    release_seed_sets = release_root / "configs" / "benchmarks" / "seed_sets.yaml"
    config_path = release_root / "configs" / "benchmarks" / "campaign.yaml"
    for path in (tooling_matrix, release_matrix, tooling_seed_sets, release_seed_sets):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    config_path.write_text("name: release\n", encoding="utf-8")
    monkeypatch.setattr(camera_ready_run_state, "get_repository_root", lambda: tooling_root)

    matrix_path = camera_ready_config._resolve_scenario_matrix_path(
        {"scenario_matrix": "configs/scenarios/matrix.yaml"},
        config_path=config_path,
        repository_root=release_root,
    )
    seed_policy = camera_ready_config._build_seed_policy(
        {"seed_policy": {"seed_sets_path": "configs/benchmarks/seed_sets.yaml"}},
        base_dir=config_path.parent,
        repository_root=release_root,
    )

    assert matrix_path == release_matrix
    assert seed_policy.seed_sets_path == release_seed_sets


def test_v02_manifest_cardinality_cannot_be_overridden() -> None:
    """The v0.2 doctor binds its matrix check to the manifest's 20,160 cells."""
    check, manifest, cfg = release_doctor._manifest_check(
        Path("configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml"),
        1,
    )
    assert check.status == "fail"
    assert "manifest-required" in check.summary
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


def test_final_doctor_rejects_unsafe_cardinality_override(monkeypatch, tmp_path: Path) -> None:
    """Final doctor mode always validates the fixed S30/H600 cardinality."""
    passed = ReleaseDoctorCheck("fixture", "pass", "safe")
    calls: list[int] = []
    monkeypatch.setattr(
        release_doctor,
        "_manifest_check",
        lambda path, expected: calls.append(expected) or (passed, object(), object()),
    )
    monkeypatch.setattr(release_doctor, "_git_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_ci_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_tag_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_release_identity_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_checkpoint_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_cluster_check", lambda *args, **kwargs: passed)
    monkeypatch.setattr(release_doctor, "_disk_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_zenodo_check", lambda *args, **kwargs: [passed])
    monkeypatch.setattr(release_doctor, "_dissertation_check", lambda *args: passed)

    report = release_doctor.collect_release_doctor_report(
        repo=tmp_path,
        manifest_path=tmp_path / "manifest.yaml",
        expected_release_sha="a" * 40,
        expected_base_sha="b" * 40,
        tag="release",
        checkpoint_receipt=None,
        private_launch_packet=None,
        dissertation=None,
        token_file=None,
        expected_cells=1,
        final=True,
    )

    assert calls == [release_doctor.FULL_RELEASE_EXPECTED_EPISODE_CELLS]
    assert report["status"] == "blocked"
    manifest_checks = [check for check in report["checks"] if check["name"] == "manifest"]
    assert manifest_checks and "unsafe override rejected" in manifest_checks[0]["summary"]


def test_final_doctor_rejects_diagnostic_checkpoint_path_map(monkeypatch, tmp_path: Path) -> None:
    """Invocation-local checkpoint substitutes cannot satisfy final publication admission."""
    passed = ReleaseDoctorCheck("fixture", "pass", "safe")
    monkeypatch.setattr(
        release_doctor,
        "_manifest_check",
        lambda *args: (passed, object(), object()),
    )
    monkeypatch.setattr(release_doctor, "_git_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_ci_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_tag_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_release_identity_check", lambda *args: passed)
    monkeypatch.setattr(
        release_doctor,
        "_checkpoint_check",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not validate remap")),
    )
    monkeypatch.setattr(release_doctor, "_cluster_check", lambda *args, **kwargs: passed)
    monkeypatch.setattr(release_doctor, "_disk_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_zenodo_check", lambda *args, **kwargs: [passed])
    monkeypatch.setattr(release_doctor, "_dissertation_check", lambda *args: passed)

    report = release_doctor.collect_release_doctor_report(
        repo=tmp_path,
        manifest_path=tmp_path / "manifest.yaml",
        expected_release_sha="a" * 40,
        expected_base_sha="b" * 40,
        tag="release",
        checkpoint_receipt=tmp_path / "receipt.json",
        checkpoint_path_map=["/remote/model.zip=checkpoints/model.zip"],
        private_launch_packet=None,
        dissertation=None,
        token_file=None,
        final=True,
    )

    assert report["status"] == "blocked"
    checkpoint_checks = [check for check in report["checks"] if check["name"] == "checkpoints"]
    assert checkpoint_checks == [
        {
            "name": "checkpoints",
            "status": "fail",
            "summary": (
                "checkpoint path remaps are diagnostic-only and cannot satisfy final publication "
                "admission"
            ),
        }
    ]


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
    assert "--expected-campaign-id" in output
    assert "--checkpoint-path-map" in output
    assert "--private-ops-repository" in output


def test_cli_checkpoint_path_map_reaches_doctor_with_repo_root(monkeypatch, tmp_path: Path) -> None:
    """The repeatable CLI mapping is passed with the selected repository root."""
    args = robot_sf_cli._build_parser().parse_args(
        [
            "release",
            "doctor",
            "--repo",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "manifest.yaml"),
            "--expected-release-sha",
            "a" * 40,
            "--expected-base-sha",
            "b" * 40,
            "--tag",
            "release",
            "--checkpoint-path-map",
            "/hpc/source.zip=checkpoints/model.zip",
        ]
    )
    captured: dict[str, object] = {}

    def fake_report(**kwargs):
        captured.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(release_cli, "collect_release_doctor_report", fake_report)
    assert release_cli.handle(args) == 0
    assert captured["checkpoint_path_map"] == ["/hpc/source.zip=checkpoints/model.zip"]
    assert captured["repo"] == tmp_path.resolve()


def test_cli_relative_manifest_resolves_against_repo_root(monkeypatch, tmp_path: Path) -> None:
    """Issue #7794: a relative --manifest must resolve against --repo, not cwd."""
    args = robot_sf_cli._build_parser().parse_args(
        [
            "release",
            "doctor",
            "--repo",
            str(tmp_path),
            "--manifest",
            "configs/release_manifest.yaml",
            "--expected-release-sha",
            "a" * 40,
            "--expected-base-sha",
            "b" * 40,
            "--tag",
            "release",
        ]
    )
    captured: dict[str, object] = {}

    def fake_report(**kwargs):
        captured.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(release_cli, "collect_release_doctor_report", fake_report)
    assert release_cli.handle(args) == 0
    assert captured["manifest_path"] == (tmp_path / "configs/release_manifest.yaml").resolve()
    assert captured["repo"] == tmp_path.resolve()


def test_cli_private_ops_repository_reaches_doctor_without_public_root_rewriting(
    monkeypatch, tmp_path: Path
) -> None:
    """The private ledger checkout remains an independent object-addressed root."""
    release_root = tmp_path / "release-worktree"
    private_root = tmp_path / "private-ops-worktree"
    release_root.mkdir()
    private_root.mkdir()
    args = robot_sf_cli._build_parser().parse_args(
        [
            "release",
            "doctor",
            "--repo",
            str(release_root),
            "--manifest",
            "manifest.yaml",
            "--expected-release-sha",
            "a" * 40,
            "--expected-base-sha",
            "b" * 40,
            "--tag",
            "release",
            "--private-ops-repository",
            str(private_root),
        ]
    )
    captured: dict[str, object] = {}

    def fake_report(**kwargs):
        captured.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(release_cli, "collect_release_doctor_report", fake_report)
    assert release_cli.handle(args) == 0
    assert captured["repo"] == release_root.resolve()
    assert captured["private_ops_repository"] == private_root


def test_cli_relative_doctor_paths_anchor_to_repo_from_tooling_cwd(
    monkeypatch, tmp_path: Path
) -> None:
    """Every relative doctor input uses the release checkout, not the tooling cwd."""
    tooling_root = tmp_path / "tooling-worktree"
    release_root = tmp_path / "release-worktree"
    tooling_root.mkdir()
    release_root.mkdir()
    monkeypatch.chdir(tooling_root)
    relative_paths = {
        "manifest_path": "configs/release_manifest.yaml",
        "checkpoint_receipt": "receipts/checkpoint.json",
        "private_launch_packet": "private/launch.yaml",
        "private_queue": "private/queue.yaml",
        "dissertation": "dissertation/release.md",
        "token_file": "private/token",
    }
    argv = [
        "release",
        "doctor",
        "--repo",
        str(release_root),
        "--manifest",
        relative_paths["manifest_path"],
        "--expected-release-sha",
        "a" * 40,
        "--expected-base-sha",
        "b" * 40,
        "--tag",
        "release",
    ]
    for option, key in (
        ("--checkpoint-receipt", "checkpoint_receipt"),
        ("--private-launch-packet", "private_launch_packet"),
        ("--private-queue", "private_queue"),
        ("--dissertation", "dissertation"),
        ("--token-file", "token_file"),
    ):
        argv.extend((option, relative_paths[key]))
    args = robot_sf_cli._build_parser().parse_args(argv)
    captured: dict[str, object] = {}

    def fake_report(**kwargs):
        captured.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(release_cli, "collect_release_doctor_report", fake_report)
    assert release_cli.handle(args) == 0
    for key, relative_path in relative_paths.items():
        assert captured[key] == (release_root / relative_path).resolve()


def test_cli_absolute_manifest_stays_absolute(monkeypatch, tmp_path: Path) -> None:
    """An absolute --manifest is honored verbatim even when --repo differs."""
    manifest = tmp_path / "manifest.yaml"
    args = robot_sf_cli._build_parser().parse_args(
        [
            "release",
            "doctor",
            "--repo",
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--expected-release-sha",
            "a" * 40,
            "--expected-base-sha",
            "b" * 40,
            "--tag",
            "release",
        ]
    )
    captured: dict[str, object] = {}

    def fake_report(**kwargs):
        captured.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(release_cli, "collect_release_doctor_report", fake_report)
    assert release_cli.handle(args) == 0
    assert captured["manifest_path"] == manifest.resolve()


def test_cli_checkpoint_path_map_runs_real_collector_and_receipt_validator(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """The public CLI validates an exact remote receipt against byte-identical local data."""
    config = tmp_path / "campaign.yaml"
    config.write_text("name: release\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoints" / "model.zip"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    remote_path = "/hpc/gpfs2/licca/checkpoints/model.zip"
    reference = SimpleNamespace(
        planner_key="ppo",
        algo="ppo",
        kind="model_path",
        value=remote_path,
        implicit=False,
    )
    cfg = SimpleNamespace(references=[reference])
    manifest = SimpleNamespace(canonical_campaign_config_path=config)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "campaign-checkpoint-staging-receipt.v1",
                "status": "ok",
                "mode": "enforced_staged",
                "stage": True,
                "submit_safe": True,
                "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "campaign_config_sha256": release_doctor._sha256(config),
                "checkpoint_registry_sha256": "0" * 64,
                "arms": [
                    {
                        "planner_key": "ppo",
                        "algo": "ppo",
                        "kind": "model_path",
                        "value": remote_path,
                        "implicit": False,
                        "status": "staged",
                        "resolved_path": remote_path,
                        "checkpoint_sha256": release_doctor._sha256(checkpoint),
                        "hash_source": "computed_file",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    passed = ReleaseDoctorCheck("fixture", "pass", "safe")
    monkeypatch.setattr(
        release_doctor,
        "_manifest_check",
        lambda *args: (passed, manifest, cfg),
    )
    monkeypatch.setattr(release_doctor, "_git_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_ci_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_tag_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_release_identity_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_cluster_check", lambda *args, **kwargs: passed)
    monkeypatch.setattr(release_doctor, "_disk_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_zenodo_check", lambda *args, **kwargs: [passed])
    monkeypatch.setattr(release_doctor, "_dissertation_check", lambda *args: passed)
    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda fixture_cfg: fixture_cfg.references,
    )
    args = robot_sf_cli._build_parser().parse_args(
        [
            "release",
            "doctor",
            "--repo",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "manifest.yaml"),
            "--expected-release-sha",
            "a" * 40,
            "--expected-base-sha",
            "b" * 40,
            "--tag",
            "release",
            "--checkpoint-receipt",
            str(receipt),
            "--checkpoint-path-map",
            f"{remote_path}=checkpoints/model.zip",
        ]
    )

    assert release_cli.handle(args) == 0
    report = json.loads(capsys.readouterr().out)
    checkpoint_checks = [check for check in report["checks"] if check["name"] == "checkpoints"]
    assert checkpoint_checks[0]["status"] == "pass"
    assert "1 checkpoint" in checkpoint_checks[0]["summary"]


def _cross_checkout_checkpoint_doctor_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    mapped_checkpoint_bytes: bytes | None,
) -> list[str]:
    """Build a tooling-checkout doctor invocation against a separate release checkout."""
    tooling_root = tmp_path / "tooling-worktree"
    release_root = tmp_path / "release-worktree"
    tooling_root.mkdir()
    (release_root / "configs").mkdir(parents=True)
    (release_root / "receipts").mkdir()
    monkeypatch.chdir(tooling_root)

    config = release_root / "configs" / "campaign.yaml"
    config.write_text("name: release\n", encoding="utf-8")
    remote_path = "/execution-host/checkpoints/model.zip"
    reference = SimpleNamespace(
        planner_key="ppo",
        algo="ppo",
        kind="model_path",
        value=remote_path,
        implicit=False,
    )
    cfg = SimpleNamespace(references=[reference])
    manifest = SimpleNamespace(canonical_campaign_config_path=config)
    expected_checkpoint = release_root / "expected-model.zip"
    expected_checkpoint.write_bytes(b"expected-checkpoint")
    receipt = release_root / "receipts" / "checkpoint.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "campaign-checkpoint-staging-receipt.v1",
                "status": "ok",
                "mode": "enforced_staged",
                "stage": True,
                "submit_safe": True,
                "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "campaign_config_sha256": release_doctor._sha256(config),
                "checkpoint_registry_sha256": "0" * 64,
                "arms": [
                    {
                        "planner_key": "ppo",
                        "algo": "ppo",
                        "kind": "model_path",
                        "value": remote_path,
                        "implicit": False,
                        "status": "staged",
                        "resolved_path": remote_path,
                        "checkpoint_sha256": release_doctor._sha256(expected_checkpoint),
                        "hash_source": "computed_file",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    passed = ReleaseDoctorCheck("fixture", "pass", "safe")
    monkeypatch.setattr(release_doctor, "_manifest_check", lambda *args: (passed, manifest, cfg))
    monkeypatch.setattr(release_doctor, "_git_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_ci_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_tag_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_release_identity_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_cluster_check", lambda *args, **kwargs: passed)
    monkeypatch.setattr(release_doctor, "_disk_check", lambda *args: passed)
    monkeypatch.setattr(release_doctor, "_zenodo_check", lambda *args, **kwargs: [passed])
    monkeypatch.setattr(release_doctor, "_dissertation_check", lambda *args: passed)
    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda fixture_cfg: fixture_cfg.references,
    )
    argv = [
        "release",
        "doctor",
        "--repo",
        str(release_root),
        "--manifest",
        "configs/manifest.yaml",
        "--expected-release-sha",
        "a" * 40,
        "--expected-base-sha",
        "b" * 40,
        "--tag",
        "release",
        "--checkpoint-receipt",
        "receipts/checkpoint.json",
    ]
    if mapped_checkpoint_bytes is not None:
        mapped_checkpoint = release_root / "checkpoints" / "model.zip"
        mapped_checkpoint.parent.mkdir()
        mapped_checkpoint.write_bytes(mapped_checkpoint_bytes)
        argv.extend(("--checkpoint-path-map", f"{remote_path}=checkpoints/model.zip"))
    return argv


def test_cross_checkout_doctor_names_verifier_location_remediation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    """Issue #7819: the public doctor directs missing bytes to the verification host."""
    argv = _cross_checkout_checkpoint_doctor_args(
        monkeypatch,
        tmp_path,
        mapped_checkpoint_bytes=None,
    )

    args = robot_sf_cli._build_parser().parse_args(argv)
    assert release_cli.handle(args) == 2
    report = json.loads(capsys.readouterr().out)
    summary = next(check["summary"] for check in report["checks"] if check["name"] == "checkpoints")
    assert "verifier-location condition, not an input mismatch" in summary
    assert "remap it with --checkpoint-path-map on this verification host" in summary
    assert "run the doctor on the execution host" in summary


def test_cross_checkout_doctor_keeps_mapped_checksum_mismatch_distinct(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    """Issue #7819: mapped bytes with the wrong digest remain a checksum failure."""
    argv = _cross_checkout_checkpoint_doctor_args(
        monkeypatch,
        tmp_path,
        mapped_checkpoint_bytes=b"wrong-checkpoint",
    )

    args = robot_sf_cli._build_parser().parse_args(argv)
    assert release_cli.handle(args) == 2
    report = json.loads(capsys.readouterr().out)
    summary = next(check["summary"] for check in report["checks"] if check["name"] == "checkpoints")
    assert "checksum changed" in summary
    assert "verifier-location" not in summary


def _write_final_packet_fixture(
    tmp_path: Path,
    *,
    source_sha: str = "a" * 40,
    tag: str = "release-tag",
    resource_profile: str = "licca",
    frozen_status: bool = False,
) -> tuple[Path, Path]:
    """Write a minimal but complete final packet/queue pair for doctor tests."""
    input_names = (
        "release_manifest",
        "canonical_campaign_config",
        "scenario_matrix",
        "public_single_node_entrypoint",
        "checkpoint_staging_receipt",
        "runtime_smoke_receipt",
        "private_wrapper",
        "release_runner",
    )
    input_digests: dict[str, str] = {}
    for name in input_names:
        path = tmp_path / "configs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{name}\n", encoding="utf-8")
        input_digests[name] = release_doctor._sha256(path)
    campaign = "campaign-1"
    identity_fields = {
        "release_manifest_sha256": input_digests["release_manifest"],
        "canonical_config_sha256": input_digests["canonical_campaign_config"],
        "scenario_matrix_sha256": input_digests["scenario_matrix"],
        "checkpoint_receipt_sha256": input_digests["checkpoint_staging_receipt"],
        "runtime_smoke_receipt_sha256": input_digests["runtime_smoke_receipt"],
        "public_entrypoint_sha256": input_digests["public_single_node_entrypoint"],
        "private_wrapper_sha256": input_digests["private_wrapper"],
        "startup_sentinel_sha256": "b" * 64,
        "admission_helper_sha256": "b" * 64,
    }
    resources = {
        "licca": {
            "cluster": "licca",
            "partition": "epyc-gpu",
            "route_id": "licca:epyc-gpu",
            "cpus": 36,
            "gpus": 1,
            "gpu_type": "A100",
            "mem_gb": 256,
            "wall_clock": "36:00:00",
            "wall_clock_seconds": 129600,
        },
        "imech192": {
            "cluster": "imech192",
            "partition": "l40s",
            "route_id": "imech192:l40s",
            "cpus": 36,
            "gpus": 1,
            "gpu_type": "L40S",
            "mem_gb": 256,
            "wall_clock": "36:00:00",
            "wall_clock_seconds": 129600,
            "qos": "l40s-gpu",
        },
    }[resource_profile]
    packet = {
        "schema": "robot-sf-launch-packet.v1",
        "packet_version": 1,
        "queue_id": "queue-1",
        "campaign_id": campaign,
        "campaign": campaign,
        "state": "ready",
        "dispatchable": True,
        "execution_contract": {
            **resources,
            "resources_exact": True,
            "release_label": "release-label",
            "force_cpu": True,
            "release_tag": tag,
            "startup_sentinel_required": True,
            "startup_prefix": 'source "$SLURM_STARTUP_SENTINEL"',
            "runtime_smoke_receipt_max_age_hours": 24,
        },
        "identity": {
            "public_source_commit": source_sha,
            **identity_fields,
        },
        "inputs": {
            name: {
                "path": f"configs/{name}",
                "sha256": input_digests[name],
                **(
                    {
                        "interface_arity": 5,
                        "fifth_argument": "exact_source_runtime_smoke_result",
                    }
                    if name == "public_single_node_entrypoint"
                    else {}
                ),
            }
            for name in input_names
        }
        | {
            "source": {
                "repository": "https://github.com/ll7/robot_sf_ll7",
                "public_commit": source_sha,
            }
        },
        "sentinel_traceability": {
            "required": True,
            "source": "$SLURM_STARTUP_SENTINEL",
            "helper": "$SLURM_STARTUP_HELPER",
            "startup_receipt": "$SLURM_STARTUP_RECEIPT",
            "admission_trace": "$SLURM_ADMISSION_RECEIPT",
            "required_identity_fields": sorted(release_doctor._REQUIRED_PACKET_TRACE_FIELDS),
        },
    }
    if frozen_status:
        packet["status"] = "admitted_frozen"
    else:
        packet["admission"] = {"status": "admitted", "dispatchable": True}
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet, sort_keys=True), encoding="utf-8")
    packet_hash = release_doctor._sha256(packet_path)
    queue_path = tmp_path / "queue.yaml"
    submit_identity = [
        f"RELEASE_LAUNCH_PACKET_SHA256={packet_hash}",
        f"RELEASE_LAUNCH_PACKET_PATH={packet_path}",
        f"RELEASE_CAMPAIGN_ID={campaign}",
        "RELEASE_LABEL=release-label",
        "RELEASE_FORCE_CPU=1",
        "RELEASE_MANIFEST_PATH=configs/release_manifest",
        "RELEASE_SCENARIO_PATH=configs/scenario_matrix",
        "RELEASE_CHECKPOINT_RECEIPT_PATH=configs/checkpoint_staging_receipt",
        "RELEASE_RUNTIME_SMOKE_RECEIPT_PATH=configs/runtime_smoke_receipt",
        f"RELEASE_EXPECTED_CPUS={resources['cpus']}",
        f"RELEASE_EXPECTED_GPUS={resources['gpus']}",
        f"RELEASE_EXPECTED_MEM_GB={resources['mem_gb']}",
        f"RELEASE_EXPECTED_WALLTIME={resources['wall_clock']}",
        *(
            f"{field}={value}"
            for field, value in (
                ("RELEASE_MANIFEST_SHA256", identity_fields["release_manifest_sha256"]),
                ("RELEASE_CONFIG_SHA256", identity_fields["canonical_config_sha256"]),
                ("RELEASE_SCENARIO_SHA256", identity_fields["scenario_matrix_sha256"]),
                (
                    "RELEASE_CHECKPOINT_RECEIPT_SHA256",
                    identity_fields["checkpoint_receipt_sha256"],
                ),
                (
                    "RELEASE_RUNTIME_SMOKE_RECEIPT_SHA256",
                    identity_fields["runtime_smoke_receipt_sha256"],
                ),
                ("RELEASE_PUBLIC_SCRIPT_SHA256", identity_fields["public_entrypoint_sha256"]),
            )
        ),
    ]
    scheduler_args = [
        "--sbatch-arg",
        f"--partition={resources['partition']}",
        "--sbatch-arg",
        f"--gres=gpu:{resources['gpu_type'].lower()}:{resources['gpus']}",
        "--sbatch-arg",
        f"--cpus-per-task={resources['cpus']}",
        "--sbatch-arg",
        f"--mem={resources['mem_gb']}G",
        "--sbatch-arg",
        f"--time={resources['wall_clock']}",
        *(["--sbatch-arg", f"--qos={resources['qos']}"] if resources.get("qos") else []),
    ]
    submit_args = (
        "--sbatch-arg --export=ALL," + ",".join(submit_identity) + " " + " ".join(scheduler_args)
    )
    queue_path.write_text(
        yaml.safe_dump(
            [
                {
                    "queue_id": "queue-1",
                    "campaign": campaign,
                    "state": "ready",
                    "expected_public_commit": source_sha,
                    "artifact_manifest": (
                        f"ops/jobs/launch_packets/{packet_path.name} sha256:{packet_hash}"
                    ),
                    "submit_args": submit_args,
                    "cluster": resources["cluster"],
                    "partition": resources["partition"],
                    "route_id": resources["route_id"],
                    **({"qos": resources["qos"]} if resources.get("qos") else {}),
                    "cpus": str(resources["cpus"]),
                    "gpus": str(resources["gpus"]),
                    "mem_gb": str(resources["mem_gb"]),
                    "estimated_elapsed_sec": str(resources["wall_clock_seconds"]),
                }
            ],
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return packet_path, queue_path


def test_final_cluster_check_validates_packet_queue_and_launch_contract(tmp_path: Path) -> None:
    """Final mode admits only a concrete packet and matching dispatch row."""
    packet, queue = _write_final_packet_fixture(tmp_path)
    check = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
        checkpoint_receipt=tmp_path / "configs" / "checkpoint_staging_receipt",
    )
    assert check.status == "pass", check.summary

    payload = json.loads(packet.read_text(encoding="utf-8"))
    payload["execution_contract"]["partition"] = "wrong"
    packet.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
        checkpoint_receipt=tmp_path / "configs" / "checkpoint_staging_receipt",
    )
    assert rejected.status == "fail"
    assert "resource contract mismatch" in rejected.summary


def test_final_cluster_check_accepts_frozen_imech192_packet(tmp_path: Path) -> None:
    """The doctor admits the exact imech192/L40S route without LiCCA defaults."""
    packet, queue = _write_final_packet_fixture(
        tmp_path, resource_profile="imech192", frozen_status=True
    )
    check = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
        checkpoint_receipt=tmp_path / "configs" / "checkpoint_staging_receipt",
    )
    assert check.status == "pass", check.summary


def test_final_cluster_check_accepts_count_only_gpu_gres(tmp_path: Path) -> None:
    """A partition-bound Slurm GRES may omit a redundant GPU type."""
    packet, queue = _write_final_packet_fixture(
        tmp_path, resource_profile="imech192", frozen_status=True
    )
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    queue_payload[0]["submit_args"] = queue_payload[0]["submit_args"].replace(
        "--gres=gpu:l40s:1", "--gres=gpu:1"
    )
    queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
    check = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
        checkpoint_receipt=tmp_path / "configs" / "checkpoint_staging_receipt",
    )
    assert check.status == "pass", check.summary


def test_scheduler_gres_rejects_composite_extra_resources() -> None:
    """A composite GRES carrying undeclared resources must be rejected."""
    assert release_doctor._scheduler_gres("gpu:l40s:1,mps:1") is None
    assert release_doctor._scheduler_gres("gpu:l40s:1") == (1, "l40s")
    assert release_doctor._scheduler_gres("gpu:1") == (1, None)


def test_scheduler_qos_undeclared_by_packet_is_rejected() -> None:
    """A --qos value the packet contract never declared must fail admission."""
    packet: dict[str, Any] = {
        "execution_contract": {
            "partition": "l40s",
            "gpus": 1,
            "gpu_type": "l40s",
            "cpus": 36,
            "mem_gb": 256,
        }
    }
    problems = release_doctor._validate_scheduler_submit_args(
        "--qos=other --gres=gpu:l40s:1 --cpus-per-task=36 --mem=256G", packet
    )
    assert any("scheduler --qos is undeclared by the packet" in p for p in problems)


def test_scheduler_qos_matching_packet_still_accepted() -> None:
    """A --qos value exactly matching the packet's concrete qos passes."""
    packet: dict[str, Any] = {
        "execution_contract": {
            "partition": "l40s",
            "gpus": 1,
            "gpu_type": "l40s",
            "cpus": 36,
            "mem_gb": 256,
            "qos": "l40s-gpu",
        }
    }
    problems = release_doctor._validate_scheduler_submit_args(
        "--qos=l40s-gpu --gres=gpu:l40s:1 --cpus-per-task=36 --mem=256G", packet
    )
    assert not any("scheduler --qos" in p for p in problems)


@pytest.mark.parametrize(
    ("drift_target", "drift_value", "summary"),
    [
        ("packet_partition", "epyc-gpu", "resource contract mismatch: partition"),
        ("queue_route_id", "licca:epyc-gpu", "resource contract mismatch: route_id"),
        ("queue_cpus", "32", "resource contract mismatch: cpus"),
    ],
)
def test_final_cluster_check_rejects_imech192_resource_drift(
    tmp_path: Path, drift_target: str, drift_value: str, summary: str
) -> None:
    """A route/resource mismatch remains blocked in either frozen store."""
    packet, queue = _write_final_packet_fixture(
        tmp_path, resource_profile="imech192", frozen_status=True
    )
    if drift_target == "packet_partition":
        packet_payload = json.loads(packet.read_text(encoding="utf-8"))
        packet_payload["execution_contract"]["partition"] = drift_value
        packet.write_text(json.dumps(packet_payload, sort_keys=True), encoding="utf-8")
    else:
        queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
        queue_payload[0]["route_id" if drift_target == "queue_route_id" else "cpus"] = drift_value
        queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert summary in rejected.summary


@pytest.mark.parametrize(
    ("original", "replacement", "summary"),
    [
        ("--partition=l40s", "--partition=epyc-gpu", "scheduler --partition"),
        ("--gres=gpu:l40s:1", "--gres=gpu:l40s:2", "scheduler --gres GPU count"),
        ("--gres=gpu:l40s:1", "--gres=gpu:a100:1", "scheduler --gres GPU type"),
        ("--cpus-per-task=36", "--cpus-per-task=32", "scheduler --cpus-per-task"),
        ("--mem=256G", "--mem=128G", "scheduler --mem"),
        ("--time=36:00:00", "--time=12:00:00", "scheduler --time"),
        ("--qos=l40s-gpu", "--qos=other", "scheduler --qos"),
    ],
)
def test_final_cluster_check_rejects_imech192_scheduler_arg_drift(
    tmp_path: Path, original: str, replacement: str, summary: str
) -> None:
    """Actual scheduler flags cannot bypass the packet-bound route contract."""
    packet, queue = _write_final_packet_fixture(
        tmp_path, resource_profile="imech192", frozen_status=True
    )
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    queue_payload[0]["submit_args"] = queue_payload[0]["submit_args"].replace(original, replacement)
    queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert summary in rejected.summary


def test_final_cluster_check_rejects_substring_packet_hash_binding(tmp_path: Path) -> None:
    """A similarly named export key cannot satisfy packet hash binding."""
    packet, queue = _write_final_packet_fixture(
        tmp_path, resource_profile="imech192", frozen_status=True
    )
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    queue_payload[0]["submit_args"] = queue_payload[0]["submit_args"].replace(
        "RELEASE_LAUNCH_PACKET_SHA256=", "XRELEASE_LAUNCH_PACKET_SHA256=", 1
    )
    queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert "RELEASE_LAUNCH_PACKET_SHA256 is missing" in rejected.summary


def test_final_cluster_check_rejects_artifact_manifest_hash_drift(tmp_path: Path) -> None:
    """A queue artifact-manifest digest cannot describe different packet bytes."""
    packet, queue = _write_final_packet_fixture(tmp_path)
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    artifact_manifest = queue_payload[0]["artifact_manifest"]
    artifact_path = artifact_manifest.split(" sha256:", 1)[0]
    queue_payload[0]["artifact_manifest"] = f"{artifact_path} sha256:{'0' * 64}"
    queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert "artifact manifest hash does not match packet" in rejected.summary


def test_final_cluster_check_rejects_digest_less_artifact_manifest(tmp_path: Path) -> None:
    """Issue #7921: a queue artifact manifest without a bound digest fails closed."""
    packet, queue = _write_final_packet_fixture(tmp_path)
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    artifact_manifest = queue_payload[0]["artifact_manifest"]
    artifact_path = artifact_manifest.split(" sha256:", 1)[0]
    # Strip the digest binding entirely (the previous bypass path).
    queue_payload[0]["artifact_manifest"] = artifact_path
    queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert "artifact manifest is not digest-bound to the packet" in rejected.summary


@pytest.mark.parametrize(
    "field",
    [
        "RELEASE_LAUNCH_PACKET_SHA256",
        "RELEASE_LAUNCH_PACKET_PATH",
        "RELEASE_MANIFEST_SHA256",
        "RELEASE_CONFIG_SHA256",
        "RELEASE_SCENARIO_SHA256",
        "RELEASE_CHECKPOINT_RECEIPT_SHA256",
        "RELEASE_RUNTIME_SMOKE_RECEIPT_SHA256",
        "RELEASE_PUBLIC_SCRIPT_SHA256",
        "RELEASE_MANIFEST_PATH",
        "RELEASE_SCENARIO_PATH",
        "RELEASE_CHECKPOINT_RECEIPT_PATH",
        "RELEASE_RUNTIME_SMOKE_RECEIPT_PATH",
        "RELEASE_CAMPAIGN_ID",
        "RELEASE_LABEL",
        "RELEASE_FORCE_CPU",
        "RELEASE_EXPECTED_CPUS",
        "RELEASE_EXPECTED_GPUS",
        "RELEASE_EXPECTED_MEM_GB",
        "RELEASE_EXPECTED_WALLTIME",
    ],
)
def test_final_cluster_check_rejects_duplicate_release_export(tmp_path: Path, field: str) -> None:
    """Every exported release identity must have one effective value."""
    packet, queue = _write_final_packet_fixture(
        tmp_path, resource_profile="imech192", frozen_status=True
    )
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    submit_args = queue_payload[0]["submit_args"]
    duplicate = "wrong" if not field.endswith("_SHA256") else "0" * 64
    queue_payload[0]["submit_args"] = submit_args.replace(
        "--export=ALL,", f"--export=ALL,{field}={duplicate},", 1
    )
    queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert f"private queue {field} is duplicated" in rejected.summary


@pytest.mark.parametrize(
    "field",
    ["RELEASE_LABEL", "RELEASE_LAUNCH_PACKET_PATH", "RELEASE_FORCE_CPU"],
)
def test_final_cluster_check_requires_wrapper_identity_exports(tmp_path: Path, field: str) -> None:
    """Every identity required by the private wrapper must be present at admission."""
    packet, queue = _write_final_packet_fixture(
        tmp_path, resource_profile="imech192", frozen_status=True
    )
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    assignments = queue_payload[0]["submit_args"].split(",")
    queue_payload[0]["submit_args"] = ",".join(
        assignment for assignment in assignments if not assignment.startswith(f"{field}=")
    )
    queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")

    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )

    assert rejected.status == "fail"
    assert f"private queue {field} is missing" in rejected.summary


@pytest.mark.parametrize(
    ("original", "replacement", "expected_problem"),
    [
        ("RELEASE_LABEL=release-label", "RELEASE_LABEL=other", "RELEASE_LABEL is not bound"),
        ("RELEASE_FORCE_CPU=1", "RELEASE_FORCE_CPU=0", "RELEASE_FORCE_CPU is not bound"),
        (
            "RELEASE_LAUNCH_PACKET_PATH=",
            "RELEASE_LAUNCH_PACKET_PATH=/wrong/other-packet.yaml",
            "packet path is not bound",
        ),
    ],
)
def test_final_cluster_check_rejects_wrapper_identity_drift(
    tmp_path: Path, original: str, replacement: str, expected_problem: str
) -> None:
    """Wrapper-required exports must remain bound to the frozen packet."""
    packet, queue = _write_final_packet_fixture(
        tmp_path, resource_profile="imech192", frozen_status=True
    )
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    submit_args = queue_payload[0]["submit_args"]
    if original.endswith("="):
        matching = next(item for item in submit_args.split(",") if item.startswith(original))
        submit_args = submit_args.replace(matching, replacement, 1)
    else:
        submit_args = submit_args.replace(original, replacement, 1)
    queue_payload[0]["submit_args"] = submit_args
    queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")

    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )

    assert rejected.status == "fail"
    assert expected_problem in rejected.summary


@pytest.mark.parametrize(
    ("submit_args", "expected_problem"),
    [
        ("'", "private queue submit_args quoting is invalid"),
        ("--sbatch-arg", "private queue --sbatch-arg is missing an option"),
        ("plain --export", "private queue --export is missing a value"),
        (
            "--sbatch-arg=--export=ALL,RELEASE_MISSING",
            "private queue RELEASE export is missing a value",
        ),
        (
            "--export=ALL,RELEASE_BAD-NAME=value",
            "private queue RELEASE export key is invalid",
        ),
    ],
)
def test_release_export_parser_rejects_malformed_arguments(
    submit_args: str, expected_problem: str
) -> None:
    """Malformed Slurm export forms must fail closed without exposing values."""
    values, problems = release_doctor._parse_release_exports(submit_args)

    assert values == {}
    assert expected_problem in problems


def test_release_export_parser_accepts_separate_export_value() -> None:
    """The parser supports Slurm's ``--export VALUE`` spelling exactly."""
    values, problems = release_doctor._parse_release_exports(
        "ignored --export ALL,RELEASE_CAMPAIGN_ID=campaign-1 --partition=l40s"
    )

    assert problems == []
    assert values == {"RELEASE_CAMPAIGN_ID": ["campaign-1"]}


def test_final_cluster_check_rejects_queue_qos_drift(tmp_path: Path) -> None:
    """A queue-level QoS, when recorded, must match the packet route."""
    packet, queue = _write_final_packet_fixture(
        tmp_path, resource_profile="imech192", frozen_status=True
    )
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    queue_payload[0]["qos"] = "other-qos"
    queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert "resource contract mismatch: qos" in rejected.summary


@pytest.mark.parametrize("value", [True, 36.0, "36.5"])
def test_final_cluster_check_rejects_non_strict_integer_queue_resource(
    tmp_path: Path, value: object
) -> None:
    """Boolean, float, and fractional resource values cannot be truncated."""
    packet, queue = _write_final_packet_fixture(
        tmp_path, resource_profile="imech192", frozen_status=True
    )
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    queue_payload[0]["cpus"] = value
    queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert "private queue resource contract mismatch: cpus" in rejected.summary


@pytest.mark.parametrize("value", [True, 0, -1, 0.0, "0", "0.5"])
def test_final_cluster_check_rejects_nonpositive_estimated_elapsed(
    tmp_path: Path, value: object
) -> None:
    """Queue duration evidence must be a strict positive integer."""
    packet, queue = _write_final_packet_fixture(
        tmp_path, resource_profile="imech192", frozen_status=True
    )
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    queue_payload[0]["estimated_elapsed_sec"] = value
    queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert "estimated_elapsed_sec must be positive" in rejected.summary


def test_final_cluster_check_rejects_nonpositive_packet_wall_clock(tmp_path: Path) -> None:
    """The frozen packet cannot admit a zero-duration wall clock."""
    packet, queue = _write_final_packet_fixture(
        tmp_path, resource_profile="imech192", frozen_status=True
    )
    packet_payload = json.loads(packet.read_text(encoding="utf-8"))
    packet_payload["execution_contract"]["wall_clock"] = "00:00:00"
    packet_payload["execution_contract"]["wall_clock_seconds"] = 0
    packet.write_text(json.dumps(packet_payload, sort_keys=True), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert "wall_clock must be positive" in rejected.summary


def test_final_cluster_check_rejects_fractional_runtime_smoke_max_age(tmp_path: Path) -> None:
    """Runtime-smoke freshness limits cannot use int() truncation."""
    packet, queue = _write_final_packet_fixture(tmp_path)
    packet_payload = json.loads(packet.read_text(encoding="utf-8"))
    packet_payload["execution_contract"]["runtime_smoke_receipt_max_age_hours"] = 24.9
    packet.write_text(json.dumps(packet_payload, sort_keys=True), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert "freshness contract is not 24 hours" in rejected.summary


def test_final_cluster_check_accepts_exact_scheduler_time_as_duration_evidence(
    tmp_path: Path,
) -> None:
    """A valid scheduler time can stand in for a legacy missing duration field."""
    packet, queue = _write_final_packet_fixture(
        tmp_path, resource_profile="imech192", frozen_status=True
    )
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    queue_payload[0].pop("estimated_elapsed_sec")
    queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
    check = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
        checkpoint_receipt=tmp_path / "configs" / "checkpoint_staging_receipt",
    )
    assert check.status == "pass", check.summary


def test_final_cluster_check_rejects_missing_duration_without_exact_scheduler_time(
    tmp_path: Path,
) -> None:
    """A queue row must retain duration evidence when scheduler time drifts."""
    packet, queue = _write_final_packet_fixture(
        tmp_path, resource_profile="imech192", frozen_status=True
    )
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    queue_payload[0].pop("estimated_elapsed_sec")
    queue_payload[0]["submit_args"] = queue_payload[0]["submit_args"].replace(
        "--time=36:00:00", "--time=12:00:00"
    )
    queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert "estimated_elapsed_sec or exact scheduler --time is required" in rejected.summary


def test_final_cluster_check_rejects_missing_public_input(tmp_path: Path) -> None:
    """Final packet admission cannot defer missing public files to the wrapper."""
    packet, queue = _write_final_packet_fixture(tmp_path)
    (tmp_path / "configs" / "release_manifest").unlink()
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert "release_manifest file is missing" in rejected.summary


@pytest.mark.parametrize("input_name", release_doctor._PUBLIC_PACKET_INPUT_NAMES)
def test_final_cluster_check_rejects_each_undeclared_public_input(
    tmp_path: Path, input_name: str
) -> None:
    """The canonical packet cannot omit any declared public input."""
    packet, queue = _write_final_packet_fixture(tmp_path)
    payload = json.loads(packet.read_text(encoding="utf-8"))
    del payload["inputs"][input_name]
    packet.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert f"inputs.{input_name} is missing" in rejected.summary


@pytest.mark.parametrize(
    ("packet_queue_id", "row_queue_id", "summary"),
    [
        ("", "queue-1", "launch packet queue_id is missing or not concrete"),
        ("pending-queue", "pending-queue", "launch packet queue_id is missing or not concrete"),
        ("queue-1", "", "private queue row queue_id is missing or not concrete"),
    ],
)
def test_final_cluster_check_requires_concrete_packet_and_row_queue_ids(
    tmp_path: Path, packet_queue_id: str, row_queue_id: str, summary: str
) -> None:
    """Queue identity cannot be inferred from an empty or placeholder ID."""
    packet, queue = _write_final_packet_fixture(tmp_path)
    packet_payload = json.loads(packet.read_text(encoding="utf-8"))
    packet_payload["queue_id"] = packet_queue_id
    packet.write_text(json.dumps(packet_payload, sort_keys=True), encoding="utf-8")
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    queue_payload[0]["queue_id"] = row_queue_id
    queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert summary in rejected.summary


def test_final_cluster_check_rejects_runtime_smoke_identity_drift(tmp_path: Path) -> None:
    """Final admission binds the queue and packet to one exact smoke result."""
    packet, queue = _write_final_packet_fixture(tmp_path)
    payload = json.loads(packet.read_text(encoding="utf-8"))
    payload["identity"]["runtime_smoke_receipt_sha256"] = "c" * 64
    packet.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert "runtime_smoke_receipt hash is not bound" in rejected.summary


def test_final_cluster_check_rejects_runtime_smoke_receipt_mutation(
    tmp_path: Path,
) -> None:
    """Final admission rejects runtime-smoke bytes changed after packet creation."""
    packet, queue = _write_final_packet_fixture(tmp_path)
    receipt = tmp_path / "configs" / "runtime_smoke_receipt"
    receipt.write_text("mutated after packet creation\n", encoding="utf-8")
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
    )
    assert rejected.status == "fail"
    assert "runtime_smoke_receipt hash does not match checkout" in rejected.summary


def test_packet_private_evidence_rejects_missing_receipt_file(tmp_path: Path) -> None:
    """A final packet without its pinned checkpoint receipt fails closed."""
    packet, queue = _write_final_packet_fixture(tmp_path)
    check = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
        checkpoint_receipt=tmp_path / "configs" / "missing-receipt.json",
    )
    assert check.status == "fail"
    assert "checkpoint receipt file is missing" in check.summary


def test_packet_private_evidence_rejects_receipt_drift(tmp_path: Path) -> None:
    """A receipt file that drifts from the packet-pinned digest fails closed."""
    packet, queue = _write_final_packet_fixture(tmp_path)
    receipt = tmp_path / "configs" / "checkpoint_staging_receipt"
    drifted = tmp_path / "configs" / "drifted-receipt.json"
    drifted.write_text(receipt.read_text(encoding="utf-8") + "\n# drifted\n", encoding="utf-8")
    check = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
        checkpoint_receipt=drifted,
    )
    assert check.status == "fail"
    assert "checkpoint receipt hash does not match packet-pinned evidence" in check.summary


def test_packet_private_evidence_accepts_pinned_receipt(tmp_path: Path) -> None:
    """An exact packet-pinned receipt passes the private-evidence check."""
    packet, queue = _write_final_packet_fixture(tmp_path)
    check = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
        checkpoint_receipt=tmp_path / "configs" / "checkpoint_staging_receipt",
    )
    assert check.status == "pass", check.summary


def test_runtime_smoke_receipt_mutation_fails_closed(tmp_path: Path) -> None:
    """Issue #7919: a post-packet runtime-smoke receipt mutation fails admission."""
    packet, queue = _write_final_packet_fixture(tmp_path)
    smoke_receipt = tmp_path / "configs" / "runtime_smoke_receipt"
    # Pass the exact pinned receipt first.
    check = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
        checkpoint_receipt=tmp_path / "configs" / "checkpoint_staging_receipt",
    )
    assert check.status == "pass", check.summary

    # Mutate the runtime-smoke receipt in the checkout after packet creation.
    smoke_receipt.write_text(
        smoke_receipt.read_text(encoding="utf-8") + "# post-packet mutation\n",
        encoding="utf-8",
    )
    rejected = release_doctor._cluster_check(
        packet,
        "a" * 40,
        final=True,
        expected_tag="release-tag",
        expected_campaign_id="campaign-1",
        queue_path=queue,
        repo=tmp_path,
        checkpoint_receipt=tmp_path / "configs" / "checkpoint_staging_receipt",
    )
    assert rejected.status == "fail"
    assert "runtime_smoke_receipt hash does not match checkout" in rejected.summary
