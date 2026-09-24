"""Focused tests for the canonical environment-manifest owner (issue #8894)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.evidence import environment_manifest as env_mod
from robot_sf.evidence.environment_manifest import (
    COMPANION_DISTRIBUTIONS,
    ENVIRONMENT_MANIFEST_SCHEMA_VERSION,
    PLATFORM_CONTRACTS,
    REASON_ACCELERATOR_UNAVAILABLE,
    REASON_CARLA_UNAVAILABLE,
    REASON_COMPANION_PACKAGE_NOT_INSTALLED,
    REASON_CUDA_UNAVAILABLE,
    REASON_DRIVER_PROBE_UNAVAILABLE,
    REASON_DRIVER_UNAVAILABLE,
    REASON_FILE_MISSING,
    REASON_GIT_UNAVAILABLE,
    REASON_NOT_A_REPOSITORY,
    REASON_NOT_SET,
    REASON_PLATFORM_CLASS_MISMATCH,
    REASON_PRIVATE_OR_CREDENTIAL_CONTENT,
    REASON_PROBE_DEPENDENCY_MISSING,
    REASON_SCHEDULER_IDENTITY_REDACTED,
    REASON_UNKNOWN_PLATFORM_CLASS,
    STATUS_DECLARED,
    STATUS_OBSERVED,
    STATUS_REDACTED,
    STATUS_UNAVAILABLE,
    EnvironmentProbes,
    RedactionContext,
    build_environment_manifest,
    evaluate_environment_manifest,
    observed,
    sanitize_text,
    sanitize_value,
    semantic_digest,
    semantic_payload,
    unavailable,
)
from scripts.tools.capture_environment_manifest import main as manifest_cli_main


def _cpu_probe() -> dict:
    return {
        "class": unavailable(REASON_PROBE_DEPENDENCY_MISSING),
        "library": unavailable(REASON_PROBE_DEPENDENCY_MISSING),
        "runtime_version": unavailable(REASON_PROBE_DEPENDENCY_MISSING),
        "driver_version": unavailable(REASON_PROBE_DEPENDENCY_MISSING),
        "devices": unavailable(REASON_PROBE_DEPENDENCY_MISSING),
        "reason": REASON_PROBE_DEPENDENCY_MISSING,
    }


def _gpu_probe(*, cuda_available: bool) -> dict:
    if cuda_available:
        return {
            "class": observed("cuda"),
            "library": observed("torch"),
            "runtime_version": observed("12.1"),
            "driver_version": observed("550.54"),
            "devices": observed(
                [{"name": "NVIDIA A100", "capability": "8.0", "total_memory_mb": 40960}]
            ),
        }
    return {
        "class": unavailable(REASON_DRIVER_UNAVAILABLE),
        "library": observed("torch"),
        "runtime_version": observed("12.1"),
        "driver_version": unavailable(REASON_DRIVER_UNAVAILABLE),
        "devices": unavailable(REASON_DRIVER_UNAVAILABLE),
        "reason": REASON_DRIVER_UNAVAILABLE,
    }


def _companions(**present: bool) -> dict:
    companions: dict = {}
    for module, distribution in COMPANION_DISTRIBUTIONS.items():
        if present.get(module):
            companions[module] = observed(
                {"module": module, "distribution": distribution, "version": "1.2.3"}
            )
        else:
            companions[module] = unavailable(REASON_COMPANION_PACKAGE_NOT_INSTALLED)
    return companions


def _repo_root(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "uv.lock").write_text("lock\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname = 'robot_sf'\n", encoding="utf-8")
    return repo


def _probes(tmp_path: Path, **overrides: object) -> tuple[Path, EnvironmentProbes]:
    repo = _repo_root(tmp_path)
    defaults: dict[str, object] = {
        "accelerator": _cpu_probe(),
        "companions": _companions(),
        "env": {},
        "scheduler_env": {},
        "hostname": "testhost",
        "home": "/home/testuser",
        "git_commit": observed("a" * 40),
        "clock": lambda: "2026-09-11T00:00:00+00:00",
    }
    defaults.update(overrides)
    return repo, EnvironmentProbes(**defaults)  # type: ignore[arg-type]


def _walk_statuses(value: object) -> list[str]:
    statuses: list[str] = []
    if isinstance(value, dict):
        if "status" in value:
            statuses.append(str(value["status"]))
        else:
            for item in value.values():
                statuses.extend(_walk_statuses(item))
    elif isinstance(value, list):
        for item in value:
            statuses.extend(_walk_statuses(item))
    return statuses


def test_registered_platform_classes_cover_required_vocabulary() -> None:
    assert {"local_cpu", "local_gpu", "slurm_cpu", "slurm_gpu", "carla"} <= set(PLATFORM_CONTRACTS)
    assert PLATFORM_CONTRACTS["local_gpu"].requires_accelerator
    assert PLATFORM_CONTRACTS["carla"].requires_carla


def test_manifest_has_schema_semantic_digest_and_all_four_statuses(tmp_path: Path) -> None:
    repo, probes = _probes(tmp_path)
    manifest = build_environment_manifest(
        "local_cpu",
        repo_root=repo,
        declared_values={"operator_note": "maintenance window"},
        probes=probes,
    )
    assert manifest["schema_version"] == ENVIRONMENT_MANIFEST_SCHEMA_VERSION
    assert manifest["semantic_digest"].startswith("sha256:")
    statuses = set(_walk_statuses(manifest))
    assert {
        STATUS_OBSERVED,
        STATUS_UNAVAILABLE,
        STATUS_REDACTED,
        STATUS_DECLARED,
    } <= statuses


def test_captures_are_deterministic_across_volatile_timestamps(tmp_path: Path) -> None:
    repo, probes = _probes(tmp_path, clock=lambda: "2026-09-11T00:00:00+00:00")
    first = build_environment_manifest("local_cpu", repo_root=repo, probes=probes)
    repo_second, probes_second = _probes(tmp_path, clock=lambda: "2027-01-01T12:30:00+00:00")
    second = build_environment_manifest("local_cpu", repo_root=repo_second, probes=probes_second)
    assert first["captured_at_utc"] != second["captured_at_utc"]
    assert first["semantic_digest"] == second["semantic_digest"]
    assert semantic_digest(first) == first["semantic_digest"]
    assert "captured_at_utc" not in semantic_payload(first)
    assert "semantic_digest" not in semantic_payload(first)


def test_missing_lockfile_is_unavailable_with_stable_reason(tmp_path: Path) -> None:
    repo, probes = _probes(tmp_path)
    (repo / "uv.lock").unlink()
    manifest = build_environment_manifest("local_cpu", repo_root=repo, probes=probes)
    lock_field = manifest["repository"]["uv.lock_sha256"]
    assert lock_field == {
        "status": STATUS_UNAVAILABLE,
        "reason": REASON_FILE_MISSING,
        "detail": "uv.lock",
    }


def test_not_a_repository_commit_is_unavailable(tmp_path: Path) -> None:
    repo, probes = _probes(tmp_path, git_commit=unavailable(REASON_NOT_A_REPOSITORY))
    manifest = build_environment_manifest("local_cpu", repo_root=repo, probes=probes)
    assert manifest["repository"]["commit"]["status"] == STATUS_UNAVAILABLE
    assert manifest["repository"]["commit"]["reason"] == REASON_NOT_A_REPOSITORY


def test_cpu_manifest_passes_local_cpu_check(tmp_path: Path) -> None:
    repo, probes = _probes(tmp_path)
    manifest = build_environment_manifest("local_cpu", repo_root=repo, probes=probes)
    result = evaluate_environment_manifest(manifest, expected_platform_class="local_cpu")
    assert result["compatible"] is True
    assert result["reasons"] == []


def test_cuda_manifest_passes_local_gpu_check(tmp_path: Path) -> None:
    repo, probes = _probes(tmp_path, accelerator=_gpu_probe(cuda_available=True))
    manifest = build_environment_manifest("local_gpu", repo_root=repo, probes=probes)
    result = evaluate_environment_manifest(manifest, expected_platform_class="local_gpu")
    assert result["compatible"] is True
    assert manifest["accelerator"]["class"] == observed("cuda")


def test_missing_driver_fails_gpu_check_with_stable_reasons(tmp_path: Path) -> None:
    repo, probes = _probes(tmp_path, accelerator=_gpu_probe(cuda_available=False))
    manifest = build_environment_manifest("local_gpu", repo_root=repo, probes=probes)
    result = evaluate_environment_manifest(manifest, expected_platform_class="local_gpu")
    assert result["compatible"] is False
    assert REASON_ACCELERATOR_UNAVAILABLE in result["reasons"]
    assert REASON_DRIVER_UNAVAILABLE in result["reasons"]


def test_platform_class_mismatch_is_reported(tmp_path: Path) -> None:
    repo, probes = _probes(tmp_path)
    manifest = build_environment_manifest("local_cpu", repo_root=repo, probes=probes)
    result = evaluate_environment_manifest(manifest, expected_platform_class="slurm_cpu")
    assert result["compatible"] is False
    assert result["reasons"] == [REASON_PLATFORM_CLASS_MISMATCH]


def test_carla_check_requires_companion_package(tmp_path: Path) -> None:
    repo, probes = _probes(tmp_path, companions=_companions())
    missing = build_environment_manifest("carla", repo_root=repo, probes=probes)
    missing_result = evaluate_environment_manifest(missing, expected_platform_class="carla")
    assert missing_result["compatible"] is False
    assert REASON_CARLA_UNAVAILABLE in missing_result["reasons"]
    assert (
        missing["packages"]["companions"]["carla"]["reason"]
        == REASON_COMPANION_PACKAGE_NOT_INSTALLED
    )

    repo_present, probes_present = _probes(tmp_path, companions=_companions(carla=True))
    present = build_environment_manifest("carla", repo_root=repo_present, probes=probes_present)
    assert evaluate_environment_manifest(present, expected_platform_class="carla")["compatible"]


def test_unknown_platform_class_is_rejected_with_stable_reason(tmp_path: Path) -> None:
    repo, probes = _probes(tmp_path)
    with pytest.raises(ValueError, match=REASON_UNKNOWN_PLATFORM_CLASS):
        build_environment_manifest("quantum_tpu", repo_root=repo, probes=probes)
    manifest = build_environment_manifest("local_cpu", repo_root=repo, probes=probes)
    result = evaluate_environment_manifest(manifest, expected_platform_class="quantum_tpu")
    assert REASON_UNKNOWN_PLATFORM_CLASS in result["reasons"]
    assert REASON_PLATFORM_CLASS_MISMATCH in result["reasons"]


def test_private_paths_hostname_and_credentials_are_redacted(tmp_path: Path) -> None:
    context = RedactionContext(home="/home/alice", hostname="secret-host")
    sanitized, changed = sanitize_text(
        "user /home/alice/work on secret-host with /scratch/private/job and "
        "https://bob:hunter2@example.com/data?X-Amz-Signature=deadbeef&token=abc123",
        context,
    )
    assert changed is True
    for secret in ("alice", "secret-host", "scratch", "hunter2", "deadbeef", "abc123"):
        assert secret not in sanitized
    assert "<home>" in sanitized
    assert "<host>" in sanitized
    assert "<private-root>" in sanitized


def test_declared_credential_like_value_is_redacted(tmp_path: Path) -> None:
    repo, probes = _probes(tmp_path)
    manifest = build_environment_manifest(
        "local_cpu",
        repo_root=repo,
        declared_values={"cluster_password": "hunter2"},
        probes=probes,
    )
    declared_field = manifest["declared"]["overrides"]["cluster_password"]
    assert declared_field["status"] == STATUS_REDACTED
    assert declared_field["reason"] == REASON_PRIVATE_OR_CREDENTIAL_CONTENT
    assert "hunter2" not in json.dumps(manifest)
    hostname_field = manifest["runtime"]["hostname"]
    assert hostname_field["status"] == STATUS_REDACTED
    assert "testhost" not in json.dumps(manifest)


def test_scheduler_identity_is_detected_but_redacted(tmp_path: Path) -> None:
    repo, probes = _probes(
        tmp_path,
        scheduler_env={
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_PARTITION": "gpu-a100",
            "SLURM_CPUS_PER_TASK": "8",
        },
    )
    manifest = build_environment_manifest("slurm_cpu", repo_root=repo, probes=probes)
    scheduler = manifest["scheduler"]
    assert scheduler["detected"] == observed(True)
    assert scheduler["job_id"]["status"] == STATUS_REDACTED
    assert scheduler["partition"]["reason"] == REASON_SCHEDULER_IDENTITY_REDACTED
    assert scheduler["cpus_per_task"] == observed("8")
    assert scheduler["nodes"] == unavailable(REASON_NOT_SET)
    assert "gpu-a100" not in json.dumps(manifest)


def test_thread_controls_only_capture_allowlisted_variables(tmp_path: Path) -> None:
    repo, probes = _probes(
        tmp_path,
        env={"OMP_NUM_THREADS": "2", "SECRET_TOKEN": "do-not-capture"},
    )
    manifest = build_environment_manifest("local_cpu", repo_root=repo, probes=probes)
    assert manifest["thread_controls"]["OMP_NUM_THREADS"] == observed("2")
    assert manifest["thread_controls"]["MKL_NUM_THREADS"] == unavailable(REASON_NOT_SET)
    assert "do-not-capture" not in json.dumps(manifest)


def test_check_cli_reports_compatible_and_incompatible(tmp_path: Path) -> None:
    repo, probes = _probes(tmp_path)
    manifest = build_environment_manifest("local_cpu", repo_root=repo, probes=probes)
    manifest_path = tmp_path / "environment.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        manifest_cli_main(
            ["check", "--manifest", str(manifest_path), "--platform-class", "local_cpu"]
        )
        == 0
    )
    assert (
        manifest_cli_main(
            ["check", "--manifest", str(manifest_path), "--platform-class", "local_gpu"]
        )
        == 1
    )
    assert (
        manifest_cli_main(["check", "--manifest", str(manifest_path), "--platform-class", "nope"])
        == 2
    )


class _FakeTorchCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(_index: int) -> str:
        return "Fake A100"

    @staticmethod
    def get_device_capability(_index: int) -> tuple[int, int]:
        return (8, 0)

    @staticmethod
    def get_device_properties(_index: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=8 * 1024 * 1024)


def _stub_cuda_subprocess(
    monkeypatch: pytest.MonkeyPatch, *, returncode: int = 0, stdout: str = "550.54\n"
) -> None:
    monkeypatch.setattr(
        env_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=returncode, stdout=stdout),
    )
    monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/nvidia-smi")


def test_probe_accelerator_real_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_mod, "try_import", lambda name: None)
    assert env_mod.probe_accelerator()["reason"] == REASON_PROBE_DEPENDENCY_MISSING

    fake_torch = SimpleNamespace(version=SimpleNamespace(cuda="12.1"), cuda=_FakeTorchCuda())
    monkeypatch.setattr(env_mod, "try_import", lambda name: fake_torch)
    _stub_cuda_subprocess(monkeypatch)
    cuda_probe = env_mod.probe_accelerator()
    assert cuda_probe["class"] == observed("cuda")
    assert cuda_probe["devices"]["value"][0]["name"] == "Fake A100"
    assert cuda_probe["driver_version"] == observed("550.54")

    _stub_cuda_subprocess(monkeypatch, returncode=1, stdout="")
    assert (
        env_mod.probe_accelerator()["driver_version"]["reason"] == REASON_DRIVER_PROBE_UNAVAILABLE
    )

    class _UnavailableCuda(_FakeTorchCuda):
        @staticmethod
        def is_available() -> bool:
            return False

    unavailable_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.1"), cuda=_UnavailableCuda()
    )
    monkeypatch.setattr(env_mod, "try_import", lambda name: unavailable_torch)
    assert env_mod.probe_accelerator()["reason"] == REASON_DRIVER_UNAVAILABLE

    cpu_built_torch = SimpleNamespace(version=SimpleNamespace(cuda=None), cuda=_UnavailableCuda())
    monkeypatch.setattr(env_mod, "try_import", lambda name: cpu_built_torch)
    assert env_mod.probe_accelerator()["reason"] == REASON_CUDA_UNAVAILABLE


def test_probe_cuda_driver_version_without_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_mod.shutil, "which", lambda name: None)
    assert env_mod._probe_cuda_driver_version() == unavailable(REASON_DRIVER_PROBE_UNAVAILABLE)


def test_probe_git_commit_real_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        env_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="b" * 40 + "\n"),
    )
    assert env_mod.probe_git_commit(tmp_path) == observed("b" * 40)

    monkeypatch.setattr(
        env_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=128, stdout=""),
    )
    assert env_mod.probe_git_commit(tmp_path)["reason"] == REASON_NOT_A_REPOSITORY

    def _raise(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(env_mod.subprocess, "run", _raise)
    assert env_mod.probe_git_commit(tmp_path)["reason"] == REASON_GIT_UNAVAILABLE


def test_probe_companions_real_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_mod.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(env_mod.importlib.metadata, "version", lambda distribution: "9.9")
    companions = env_mod.probe_companions()
    assert companions["carla"] == observed(
        {"module": "carla", "distribution": "carla", "version": "9.9"}
    )

    monkeypatch.setattr(env_mod.importlib.util, "find_spec", lambda name: None)
    assert env_mod.probe_companions()["carla"]["reason"] == REASON_COMPANION_PACKAGE_NOT_INSTALLED


def test_collect_package_identity_missing_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(distribution: str) -> None:
        raise env_mod.importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(env_mod.importlib.metadata, "version", _raise)
    section = env_mod.collect_package_identity({}, RedactionContext())
    assert section["distribution_version"] == unavailable("not_installed")


def test_sanitize_value_handles_sequences() -> None:
    context = RedactionContext(home="/home/alice", hostname="secret-host")
    sanitized, changed = sanitize_value(
        ["/home/alice/data", {"root": "/scratch/cluster/x"}], context
    )
    assert changed is True
    serialized = json.dumps(sanitized)
    assert "alice" not in serialized
    assert "scratch" not in serialized
