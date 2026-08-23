"""Tests for full benchmark-release doctor admission checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

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
        f"RELEASE_CAMPAIGN_ID={campaign}",
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
    )
    assert check.status == "pass", check.summary


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


@pytest.mark.parametrize(
    "field",
    [
        "RELEASE_LAUNCH_PACKET_SHA256",
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
