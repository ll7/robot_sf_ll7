"""Tests for the deterministic cross-host execution-environment comparator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/validation/compare_execution_environments.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures/cross_host_environments"

_SPEC = importlib.util.spec_from_file_location("_compare_execution_environments", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _host(label: str = "host-a") -> dict[str, Any]:
    return {
        "schema_version": _MODULE.HOST_SCHEMA_VERSION,
        "host_label": label,
        "os": {"system": "Linux", "architecture_class": "linux-x86_64", "libc": "glibc-2.39"},
        "python": {"implementation": "CPython", "version": "3.12.7", "abi_tag": "cp312"},
        "distributions": {
            "numpy": {"version": "2.2.1", "build": "cp312-manylinux_x86_64"},
            "torch": {"version": "2.6.0", "build": "cp312-manylinux_x86_64"},
        },
        "editable": {"robot-sf": {"commit": "a" * 40, "dirty": False}},
        "native_extensions": {"robot_sf._native": {"sha256": "b" * 64, "compiler": "GCC 13.2.0"}},
        "threading": {"blas_backend": "openblas-pthreads", "omp_num_threads": "1"},
        "accelerators": {"torch": {"cuda_runtime": "12.6", "driver": "560.35.05"}},
        "locale": {"lc_all": "C.UTF-8", "timezone": "UTC"},
        "filesystem": {"path_separator": "/", "case_sensitive": True},
    }


def _requirements(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": _MODULE.REQUIREMENTS_SCHEMA_VERSION,
        "workload_id": "cpu-smoke",
        "not_applicable_fields": ["accelerators"],
        "material_fields": [],
        "compatible_variations": [],
    }
    payload.update(overrides)
    return payload


def _compare(host_a: dict[str, Any], host_b: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    return _MODULE.compare_environments(
        host_a, host_b, _MODULE.parse_requirements(_requirements(**overrides))
    )


def _row(report: dict[str, Any], field: str) -> dict[str, Any]:
    return next(row for row in report["fields"] if row["field"] == field)


def _kinds(report: dict[str, Any]) -> set[str]:
    return {item["kind"] for item in report["detections"]}


def test_identical_cpu_hosts_are_equivalent_for_workload() -> None:
    report = _compare(_host("host-a"), _host("host-b"))
    assert report["comparison_status"] == "equivalent_for_workload"
    assert report["can_claim_cross_host_reproducibility"] is True
    assert report["output_equivalence_inferred"] is False
    assert report["classification_counts"]["exact_match"] > 0
    evidence = {"episodes": 3, "paired_delta_mean": 0.01}
    with_evidence = _MODULE.compare_environments(
        _host("host-a"), _host("host-b"), _MODULE.parse_requirements(_requirements()), evidence
    )
    assert with_evidence["comparison_status"] == report["comparison_status"]
    assert with_evidence["empirical_repeat_evidence"] == evidence


def test_workload_materiality_gates_non_equivalent_differences() -> None:
    arch = _host("host-b")
    arch["os"]["architecture_class"] = "linux-aarch64"
    arch_report = _compare(_host(), arch, material_fields=["os/architecture_class"])
    assert _row(arch_report, "os/architecture_class")["classification"] == "different_material"
    assert arch_report["comparison_status"] == "not_equivalent"
    assert "platform_marker_differs" in _kinds(arch_report)
    threads = _host("host-b")
    threads["threading"] = {"blas_backend": "mkl", "omp_num_threads": "8"}
    thread_report = _compare(_host(), threads, material_fields=["threading"])
    assert _row(thread_report, "threading/blas_backend")["classification"] == "different_material"
    assert (
        _row(thread_report, "threading/omp_num_threads")["classification"] == "different_material"
    )
    hidden_a = _host("host-a")
    hidden_b = _host("host-b")
    hidden_a["threading"]["omp_num_threads"] = "unset"
    hidden_b["threading"]["omp_num_threads"] = "unset"
    hidden_report = _compare(hidden_a, hidden_b)
    hidden_row = _row(hidden_report, "threading/omp_num_threads")
    assert hidden_row["classification"] == "unavailable"
    assert hidden_row["rationale"] == "hidden_thread_default"
    assert hidden_report["comparison_status"] == "not_comparable"
    gpu = _host("host-b")
    gpu["accelerators"]["torch"]["cuda_runtime"] = "12.4"
    gpu["accelerators"]["torch"]["driver"] = "550.54.15"
    cpu_report = _compare(_host(), gpu)
    assert cpu_report["comparison_status"] == "equivalent_for_workload"
    assert _row(cpu_report, "accelerators/torch/cuda_runtime")["classification"] == "not_applicable"
    gpu_report = _compare(_host(), gpu, not_applicable_fields=[], material_fields=["accelerators"])
    assert gpu_report["comparison_status"] == "not_equivalent"
    gpu_row = _row(gpu_report, "accelerators/torch/cuda_runtime")
    assert gpu_row["classification"] == "different_material"


def test_source_identical_build_differences_are_material() -> None:
    host_b = _host("host-b")
    host_b["distributions"]["numpy"]["build"] = "cp312-manylinux_aarch64"
    host_b["native_extensions"]["robot_sf._native"]["sha256"] = "c" * 64
    report = _compare(_host(), host_b)
    assert _row(report, "distributions/numpy/version")["classification"] == "exact_match"
    assert _row(report, "distributions/numpy/build")["classification"] == "different_material"
    native = _row(report, "native_extensions/robot_sf._native/sha256")
    assert native["classification"] == "different_material"
    assert "source_identical_build_different" in _kinds(report)
    suffix = _host("host-b")
    suffix["distributions"]["numpy"]["version"] = "2.2.1+cpu"
    suffix_report = _compare(_host(), suffix)
    assert _row(suffix_report, "distributions/numpy/version")["classification"] == (
        "different_unclassified"
    )


def test_missing_and_redacted_fields_are_not_comparable() -> None:
    missing = _host("host-b")
    del missing["distributions"]["numpy"]
    missing_report = _compare(_host(), missing)
    assert _row(missing_report, "distributions/numpy/version")["classification"] == "unavailable"
    assert missing_report["comparison_status"] == "not_comparable"
    assert "missing_field" in _kinds(missing_report)
    redacted = _host("host-b")
    redacted["distributions"]["numpy"]["version"] = "<redacted>"
    redacted_report = _compare(_host(), redacted)
    row = _row(redacted_report, "distributions/numpy/version")
    assert row["classification"] == "redacted_not_comparable"
    assert redacted_report["comparison_status"] == "not_comparable"
    assert "redacted_field" in _kinds(redacted_report)


def test_declared_compatible_variation_records_rationale() -> None:
    host_b = _host("host-b")
    host_b["threading"]["omp_num_threads"] = "8"
    declaration = {
        "field": "threading/omp_num_threads",
        "host_a_values": ["1"],
        "host_b_values": ["8"],
        "rationale": "smoke workload runs one episode at a time on both hosts",
    }
    report = _compare(_host(), host_b, compatible_variations=[declaration])
    assert _row(report, "threading/omp_num_threads")["classification"] == "compatible_declared"
    assert report["comparison_status"] == "equivalent_for_workload"
    assert report["declared_compatible_differences"] == [
        {"field": "threading/omp_num_threads", "rationale": declaration["rationale"]}
    ]
    host_b["threading"]["omp_num_threads"] = "3"
    unsatisfied = _compare(_host(), host_b, compatible_variations=[declaration])
    assert _row(unsatisfied, "threading/omp_num_threads")["classification"] == (
        "different_unclassified"
    )
    assert "declaration_not_satisfied" in _kinds(unsatisfied)


def test_private_identity_and_ambiguous_requirements_fail_closed() -> None:
    host_a = _host("host-a")
    host_a["filesystem"]["root"] = "/home/alice/robot_sf"
    report = _compare(host_a, _host("host-b"))
    assert _row(report, "filesystem/root")["host_a_value"] == "/home/<user>/robot_sf"
    assert "alice" not in json.dumps(report)
    dirty_a = _host("host-a")
    dirty_b = _host("host-b")
    dirty_a["editable"]["robot-sf"]["dirty"] = True
    dirty_b["editable"]["robot-sf"]["dirty"] = True
    dirty_a["container"] = {"image_tag": "runner:latest"}
    dirty_b["container"] = {"image_tag": "runner:latest"}
    dirty_report = _compare(dirty_a, dirty_b)
    assert _row(dirty_report, "editable/robot-sf/dirty")["classification"] == "unavailable"
    assert {"dirty_editable_package", "mutable_container_tag"} <= _kinds(dirty_report)
    bad_key = _host("host-a")
    bad_key["hostname"] = "lab-node-7"
    with pytest.raises(_MODULE.ContractError):
        _MODULE.parse_host_manifest(bad_key)
    bad_nested = _host("host-a")
    bad_nested["environment"] = {"USER": "alice"}
    with pytest.raises(_MODULE.ContractError):
        _MODULE.parse_host_manifest(bad_nested)
    with pytest.raises(_MODULE.ContractError):
        _MODULE.parse_requirements(
            _requirements(
                not_applicable_fields=["threading"], material_fields=["threading/omp_num_threads"]
            )
        )
    with pytest.raises(_MODULE.ContractError):
        _MODULE.parse_requirements(
            _requirements(
                compatible_variations=[
                    {
                        "field": "threading/omp_num_threads",
                        "host_a_values": ["1"],
                        "host_b_values": ["8"],
                        "rationale": "declared",
                        "extra": True,
                    }
                ]
            )
        )


def test_private_identity_is_sanitized_inside_mapping_values() -> None:
    host_a = _host("host-a")
    host_b = _host("host-b")
    host_a["capture_metadata"] = [{"path": "/home/alice/robot_sf", "endpoint": "alice@192.0.2.10"}]
    host_b["capture_metadata"] = [{"path": "/home/bob/robot_sf", "endpoint": "bob@192.0.2.11"}]

    report = _compare(host_a, host_b)

    serialized = json.dumps(report)
    assert "/home/<user>/robot_sf" in serialized
    assert "<user>@<host>" in serialized
    assert "alice" not in serialized
    assert "bob" not in serialized
    assert "192.0.2.10" not in serialized
    assert "192.0.2.11" not in serialized


def test_cli_fixture_scenarios_are_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    args = [
        "--host-a",
        str(FIXTURES / "host_a.json"),
        "--host-b",
        str(FIXTURES / "host_b.json"),
        "--requirements",
    ]
    assert _MODULE.main([*args, str(FIXTURES / "requirements_cpu.json")]) == 0
    first = capsys.readouterr().out
    cpu_report = json.loads(first)
    assert cpu_report["comparison_status"] == "equivalent_for_workload"
    assert _MODULE.main([*args, str(FIXTURES / "requirements_cpu.json")]) == 0
    assert capsys.readouterr().out == first
    assert _MODULE.main([*args, str(FIXTURES / "requirements_gpu.json")]) == 2
    gpu_report = json.loads(capsys.readouterr().out)
    assert gpu_report["comparison_status"] == "not_equivalent"
    gpu_row = _row(gpu_report, "accelerators/torch/cuda_runtime")
    assert gpu_row["classification"] == "different_material"
    assert "output equivalence" in _MODULE.render_markdown(cpu_report)
