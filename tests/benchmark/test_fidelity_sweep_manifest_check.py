"""Tests for issue #3207 dry-run fidelity manifest checker summaries."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.fidelity_sensitivity import load_fidelity_sensitivity_config
from robot_sf.benchmark.fidelity_sweep_manifest import (
    FIDELITY_SWEEP_MANIFEST_CHECK_SCHEMA,
    ManifestOptions,
    build_fidelity_sweep_manifest,
    check_fidelity_sweep_manifest,
    write_fidelity_sweep_manifest,
    write_fidelity_sweep_manifest_check,
)

if TYPE_CHECKING:
    from types import ModuleType


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = "configs/research/fidelity_sensitivity_v1.yaml"


def _repo_manifest() -> dict[str, object]:
    config = load_fidelity_sensitivity_config(_CONFIG_PATH)
    return build_fidelity_sweep_manifest(
        config,
        options=ManifestOptions(config_path=_CONFIG_PATH, git_head="abc1234"),
    )


def _load_manifest_builder() -> ModuleType:
    module_path = _REPO_ROOT / "scripts" / "benchmark" / "build_fidelity_sweep_manifest.py"
    spec = importlib.util.spec_from_file_location("build_fidelity_sweep_manifest", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load builder module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_fidelity_sweep_manifest"] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_checker_summarizes_factor_coverage_without_evidence_upgrade() -> None:
    """Checker reports factor coverage while preserving no-claim boundaries."""
    manifest = _repo_manifest()

    summary = check_fidelity_sweep_manifest(manifest)

    assert summary["schema_version"] == FIDELITY_SWEEP_MANIFEST_CHECK_SCHEMA
    assert summary["status"] == "manifest_check_only"
    assert summary["evidence_status"] == "not_benchmark_evidence"
    assert "does not run sensitivity studies" in summary["claim_boundary"]
    assert summary["passes"] is True
    assert summary["violations"] == []
    assert summary["axis_count"] == len(manifest["axes"])
    assert summary["variant_count"] == sum(len(axis["variants"]) for axis in manifest["axes"])
    assert summary["unresolved_runtime_binding_count"] == summary["variant_count"]
    assert summary["payload_kind_counts"]["patch"] >= 1
    assert summary["payload_kind_counts"]["observation_noise"] >= 1
    assert [axis["key"] for axis in summary["axis_summaries"]] == [
        axis["key"] for axis in manifest["axes"]
    ]


def test_manifest_checker_fails_closed_on_evidence_boundary_violation() -> None:
    """Checker surfaces evidence-boundary regressions instead of silently passing."""
    manifest = _repo_manifest()
    manifest["evidence_status"] = "benchmark_evidence"
    manifest["claim_boundary"] = "benchmark evidence"
    manifest["axes"][0]["variants"][0]["runtime_binding_status"] = "bound_runtime_patch"

    summary = check_fidelity_sweep_manifest(manifest)

    assert summary["passes"] is False
    assert "manifest evidence_status must remain not_benchmark_evidence" in summary["violations"]
    assert any("not benchmark evidence" in violation for violation in summary["violations"])
    assert any("unresolved runtime binding" in violation for violation in summary["violations"])


def test_manifest_checker_json_output_is_deterministic(tmp_path: Path) -> None:
    """Checker JSON writes deterministically alongside dry-run manifest output."""
    manifest = _repo_manifest()
    summary = check_fidelity_sweep_manifest(manifest)

    manifest_path = write_fidelity_sweep_manifest(manifest, tmp_path)
    check_path = write_fidelity_sweep_manifest_check(summary, tmp_path)
    reloaded_summary = json.loads(check_path.read_text(encoding="utf-8"))

    assert manifest_path.name == "fidelity_sweep_manifest.json"
    assert check_path.name == "fidelity_sweep_manifest_check.json"
    assert reloaded_summary == summary
    assert check_path.read_text(encoding="utf-8") == (
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def test_manifest_builder_cli_writes_check_summary_when_requested(
    tmp_path: Path, monkeypatch
) -> None:
    """CLI check mode writes a checker summary without changing dry-run behavior."""
    builder = _load_manifest_builder()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_fidelity_sweep_manifest.py",
            "--dry-run",
            "--check",
            "--out",
            str(tmp_path),
        ],
    )

    assert builder.main() == 0
    manifest_path = tmp_path / "fidelity_sweep_manifest.json"
    check_path = tmp_path / "fidelity_sweep_manifest_check.json"
    check_summary = json.loads(check_path.read_text(encoding="utf-8"))

    assert manifest_path.exists()
    assert check_summary["schema_version"] == FIDELITY_SWEEP_MANIFEST_CHECK_SCHEMA
    assert check_summary["status"] == "manifest_check_only"
    assert check_summary["passes"] is True
