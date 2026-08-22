"""Tests for the executable research campaign manifest packet runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.validation.run_research_campaign_manifest import (
    ManifestError,
    _repo_declared_path,
    evaluate_research_manifest_answerability,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "validation" / "run_research_campaign_manifest.py"
EXAMPLE_MANIFEST = REPO_ROOT / "configs" / "benchmarks" / "research_campaign_manifest.example.yaml"


def _copy_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(EXAMPLE_MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    return manifest_path


def _run_manifest(manifest_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_declared_path_rejects_symlink_escape(tmp_path: Path) -> None:
    """Manifest-bound paths cannot resolve outside the repository root."""
    outside = tmp_path.parent / "answerability-manifest-outside.yaml"
    outside.write_text("outside: true\n", encoding="utf-8")
    link = tmp_path / "config.yaml"
    link.symlink_to(outside)

    with pytest.raises(ManifestError, match="resolve within the repository root"):
        _repo_declared_path("config.yaml", "scenario_suite.campaign_config", repo_root=tmp_path)


def test_run_research_campaign_manifest_writes_packet(tmp_path: Path) -> None:
    """The example contract should produce the canonical compact packet files."""
    manifest_path = _copy_manifest(tmp_path)
    output_dir = tmp_path / "packet"

    completed = _run_manifest(manifest_path, output_dir)

    assert completed.returncode == 0, completed.stderr
    for file_name in (
        "manifest_resolved.json",
        "rows.jsonl",
        "campaign_table.csv",
        "summary.json",
        "report.md",
        "context_note.md",
    ):
        assert (output_dir / file_name).exists()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["campaign_id"] == "issue_3062_example_research_campaign"
    assert summary["final_decision"] == "diagnostic"
    assert summary["answerability"]["state"] == "diagnostic_only"
    assert summary["row_status_summary"] == {"diagnostic_only": 4}
    assert summary["validation_results"][0]["executed"] is False

    rows = [
        json.loads(line)
        for line in (output_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {(row["planner_id"], row["seed"]) for row in rows} == {
        ("social_force", 101),
        ("social_force", 202),
        ("prediction_planner", 101),
        ("prediction_planner", 202),
    }


def test_evaluate_research_manifest_answerability_reuses_proof_without_writing_packet(
    tmp_path: Path,
) -> None:
    """The production admission seam evaluates proof without creating campaign output."""
    manifest_path = _copy_manifest(tmp_path)

    report = evaluate_research_manifest_answerability(manifest_path)

    assert report["answerability"]["state"] == "diagnostic_only"
    assert report["answerability_proof"]["status"] == "completed"
    assert not (tmp_path / "packet").exists()


def test_evaluate_research_manifest_answerability_requires_proof_surface_declarations(
    tmp_path: Path,
) -> None:
    """Production admission cannot bypass the executable proof-surface contract."""
    manifest_path = _copy_manifest(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["answerability"].pop("proof_surfaces")
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    with pytest.raises(ManifestError, match=r"answerability\.proof_surfaces is required"):
        evaluate_research_manifest_answerability(manifest_path)


def test_evaluate_research_manifest_answerability_rejects_config_mismatch(
    tmp_path: Path,
) -> None:
    """Strict admission must bind the proof manifest to the requested camera config."""
    manifest_path = _copy_manifest(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["answerability"]["design"]["mode"] = "decision_capable"
    manifest["answerability"]["artifacts"]["durability_status"] = "ready"
    manifest["scenario_suite"]["campaign_config"] = (
        "configs/benchmarks/issue_3425_empirical_vertical_slice_smoke.yaml"
    )
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    with pytest.raises(ManifestError, match="does not match the camera-ready config"):
        evaluate_research_manifest_answerability(
            manifest_path,
            expected_campaign_config=(
                REPO_ROOT / "configs/benchmarks/issue_3425_empirical_vertical_slice_manifest.yaml"
            ),
        )


def test_run_research_campaign_manifest_requires_claim_boundary(tmp_path: Path) -> None:
    """A manifest without a claim boundary must fail before writing evidence."""
    manifest_path = _copy_manifest(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["campaign"]["claim_boundary"] = ""
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    completed = _run_manifest(manifest_path, tmp_path / "packet")

    assert completed.returncode == 2
    assert "campaign.claim_boundary is required" in completed.stderr


def test_run_research_campaign_manifest_rejects_output_as_durable_evidence(tmp_path: Path) -> None:
    """Durable evidence must not point back into disposable output paths."""
    manifest_path = _copy_manifest(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["durable_evidence"]["plan"]["path"] = "output/benchmarks/not_durable/README.md"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    completed = _run_manifest(manifest_path, tmp_path / "packet")

    assert completed.returncode == 2
    assert "durable_evidence.plan.path must not point into output/" in completed.stderr


def test_run_research_campaign_manifest_rejects_required_path_traversal(tmp_path: Path) -> None:
    """Configured output checks must reject paths that escape the local root."""
    manifest_path = _copy_manifest(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["required_paths"] = ["../escape.json"]
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "packet"),
            "--require-configured-outputs",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "outputs.required_paths[] must not traverse parent directories" in completed.stderr


def test_run_research_campaign_manifest_requires_mapping_sections(tmp_path: Path) -> None:
    """Required top-level sections must have mapping values."""
    manifest_path = _copy_manifest(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["metrics"] = None
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    completed = _run_manifest(manifest_path, tmp_path / "packet")

    assert completed.returncode == 2
    assert "metrics must be a mapping" in completed.stderr


@pytest.mark.parametrize(
    "status",
    [
        "diagnostic_only",
        "fallback",
        "degraded",
        "not_available",
        "unavailable",
        "not_run",
        "failed",
        "blocked",
    ],
)
def test_run_research_campaign_manifest_rejects_caveated_success_status(
    tmp_path: Path, status: str
) -> None:
    """Caveated or non-executed rows cannot be declared success by manifest authors."""
    manifest_path = _copy_manifest(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if status not in manifest["row_status_policy"]["allowed_values"]:
        manifest["row_status_policy"]["allowed_values"].append(status)
    manifest["row_status_policy"]["success_values"] = [status]
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    completed = _run_manifest(manifest_path, tmp_path / "packet")

    assert completed.returncode == 2
    assert "Fail-closed row statuses cannot be success values" in completed.stderr


def test_run_research_campaign_manifest_requires_planner_row_mappings(tmp_path: Path) -> None:
    """Planner rows must be mappings before the runner builds packet rows."""
    manifest_path = _copy_manifest(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["planners"]["rows"] = ["social_force"]
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    completed = _run_manifest(manifest_path, tmp_path / "packet")

    assert completed.returncode == 2
    assert "planners.rows[0] must be a mapping" in completed.stderr


def test_run_research_campaign_manifest_treats_null_optional_commands_as_empty(
    tmp_path: Path,
) -> None:
    """Explicit null validation commands should behave like no commands configured."""
    manifest_path = _copy_manifest(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["validation"]["commands"] = None
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    completed = _run_manifest(manifest_path, tmp_path / "packet")

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((tmp_path / "packet" / "summary.json").read_text(encoding="utf-8"))
    assert summary["validation_results"] == []


def test_run_research_campaign_manifest_can_require_decision_capable_answerability(
    tmp_path: Path,
) -> None:
    """The canonical runner can fail closed before a decision-capable launch."""
    manifest_path = _copy_manifest(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "packet"),
            "--require-answerable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "answerability gate requires state=answerable" in completed.stderr
