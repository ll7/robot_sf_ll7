"""Tests issue #4018 density-curriculum comparison readiness."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.training.density_curriculum_readiness import (
    evaluate_density_curriculum_readiness,
)
from scripts.training.check_density_curriculum_readiness import main

if TYPE_CHECKING:
    from pathlib import Path


def _manifest(*, dry_run: bool = False) -> dict[str, object]:
    return {
        "schema_version": "density_curriculum_comparison.v1",
        "issue": "ll7/robot_sf_ll7#4018",
        "claim_boundary": "diagnostic harness only; no benchmark training-result claim",
        "dry_run": dry_run,
        "curriculum": {
            "path": "configs/training/ppo/ablations/issue_4018_density_curriculum_smoke.yaml",
            "policy_id": "issue_4018_density_curriculum_smoke",
            "total_timesteps": 4096,
            "density_curriculum_enabled": True,
        },
        "baseline": {
            "path": "configs/training/ppo/ablations/issue_4018_fixed_density_smoke.yaml",
            "policy_id": "issue_4018_fixed_density_smoke",
            "total_timesteps": 4096,
            "density_curriculum_enabled": False,
        },
        "artifacts": {
            "curriculum_checkpoint": "output/issue_4018/curriculum/best_model.zip",
            "baseline_checkpoint": "output/issue_4018/baseline/best_model.zip",
        },
    }


def _write_manifest(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "comparison_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_readiness_accepts_completed_paired_manifest(tmp_path: Path) -> None:
    """Completed paired manifests become diagnostic-smoke ready."""
    path = _write_manifest(tmp_path, _manifest())

    readiness = evaluate_density_curriculum_readiness(path).to_dict()

    assert readiness["schema_version"] == "issue_4018.density_curriculum_readiness.v1"
    assert readiness["status"] == "ready_diagnostic_smoke"
    assert readiness["blockers"] == []
    assert "not benchmark evidence" in readiness["claim_boundary"]
    assert readiness["comparison"]["curriculum"]["density_curriculum_enabled"] is True
    assert readiness["comparison"]["baseline"]["density_curriculum_enabled"] is False


def test_readiness_blocks_dry_run_manifest_without_training_artifacts(tmp_path: Path) -> None:
    """Dry-run comparison manifests are useful setup evidence but not run-ready evidence."""
    payload = _manifest(dry_run=True)
    payload.pop("artifacts")
    path = _write_manifest(tmp_path, payload)

    readiness = evaluate_density_curriculum_readiness(path).to_dict()

    assert readiness["status"] == "blocked"
    assert "manifest is dry_run" in " ".join(readiness["blockers"])
    assert "artifacts mapping" in " ".join(readiness["blockers"])


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload.__setitem__("schema_version", "wrong.v1"),
            "schema_version",
        ),
        (
            lambda payload: payload["baseline"].__setitem__("density_curriculum_enabled", True),
            "baseline arm must disable",
        ),
        (
            lambda payload: payload["baseline"].__setitem__("total_timesteps", 1024),
            "total_timesteps",
        ),
    ],
)
def test_readiness_fails_closed_on_contract_drift(
    tmp_path: Path,
    mutate,
    match: str,
) -> None:
    """Comparison contract drift is reported as a blocker."""
    payload = _manifest()
    mutate(payload)
    path = _write_manifest(tmp_path, payload)

    readiness = evaluate_density_curriculum_readiness(path)

    assert readiness.status == "blocked"
    assert match in " ".join(readiness.blockers)


def test_cli_writes_packet_and_uses_fail_closed_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI emits a JSON packet and returns non-zero while dry-run blockers remain."""
    path = _write_manifest(tmp_path, _manifest(dry_run=True))
    output = tmp_path / "readiness.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_density_curriculum_readiness.py",
            str(path),
            "--output",
            str(output),
        ],
    )

    assert main() == 1
    packet = json.loads(output.read_text(encoding="utf-8"))
    assert packet["status"] == "blocked"
