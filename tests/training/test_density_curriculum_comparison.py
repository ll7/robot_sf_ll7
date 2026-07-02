"""Tests for the issue #4018 density curriculum comparison launcher."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.training.run_density_curriculum_comparison import main

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_comparison_dry_run_writes_claim_bounded_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Dry-run validates both configs and records no-claim comparison metadata."""
    output_dir = tmp_path / "comparison"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_density_curriculum_comparison.py",
            "--curriculum-config",
            str(
                REPO_ROOT
                / "configs/training/ppo/ablations/issue_4018_density_curriculum_smoke.yaml"
            ),
            "--baseline-config",
            str(REPO_ROOT / "configs/training/ppo/ablations/issue_4018_fixed_density_smoke.yaml"),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
    )

    assert main() == 0

    manifest = json.loads((output_dir / "comparison_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert "no benchmark" in manifest["claim_boundary"]
    assert manifest["curriculum"]["density_curriculum_enabled"] is True
    assert manifest["baseline"]["density_curriculum_enabled"] is False
    assert manifest["curriculum"]["total_timesteps"] == manifest["baseline"]["total_timesteps"]
