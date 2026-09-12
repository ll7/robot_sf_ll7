"""Tests for the durable SLURM agent contract (issue #8941).

The scoped SLURM playbook keeps cluster safety, custody, failure classification, and evidence
boundaries durable, while campaign-specific environment variables, horizons, and progression rules
are mapped to their campaign owner instead of being frozen into repository law.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SLURM_PLAYBOOK = REPO_ROOT / "SLURM" / "AGENTS.md"

DURABLE_SECTIONS = (
    "## Configuration And Submission",
    "## Custody And Identity",
    "## Failure Classification",
    "## Evidence Boundary",
    "## Transient Campaign Guidance",
    "## Insight Capture",
)


def test_durable_safeguards_remain_prominent() -> None:
    """Cluster safety and evidence safeguards stay in the scoped contract."""
    text = SLURM_PLAYBOOK.read_text(encoding="utf-8")
    for section in DURABLE_SECTIONS:
        assert section in text, section
    assert "private operations overlay" in text
    assert "actionable" in text
    assert "never success evidence" in text
    assert "preserve logs" in text.lower()


def test_campaign_specific_policy_is_not_frozen_as_law() -> None:
    """Issue-791 env vars, WandB policy, and horizon bands leave the durable contract."""
    text = SLURM_PLAYBOOK.read_text(encoding="utf-8")
    for fragment in ("ISSUE791_", "ISSUE791_WANDB_POLICY", "8k", "128k", "1m+"):
        assert fragment not in text, fragment


def test_transient_guidance_has_owner_and_review_condition() -> None:
    """Campaign guidance names an owner and a review condition instead of being permanent."""
    text = SLURM_PLAYBOOK.read_text(encoding="utf-8")
    assert "[`configs/training/`](../configs/training/)" in text
    assert "[context record via the context index](../docs/context/README.md)" in text
    assert "Review condition" in text
    assert (REPO_ROOT / "configs" / "training").is_dir()
    assert (REPO_ROOT / "docs" / "context" / "README.md").is_file()


def test_referenced_public_wrappers_exist() -> None:
    """Every wrapper named as a durable default exists in the repository."""
    text = SLURM_PLAYBOOK.read_text(encoding="utf-8")
    for relative in ("scripts/dev/sbatch_use_max_time.sh",):
        assert relative in text
        assert (REPO_ROOT / relative).is_file(), relative


def test_missing_private_overlay_fails_closed_with_actionable_message(tmp_path: Path) -> None:
    """Auxme submission stops before launch when its private overlay is unavailable."""
    environment = os.environ.copy()
    environment["ROBOT_SF_PRIVATE_OPS"] = str(tmp_path / "missing-private-ops")
    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "dev" / "sbatch_auxme_issue791.sh"), "--help"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 127
    assert "Private operations helper not found:" in output
    assert "Configure the private Slurm/Auxme overlay" in output


def test_insight_capture_is_conditional() -> None:
    """Insight capture is required for material findings, not after every informative run."""
    text = SLURM_PLAYBOOK.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "no material, reusable finding" in normalized
    assert "documentation edit when a run produced no material" in normalized
