"""Tests for the durable SLURM agent contract (issue #8941).

The scoped SLURM playbook keeps cluster safety, custody, failure classification, and evidence
boundaries durable, while campaign-specific environment variables, horizons, and progression rules
are mapped to their campaign owner instead of being frozen into repository law.
"""

from __future__ import annotations

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
    assert "configs/training/" in text
    assert "Review condition" in text
    assert "issue-791 campaign context note" in text


def test_referenced_public_wrappers_exist() -> None:
    """Every wrapper named as a durable default exists in the repository."""
    text = SLURM_PLAYBOOK.read_text(encoding="utf-8")
    for relative in ("scripts/dev/sbatch_use_max_time.sh",):
        assert relative in text
        assert (REPO_ROOT / relative).is_file(), relative


def test_insight_capture_is_conditional() -> None:
    """Insight capture is required for material findings, not after every informative run."""
    text = SLURM_PLAYBOOK.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "no material, reusable finding" in normalized
    assert "documentation edit when a run produced no material" in normalized
