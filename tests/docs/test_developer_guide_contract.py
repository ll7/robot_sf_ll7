"""Contract tests for the concise developer-guide landing page."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LANDING = ROOT / "docs" / "dev_guide.md"
REFERENCE = ROOT / "docs" / "dev_guide_reference.md"
MIGRATION = ROOT / "docs" / "dev_guide_anchor_migration.yaml"


def _anchor_ids(text: str) -> set[str]:
    """Return explicit compatibility anchors from a Markdown document."""

    return set(re.findall(r'<a id="([^"]+)"></a>', text))


def test_landing_page_is_concise_and_reaches_task_guides() -> None:
    """The first-use page stays small and exposes the canonical workflow owners."""

    text = LANDING.read_text(encoding="utf-8")
    nonblank_lines = [line for line in text.splitlines() if line.strip()]
    assert len(nonblank_lines) <= 250
    assert "./dev/worktree_lifecycle.md" in text
    assert "./dev/local_ci.md" in text
    assert "./developer-guide.md" in text
    assert "./dev_guide_reference.md" in text


def test_anchor_migration_preserves_known_inbound_fragments() -> None:
    """Known old fragments resolve through explicit stubs and mapped targets."""

    landing_text = LANDING.read_text(encoding="utf-8")
    reference_text = REFERENCE.read_text(encoding="utf-8")
    payload = yaml.safe_load(MIGRATION.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "dev_guide_anchor_migration.v1"
    anchors = payload["anchors"]
    assert set(anchors) <= _anchor_ids(landing_text)
    assert "docs/dev_guide.md" in reference_text
    for target in anchors.values():
        target_path = target.split("#", maxsplit=1)[0]
        assert (ROOT / target_path).exists(), target


def test_landing_retains_boot_and_safety_discoverability() -> None:
    """The compatibility page keeps the rules that old agent/docs tests discover there."""

    text = LANDING.read_text(encoding="utf-8")
    for fragment in (
        "docs/context/issue_713_batch_first_issue_workflow.md",
        "issue_728_coding_agents_compatibility.md",
        "Canonical skills live in `.agents/skills/`",
        "degraded execution is diagnostic only, never success evidence.",
    ):
        assert fragment in text
