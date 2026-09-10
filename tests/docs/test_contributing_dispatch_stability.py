"""Guard CONTRIBUTING.md against volatile dispatch claims (issue #8713)."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRIBUTING = REPO_ROOT / "CONTRIBUTING.md"
EXAMPLES_README = REPO_ROOT / "examples" / "README.md"


def _dispatch_section() -> str:
    text = CONTRIBUTING.read_text(encoding="utf-8")
    start = text.index("## Issue state labels and dispatch")
    end = text.index("### Auditing dispatch state")
    return text[start:end]


def test_dispatch_examples_name_no_live_issue() -> None:
    """Worked dispatch examples must stay timeless (no live issue numbers)."""
    section = _dispatch_section()
    assert re.search(r"#\d{3,}", section) is None


def test_examples_instructions_forbid_hand_editing_generated_readme() -> None:
    """Contribution steps must route through the manifest and regeneration."""
    text = CONTRIBUTING.read_text(encoding="utf-8")
    assert "examples/examples_manifest.yaml" in text
    assert "render_examples_readme.py" in text
    assert "Never hand-edit the generated file" in text


def test_examples_readme_declares_generated() -> None:
    """The generated README must keep declaring its generated status."""
    header = EXAMPLES_README.read_text(encoding="utf-8")[:600]
    assert "auto-generated" in header
    assert "examples_manifest.yaml" in header or "manifest" in header
