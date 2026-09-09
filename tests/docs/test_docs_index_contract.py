"""Contract tests for docs/README.md index landing page."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_README = ROOT / "docs" / "README.md"


def test_docs_readme_avoids_stale_completion_and_counts() -> None:
    """docs/README.md should not retain stale completion tags, test counts, or conflicting subcommand counts."""
    assert DOCS_README.exists()
    content = DOCS_README.read_text(encoding="utf-8")

    # Heading and TOC should not contain stale '(Complete)' marker
    assert "## 🚀 Social Navigation Benchmark Platform\n" in content
    assert "## 🚀 Social Navigation Benchmark Platform (Complete)" not in content
    assert (
        "- [🚀 Social Navigation Benchmark Platform](#-social-navigation-benchmark-platform)"
        in content
    )
    assert (
        "- [🚀 Social Navigation Benchmark Platform (Complete)](#-social-navigation-benchmark-platform-complete)"
        not in content
    )

    # Should not contain stale hard-coded test count
    assert "108 tests passing" not in content

    # Should not contain contradictory / transient subcommand counts
    assert "15 CLI subcommands" not in content
    assert "30 subcommands" not in content

    # Should contain durable phrasing
    assert "CLI subcommands with examples" in content
    assert "CLI subcommands covering full experiment workflows" in content
