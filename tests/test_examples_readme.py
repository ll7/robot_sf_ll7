"""Guard generated examples README catalog quality."""

from __future__ import annotations

import re
from collections import Counter

from robot_sf.examples import load_manifest
from scripts.validation.render_examples_readme import build_markdown


def test_generated_examples_readme_has_unique_advanced_numeric_labels() -> None:
    """Advanced example labels should not reuse the same numeric prefix."""
    markdown = build_markdown(
        load_manifest(),
        include_archived=False,
        include_ci_column=True,
    )

    numeric_labels = re.findall(r"\| \[(\d+) [^\]]+\]\(\./advanced/", markdown)
    counts = Counter(numeric_labels)

    assert {number: count for number, count in counts.items() if count > 1} == {}


def test_generated_examples_readme_does_not_include_known_catalog_typos() -> None:
    """Generated CI notes should not carry known wording typos."""
    markdown = build_markdown(
        load_manifest(),
        include_archived=False,
        include_ci_column=True,
    )

    assert "relevatn" not in markdown
