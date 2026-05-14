"""Guard generated examples README catalog quality."""

from __future__ import annotations

import re
from collections import Counter

from robot_sf.examples import load_manifest
from scripts.validation.render_examples_readme import build_markdown


def test_generated_examples_readme_has_unique_numbered_labels() -> None:
    """Numbered example labels should not reuse the same numeric prefix."""
    markdown = build_markdown(
        load_manifest(),
        include_archived=False,
        include_ci_column=True,
    )

    numbered_labels = re.findall(r"\| \[(\d+) [^\]]+\]\(\./([^/]+)/", markdown)
    counts = Counter((category, number) for number, category in numbered_labels)

    assert {
        f"{category}:{number}": count
        for (category, number), count in counts.items()
        if count > 1
    } == {}


def test_generated_examples_readme_does_not_include_known_catalog_typos() -> None:
    """Generated CI notes should not carry known wording typos."""
    markdown = build_markdown(
        load_manifest(),
        include_archived=False,
        include_ci_column=True,
    )

    assert "relevatn" not in markdown
