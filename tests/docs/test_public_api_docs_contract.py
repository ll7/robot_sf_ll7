"""Contract and regression tests for public API and stability policy documentation.

Ensures:
- Exactly one canonical public API policy document exists (docs/public_api.md).
- Stale proposal and pending-sign-off language from #5801/#8245 is eliminated.
- docs/api/stable_public_api.md acts as a clean redirect/forwarding pointer.
- Top-level facade exports match robot_sf.__all__.
- Sphinx documentation navigation links resolve correctly.
"""

from __future__ import annotations

from pathlib import Path

import robot_sf

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DOC = REPO_ROOT / "docs" / "public_api.md"
REDIRECT_DOC = REPO_ROOT / "docs" / "api" / "stable_public_api.md"
SPHINX_API_INDEX = REPO_ROOT / "docs" / "api" / "index.rst"
SPHINX_ROOT_INDEX = REPO_ROOT / "docs" / "index.rst"
ENV_REGISTRY_FILE = REPO_ROOT / "robot_sf" / "gym_env" / "env_registry.py"

STALE_PROPOSAL_PHRASES = (
    "pending maintainer sign-off",
    "proposal for issue #5801",
    "proposal pending closed issue",
    "Public surface (proposed)",
    "Deprecation policy (proposed)",
    "maintainer-review proposal",
)

CANONICAL_POLICY_HEADINGS = (
    "## Stability Levels",
    "## Top-Level Entry Points",
    "## Supported Python Modules Catalog",
    "## Internal Architecture Boundaries",
    "## Lifecycle and Deprecation Policy",
)


def test_canonical_public_api_page_exists_and_documents_all_exports() -> None:
    """docs/public_api.md is canonical and documents all top-level public facade exports."""
    assert CANONICAL_DOC.is_file(), f"{CANONICAL_DOC} must exist as canonical public API doc"
    text = CANONICAL_DOC.read_text(encoding="utf-8")

    assert "# Robot SF Public API" in text
    assert "## Stability Levels" in text
    assert "## Top-Level Entry Points" in text
    assert "## CLI Public Surface" in text
    assert "## Supported Python Modules Catalog" in text
    assert "## Internal Architecture Boundaries" in text
    assert "## Lifecycle and Deprecation Policy" in text
    assert "`release`" in text
    assert "./RELEASE.md" in text
    assert "maintainer-facing" in text

    # Verify all symbols exported in robot_sf.__all__ are documented
    for export_name in robot_sf.__all__:
        assert export_name in text, (
            f"Export '{export_name}' from robot_sf.__all__ missing from {CANONICAL_DOC}"
        )

    # Verify stability levels
    for level in ("stable", "beta", "experimental"):
        assert f"`{level}`" in text


def test_no_duplicate_canonical_stability_policy_page() -> None:
    """docs/api/stable_public_api.md redirects to docs/public_api.md with no duplicate policy definitions."""
    assert REDIRECT_DOC.is_file()
    redirect_text = REDIRECT_DOC.read_text(encoding="utf-8")

    assert "[Robot SF Public API](../public_api.md)" in redirect_text
    assert "consolidated into the canonical" in redirect_text.lower()
    # The former page is a forwarding pointer, not a second policy owner. Keep
    # this contract heading-based so equivalent prose cannot silently duplicate
    # the canonical policy.
    assert "## " not in redirect_text
    for heading in CANONICAL_POLICY_HEADINGS:
        assert heading.lower() not in redirect_text.lower()


def test_no_stale_proposal_language() -> None:
    """Neither doc may contain stale proposal or pending sign-off language."""
    canonical_text = CANONICAL_DOC.read_text(encoding="utf-8")
    redirect_text = REDIRECT_DOC.read_text(encoding="utf-8")
    api_index_text = SPHINX_API_INDEX.read_text(encoding="utf-8")

    for phrase in STALE_PROPOSAL_PHRASES:
        assert phrase.lower() not in canonical_text.lower(), (
            f"Stale phrase '{phrase}' in {CANONICAL_DOC}"
        )
        assert phrase.lower() not in redirect_text.lower(), (
            f"Stale phrase '{phrase}' in {REDIRECT_DOC}"
        )
        assert phrase.lower() not in api_index_text.lower(), (
            f"Stale phrase '{phrase}' in {SPHINX_API_INDEX}"
        )


def test_env_registry_references_canonical_doc() -> None:
    """env_registry.py docstring references docs/public_api.md."""
    text = ENV_REGISTRY_FILE.read_text(encoding="utf-8")
    assert "docs/public_api.md" in text
    assert "docs/api/stable_public_api.md" not in text


def test_sphinx_navigation_indices_include_public_api() -> None:
    """Sphinx root and API indices properly reference Public API."""
    root_text = SPHINX_ROOT_INDEX.read_text(encoding="utf-8")
    api_text = SPHINX_API_INDEX.read_text(encoding="utf-8")

    assert "Public API <public_api>" in root_text
    assert ":doc:" in api_text
