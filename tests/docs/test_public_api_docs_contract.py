"""Contract and regression tests for public API and stability policy documentation.

Ensures:
- Exactly one canonical public API policy document exists (docs/public_api.md).
- Stale proposal and pending-sign-off language from #5801/#8245 is eliminated.
- docs/api/stable_public_api.md acts as a clean redirect/forwarding pointer.
- Top-level facade exports match robot_sf.__all__.
- Sphinx documentation navigation links resolve correctly.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import robot_sf

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DOC = REPO_ROOT / "docs" / "public_api.md"
REDIRECT_DOC = REPO_ROOT / "docs" / "api" / "stable_public_api.md"
SPHINX_API_INDEX = REPO_ROOT / "docs" / "api" / "index.rst"
SPHINX_ROOT_INDEX = REPO_ROOT / "docs" / "index.rst"
ENV_REGISTRY_FILE = REPO_ROOT / "robot_sf" / "gym_env" / "env_registry.py"

MANIFEST_START = "<!-- public-api-manifest:start -->"
MANIFEST_END = "<!-- public-api-manifest:end -->"
ALLOWED_STABILITY_LEVELS = frozenset({"stable", "beta", "experimental"})


@dataclass(frozen=True)
class PublicApiManifestEntry:
    """Structured representation of an entry in the public API manifest table."""

    symbol: str
    stability: str
    kind: str
    source: str


def parse_public_api_manifest(doc_text: str) -> list[PublicApiManifestEntry]:
    """Parse the public API manifest table from the canonical markdown document.

    Args:
        doc_text: Raw markdown text of docs/public_api.md.

    Returns:
        List of parsed PublicApiManifestEntry objects.

    Raises:
        ValueError: If manifest markers are missing, malformed, or if table
            rows are invalid or duplicate.
    """
    if MANIFEST_START not in doc_text or MANIFEST_END not in doc_text:
        raise ValueError(
            f"Manifest markers '{MANIFEST_START}' and '{MANIFEST_END}' must be present in doc"
        )
    start_idx = doc_text.index(MANIFEST_START) + len(MANIFEST_START)
    end_idx = doc_text.index(MANIFEST_END)
    if end_idx < start_idx:
        raise ValueError("Manifest end marker appears before start marker")

    table_text = doc_text[start_idx:end_idx].strip()
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("Manifest table must contain header, delimiter, and at least one row")

    header = [c.strip() for c in lines[0].split("|")[1:-1]]
    expected_header = ["Symbol", "Stability", "Kind", "Source"]
    if header != expected_header:
        raise ValueError(f"Unexpected manifest table header: {header}, expected {expected_header}")

    entries: list[PublicApiManifestEntry] = []
    seen_symbols: set[str] = set()

    for line in lines[2:]:  # skip header and delimiter
        cols = [c.strip().strip("`") for c in line.split("|")[1:-1]]
        if len(cols) != 4:
            raise ValueError(
                f"Malformed manifest row: '{line}' (expected 4 columns, got {len(cols)})"
            )
        symbol, stability, kind, source = cols
        if not symbol:
            raise ValueError(f"Empty symbol in manifest row: '{line}'")
        if symbol in seen_symbols:
            raise ValueError(f"Duplicate symbol in manifest: '{symbol}'")
        seen_symbols.add(symbol)
        if stability not in ALLOWED_STABILITY_LEVELS:
            raise ValueError(
                f"Invalid stability class '{stability}' for symbol '{symbol}'. "
                f"Allowed: {sorted(ALLOWED_STABILITY_LEVELS)}"
            )
        entries.append(
            PublicApiManifestEntry(
                symbol=symbol,
                stability=stability,
                kind=kind,
                source=source,
            )
        )
    return entries


def inventory_public_exports(package: Any = robot_sf) -> dict[str, Any]:
    """Deterministically inventory public exports from package __all__, dir, and lazy resolution.

    Args:
        package: The module/package to inventory (default: robot_sf).

    Returns:
        Dict mapping export name to the resolved object.

    Raises:
        AssertionError: If __all__ contains duplicates, if any export in __all__ is missing
            from dir(), or if any export fails lazy resolution.
    """
    all_exports = list(package.__all__)
    if len(all_exports) != len(set(all_exports)):
        raise AssertionError(f"package.__all__ contains duplicate symbols: {all_exports}")

    dir_names = set(dir(package))
    inventory: dict[str, Any] = {}
    for name in all_exports:
        if name not in dir_names:
            pkg_name = getattr(package, "__name__", "package")
            raise AssertionError(f"Export '{name}' in __all__ is missing from dir({pkg_name})")
        try:
            resolved = getattr(package, name)
        except Exception as err:
            raise AssertionError(
                f"Export '{name}' failed lazy attribute resolution: {err}"
            ) from err
        inventory[name] = resolved
    return inventory


def verify_public_api_parity(
    manifest_entries: list[PublicApiManifestEntry],
    package: Any = robot_sf,
) -> dict[str, Any]:
    """Verify exact one-to-one parity between package exports and documentation manifest.

    Returns:
        Dict with inventory of resolved exports.

    Raises:
        AssertionError: If there are undocumented exports or stale documented symbols.
    """
    inventory = inventory_public_exports(package)
    exported_symbols = set(inventory.keys())
    documented_symbols = {entry.symbol for entry in manifest_entries}

    undocumented = exported_symbols - documented_symbols
    stale = documented_symbols - exported_symbols

    if undocumented or stale:
        errors = []
        if undocumented:
            errors.append(f"Undocumented exports in code: {sorted(undocumented)}")
        if stale:
            errors.append(f"Stale documented symbols in manifest: {sorted(stale)}")
        raise AssertionError("Public API parity mismatch:\n" + "\n".join(errors))
    return inventory


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


def test_public_api_manifest_exact_parity() -> None:
    """Verify one-to-one parity between robot_sf exports and public API manifest table."""
    text = CANONICAL_DOC.read_text(encoding="utf-8")
    manifest_entries = parse_public_api_manifest(text)

    # Parity check: exact one-to-one match between code inventory and manifest
    inventory = verify_public_api_parity(manifest_entries, robot_sf)
    assert len(inventory) == len(robot_sf.__all__)

    # Every manifest entry must have a valid stability class and resolve
    for entry in manifest_entries:
        assert entry.stability in ALLOWED_STABILITY_LEVELS
        assert entry.symbol in inventory
        assert inventory[entry.symbol] is not None

    # Supported names consistency across __all__, __dir__, and resolution
    dir_names = set(dir(robot_sf))
    for entry in manifest_entries:
        assert entry.symbol in dir_names
    assert set(robot_sf.__all__) <= dir_names


def test_negative_fixture_undocumented_export() -> None:
    """Negative fixture: verify parity fails when a code export is missing from docs."""

    class MockPackage:
        __name__ = "mock_robot_sf"
        __all__ = ["EpisodeRecord", "undocumented_symbol"]
        EpisodeRecord = object()
        undocumented_symbol = object()

        def __dir__(self) -> list[str]:
            return ["EpisodeRecord", "undocumented_symbol"]

    manifest_entries = [PublicApiManifestEntry("EpisodeRecord", "stable", "class", "robot_sf.api")]
    with pytest.raises(
        AssertionError, match=r"Undocumented exports in code: \['undocumented_symbol'\]"
    ):
        verify_public_api_parity(manifest_entries, MockPackage())


def test_negative_fixture_stale_documented_symbol() -> None:
    """Negative fixture: verify parity fails when docs include a stale export."""

    class MockPackage:
        __name__ = "mock_robot_sf"
        __all__ = ["EpisodeRecord"]
        EpisodeRecord = object()

        def __dir__(self) -> list[str]:
            return ["EpisodeRecord"]

    manifest_entries = [
        PublicApiManifestEntry("EpisodeRecord", "stable", "class", "robot_sf.api"),
        PublicApiManifestEntry("stale_symbol", "stable", "function", "robot_sf.api"),
    ]
    with pytest.raises(
        AssertionError, match=r"Stale documented symbols in manifest: \['stale_symbol'\]"
    ):
        verify_public_api_parity(manifest_entries, MockPackage())


def test_negative_fixture_failing_lazy_resolver() -> None:
    """Negative fixture: verify inventory fails when lazy resolution raises AttributeError."""

    class BrokenPackage:
        __name__ = "broken_robot_sf"
        __all__ = ["broken_symbol"]

        def __dir__(self) -> list[str]:
            return ["broken_symbol"]

        def __getattr__(self, name: str) -> Any:
            raise AttributeError(f"module has no attribute {name!r}")

    with pytest.raises(
        AssertionError, match=r"Export 'broken_symbol' failed lazy attribute resolution"
    ):
        inventory_public_exports(BrokenPackage())


def test_negative_fixture_duplicate_manifest_symbol() -> None:
    """Negative fixture: verify manifest parser rejects duplicate symbol rows."""
    duplicate_md = (
        "<!-- public-api-manifest:start -->\n"
        "| Symbol | Stability | Kind | Source |\n"
        "| --- | --- | --- | --- |\n"
        "| `make_env` | stable | function | `robot_sf.api` |\n"
        "| `make_env` | stable | function | `robot_sf.api` |\n"
        "<!-- public-api-manifest:end -->\n"
    )
    with pytest.raises(ValueError, match="Duplicate symbol in manifest: 'make_env'"):
        parse_public_api_manifest(duplicate_md)


def test_negative_fixture_invalid_stability_class() -> None:
    """Negative fixture: verify manifest parser rejects invalid stability levels."""
    invalid_tier_md = (
        "<!-- public-api-manifest:start -->\n"
        "| Symbol | Stability | Kind | Source |\n"
        "| --- | --- | --- | --- |\n"
        "| `make_env` | unsupported_level | function | `robot_sf.api` |\n"
        "<!-- public-api-manifest:end -->\n"
    )
    with pytest.raises(ValueError, match="Invalid stability class 'unsupported_level'"):
        parse_public_api_manifest(invalid_tier_md)


def test_negative_fixture_malformed_manifest_markers() -> None:
    """Negative fixture: verify parser rejects documents missing manifest markers."""
    with pytest.raises(ValueError, match=r"Manifest markers.*must be present"):
        parse_public_api_manifest("No manifest table here")


def test_negative_fixture_heavy_eager_import() -> None:
    """Negative fixture: verify import-weight check fails when a heavy dependency is injected."""
    code = (
        "import sys, robot_sf; "
        "sys.modules['torch'] = object(); "
        "heavy = [m for m in ('pygame', 'torch', 'stable_baselines3', 'carla', 'playwright', 'ray', 'tensorflow') if m in sys.modules]; "
        "assert not heavy, f'Heavy modules imported: {heavy}'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Heavy modules imported: ['torch']" in result.stderr
