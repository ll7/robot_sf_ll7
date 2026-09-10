"""Manifest schema, validator, and README generator contracts (issue #8731).

Covers the versioned ``examples_manifest.yaml`` schema: sentinel prerequisite
normalization, malformed-sentinel rejection, unique path/name identity, known
categories, normalized tags, valid runtime class, ``ci_enabled``/``ci_reason``
consistency, doc-target existence, complete maintained-file coverage via
explicit versioned exclusions, deterministic coded validator output, and
byte-stable README generation.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from robot_sf.examples.manifest_loader import (
    MANIFEST_SCHEMA_VERSION,
    ManifestValidationError,
    load_manifest,
)
from scripts.validation import validate_examples_manifest as validator
from scripts.validation.render_examples_readme import build_markdown

REPO_ROOT = Path(__file__).resolve().parents[2]

_BASE_CATEGORIES = [
    {
        "slug": "uncategorized",
        "title": "Uncategorized",
        "description": "d",
        "order": 0,
        "ci_default": True,
    },
    {
        "slug": "quickstart",
        "title": "Quickstart",
        "description": "d",
        "order": 1,
        "ci_default": True,
    },
]


def _write_example(examples_dir: Path, relative: str, docstring: str = "A demo.") -> None:
    """Create an example ``.py`` file with a module docstring."""

    target = examples_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f'"""{docstring}"""\n', encoding="utf-8")


def _write_manifest(
    tmp_path: Path,
    *,
    examples: list[dict],
    categories: list[dict] | None = None,
    exclusions: list[dict] | None = None,
    schema_version: int | None = None,
    extra_top: dict | None = None,
) -> Path:
    """Write a minimal manifest under ``tmp_path/examples`` and return its path."""

    examples_dir = tmp_path / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "version": "0.1.0",
        "categories": categories if categories is not None else _BASE_CATEGORIES,
        "examples": examples,
    }
    if schema_version is not None:
        payload["schema_version"] = schema_version
    if exclusions is not None:
        payload["exclusions"] = exclusions
    if extra_top:
        payload.update(extra_top)
    manifest_path = examples_dir / "examples_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return manifest_path


def _demo_entry(**overrides) -> dict:
    """Return a minimal valid example entry."""

    entry: dict = {
        "path": "quickstart/demo.py",
        "name": "Demo",
        "summary": "A demo.",
        "category_slug": "quickstart",
        "prerequisites": [],
        "ci_enabled": True,
        "ci_reason": None,
        "doc_reference": None,
        "tags": ["demo"],
    }
    entry.update(overrides)
    return entry


def test_schema_version_defaults_to_one() -> None:
    """Manifests without ``schema_version`` load as the canonical version."""

    manifest = load_manifest()
    assert manifest.schema_version == MANIFEST_SCHEMA_VERSION == 1


def test_unsupported_schema_version_rejected(tmp_path: Path) -> None:
    """Unknown schema versions fail closed."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    manifest_path = _write_manifest(tmp_path, examples=[_demo_entry()], schema_version=999)
    with pytest.raises(ManifestValidationError, match="schema_version"):
        load_manifest(manifest_path)


@pytest.mark.parametrize(
    "raw",
    [["None"], ["none"], ["NONE"], ["Null"], ["NULL"], ["n/a"], ["N/A"], ["-"], [" None "]],
)
def test_sentinel_prerequisite_spellings_normalize_to_empty(tmp_path: Path, raw: list) -> None:
    """Sentinel prerequisite spellings normalize to ``[]``."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    manifest_path = _write_manifest(tmp_path, examples=[_demo_entry(prerequisites=raw)])
    manifest = load_manifest(manifest_path)
    assert manifest.examples[0].prerequisites == ()


def test_yaml_null_prerequisite_normalizes_to_empty(tmp_path: Path) -> None:
    """YAML ``null`` entries inside prerequisites normalize to ``[]``."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    manifest_path = _write_manifest(tmp_path, examples=[_demo_entry(prerequisites=[None])])
    manifest = load_manifest(manifest_path)
    assert manifest.examples[0].prerequisites == ()


def test_mixed_sentinel_and_real_prerequisite_keeps_real(tmp_path: Path) -> None:
    """Sentinels filter out while real prerequisites survive."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    manifest_path = _write_manifest(
        tmp_path,
        examples=[_demo_entry(prerequisites=["None", "maps/svg_maps/debug_06.svg"])],
    )
    manifest = load_manifest(manifest_path)
    assert manifest.examples[0].prerequisites == ("maps/svg_maps/debug_06.svg",)


@pytest.mark.parametrize(
    "raw",
    ["None", "none", [""], ["   "], [123], [{"path": "x"}], [["nested"]]],
)
def test_malformed_prerequisite_rejected(tmp_path: Path, raw) -> None:
    """Malformed sentinels (bare strings, non-strings, empties) are rejected."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    manifest_path = _write_manifest(tmp_path, examples=[_demo_entry(prerequisites=raw)])
    with pytest.raises(ManifestValidationError, match="E_MALFORMED_PREREQUISITE"):
        load_manifest(manifest_path)


def test_duplicate_path_rejected(tmp_path: Path) -> None:
    """Two entries must not share one manifest path."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    entries = [_demo_entry(), _demo_entry(name="Other Demo")]
    manifest_path = _write_manifest(tmp_path, examples=entries)
    with pytest.raises(ManifestValidationError, match="E_DUPLICATE_PATH"):
        load_manifest(manifest_path)


def test_duplicate_name_rejected(tmp_path: Path) -> None:
    """Two entries must not share one display name."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    _write_example(tmp_path / "examples", "quickstart/other.py")
    entries = [_demo_entry(), _demo_entry(path="quickstart/other.py")]
    manifest_path = _write_manifest(tmp_path, examples=entries)
    with pytest.raises(ManifestValidationError, match="E_DUPLICATE_NAME"):
        load_manifest(manifest_path)


def test_unknown_category_rejected(tmp_path: Path) -> None:
    """Entries referencing an unknown category fail with a stable message."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    manifest_path = _write_manifest(tmp_path, examples=[_demo_entry(category_slug="nope-missing")])
    with pytest.raises(ManifestValidationError, match="unknown category"):
        load_manifest(manifest_path)


def test_absent_path_rejected(tmp_path: Path) -> None:
    """``validate_paths=True`` rejects manifest paths missing from disk."""

    manifest_path = _write_manifest(tmp_path, examples=[_demo_entry()])
    with pytest.raises(ManifestValidationError, match="missing from filesystem"):
        load_manifest(manifest_path, validate_paths=True)


def test_ci_reason_missing_rejected(tmp_path: Path) -> None:
    """CI-disabled entries without a reason are rejected."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    manifest_path = _write_manifest(
        tmp_path, examples=[_demo_entry(ci_enabled=False, ci_reason=None)]
    )
    with pytest.raises(ManifestValidationError, match="missing ci_reason"):
        load_manifest(manifest_path)


def test_ci_reason_contradiction_rejected(tmp_path: Path) -> None:
    """CI-enabled entries must not declare a ``ci_reason``."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    manifest_path = _write_manifest(
        tmp_path, examples=[_demo_entry(ci_enabled=True, ci_reason="nope")]
    )
    with pytest.raises(ManifestValidationError, match="E_CI_REASON_CONTRADICTION"):
        load_manifest(manifest_path)


@pytest.mark.parametrize(
    "tags",
    [["Backend"], [" backend"], ["backend "], [""], [["x"]], ["demo", "demo"], ["has space"]],
)
def test_tags_must_be_normalized(tmp_path: Path, tags: list) -> None:
    """Tags must already be normalized lowercase ``[a-z0-9][a-z0-9_-]*``."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    manifest_path = _write_manifest(tmp_path, examples=[_demo_entry(tags=tags)])
    with pytest.raises(ManifestValidationError, match="E_TAG_NOT_NORMALIZED"):
        load_manifest(manifest_path)


def test_runtime_class_sentinel_rejected(tmp_path: Path) -> None:
    """``expected_runtime`` must never be a prerequisite-style sentinel."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    manifest_path = _write_manifest(tmp_path, examples=[_demo_entry(expected_runtime="None")])
    with pytest.raises(ManifestValidationError, match="E_RUNTIME_CLASS_INVALID"):
        load_manifest(manifest_path)


def test_unregistered_file_detected(tmp_path: Path) -> None:
    """Files on disk without registration or exclusion get ``E_UNREGISTERED_FILE``."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    _write_example(tmp_path / "examples", "quickstart/stowaway.py", "A stowaway.")
    manifest_path = _write_manifest(tmp_path, examples=[_demo_entry()])
    manifest = load_manifest(manifest_path)
    errors = validator._check_manifest_coverage(manifest, tmp_path / "examples")
    assert any("E_UNREGISTERED_FILE" in error and "stowaway" in error for error in errors)


def test_exclusion_covers_mirror_file(tmp_path: Path) -> None:
    """Versioned exclusions with rationale silence coverage for mirrors/helpers."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    _write_example(tmp_path / "examples", "legacy_shim.py", "A shim.")
    manifest_path = _write_manifest(
        tmp_path,
        examples=[_demo_entry()],
        exclusions=[{"path": "legacy_shim.py", "kind": "mirror", "reason": "Shim for demo."}],
    )
    manifest = load_manifest(manifest_path, validate_paths=True)
    assert manifest.excluded_paths() == frozenset({"legacy_shim.py"})
    errors = validator._check_manifest_coverage(manifest, tmp_path / "examples")
    assert errors == []


def test_exclusion_without_reason_rejected(tmp_path: Path) -> None:
    """Exclusions require an explicit rationale."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    manifest_path = _write_manifest(
        tmp_path,
        examples=[_demo_entry()],
        exclusions=[{"path": "quickstart/demo.py", "kind": "mirror", "reason": ""}],
    )
    with pytest.raises(ManifestValidationError, match="non-empty string"):
        load_manifest(manifest_path)


def test_doc_reference_missing_detected(tmp_path: Path) -> None:
    """``doc_reference`` targets must exist in the repository."""

    _write_example(tmp_path / "examples", "quickstart/demo.py")
    manifest_path = _write_manifest(
        tmp_path, examples=[_demo_entry(doc_reference="docs/does-not-exist-xyz.md")]
    )
    manifest = load_manifest(manifest_path)
    errors = validator._check_doc_references(manifest)
    assert any("E_DOC_REFERENCE_MISSING" in error for error in errors)


def test_every_maintained_file_registered_or_excluded() -> None:
    """Zero unregistered maintained examples in the real repository."""

    manifest = load_manifest(validate_paths=True)
    errors = validator._check_manifest_coverage(manifest, manifest.examples_root)
    assert errors == []


def test_real_manifest_has_no_sentinel_prerequisites() -> None:
    """Zero malformed prerequisite sentinels in the canonical manifest."""

    from robot_sf.examples.manifest_loader import PREREQUISITE_SENTINELS

    manifest = load_manifest()
    for example in manifest.examples:
        for prerequisite in example.prerequisites:
            assert prerequisite.strip() == prerequisite
            assert prerequisite.lower() not in PREREQUISITE_SENTINELS


def test_real_manifest_coverage_counts() -> None:
    """The canonical manifest registers every maintained example exactly once."""

    manifest = load_manifest(validate_paths=True)
    discovered = {
        path.relative_to(manifest.examples_root).as_posix()
        for path in manifest.examples_root.rglob("*.py")
        if path.name != "__init__.py"
    }
    registered = {example.path.as_posix() for example in manifest.examples}
    registered |= set(manifest.excluded_paths())
    assert discovered == registered
    assert len(registered) == len(discovered)
    assert len({example.path.as_posix() for example in manifest.examples}) == len(manifest.examples)


def test_validator_output_is_deterministic() -> None:
    """Validator errors carry ``path: [CODE]`` and sort deterministically."""

    manifest = load_manifest()
    first = sorted(validator._check_ci_consistency(manifest))
    second = sorted(validator._check_ci_consistency(manifest))
    assert first == second
    for error in (
        validator._check_category_directory_alignment(manifest)
        + validator._check_doc_references(manifest)
        + validator._check_tags_normalized(manifest)
        + validator._check_runtime_class(manifest)
    ):
        assert ": [E_" in error


def test_generator_is_byte_stable() -> None:
    """Repeated README generation is byte-identical."""

    manifest = load_manifest()
    first = build_markdown(manifest, include_archived=False, include_ci_column=True)
    second = build_markdown(manifest, include_archived=False, include_ci_column=True)
    assert first == second
    assert first.endswith("\n")
    assert "auto-generated" in first


def test_generated_readme_matches_committed_file() -> None:
    """The committed ``examples/README.md`` matches the canonical generator."""

    manifest = load_manifest()
    generated = build_markdown(manifest, include_archived=False, include_ci_column=True)
    committed = (REPO_ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    assert generated == committed


def test_canonical_readme_has_no_sentinel_cells() -> None:
    """Generated tables render empty prerequisites as ``_None_``, never ``None``."""

    manifest = load_manifest()
    markdown = build_markdown(manifest, include_archived=False, include_ci_column=True)
    sentinel_cells = [
        line for line in markdown.splitlines() if line.startswith("| ") and "| None |" in line
    ]
    assert sentinel_cells == []


def test_docstring_contract_matches_summaries() -> None:
    """Every registered example docstring first line matches its summary."""

    manifest = load_manifest()
    errors, _ = validator._check_docstrings(manifest, allow_missing=False)
    assert errors == []


def test_manifest_parses_expected_runtime_field(tmp_path: Path) -> None:
    """The loader surfaces the optional runtime-class (``expected_runtime``) field."""

    manifest_text = dedent(
        """
        version: 0.1.0
        categories:
          - slug: quickstart
            title: "Quickstart"
            description: "d"
            order: 1
            ci_default: true
        examples:
          - path: quickstart/demo.py
            name: "Demo"
            summary: "A demo."
            category_slug: quickstart
            ci_enabled: true
            ci_reason: null
            doc_reference: null
            tags: [demo]
            expected_runtime: "~5s"
        """
    ).strip()
    examples_dir = tmp_path / "examples" / "quickstart"
    examples_dir.mkdir(parents=True)
    (examples_dir / "demo.py").write_text('"""A demo."""\n', encoding="utf-8")
    manifest_path = tmp_path / "examples" / "examples_manifest.yaml"
    manifest_path.write_text(manifest_text, encoding="utf-8")

    manifest = load_manifest(manifest_path, validate_paths=True)
    assert manifest.examples[0].expected_runtime == "~5s"
