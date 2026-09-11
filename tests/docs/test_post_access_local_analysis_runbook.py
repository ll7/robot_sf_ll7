"""Consistency contract for the post-access local-analysis runbook (issue #8899)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = REPO_ROOT / "docs/post_access_local_analysis_runbook.md"
ENTRY_POINTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs/adoption_path.md",
    REPO_ROOT / "docs/context/artifact_evidence_vocabulary.md",
    REPO_ROOT / "docs/troubleshooting/doctor.md",
    REPO_ROOT / "CONTRIBUTING.md",
)
SOURCE_PATHS = (
    "scripts/dev/check_runtime_requirements.sh",
    "scripts/dev/check_carla_runtime.sh",
    "scripts/generate_figures.py",
    "scripts/repro/capture_release_environment.py",
    "scripts/repro/verify_release_checksums.py",
    "scripts/reporting/generate_result_card.py",
    "scripts/research/generate_report.py",
    "scripts/tools/chunk_manifest.py",
    "scripts/tools/lineage_index.py",
    "scripts/validation/check_legacy_ppo_snapshot_parity.py",
    "scripts/validation/check_local_model_artifacts.py",
    "scripts/validation/compare_execution_environments.py",
    "scripts/validation/validate_artifact_catalog.py",
    "examples/advanced/36_optional_capability_handling.py",
    "examples/advanced/39_failed_campaign_diagnosis.py",
    "examples/advanced/41_artifact_preservation.py",
    "configs/baselines/example_matrix.yaml",
    "model/registry.yaml",
    "robot_sf/benchmark/schemas/episode.schema.v1.json",
)
REQUIRED_ANCHOR_LINKS = (
    "dev_guide.md#setup",
    "dev_runtime_requirements.md#headless-rendering",
    "troubleshooting/doctor.md#statuses-and-exit-code",
    "ENVIRONMENT.md#tiny-batch-smoke",
    "model/registry.md#using-models-from-the-registry",
    "artifact_retention_and_cleanup.md#4-checked-workflows",
)
REASON_CODES = (
    "core_available",
    "extra_missing",
    "model_unavailable",
    "model_unknown",
    "dataset_unavailable",
    "runtime_unsupported",
)


def _links(text: str) -> list[str]:
    """Return inline Markdown link destinations."""
    return re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text)


def _github_slug(heading: str) -> str:
    """Build the simple GitHub heading slug used by the checked anchors."""
    heading = re.sub(r"[`<].*?[`>]", "", heading).lower()
    heading = re.sub(r"[^\w\s-]", "", heading)
    return re.sub(r"\s+", "-", heading.strip())


def _anchors(path: Path) -> set[str]:
    """Return explicit and heading-derived anchors for one Markdown page."""
    text = path.read_text(encoding="utf-8")
    explicit = set(re.findall(r'<a\s+id="([^"]+)"></a>', text))
    headings = {
        _github_slug(match.group(1)) for match in re.finditer(r"^#{1,6}\s+(.+)$", text, re.M)
    }
    return explicit | headings


def test_runbook_links_resolve_to_existing_files_and_anchors() -> None:
    """Every runbook-local Markdown link resolves against the current checkout."""
    text = RUNBOOK.read_text(encoding="utf-8")
    assert RUNBOOK.is_file()
    for destination in _links(text):
        parsed = urlsplit(unquote(destination.strip().strip("<>")))
        if parsed.scheme or parsed.netloc:
            continue
        target = (RUNBOOK.parent / parsed.path).resolve() if parsed.path else RUNBOOK
        assert target.is_file(), destination
        assert REPO_ROOT in target.parents or target == REPO_ROOT, destination
        if parsed.fragment:
            assert parsed.fragment in _anchors(target), destination


def test_runbook_has_source_bound_paths_profiles_and_reason_codes() -> None:
    """Commands and vocabulary in the runbook remain bound to current owners."""
    text = RUNBOOK.read_text(encoding="utf-8")
    for relative in SOURCE_PATHS:
        assert (REPO_ROOT / relative).is_file(), relative
        assert relative in text, relative
    for anchor in REQUIRED_ANCHOR_LINKS:
        assert anchor in text, anchor
    for profile in ("minimal", "analysis", "visualization", "model-loading"):
        assert f"`{profile}`" in text
    for code in REASON_CODES:
        assert f"`{code}`" in text
    assert "uv sync --extra benchmark --extra analytics" in text
    assert "uv sync --extra training" in text
    assert "uv sync --all-extras --group carla" in text
    assert all(variable in text for variable in ("DISPLAY", "MPLBACKEND", "SDL_VIDEODRIVER"))
    assert "ROBOT_SF_VALIDATE_VISUALS" in text
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = set(project["project"]["optional-dependencies"])
    assert {"benchmark", "analytics", "viz", "training", "gpu"} <= extras
    capability_source = (
        REPO_ROOT / "examples/advanced/36_optional_capability_handling.py"
    ).read_text(encoding="utf-8")
    for code in REASON_CODES:
        assert f'reason_code="{code}"' in capability_source


def test_entry_points_and_fail_closed_boundaries_are_explicit() -> None:
    """The five requested entry points route to the runbook and preserve boundaries."""
    for path in ENTRY_POINTS:
        assert "post_access_local_analysis_runbook.md" in path.read_text(encoding="utf-8"), path
    text = RUNBOOK.read_text(encoding="utf-8")
    for status in ("ok", "skipped", "missing_optional", "failed", "not_available", "blocked"):
        assert f"`{status}`" in text, status
    assert "does not establish planner quality" in text
    assert "no benchmark or paper-facing claim" in text
    assert "Do not paste credentials" in text
    assert "/home/" not in text
    assert "/scratch/" not in text
    assert "40_restore_campaign_capsule.py" not in text
    assert "39_artifact_preservation.py" not in text


@pytest.mark.parametrize(
    "path",
    (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs/adoption_path.md",
        REPO_ROOT / "docs/context/artifact_evidence_vocabulary.md",
        REPO_ROOT / "docs/troubleshooting/doctor.md",
        REPO_ROOT / "CONTRIBUTING.md",
    ),
)
def test_entry_point_link_destination_is_not_dangling(path: Path) -> None:
    """Each entry point names the new runbook file, not a future companion path."""
    destination = next(
        (
            link
            for link in _links(path.read_text(encoding="utf-8"))
            if "post_access_local_analysis_runbook" in link
        ),
        None,
    )
    assert destination is not None
    parsed = urlsplit(destination)
    assert (path.parent / parsed.path).resolve() == RUNBOOK.resolve()
