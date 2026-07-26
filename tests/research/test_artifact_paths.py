"""Direct contract tests for ``robot_sf.research.artifact_paths`` (issue #6365).

These tests lock the public path/tree contract of the artifact-path helpers:
artifact-root overrides, research-reports composition, deterministic report-id
timestamps with spaces/hyphens/punctuation sanitization, report-tree creation
(generated and explicit roots), idempotent creation, and ``get_output_paths``
without hidden filesystem writes.

All filesystem effects stay inside the pytest ``tmp_path``. The default artifact
root (``output``) is only ever asserted by value, never materialised, and every
generated-root exercise pins ``ROBOT_SF_ARTIFACT_ROOT`` to ``tmp_path`` so no
repository ``output/`` directory is ever created.
"""

from __future__ import annotations

import re
from datetime import UTC
from datetime import datetime as _real_datetime
from pathlib import Path

import pytest

from robot_sf.research import artifact_paths as ap

# Deterministic clock for report ids. The fake ``datetime`` injected into the
# module returns this moment from ``now()``; the expected timestamp string is
# derived from it so the assertions cannot drift from the format.
_FROZEN_NOW = _real_datetime(2026, 7, 26, 14, 30, 45, tzinfo=UTC)
EXPECTED_TIMESTAMP = _FROZEN_NOW.strftime("%Y%m%d_%H%M%S")  # "20260726_143045"


class _FrozenDatetime:
    """Stand-in for ``datetime`` so report ids are deterministic under test."""

    @classmethod
    def now(cls) -> _real_datetime:
        return _FROZEN_NOW


@pytest.fixture
def freeze_report_time(monkeypatch) -> str:
    """Freeze the report-id clock inside ``artifact_paths`` and return the stamp."""
    monkeypatch.setattr(ap, "datetime", _FrozenDatetime)
    return EXPECTED_TIMESTAMP


# ---------------------------------------------------------------------------
# Artifact root and research-reports composition
# ---------------------------------------------------------------------------


def test_get_artifact_root_defaults_to_output(monkeypatch) -> None:
    """Without an override the canonical root is the literal ``output`` path."""
    monkeypatch.delenv("ROBOT_SF_ARTIFACT_ROOT", raising=False)
    assert ap.get_artifact_root() == Path("output")


def test_get_artifact_root_respects_env_override(monkeypatch, tmp_path: Path) -> None:
    """``ROBOT_SF_ARTIFACT_ROOT`` fully replaces the default root."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    assert ap.get_artifact_root() == tmp_path


def test_get_research_reports_root_defaults_under_output(monkeypatch) -> None:
    """The reports root composes as ``<artifact_root>/research_reports``."""
    monkeypatch.delenv("ROBOT_SF_ARTIFACT_ROOT", raising=False)
    assert ap.get_research_reports_root() == Path("output") / "research_reports"


def test_get_research_reports_root_composes_overridden_root(monkeypatch, tmp_path: Path) -> None:
    """The reports root tracks the overridden artifact root."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    assert ap.get_research_reports_root() == tmp_path / "research_reports"


# ---------------------------------------------------------------------------
# Report id generation: deterministic timestamp + sanitization
# ---------------------------------------------------------------------------


def test_generate_report_id_starts_with_timestamp_shape(freeze_report_time: str) -> None:
    """The id opens with a ``YYYYMMDD_HHMMSS_`` timestamp prefix."""
    report_id = ap.generate_report_id("anything")
    assert report_id.startswith(f"{EXPECTED_TIMESTAMP}_")
    assert re.match(r"^\d{8}_\d{6}_", report_id)


@pytest.mark.parametrize(
    ("experiment_name", "sanitized_suffix"),
    [
        # Lowercasing and spaces -> underscores.
        ("BC Ablation Study", "bc_ablation_study"),
        # Hyphens collapse to underscores.
        ("my-experiment", "my_experiment"),
        # Punctuation stripped, hyphens and spaces both handled.
        ("Exp #1: Trial-2!", "exp_1_trial_2"),
        # Pure casing change, no separators.
        ("UPPER", "upper"),
        # Empty label: separator still emitted (locks observed behavior).
        ("", ""),
        # Multiple spaces are not collapsed; each becomes its own underscore.
        ("  Hello  World  ", "__hello__world__"),
    ],
)
def test_generate_report_id_sanitization(
    freeze_report_time: str, experiment_name: str, sanitized_suffix: str
) -> None:
    """Report ids sanitise casing, spaces, hyphens, and punctuation deterministically."""
    assert ap.generate_report_id(experiment_name) == f"{EXPECTED_TIMESTAMP}_{sanitized_suffix}"


# ---------------------------------------------------------------------------
# ensure_report_tree: generated and explicit roots, structure, idempotency
# ---------------------------------------------------------------------------

EXPECTED_TREE_KEYS = ("root", "report", "report_tex", "metadata", "figures", "data", "configs")


def test_ensure_report_tree_generated_root_under_artifact_root(
    monkeypatch, freeze_report_time: str, tmp_path: Path
) -> None:
    """A generated root lands under ``<artifact_root>/research_reports/<report_id>``."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))

    paths = ap.ensure_report_tree("My Exp")

    expected_root = tmp_path / "research_reports" / f"{EXPECTED_TIMESTAMP}_my_exp"
    assert paths["root"] == expected_root
    # Directories are actually created on disk.
    assert expected_root.is_dir()
    assert (expected_root / "figures").is_dir()
    assert (expected_root / "data").is_dir()
    assert (expected_root / "configs").is_dir()
    # The directory tree stays inside tmp_path (no repository output/ created).
    assert "output" not in str(expected_root.resolve())
    assert expected_root.resolve().is_relative_to(tmp_path.resolve())


def test_ensure_report_tree_paths_are_layout_not_yet_files(
    monkeypatch, freeze_report_time: str, tmp_path: Path
) -> None:
    """``ensure_report_tree`` returns file layouts but only creates directories."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))

    paths = ap.ensure_report_tree("layout")

    root = paths["root"]
    assert paths["report"] == root / "report.md"
    assert paths["report_tex"] == root / "report.tex"
    assert paths["metadata"] == root / "metadata.json"
    assert paths["figures"] == root / "figures"
    assert paths["data"] == root / "data"
    assert paths["configs"] == root / "configs"
    # The reported file paths are not materialised; only the directories exist.
    assert not paths["report"].exists()
    assert not paths["report_tex"].exists()
    assert not paths["metadata"].exists()
    assert paths["figures"].is_dir()
    assert paths["data"].is_dir()
    assert paths["configs"].is_dir()


def test_ensure_report_tree_returns_all_expected_keys(monkeypatch, tmp_path: Path) -> None:
    """The returned mapping exposes exactly the documented logical names."""
    monkeypatch.delenv("ROBOT_SF_ARTIFACT_ROOT", raising=False)

    paths = ap.ensure_report_tree("keys", output_override=tmp_path / "report")

    assert tuple(paths.keys()) == EXPECTED_TREE_KEYS


def test_ensure_report_tree_explicit_output_override(monkeypatch, tmp_path: Path) -> None:
    """An explicit ``output_override`` is used verbatim, bypassing the reports root."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path / "unused"))

    override = tmp_path / "custom_report"
    paths = ap.ensure_report_tree("ignored-name", output_override=override)

    assert paths["root"] == override
    assert "research_reports" not in str(override)
    # Tree is created at the override location.
    assert override.is_dir()
    assert (override / "figures").is_dir()
    assert (override / "data").is_dir()
    assert (override / "configs").is_dir()
    # The env-pinned reports root is not touched.
    assert not (tmp_path / "unused" / "research_reports").exists()


def test_ensure_report_tree_is_idempotent(tmp_path: Path) -> None:
    """Calling twice on the same root does not raise and keeps the tree intact."""
    override = tmp_path / "report"

    first = ap.ensure_report_tree("once", output_override=override)
    second = ap.ensure_report_tree("again", output_override=override)

    assert first == second
    assert override.is_dir()
    for sub in ("figures", "data", "configs"):
        assert (override / sub).is_dir()


# ---------------------------------------------------------------------------
# get_output_paths: pure mapping, no hidden writes
# ---------------------------------------------------------------------------


def test_get_output_paths_matches_ensure_report_tree_layout(tmp_path: Path) -> None:
    """``get_output_paths`` reproduces the layout returned by ``ensure_report_tree``."""
    root = tmp_path / "report"
    built = ap.ensure_report_tree("source", output_override=root)

    assert ap.get_output_paths(root) == built


def test_get_output_paths_has_expected_keys(tmp_path: Path) -> None:
    """The mapping exposes exactly the documented logical names."""
    paths = ap.get_output_paths(tmp_path / "report")

    assert tuple(paths.keys()) == EXPECTED_TREE_KEYS


def test_get_output_paths_creates_no_directories(tmp_path: Path) -> None:
    """``get_output_paths`` is a pure mapping and writes nothing to disk."""
    root = tmp_path / "never_created"

    paths = ap.get_output_paths(root)

    assert paths["root"] == root
    assert paths["report"] == root / "report.md"
    assert paths["report_tex"] == root / "report.tex"
    assert paths["metadata"] == root / "metadata.json"
    assert paths["figures"] == root / "figures"
    assert paths["data"] == root / "data"
    assert paths["configs"] == root / "configs"
    # Nothing was created: neither root nor any subdirectory.
    assert not root.exists()
    assert not (root / "figures").exists()
    assert not (root / "data").exists()
    assert not (root / "configs").exists()
