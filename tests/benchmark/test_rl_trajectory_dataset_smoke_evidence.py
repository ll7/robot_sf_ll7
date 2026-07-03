"""Evidence-closure regression tests for the issue #4011 RL trajectory smoke bundle.

These tests pin the committed ``RLTrajectoryDataset.v1`` smoke evidence bundle under
``docs/context/evidence/`` to the canonical recorder/loader contract so downstream offline-RL
work can depend on the artifact. They fail closed if the committed preview, checksum, or manifest
drift out of sync with the recorder in ``robot_sf/benchmark/rl_trajectory_dataset.py``.

Scope: closure of the existing merged pipeline (#4145) only. No dataset is regenerated from a
live runner here; the committed bundle is validated in place.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.rl_trajectory_dataset import (
    RL_TRAJECTORY_DATASET_MANIFEST_SCHEMA_VERSION,
    RL_TRAJECTORY_DATASET_SCHEMA_VERSION,
    build_rl_trajectory_manifest,
    compute_return_to_go,
    flatten_rl_trajectory_episodes,
    load_rl_trajectory_dataset,
    sha256_file,
    validate_rl_trajectory_episode,
)
from robot_sf.benchmark.schemas.rl_trajectory_dataset_schema import (
    validate_rl_trajectory_dataset_manifest,
)

# Repository-root-relative location of the committed smoke evidence bundle.
_EVIDENCE_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "context"
    / "evidence"
    / "issue_4011_rl_trajectory_dataset_smoke_2026-07-02"
)
_PREVIEW_PATH = _EVIDENCE_DIR / "issue_4011_smoke.preview.jsonl"
_MANIFEST_PATH = _EVIDENCE_DIR / "issue_4011_smoke.manifest.json"

# Provenance keys the smoke manifest must expose so downstream work can trace the artifact.
_REQUIRED_PROVENANCE_KEYS = (
    "artifact_durability",
    "git_commit",
    "return_convention",
    "reward_convention",
    "source_jsonl",
    "source_sha256",
)


def _load_committed_manifest() -> dict:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def test_smoke_evidence_bundle_files_present() -> None:
    """The committed smoke bundle exposes both the preview dataset and its manifest."""
    assert _PREVIEW_PATH.is_file(), f"missing committed preview: {_PREVIEW_PATH}"
    assert _MANIFEST_PATH.is_file(), f"missing committed manifest: {_MANIFEST_PATH}"


def test_committed_preview_loads_and_validates() -> None:
    """The committed preview loads through the canonical loader and passes episode validation."""
    episodes = load_rl_trajectory_dataset(_PREVIEW_PATH)
    assert episodes, "committed preview must contain at least one episode"
    for episode in episodes:
        # Loader already validates, but assert explicitly to pin the contract at this boundary.
        validate_rl_trajectory_episode(episode)
        assert list(episode.return_to_go) == compute_return_to_go(episode.rewards)


def test_committed_manifest_passes_schema_and_semantics() -> None:
    """The committed manifest satisfies the JSON Schema and split-leakage semantics."""
    manifest = _load_committed_manifest()
    validate_rl_trajectory_dataset_manifest(manifest)
    assert manifest["schema_version"] == RL_TRAJECTORY_DATASET_MANIFEST_SCHEMA_VERSION
    assert manifest["dataset_schema_version"] == RL_TRAJECTORY_DATASET_SCHEMA_VERSION
    assert manifest["dataset_path"] == _PREVIEW_PATH.name


def test_manifest_checksum_matches_committed_preview() -> None:
    """The manifest ``dataset_sha256`` matches the committed preview, failing closed on drift."""
    manifest = _load_committed_manifest()
    assert sha256_file(_PREVIEW_PATH) == manifest["dataset_sha256"], (
        "committed preview checksum drifted from the manifest; regenerate the smoke bundle "
        "with scripts/benchmark/record_rl_trajectory_dataset.py"
    )


def test_committed_manifest_is_reproducible_from_recorder() -> None:
    """Rebuilding the manifest from the committed preview reproduces the committed manifest.

    This closes the loop: the durable evidence manifest is exactly what the canonical recorder
    ``build_rl_trajectory_manifest`` derives from the committed dataset. Only the non-derived
    inputs (timestamp and provenance) are supplied from the committed manifest.
    """
    committed = _load_committed_manifest()
    episodes = load_rl_trajectory_dataset(_PREVIEW_PATH)
    rebuilt = build_rl_trajectory_manifest(
        dataset_id=committed["dataset_id"],
        dataset_path=_PREVIEW_PATH,
        episodes=episodes,
        created_at_utc=committed["created_at_utc"],
        provenance=committed["provenance"],
    )
    assert rebuilt == committed, (
        "committed manifest is not reproducible from the recorder; the committed evidence "
        "drifted from robot_sf/benchmark/rl_trajectory_dataset.py"
    )


def test_manifest_provenance_exposes_required_keys() -> None:
    """The manifest provenance carries the keys downstream offline-RL work needs to trace it."""
    manifest = _load_committed_manifest()
    provenance = manifest["provenance"]
    missing = [key for key in _REQUIRED_PROVENANCE_KEYS if key not in provenance]
    assert not missing, f"manifest provenance missing required keys: {missing}"


def test_manifest_counts_agree_with_loaded_episodes() -> None:
    """Manifest episode/step counts agree with the flattened committed dataset transitions."""
    manifest = _load_committed_manifest()
    episodes = load_rl_trajectory_dataset(_PREVIEW_PATH)
    flattened = flatten_rl_trajectory_episodes(episodes)

    assert manifest["episode_count"] == len(episodes)
    assert manifest["step_count"] == len(flattened["rewards"])
    assert manifest["step_count"] == sum(episode.step_count for episode in episodes)


@pytest.mark.parametrize("split_name", ["train", "validation", "test"])
def test_manifest_exposes_all_split_names(split_name: str) -> None:
    """All three split summaries stay present so future split logic remains stable."""
    manifest = _load_committed_manifest()
    assert split_name in manifest["splits"]
