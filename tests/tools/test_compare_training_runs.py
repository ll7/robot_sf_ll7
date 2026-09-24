"""Tests for comparing training runs and resolving manifest files."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from scripts.tools import compare_training_runs as ctr


def test_load_training_run_falls_back_to_prefixed_manifest(tmp_path: Path, monkeypatch):
    # Point artifact root to temp location
    """Verify manifest resolution falls back to prefix-matched filenames when exact match is absent.

    Args:
        tmp_path: Temporary directory fixture serving as the artifact root.
        monkeypatch: Pytest monkeypatch fixture to redirect ROBOT_SF_ARTIFACT_ROOT.
    """
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    runs_dir = tmp_path / "benchmarks" / "ppo_imitation" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest = runs_dir / "runA_variant.json"
    manifest.write_text(
        json.dumps({"run_id": "runA", "metrics": {"foo": {"mean": 1}}}), encoding="utf-8"
    )

    loaded = ctr._load_training_run("runA")
    assert loaded["run_id"] == "runA"
    assert loaded["metrics"]["foo"]["mean"] == 1


def test_load_training_run_searches_nested_timestamp_root(tmp_path: Path, monkeypatch):
    """Verify manifest resolution recursively discovers runs under nested timestamp directories.

    Args:
        tmp_path: Temporary directory fixture serving as the artifact root.
        monkeypatch: Pytest monkeypatch fixture to redirect ROBOT_SF_ARTIFACT_ROOT.
    """
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    nested = tmp_path / "benchmarks" / "ts123" / "benchmarks" / "ppo_imitation" / "runs"
    nested.mkdir(parents=True, exist_ok=True)
    manifest = nested / "runB.json"
    manifest.write_text(
        json.dumps({"run_id": "runB", "metrics": {"bar": {"mean": 2}}}), encoding="utf-8"
    )

    loaded = ctr._load_training_run("runB")
    assert loaded["run_id"] == "runB"
    assert loaded["metrics"]["bar"]["mean"] == 2


def test_load_training_run_falls_back_to_newest_when_missing(tmp_path: Path, monkeypatch):
    """Verify manifest resolution falls back to the newest available run when requested run is missing.

    Args:
        tmp_path: Temporary directory fixture serving as the artifact root.
        monkeypatch: Pytest monkeypatch fixture to redirect ROBOT_SF_ARTIFACT_ROOT.
    """
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    runs_dir = tmp_path / "benchmarks" / "ppo_imitation" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest = runs_dir / "other_run.json"
    manifest.write_text(
        json.dumps({"run_id": "other_run", "metrics": {"baz": {"mean": 3}}}), encoding="utf-8"
    )

    loaded = ctr._load_training_run("nonexistent_run")
    assert loaded["run_id"] == "other_run"
