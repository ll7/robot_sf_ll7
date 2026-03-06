"""Tests for model-registry W&B latest-run helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from robot_sf.models import registry

if TYPE_CHECKING:
    from pathlib import Path


class _FakeFile:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRun:
    def __init__(
        self,
        *,
        run_id: str,
        name: str,
        created_at: str,
        group: str,
        job_type: str,
        state: str,
        tags: tuple[str, ...],
        files: tuple[str, ...],
    ) -> None:
        self.id = run_id
        self.name = name
        self.created_at = created_at
        self.group = group
        self.job_type = job_type
        self.state = state
        self.tags = tags
        self._files = files

    def files(self):
        return [_FakeFile(name) for name in self._files]


def test_find_latest_wandb_model_filters_and_picks_newest(monkeypatch) -> None:
    """Latest W&B model selection should honor filters and choose the newest matching run."""

    class _Api:
        def runs(self, path: str):
            assert path == "ll7/robot_sf"
            return [
                _FakeRun(
                    run_id="older",
                    name="ppo_prefix_old",
                    created_at="2026-03-05T10:00:00Z",
                    group="issue-576",
                    job_type="expert-ppo",
                    state="finished",
                    tags=("ppo", "overnight"),
                    files=("model.zip",),
                ),
                _FakeRun(
                    run_id="newer",
                    name="ppo_prefix_new",
                    created_at="2026-03-05T12:00:00Z",
                    group="issue-576",
                    job_type="expert-ppo",
                    state="running",
                    tags=("ppo", "overnight"),
                    files=("model.zip", "output.log"),
                ),
                _FakeRun(
                    run_id="wrong",
                    name="other_name",
                    created_at="2026-03-05T13:00:00Z",
                    group="other",
                    job_type="expert-ppo",
                    state="finished",
                    tags=("ppo",),
                    files=("model.zip",),
                ),
            ]

    monkeypatch.setattr(registry, "wandb", SimpleNamespace(Api=_Api))
    selected = registry.find_latest_wandb_model(
        entity="ll7",
        project="robot_sf",
        group="issue-576",
        job_type="expert-ppo",
        name_prefix="ppo_prefix",
        tags=("ppo",),
        allowed_states=("finished", "running"),
    )
    assert selected.run_id == "newer"
    assert selected.run_path == "ll7/robot_sf/newer"


def test_resolve_latest_wandb_model_downloads_selected_file(monkeypatch, tmp_path: Path) -> None:
    """Latest W&B helper should route the chosen run through the existing download helper."""
    monkeypatch.setattr(
        registry,
        "find_latest_wandb_model",
        lambda **kwargs: registry.WandbLatestModel(
            run_id="abc123",
            run_path="ll7/robot_sf/abc123",
            run_name="ppo_latest",
            job_type="expert-ppo",
            group="issue-576",
            state="finished",
            created_at="2026-03-05T12:00:00Z",
            file_name="model.zip",
        ),
    )
    expected = tmp_path / "model.zip"
    monkeypatch.setattr(
        registry,
        "_download_from_wandb",
        lambda entry, cache_dir=None: expected,
    )
    resolved, selected = registry.resolve_latest_wandb_model(
        entity="ll7",
        project="robot_sf",
        cache_dir=tmp_path,
    )
    assert resolved == expected
    assert selected.run_id == "abc123"


def test_find_latest_wandb_model_raises_when_no_run_matches(monkeypatch) -> None:
    """Latest selection should fail cleanly when no run exposes the requested model file."""

    class _Api:
        def runs(self, path: str):
            assert path == "ll7/robot_sf"
            return [
                _FakeRun(
                    run_id="missing-file",
                    name="ppo_prefix_old",
                    created_at="2026-03-05T10:00:00Z",
                    group="issue-576",
                    job_type="expert-ppo",
                    state="finished",
                    tags=("ppo",),
                    files=("output.log",),
                )
            ]

    monkeypatch.setattr(registry, "wandb", SimpleNamespace(Api=_Api))
    with pytest.raises(FileNotFoundError, match="No W&B run matched latest-model query"):
        registry.find_latest_wandb_model(
            entity="ll7",
            project="robot_sf",
            group="issue-576",
            job_type="expert-ppo",
            name_prefix="ppo_prefix",
        )
