"""Tests for model-registry W&B latest-run helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.models import registry


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


def test_load_registry_skips_invalid_entries_and_reads_models(tmp_path: Path) -> None:
    """Registry loader should skip malformed rows and keep valid model ids."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
version: 1
models:
  - ignored
  - local_path: output/model_cache/missing/model.zip
  - model_id: valid_model
    local_path: output/model_cache/valid/model.zip
""".strip()
        + "\n",
        encoding="utf-8",
    )

    loaded = registry.load_registry(registry_path)
    assert list(loaded) == ["valid_model"]


def test_load_registry_rejects_duplicate_model_ids(tmp_path: Path) -> None:
    """Duplicate model ids should fail fast instead of silently overriding."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
version: 1
models:
  - model_id: duplicated
    local_path: output/model_cache/a/model.zip
  - model_id: duplicated
    local_path: output/model_cache/b/model.zip
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate model_id"):
        registry.load_registry(registry_path)


def test_get_registry_entry_raises_for_unknown_model(tmp_path: Path) -> None:
    """Unknown model ids should raise a clear KeyError."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("version: 1\nmodels: []\n", encoding="utf-8")
    with pytest.raises(KeyError, match="Unknown model_id"):
        registry.get_registry_entry("missing", registry_path)


def test_resolve_model_path_prefers_existing_local_path(tmp_path: Path) -> None:
    """Local registry artifacts should be returned without invoking W&B download."""
    local_model = tmp_path / "weights" / "model.zip"
    local_model.parent.mkdir(parents=True)
    local_model.write_text("checkpoint", encoding="utf-8")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        f"version: 1\nmodels:\n  - model_id: local_model\n    local_path: {local_model.as_posix()}\n",
        encoding="utf-8",
    )

    resolved = registry.resolve_model_path("local_model", registry_path=registry_path)
    assert resolved == local_model


def test_resolve_model_path_rejects_missing_local_path_when_download_disabled(
    tmp_path: Path,
) -> None:
    """Missing local artifacts should fail cleanly when downloads are disabled."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        "version: 1\nmodels:\n  - model_id: local_model\n    local_path: missing/model.zip\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="downloads are disabled"):
        registry.resolve_model_path(
            "local_model",
            registry_path=registry_path,
            allow_download=False,
        )


def test_download_from_wandb_uses_cached_path(monkeypatch, tmp_path: Path) -> None:
    """Cached W&B downloads should be reused without hitting the API."""
    cache_dir = tmp_path / "cache"
    cached = cache_dir / "demo" / "model.zip"
    cached.parent.mkdir(parents=True)
    cached.write_text("checkpoint", encoding="utf-8")

    api_called = {"value": False}

    class _Api:
        def run(self, path: str):
            api_called["value"] = True
            raise AssertionError(path)

    monkeypatch.setattr(registry, "wandb", SimpleNamespace(Api=_Api))
    resolved = registry._download_from_wandb(
        {"model_id": "demo", "wandb_run_path": "ll7/robot_sf/demo", "wandb_file": "model.zip"},
        cache_dir=cache_dir,
    )
    assert resolved == cached
    assert api_called["value"] is False


def test_download_from_wandb_builds_run_path_from_split_fields(monkeypatch, tmp_path: Path) -> None:
    """Download helper should support registry rows with separate entity/project/run id fields."""
    downloaded = tmp_path / "cache" / "demo" / "model.zip"

    class _RunFile:
        def download(self, *, root: str, replace: bool):
            assert replace is True
            path = Path(root) / "model.zip"
            path.write_text("checkpoint", encoding="utf-8")
            return path

    class _Run:
        def file(self, name: str):
            assert name == "model.zip"
            return _RunFile()

    class _Api:
        def run(self, path: str):
            assert path == "ll7/robot_sf/demo"
            return _Run()

    monkeypatch.setattr(registry, "wandb", SimpleNamespace(Api=_Api))
    resolved = registry._download_from_wandb(
        {
            "model_id": "demo",
            "wandb_entity": "ll7",
            "wandb_project": "robot_sf",
            "wandb_run_id": "demo",
        },
        cache_dir=tmp_path / "cache",
    )
    assert resolved == downloaded


def test_download_from_wandb_rejects_missing_run_metadata(tmp_path: Path) -> None:
    """Download helper should fail clearly when the registry row lacks W&B location metadata."""
    with pytest.raises(ValueError, match="missing wandb_run_path"):
        registry._download_from_wandb({"model_id": "demo"}, cache_dir=tmp_path / "cache")


def test_upsert_registry_entry_appends_and_replaces(tmp_path: Path) -> None:
    """Upsert should create the registry file, then replace matching rows in place."""
    registry_path = tmp_path / "registry.yaml"

    first = {"model_id": "demo", "local_path": "output/model_cache/demo/model.zip"}
    registry.upsert_registry_entry(first, registry_path=registry_path)
    loaded_first = registry.load_registry(registry_path)
    assert loaded_first["demo"]["local_path"] == "output/model_cache/demo/model.zip"

    replacement = {"model_id": "demo", "local_path": "output/model_cache/demo/model_v2.zip"}
    registry.upsert_registry_entry(replacement, registry_path=registry_path)
    loaded_second = registry.load_registry(registry_path)
    assert list(loaded_second) == ["demo"]
    assert loaded_second["demo"]["local_path"] == "output/model_cache/demo/model_v2.zip"


def test_upsert_registry_entry_requires_model_id(tmp_path: Path) -> None:
    """Upsert should reject entries without a model id."""
    with pytest.raises(ValueError, match="must include a model_id"):
        registry.upsert_registry_entry(
            {"local_path": "output/model_cache/demo/model.zip"},
            registry_path=tmp_path / "registry.yaml",
        )
