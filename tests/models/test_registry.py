"""Tests for model-registry W&B latest-run helpers."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.models import registry


class _FakeFile:
    """W&B file stub exposing a name."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRun:
    """W&B run stub exposing metadata and file listing."""

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
        """Return fake W&B files for this run."""
        return [_FakeFile(name) for name in self._files]


def test_find_latest_wandb_model_filters_and_picks_newest(monkeypatch) -> None:
    """Latest W&B model selection should honor filters and choose the newest matching run."""

    class _Api:
        """W&B API stub returning several candidate runs."""

        def runs(self, path: str):
            """Return runs for latest-model filtering tests."""
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
        """W&B API stub returning no usable model files."""

        def runs(self, path: str):
            """Return runs that should not match the requested file."""
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


def test_canonical_registry_entry_has_public_release_pointer() -> None:
    """The canonical PPO benchmark checkpoint should be publicly hydratable."""
    entry = registry.get_registry_entry(
        "ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417"
    )
    assert entry["github_release"] == {
        "repository": "ll7/robot_sf_ll7",
        "tag": "artifact/models-2026-05-registry-v1",
        "asset_name": (
            "ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417-model.zip"
        ),
        "file_name": "model.zip",
        "sha256": "2b30df812bfcc737924b126b0763d69c567fe20716dc1c1eba8f56f926b49c1d",
        "size_bytes": 93662266,
        "metadata_asset_name": (
            "ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417"
            "-metadata.json"
        ),
    }


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


def test_resolve_model_path_rejects_missing_local_only_entry_with_migration_guidance(
    tmp_path: Path,
) -> None:
    """Local-only entries should fail fast with an explicit replacement hint."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        (
            "version: 1\nmodels:\n"
            "  - model_id: predictive_proxy_selected_v2_full\n"
            "    local_path: missing/predictive_model.pt\n"
            "    local_only: true\n"
            "    replacement_model_id: predictive_proxy_selected_v2_xl_ego\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="local-only.*predictive_proxy_selected_v2_xl_ego"):
        registry.resolve_model_path(
            "predictive_proxy_selected_v2_full",
            registry_path=registry_path,
        )


def test_resolve_model_path_downloads_github_release_before_wandb(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Public GitHub release pointers should hydrate without requiring W&B credentials."""
    payload = b"public checkpoint"
    expected_sha256 = registry.hashlib.sha256(payload).hexdigest()
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        (
            "version: 1\nmodels:\n"
            "  - model_id: public_model\n"
            "    local_path: missing/model.zip\n"
            "    wandb_run_path: ll7/robot_sf/private\n"
            "    wandb_file: model.zip\n"
            "    github_release:\n"
            "      repository: ll7/robot_sf_ll7\n"
            "      tag: artifact/models-2026-05-registry-v1\n"
            "      asset_name: public_model-model.zip\n"
            "      file_name: model.zip\n"
            f"      sha256: {expected_sha256}\n"
        ),
        encoding="utf-8",
    )
    opened_urls: list[str] = []

    class _Response(io.BytesIO):
        """Context-manager byte stream returned by fake urlopen."""

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            self.close()

    def _fake_urlopen(url: str):
        opened_urls.append(url)
        return _Response(payload)

    monkeypatch.setattr(registry.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(registry, "wandb", None)

    resolved = registry.resolve_model_path(
        "public_model",
        registry_path=registry_path,
        cache_dir=tmp_path / "cache",
    )

    assert resolved == tmp_path / "cache" / "public_model" / "model.zip"
    assert resolved.read_bytes() == payload
    assert opened_urls == [
        "https://github.com/ll7/robot_sf_ll7/releases/download/"
        "artifact/models-2026-05-registry-v1/public_model-model.zip"
    ]


def test_download_from_github_release_reuses_valid_cached_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A cached public artifact should be reused only when its checksum matches."""
    registry._LOGGED_CACHED_MODEL_ARTIFACTS.clear()
    cache_dir = tmp_path / "cache"
    cached = cache_dir / "public_model" / "model.zip"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"public checkpoint")

    def _fail_urlopen(url: str):  # pragma: no cover - should not be reached
        raise AssertionError(url)

    monkeypatch.setattr(registry.urllib.request, "urlopen", _fail_urlopen)
    resolved = registry._download_from_github_release(
        {
            "model_id": "public_model",
            "github_release": {
                "repository": "ll7/robot_sf_ll7",
                "tag": "artifact/models-2026-05-registry-v1",
                "asset_name": "public_model-model.zip",
                "file_name": "model.zip",
                "sha256": registry.hashlib.sha256(b"public checkpoint").hexdigest(),
            },
        },
        cache_dir=cache_dir,
    )

    assert resolved == cached
    registry._LOGGED_CACHED_MODEL_ARTIFACTS.clear()


def test_download_from_github_release_rejects_checksum_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Downloaded public artifacts should be immutable through SHA256 verification."""

    class _Response(io.BytesIO):
        """Context-manager byte stream returned by fake urlopen."""

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            self.close()

    monkeypatch.setattr(
        registry.urllib.request,
        "urlopen",
        lambda url: _Response(b"tampered checkpoint"),
    )

    with pytest.raises(ValueError, match="expected 0000"):
        registry._download_from_github_release(
            {
                "model_id": "public_model",
                "github_release": {
                    "repository": "ll7/robot_sf_ll7",
                    "tag": "artifact/models-2026-05-registry-v1",
                    "asset_name": "public_model-model.zip",
                    "file_name": "model.zip",
                    "sha256": "0000",
                },
            },
            cache_dir=tmp_path / "cache",
        )


def test_download_from_wandb_uses_cached_path(monkeypatch, tmp_path: Path) -> None:
    """Cached W&B downloads should be reused without hitting the API."""
    registry._LOGGED_CACHED_MODEL_ARTIFACTS.clear()
    cache_dir = tmp_path / "cache"
    cached = cache_dir / "demo" / "model.zip"
    cached.parent.mkdir(parents=True)
    cached.write_text("checkpoint", encoding="utf-8")

    api_called = {"value": False}

    class _Api:
        """W&B API stub that should not be called on cache hits."""

        def run(self, path: str):
            """Fail if a cached download still hits the API."""
            api_called["value"] = True
            raise AssertionError(path)

    monkeypatch.setattr(registry, "wandb", SimpleNamespace(Api=_Api))
    resolved = registry._download_from_wandb(
        {"model_id": "demo", "wandb_run_path": "ll7/robot_sf/demo", "wandb_file": "model.zip"},
        cache_dir=cache_dir,
    )
    assert resolved == cached
    assert api_called["value"] is False
    registry._LOGGED_CACHED_MODEL_ARTIFACTS.clear()


def test_download_from_wandb_logs_cached_path_once(monkeypatch, tmp_path: Path) -> None:
    """Repeated cache hits for the same artifact should emit only one info log."""
    registry._LOGGED_CACHED_MODEL_ARTIFACTS.clear()
    cache_dir = tmp_path / "cache"
    cached = cache_dir / "demo" / "model.zip"
    cached.parent.mkdir(parents=True)
    cached.write_text("checkpoint", encoding="utf-8")
    messages: list[str] = []

    def _fake_info(message: str, *args) -> None:
        """Record formatted cache-hit log messages."""
        messages.append(message.format(*args) if args else message)

    class _Api:
        """W&B API stub that should not be called for cached artifacts."""

        def run(self, path: str):
            """Fail if the cache-hit path asks W&B for a run."""
            raise AssertionError(path)

    monkeypatch.setattr(registry, "wandb", SimpleNamespace(Api=_Api))
    monkeypatch.setattr(registry.logger, "info", _fake_info)

    for _ in range(3):
        assert (
            registry._download_from_wandb(
                {
                    "model_id": "demo",
                    "wandb_run_path": "ll7/robot_sf/demo",
                    "wandb_file": "model.zip",
                },
                cache_dir=cache_dir,
            )
            == cached
        )

    assert messages == [f"Using cached model artifact: {cached}"]
    registry._LOGGED_CACHED_MODEL_ARTIFACTS.clear()


def test_download_from_wandb_builds_run_path_from_split_fields(monkeypatch, tmp_path: Path) -> None:
    """Download helper should support registry rows with separate entity/project/run id fields."""
    downloaded = tmp_path / "cache" / "demo" / "model.zip"

    class _RunFile:
        """W&B run-file stub that writes a model file on download."""

        def download(self, *, root: str, replace: bool):
            """Write the requested model file into the cache root."""
            assert replace is True
            path = Path(root) / "model.zip"
            path.write_text("checkpoint", encoding="utf-8")
            return path

    class _Run:
        """W&B run stub exposing a named file."""

        def file(self, name: str):
            """Return the fake run file for model.zip."""
            assert name == "model.zip"
            return _RunFile()

    class _Api:
        """W&B API stub resolving split entity/project/run fields."""

        def run(self, path: str):
            """Return the fake run for the expected path."""
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


def test_download_from_wandb_prefers_artifact_path(monkeypatch, tmp_path: Path) -> None:
    """Durable W&B artifact paths should be used before run-file downloads."""
    downloaded = tmp_path / "cache" / "demo" / "model.zip"
    calls: list[tuple[str, str]] = []

    class _Artifact:
        """W&B artifact stub that writes a model file on download."""

        def download(self, *, root: str):
            """Write the model file and return the artifact root."""
            path = Path(root) / "model.zip"
            path.write_text("checkpoint", encoding="utf-8")
            return str(Path(root))

    class _Api:
        """W&B API stub preferring artifact downloads."""

        def artifact(self, path: str):
            """Record and return the requested artifact."""
            calls.append(("artifact", path))
            return _Artifact()

        def run(self, path: str):  # pragma: no cover - should not be reached
            """Fail if artifact download falls back to run download."""
            calls.append(("run", path))
            raise AssertionError(path)

    monkeypatch.setattr(registry, "wandb", SimpleNamespace(Api=_Api))
    resolved = registry._download_from_wandb(
        {
            "model_id": "demo",
            "wandb_artifact_path": "ll7/robot_sf/demo-best:v1",
            "wandb_run_path": "ll7/robot_sf/demo-run",
            "wandb_file": "model.zip",
        },
        cache_dir=tmp_path / "cache",
    )
    assert resolved == downloaded
    assert calls == [("artifact", "ll7/robot_sf/demo-best:v1")]


def test_download_from_wandb_rejects_missing_run_metadata(tmp_path: Path) -> None:
    """Download helper should fail clearly when the registry row lacks W&B location metadata."""
    with pytest.raises(ValueError, match="missing wandb_artifact_path"):
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
