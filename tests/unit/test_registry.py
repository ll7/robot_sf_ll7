"""Tests for model-registry W&B latest-run helpers."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

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


def test_promoted_learned_policy_requires_observation_track_metadata() -> None:
    """Benchmark-promoted learned checkpoints must declare their observation contract."""
    entry = {
        "model_id": "promoted_ppo",
        "tags": ["ppo", "promoted"],
        "benchmark_promotion": {"claim_boundary": "benchmark_promoted"},
    }

    issues = registry.validate_registry_entry_benchmark_promotion(entry)
    paths = {issue.path for issue in issues}

    assert "benchmark_promotion.benchmark_track" in paths
    assert "benchmark_promotion.allowed_observation_keys" in paths
    assert "benchmark_promotion.privileged_input_status" in paths


def test_research_only_registry_boundary_is_allowed_with_reason() -> None:
    """Non-benchmark learned checkpoints can stay in the registry with a clear claim boundary."""
    entry = {
        "model_id": "lidar_smoke_candidate",
        "tags": ["learned-policy", "candidate"],
        "benchmark_promotion": {
            "claim_boundary": "smoke_only",
            "non_benchmark_reason": "LiDAR launch packet only; no benchmark claim.",
        },
    }

    assert registry.validate_registry_entry_benchmark_promotion(entry) == []


def test_repository_registry_benchmark_promotion_metadata_is_valid() -> None:
    """Tracked registry entries should satisfy benchmark-promotion claim boundaries."""
    registry_path = Path(__file__).resolve().parents[2] / "model" / "registry.yaml"
    entries = registry.load_registry(registry_path)

    issues = {
        model_id: registry.validate_registry_entry_benchmark_promotion(entry)
        for model_id, entry in entries.items()
    }

    assert {model_id: row_issues for model_id, row_issues in issues.items() if row_issues} == {}


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


def test_resolve_model_path_verifies_existing_release_cache(monkeypatch, tmp_path: Path) -> None:
    """Release-backed cache paths must not bypass their registry checksum."""
    monkeypatch.chdir(tmp_path)
    payload = b"verified-checkpoint"
    expected_sha = hashlib.sha256(payload).hexdigest()
    cached = tmp_path / "output/model_cache/public_model/public_model-model.zip"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"substituted-checkpoint")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        f"""
version: 1
models:
  - model_id: public_model
    local_path: output/model_cache/public_model/public_model-model.zip
    github_release:
      repo: ll7/robot_sf_ll7
      tag: artifact/models-test
      asset_name: public_model-model.zip
      sha256: {expected_sha}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="not found locally"):
        registry.resolve_model_path(
            "public_model", registry_path=registry_path, allow_download=False
        )

    cached.write_bytes(payload)
    resolved = registry.resolve_model_path(
        "public_model", registry_path=registry_path, allow_download=False
    )
    assert resolved == cached


def test_resolve_model_path_preserves_unpacked_release_local_path(
    monkeypatch, tmp_path: Path
) -> None:
    """Unpacked local paths remain usable when release metadata names an archive bundle."""
    monkeypatch.chdir(tmp_path)
    local_model = tmp_path / "model/checkpoint/network.meta"
    local_model.parent.mkdir(parents=True)
    local_model.write_text("unpacked-checkpoint", encoding="utf-8")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
version: 1
models:
  - model_id: bundled_model
    local_path: model/checkpoint/network.meta
    github_release:
      repo: ll7/robot_sf_ll7
      tag: artifact/models-test
      asset_name: bundled_model-checkpoint.tar.gz
      sha256: 0000000000000000000000000000000000000000000000000000000000000000
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert (
        registry.resolve_model_path(
            "bundled_model", registry_path=registry_path, allow_download=False
        )
        == local_model
    )


def test_predictive_proxy_registry_paths_match_release_assets() -> None:
    """Predictive proxy cache pointers should name their release assets exactly."""
    registry_path = Path(__file__).resolve().parents[2] / "model" / "registry.yaml"
    entries = registry.load_registry(registry_path)

    for model_id in ("predictive_proxy_selected_v1", "predictive_proxy_selected_v2_full"):
        entry = entries[model_id]
        release = entry["github_release"]
        assert Path(entry["local_path"]).parts[:3] == ("output", "model_cache", model_id)
        assert Path(entry["local_path"]).name == release["asset_name"]


def test_resolve_model_path_downloads_github_release_asset(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Public GitHub release assets should be preferred over W&B downloads."""
    payload = b"checkpoint"
    source = tmp_path / "source.zip"
    source.write_bytes(payload)
    expected_sha = registry._sha256(source)
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        f"""
version: 1
models:
  - model_id: public_model
    local_path: missing/model.zip
    wandb_run_path: ll7/robot_sf/private
    github_release:
      repo: ll7/robot_sf_ll7
      tag: artifact/models-test
      asset_name: public_model-model.zip
      sha256: {expected_sha}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    class _Response:
        """Context-manager response returning payload chunks."""

        def __init__(self, data: bytes) -> None:
            self._data = data
            self._offset = 0

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def read(self, size: int) -> bytes:
            chunk = self._data[self._offset : self._offset + size]
            self._offset += len(chunk)
            return chunk

    urls: list[str] = []

    def _fake_urlopen(url: str, timeout: int):
        """Capture the release URL and return the fake payload."""
        assert timeout == 60
        urls.append(url)
        return _Response(payload)

    def _fail_wandb_download(entry, *, cache_dir):
        """Fail if GitHub release resolution falls back to W&B."""
        raise AssertionError(entry)

    monkeypatch.setattr(registry, "urlopen", _fake_urlopen)
    monkeypatch.setattr(registry, "_download_from_wandb", _fail_wandb_download)

    resolved = registry.resolve_model_path(
        "public_model",
        registry_path=registry_path,
        cache_dir=tmp_path / "cache",
    )

    assert resolved == tmp_path / "cache" / "public_model" / "public_model-model.zip"
    assert resolved.read_bytes() == payload
    assert urls == [
        "https://github.com/ll7/robot_sf_ll7/releases/download/"
        "artifact/models-test/public_model-model.zip"
    ]


def test_github_release_download_retries_429_and_honors_retry_after(
    monkeypatch, tmp_path: Path
) -> None:
    """Transient rate limits retry with the bounded server-provided delay."""
    payload = b"checkpoint"

    class _Response:
        def __init__(self) -> None:
            self._done = False

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def read(self, size: int) -> bytes:
            if self._done:
                return b""
            self._done = True
            return payload

    calls: list[int] = []

    def fake_urlopen(url: str, *, timeout: int):
        calls.append(timeout)
        if len(calls) == 1:
            raise HTTPError(url, 429, "rate limited", {"Retry-After": "7"}, None)
        return _Response()

    sleeps: list[float] = []
    monkeypatch.setattr(registry, "urlopen", fake_urlopen)
    target = tmp_path / "cache" / "model.zip"
    target.parent.mkdir(parents=True)
    registry._stream_download_url(
        "https://github.com/ll7/robot_sf_ll7/releases/download/tag/model.zip",
        target,
        max_attempts=2,
        backoff_seconds=1.0,
        sleep=sleeps.append,
    )

    assert calls == [60, 60]
    assert sleeps == [7.0]
    assert target.read_bytes() == payload
    assert list(target.parent.glob("*.part")) == []


def test_github_release_retry_after_is_capped(monkeypatch, tmp_path: Path) -> None:
    """A hostile or accidental long Retry-After cannot defeat the retry time bound."""
    target = tmp_path / "cache" / "model.zip"
    target.parent.mkdir(parents=True)
    sleeps: list[float] = []

    def fake_urlopen(url: str, *, timeout: int):
        raise HTTPError(url, 429, "rate limited", {"Retry-After": "999"}, None)

    monkeypatch.setattr(registry, "urlopen", fake_urlopen)
    with pytest.raises(HTTPError):
        registry._stream_download_url(
            "https://github.com/ll7/robot_sf_ll7/releases/download/tag/model.zip",
            target,
            max_attempts=2,
            max_backoff_seconds=2.0,
            sleep=sleeps.append,
        )

    assert sleeps == [2.0]


@pytest.mark.parametrize("retry_after", [None, "not-a-delay"])
def test_github_release_download_uses_bounded_backoff_for_missing_or_malformed_retry_after(
    monkeypatch, tmp_path: Path, retry_after: str | None
) -> None:
    """Invalid server delay headers fall back to the bounded local backoff."""
    calls: list[int] = []

    def fake_urlopen(url: str, *, timeout: int):
        calls.append(timeout)
        headers = {} if retry_after is None else {"Retry-After": retry_after}
        raise HTTPError(url, 503, "temporarily unavailable", headers, None)

    sleeps: list[float] = []
    monkeypatch.setattr(registry, "urlopen", fake_urlopen)
    target = tmp_path / "cache" / "model.zip"
    with pytest.raises(HTTPError):
        registry._stream_download_url(
            "https://github.com/ll7/robot_sf_ll7/releases/download/tag/model.zip",
            target,
            max_attempts=3,
            backoff_seconds=0.5,
            max_backoff_seconds=2.0,
            sleep=sleeps.append,
        )

    assert calls == [60, 60, 60]
    assert sleeps == [0.5, 1.0]
    assert not target.exists()
    assert list(target.parent.glob("*.part")) == []


@pytest.mark.parametrize(
    ("retry_delta", "expected_delay"),
    [(timedelta(seconds=45), 45.0), (timedelta(seconds=-5), 0.0)],
    ids=["future-date", "past-date"],
)
def test_retry_after_http_date_uses_injected_clock(
    monkeypatch, tmp_path: Path, retry_delta: timedelta, expected_delay: float
) -> None:
    """HTTP-date retry headers honor a deterministic clock, including past dates."""
    current = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
    target = tmp_path / "cache" / "model.zip"
    target.parent.mkdir(parents=True)
    sleeps: list[float] = []
    calls: list[int] = []

    class _Response:
        def __init__(self) -> None:
            self._done = False

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def read(self, size: int) -> bytes:
            if self._done:
                return b""
            self._done = True
            return b"checkpoint"

    def fake_urlopen(url: str, *, timeout: int):
        calls.append(timeout)
        if len(calls) == 1:
            raise HTTPError(
                url,
                429,
                "rate limited",
                {"Retry-After": format_datetime(current + retry_delta, usegmt=True)},
                None,
            )
        return _Response()

    monkeypatch.setattr(registry, "urlopen", fake_urlopen)
    registry._stream_download_url(
        "https://github.com/ll7/robot_sf_ll7/releases/download/tag/model.zip",
        target,
        max_attempts=2,
        max_backoff_seconds=30.0,
        sleep=sleeps.append,
        clock=lambda: current,
    )

    assert calls == [60, 60]
    assert sleeps == [min(expected_delay, 30.0)]
    assert target.read_bytes() == b"checkpoint"


def test_github_release_cache_reuses_only_matching_sha256(monkeypatch, tmp_path: Path) -> None:
    """A corrupted cache entry is rejected before a verified replacement download."""
    payload = b"verified-checkpoint"
    source = tmp_path / "source"
    source.write_bytes(payload)
    expected_sha = registry._sha256(source)
    cache_dir = tmp_path / "cache"
    cached = cache_dir / "public_model" / "public_model-model.zip"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"corrupt")
    calls: list[str] = []

    class _Response:
        def __init__(self) -> None:
            self._done = False

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def read(self, size: int) -> bytes:
            if self._done:
                return b""
            self._done = True
            return payload

    def fake_urlopen(url: str, *, timeout: int):
        calls.append(url)
        return _Response()

    monkeypatch.setattr(registry, "urlopen", fake_urlopen)
    entry = {
        "model_id": "public_model",
        "github_release": {
            "url": "https://github.com/ll7/robot_sf_ll7/releases/download/tag/public_model-model.zip",
            "asset_name": "public_model-model.zip",
            "sha256": expected_sha,
        },
    }

    resolved = registry._download_from_github_release(entry, cache_dir=cache_dir)

    assert resolved == cached
    assert cached.read_bytes() == payload
    assert calls == [entry["github_release"]["url"]]


def test_github_release_exhaustion_is_marked_external_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    """Release retrieval failure stays a blocking external result at the registry boundary."""
    error = HTTPError(
        "https://github.com/ll7/robot_sf_ll7/releases/download/tag/model.zip", 503, "down", {}, None
    )

    def fail_stream(*args, **kwargs):
        raise error

    monkeypatch.setattr(registry, "_stream_download_url", fail_stream)
    entry = {
        "model_id": "public_model",
        "github_release": {
            "url": error.url,
            "asset_name": "model.zip",
            "sha256": "0" * 64,
        },
    }

    with pytest.raises(RuntimeError, match="external_unavailable"):
        registry._download_from_github_release(entry, cache_dir=tmp_path / "cache")


def test_resolve_model_path_rejects_bad_github_release_checksum(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Release downloads must match the registry checksum when one is recorded."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
version: 1
models:
  - model_id: public_model
    local_path: missing/model.zip
    github_release:
      url: https://github.com/ll7/robot_sf_ll7/releases/download/tag/public_model-model.zip
      asset_name: public_model-model.zip
      sha256: 0000000000000000000000000000000000000000000000000000000000000000
""".strip()
        + "\n",
        encoding="utf-8",
    )

    class _Response:
        """Context-manager response returning one invalid payload."""

        def __init__(self) -> None:
            self._done = False

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def read(self, size: int) -> bytes:
            if self._done:
                return b""
            self._done = True
            return b"wrong"

    monkeypatch.setattr(registry, "urlopen", lambda url, timeout: _Response())
    with pytest.raises(ValueError, match="Checksum mismatch"):
        registry.resolve_model_path(
            "public_model",
            registry_path=registry_path,
            cache_dir=tmp_path / "cache",
        )


def test_resolve_model_path_requires_github_release_checksum(tmp_path: Path) -> None:
    """Release-backed registry rows must fail closed without an expected checksum."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
version: 1
models:
  - model_id: public_model
    local_path: missing/model.zip
    github_release:
      url: https://github.com/ll7/robot_sf_ll7/releases/download/tag/public_model-model.zip
      asset_name: public_model-model.zip
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="github_release.sha256 is required"):
        registry.resolve_model_path(
            "public_model",
            registry_path=registry_path,
            cache_dir=tmp_path / "cache",
        )


def test_resolve_model_path_rejects_untrusted_github_release_url(tmp_path: Path) -> None:
    """Release-backed downloads should only use trusted GitHub HTTPS URLs."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
version: 1
models:
  - model_id: public_model
    local_path: missing/model.zip
    github_release:
      url: https://example.com/ll7/robot_sf_ll7/releases/download/tag/public_model-model.zip
      asset_name: public_model-model.zip
      sha256: 0000000000000000000000000000000000000000000000000000000000000000
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="https://github.com URL"):
        registry.resolve_model_path(
            "public_model",
            registry_path=registry_path,
            cache_dir=tmp_path / "cache",
        )


def test_resolve_model_path_rejects_release_asset_path_traversal(tmp_path: Path) -> None:
    """Release asset names should not be able to escape the model cache directory."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
version: 1
models:
  - model_id: public_model
    local_path: missing/model.zip
    github_release:
      url: https://github.com/ll7/robot_sf_ll7/releases/download/tag/public_model-model.zip
      asset_name: ../public_model-model.zip
      sha256: 0000000000000000000000000000000000000000000000000000000000000000
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="asset_name must be a file name"):
        registry.resolve_model_path(
            "public_model",
            registry_path=registry_path,
            cache_dir=tmp_path / "cache",
        )


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


def test_download_from_wandb_rejects_missing_run_metadata(monkeypatch, tmp_path: Path) -> None:
    """Download helper should fail clearly when the registry row lacks W&B location metadata."""
    monkeypatch.setattr(registry, "wandb", SimpleNamespace(Api=lambda: None))
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
