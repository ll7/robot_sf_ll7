"""Tests for the reusable model-asset preflight (issue #6189).

These tests exercise the shared preflight primitive with a fake registry entry
and a fake ``urlopen`` -- no real network. They cover:

* A: the preflight resolves + checksum-verifies a required model into the cache.
* C: the preflight fails loudly after bounded retries when the asset is
  unavailable (never a silent fallback).

plus the atomic-download and config-extraction guarantees the primitive relies
on. The end-to-end offline exact-repeat determinism proof (test B) lives in
``tests/benchmark/test_exact_repeat_executor.py`` next to the campaign fixtures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.error import URLError

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.models import registry
from robot_sf.models.preflight import (
    ModelPreflightError,
    preflight_model,
    preflight_models,
    required_model_ids_for_config,
)


class _Response:
    """Minimal context-manager HTTP response returning ``payload`` in chunks."""

    def __init__(self, payload: bytes) -> None:
        self._data = payload
        self._offset = 0

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def _write_registry(tmp_path: Path, expected_sha: str) -> Path:
    """Write a single-entry registry pointing at a GitHub release asset."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        f"""
version: 1
models:
  - model_id: preflight_model
    local_path: missing/model.pt
    github_release:
      repo: ll7/robot_sf_ll7
      tag: artifact/models-test
      asset_name: preflight_model-model.pt
      sha256: {expected_sha}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return registry_path


# --- Test A: preflight resolves + checksum-verifies into cache ---------------


def test_preflight_resolves_and_checksum_verifies_into_cache(monkeypatch, tmp_path: Path) -> None:
    """A required model is staged into the cache with its registry SHA-256 verified."""
    payload = b"predictive-checkpoint-bytes"
    source = tmp_path / "source.pt"
    source.write_bytes(payload)
    expected_sha = registry._sha256(source)
    registry_path = _write_registry(tmp_path, expected_sha)

    calls: list[str] = []

    def _fake_urlopen(url: str, timeout: int):
        calls.append(url)
        return _Response(payload)

    monkeypatch.setattr(registry, "urlopen", _fake_urlopen)

    cache_dir = tmp_path / "cache"
    resolved = preflight_model(
        "preflight_model",
        registry_path=registry_path,
        cache_dir=cache_dir,
        backoff_seconds=0.0,
    )

    assert resolved == cache_dir / "preflight_model" / "preflight_model-model.pt"
    assert resolved.read_bytes() == payload
    assert registry._sha256(resolved) == expected_sha
    assert calls, "preflight must fetch the asset when it is not already cached"

    # Second call reuses the verified cache and performs no further download.
    calls.clear()
    again = preflight_model(
        "preflight_model",
        registry_path=registry_path,
        cache_dir=cache_dir,
        backoff_seconds=0.0,
    )
    assert again == resolved
    assert calls == [], "a cached, checksum-valid asset must not be re-downloaded"

    # The timed loop disables downloads, but must still resolve this verified cache hit.
    monkeypatch.setenv("ROBOT_SF_DISABLE_MODEL_DOWNLOADS", "1")
    offline = registry.resolve_model_path(
        "preflight_model", registry_path=registry_path, cache_dir=cache_dir
    )
    assert offline == resolved
    assert calls == [], "offline resolution must reuse the verified cache without networking"


def test_preflight_models_stages_each_required_id(monkeypatch, tmp_path: Path) -> None:
    """``preflight_models`` returns a verified cache path for every required id."""
    payload = b"checkpoint"
    source = tmp_path / "s.pt"
    source.write_bytes(payload)
    expected_sha = registry._sha256(source)
    registry_path = _write_registry(tmp_path, expected_sha)
    monkeypatch.setattr(registry, "urlopen", lambda url, timeout: _Response(payload))

    resolved = preflight_models(
        ["preflight_model", "preflight_model"],  # duplicate is de-duplicated
        registry_path=registry_path,
        cache_dir=tmp_path / "cache",
        backoff_seconds=0.0,
    )
    assert list(resolved) == ["preflight_model"]
    assert resolved["preflight_model"].is_file()


# --- Atomicity: a failed download never leaves a partial file at the cache path


def test_failed_download_leaves_no_partial_cache_file(monkeypatch, tmp_path: Path) -> None:
    """A mid-stream download failure must not leave a corrupt file at the cache path."""
    registry_path = _write_registry(tmp_path, "0" * 64)

    class _BrokenResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            raise URLError("connection reset mid-stream")

    monkeypatch.setattr(registry, "urlopen", lambda url, timeout: _BrokenResponse())

    cache_dir = tmp_path / "cache"
    with pytest.raises(ModelPreflightError):
        preflight_model(
            "preflight_model",
            registry_path=registry_path,
            cache_dir=cache_dir,
            max_attempts=1,
            backoff_seconds=0.0,
        )

    cached_path = cache_dir / "preflight_model" / "preflight_model-model.pt"
    assert not cached_path.exists(), "no partial artifact may remain at the cache path"
    # No leftover temp part files either.
    assert list((cache_dir / "preflight_model").glob("*.part")) == []


def test_bad_checksum_never_publishes_a_cache_file(monkeypatch, tmp_path: Path) -> None:
    """Checksum verification occurs before atomic publication to the shared cache."""
    target = tmp_path / "cache" / "model.pt"
    target.parent.mkdir()
    monkeypatch.setattr(registry, "urlopen", lambda url, timeout: _Response(b"untrusted"))

    with pytest.raises(ValueError, match="Checksum mismatch"):
        registry._stream_download_url(
            "https://github.com/ll7/robot_sf_ll7/releases/download/tag/model.pt",
            target,
            expected_sha256="0" * 64,
        )

    assert not target.exists()
    assert list(target.parent.glob("*.part")) == []


def test_runtime_download_guard_blocks_late_model_resolution(monkeypatch, tmp_path: Path) -> None:
    """Timed execution cannot turn a cache miss into an on-demand download."""
    registry_path = _write_registry(tmp_path, "0" * 64)
    calls: list[str] = []
    monkeypatch.setattr(registry, "urlopen", lambda url, timeout: calls.append(url))
    monkeypatch.setenv("ROBOT_SF_DISABLE_MODEL_DOWNLOADS", "1")

    with pytest.raises(FileNotFoundError, match="downloads are disabled"):
        registry.resolve_model_path(
            "preflight_model", registry_path=registry_path, cache_dir=tmp_path / "cache"
        )

    assert calls == []


# --- Test C: preflight fails loudly after bounded retries --------------------


def test_preflight_fails_loudly_after_bounded_retries(monkeypatch, tmp_path: Path) -> None:
    """Persistent unavailability raises ModelPreflightError after exactly N attempts."""
    registry_path = _write_registry(tmp_path, "0" * 64)

    attempts: list[str] = []

    def _always_fail(url: str, timeout: int):
        attempts.append(url)
        raise URLError("network unavailable")

    monkeypatch.setattr(registry, "urlopen", _always_fail)

    slept: list[float] = []

    with pytest.raises(ModelPreflightError) as excinfo:
        preflight_model(
            "preflight_model",
            registry_path=registry_path,
            cache_dir=tmp_path / "cache",
            max_attempts=3,
            backoff_seconds=0.5,
            sleep=slept.append,
        )

    # Exactly max_attempts downloads were attempted (bounded, not infinite).
    assert len(attempts) == 3
    # Backoff slept between attempts only (N-1 times), with linear growth.
    assert slept == [0.5, 1.0]
    assert "after 3 attempt(s)" in str(excinfo.value)


def test_preflight_rejects_non_positive_max_attempts(tmp_path: Path) -> None:
    """A non-positive attempt budget is a programming error, surfaced immediately."""
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        preflight_model("preflight_model", max_attempts=0)


# --- Config extraction -------------------------------------------------------


def test_required_model_ids_for_config_honours_predictive_gate() -> None:
    """Predictive-foresight model id is required only when its gate flag is enabled."""
    enabled = {
        "model_id": "ppo_expert",
        "predictive_foresight_enabled": True,
        "predictive_foresight_model_id": "predictive_proxy_selected_v2_full",
    }
    assert required_model_ids_for_config(enabled) == [
        "ppo_expert",
        "predictive_proxy_selected_v2_full",
    ]

    disabled = dict(enabled, predictive_foresight_enabled=False)
    assert required_model_ids_for_config(disabled) == ["ppo_expert"]


def test_required_model_ids_for_config_walks_nested_arms() -> None:
    """Model ids are collected from nested planner/arm structures and de-duplicated."""
    config = {
        "planners": [
            {"algo_config": {"model_id": "a"}},
            {"algo_config": {"sacadrl_model_id": "b", "model_id": "a"}},
            {"algo_config": {"predictive_model_id": "c"}},
        ]
    }
    assert required_model_ids_for_config(config) == ["a", "b", "c"]
