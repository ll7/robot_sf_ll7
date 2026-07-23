"""Tests for the CI-visible model preflight entry point (issue #6189)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

import scripts.dev.prefetch_models as prefetch_cli
from scripts.dev.prefetch_models import main, prefetch_models


def test_prefetch_models_reports_downloaded_and_present_status(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    """A first run stages the model (downloaded); a second run reuses the cache."""
    staged_path = tmp_path / "cache" / "demo" / "model.zip"
    staged_path.parent.mkdir(parents=True)
    staged_path.write_bytes(b"payload")
    expected_sha = "deadbeef" * 8

    call_log: list[str] = []

    def _fake_prefetch(model_id, *, registry_path=None, cache_dir):
        call_log.append(model_id)
        return staged_path, expected_sha

    monkeypatch.setattr(prefetch_cli, "prefetch_model", _fake_prefetch)

    # Second call is detected as cached-reuse because the cache dir already holds
    # the asset file before prefetch runs.
    results = prefetch_models(
        ["demo"],
        registry=None,
        cache_dir=tmp_path / "cache",
    )

    assert results == [
        {
            "model_id": "demo",
            "ok": True,
            "cached_reused": True,
            "status": "present",
            "path": str(staged_path),
            "sha256": expected_sha,
        }
    ]
    assert call_log == ["demo"]


def test_prefetch_models_records_loud_setup_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A failing stage surfaces a non-ok row with the exception type and message."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    def _boom(model_id, *, registry_path=None, cache_dir):
        raise RuntimeError("after 3 attempt(s): network down")

    monkeypatch.setattr(prefetch_cli, "prefetch_model", _boom)

    results = prefetch_models(["demo"], registry=None, cache_dir=cache_dir)

    assert len(results) == 1
    row = results[0]
    assert row["ok"] is False
    assert row["status"] == "prefetch_failed"
    assert row["error_type"] == "RuntimeError"
    assert "after 3 attempt" in row["error"]


def test_main_writes_manifest_and_exits_nonzero_on_failure(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    """main writes a JSON manifest and exits 1 when any model fails to stage."""
    cache_dir = tmp_path / "cache"
    manifest = tmp_path / "manifest.json"

    def _boom(model_id, *, registry_path=None, cache_dir):
        raise RuntimeError("network down")

    monkeypatch.setattr(prefetch_cli, "prefetch_model", _boom)

    rc = main(
        [
            "--model-id",
            "demo",
            "--cache-dir",
            str(cache_dir),
            "--manifest",
            str(manifest),
            "--format",
            "json",
        ]
    )

    assert rc == 1
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema"] == "model_prefetch_manifest.v1"
    assert payload["all_ok"] is False
    assert payload["results"][0]["model_id"] == "demo"

    captured = capsys.readouterr().out
    assert json.loads(captured)["all_ok"] is False


def test_main_exits_zero_when_all_models_stage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """main exits 0 and marks all_ok when every model stages successfully."""
    cache_dir = tmp_path / "cache"
    manifest = tmp_path / "manifest.json"

    def _ok(model_id, *, registry_path=None, cache_dir):
        path = cache_dir / model_id / "model.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return path, "ab" * 32

    monkeypatch.setattr(prefetch_cli, "prefetch_model", _ok)

    rc = main(
        [
            "--model-id",
            "predictive_proxy_selected_v2_full",
            "--cache-dir",
            str(cache_dir),
            "--manifest",
            str(manifest),
        ]
    )

    assert rc == 0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["all_ok"] is True
    assert payload["results"][0]["status"] == "downloaded"


def test_main_requires_model_id() -> None:
    """At least one --model-id is required so the preflight never no-ops silently."""
    with pytest.raises(SystemExit):
        main([])
