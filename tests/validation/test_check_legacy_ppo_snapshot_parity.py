"""Tests for legacy PPO snapshot parity inventory and smoke checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from gymnasium import spaces

from scripts.validation import check_legacy_ppo_snapshot_parity as checker


def _write_registry(path: Path, entries: list[dict]) -> None:
    path.write_text(yaml.safe_dump({"version": 1, "models": entries}), encoding="utf-8")


def _entry(model_id: str, *, release: dict | None = None) -> dict:
    return {
        "model_id": model_id,
        "local_path": f"output/model_cache/{model_id}/model.zip",
        "tags": ["ppo", "br06"],
        "github_release": release
        if release is not None
        else {
            "repo": "ll7/robot_sf_ll7",
            "tag": "artifact/models-2026-05-registry-v1",
            "asset_name": f"{model_id}-model.zip",
            "sha256": "a" * 64,
            "size_bytes": 123,
        },
    }


def test_inventory_marks_supported_legacy_registry_entries() -> None:
    """Repo registry should keep all supported legacy PPO rows durable."""
    repo_root = Path(__file__).resolve().parents[2]
    rows = checker.build_inventory(
        repo_root=repo_root,
        registry_path=repo_root / "model" / "registry.yaml",
    )

    supported = {
        row.identifier: row
        for row in rows
        if row.identifier in checker.SUPPORTED_LEGACY_PPO_MODEL_IDS
    }

    assert set(supported) == set(checker.SUPPORTED_LEGACY_PPO_MODEL_IDS)
    assert {row.status for row in supported.values()} == {"supported"}
    assert all(row.durable_uri.startswith("https://github.com/") for row in supported.values())


def test_inventory_fails_supported_entry_without_durable_release(tmp_path: Path) -> None:
    """Supported legacy rows fail closed when release checksum metadata is absent."""
    registry_path = tmp_path / "registry.yaml"
    model_id = checker.SUPPORTED_LEGACY_PPO_MODEL_IDS[0]
    _write_registry(registry_path, [_entry(model_id, release={"asset_name": "model.zip"})])

    rows = checker.build_inventory(repo_root=tmp_path, registry_path=registry_path)

    target = next(row for row in rows if row.identifier == model_id)
    assert target.status == "unsupported_missing_durable_pointer"
    assert "sha256" in target.reason


def test_inventory_records_root_local_snapshots_as_unsupported(tmp_path: Path) -> None:
    """Root-local debug checkpoints should be explicit unsupported rows."""
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "run_043.zip").write_text("debug checkpoint", encoding="utf-8")
    registry_path = tmp_path / "registry.yaml"
    _write_registry(
        registry_path, [_entry(model_id) for model_id in checker.SUPPORTED_LEGACY_PPO_MODEL_IDS]
    )

    rows = checker.build_inventory(repo_root=tmp_path, registry_path=registry_path)

    root_row = next(row for row in rows if row.identifier == "model/run_043.zip")
    assert root_row.status == "unsupported_local_only"
    assert root_row.source == "root_local_file"
    assert "no durable registry provenance" in root_row.reason


def test_cli_json_inventory_reports_ok_for_repo_registry(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Inventory-only JSON output should be parseable and pass for the repo registry."""
    repo_root = Path(__file__).resolve().parents[2]

    exit_code = checker.main(
        [
            "--repo-root",
            str(repo_root),
            "--registry-path",
            str(repo_root / "model" / "registry.yaml"),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema"] == "legacy_ppo_snapshot_parity.v1"
    assert payload["status"] == "ok"
    assert payload["blocking_rows"] == []


def test_run_model_step_smoke_uses_factory_model_prediction_and_gymnasium_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The opt-in smoke should load, predict, and step through Gymnasium once."""
    registry_path = tmp_path / "registry.yaml"
    model_id = checker.SUPPORTED_LEGACY_PPO_MODEL_IDS[0]
    model_path = tmp_path / "model.zip"
    model_path.write_text("checkpoint", encoding="utf-8")
    _write_registry(registry_path, [_entry(model_id)])

    class FakeModel:
        def predict(self, obs, deterministic=True):
            assert deterministic is True
            assert np.asarray(obs).shape == (2,)
            return np.array([0.1, 0.0], dtype=np.float32), None

    class FakeEnv:
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        def reset(self, seed=None):
            assert seed == 3469
            return np.array([0.0, 0.5], dtype=np.float32), {"seed": seed}

        def step(self, action):
            assert self.action_space.contains(action)
            return np.array([0.1, 0.5], dtype=np.float32), 1.25, False, False, {"ok": True}

        def close(self):
            pass

    monkeypatch.setattr(checker, "resolve_model_path", lambda *args, **kwargs: model_path)
    monkeypatch.setattr(checker, "_make_smoke_env", lambda seed: FakeEnv())
    monkeypatch.setattr(checker, "_load_ppo_model", lambda path: FakeModel())

    report = checker.run_model_step_smoke(
        model_id=model_id,
        repo_root=tmp_path,
        registry_path=registry_path,
        allow_download=False,
        seed=3469,
    )

    assert report.status == "ok"
    assert report.action_shape == (2,)
    assert report.reward_type == "float"
    assert report.terminated_type == "bool"
    assert report.truncated_type == "bool"
    assert report.info_keys == ("ok",)
