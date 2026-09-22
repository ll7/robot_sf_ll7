"""Focused contract tests for the offline checkpoint cold-load example."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "examples/advanced/42_offline_checkpoint_cold_load.py"
SPEC = importlib.util.spec_from_file_location("offline_cold_load_example", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _copy_fixture(tmp_path: Path, mutate=None) -> Path:
    fixture = tmp_path / "fixture"
    shutil.copytree(MODULE.DEFAULT_FIXTURE.parent, fixture)
    path = fixture / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutate:
        mutate(payload, fixture)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _error(path: Path) -> str:
    with pytest.raises(MODULE.ColdLoadError) as caught:
        MODULE.run_cold_load(path)
    return caught.value.code


def test_positive_load_is_direct_sb3_deterministic_and_cleans_root(tmp_path: Path) -> None:
    root = tmp_path / "isolated"
    report = MODULE.run_cold_load(MODULE.DEFAULT_FIXTURE, isolated_root=root, assert_isolated_cache=True)  # fmt: skip
    assert report["status"] == "pass"
    assert (report["load_mode"], report["source_config_status"], report["downloads_allowed"]) == ("direct_sb3", "unavailable", False)  # fmt: skip
    assert report["validation"]["parameters_finite"] and report["validation"]["feature_extractor_class"] == "stable_baselines3.common.torch_layers.FlattenExtractor"  # fmt: skip
    assert report["action"]["shape"] == [2] and report["action"]["within_bounds"] is True
    assert root.exists() is False


def test_missing_companion_state_fails_closed(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    (path.parent / "normalizer.json").unlink()
    assert _error(path) == "missing_companion_state"


def test_unavailable_dependency_fails_closed(tmp_path: Path, monkeypatch) -> None:
    def missing(_: str) -> str:
        raise MODULE.PackageNotFoundError("synthetic dependency")

    monkeypatch.setattr(MODULE, "package_version", missing)
    assert _error(_copy_fixture(tmp_path)) == "dependency_unavailable"


def test_hidden_cache_path_is_rejected(tmp_path: Path, monkeypatch) -> None:
    path = _copy_fixture(tmp_path)
    hidden = tmp_path / "ambient-cache" / "synthetic_ppo_policy.zip"
    hidden.parent.mkdir()
    shutil.copyfile(MODULE.DEFAULT_FIXTURE.parent / hidden.name, hidden)
    monkeypatch.setattr(MODULE, "resolve_model_path", lambda *args, **kwargs: hidden)
    assert _error(path) == "cache_resolution_failed"


def _invalid_normalizer(payload: dict, fixture: Path) -> None:
    normalizer = fixture / "normalizer.json"
    state = json.loads(normalizer.read_text(encoding="utf-8"))
    state["variance"][0] = 0.0
    normalizer.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(normalizer.read_bytes()).hexdigest()
    payload["artifact"]["normalizer"].update(
        {"sha256": digest, "size_bytes": normalizer.stat().st_size}
    )
    payload["normalizer"]["sha256"] = digest


def _out_of_bounds_normalizer(payload: dict, fixture: Path) -> None:
    normalizer = fixture / "normalizer.json"
    normalizer.write_bytes(normalizer.read_bytes().replace(b"[0.0, 0.0, 0.0, 0.0]", b"[-100.0, -100.0, -100.0, -100.0]"))  # fmt: skip
    digest = hashlib.sha256(normalizer.read_bytes()).hexdigest()
    payload["artifact"]["normalizer"].update({"sha256": digest, "size_bytes": normalizer.stat().st_size})  # fmt: skip
    payload["normalizer"]["sha256"] = digest


def _resize(section: str):
    return lambda payload, _: payload[section].update(
        {"shape": [3], "low": [-1.0] * 3, "high": [1.0] * 3}
    )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda payload, _: payload.update({"schema": "offline_cold_load_manifest.v2"}),
            "unsupported_schema",
        ),
        (_resize("observation"), "normalizer_shape_mismatch"),
        (_resize("action"), "action_contract_mismatch"),
        (
            lambda payload, _: payload["artifact"]["checkpoint"].update(
                {"sha256": hashlib.sha256(b"not-the-checkpoint").hexdigest()}
            ),
            "digest_mismatch",
        ),
        (
            lambda payload, _: payload["dependencies"][0].update({"version": "0.0.0"}),
            "dependency_manifest_invalid",
        ),
        (_out_of_bounds_normalizer, "normalized_observation_out_of_bounds"),
        (
            lambda payload, _: payload["loader"].update({"allow_download": True}),
            "cache_policy_invalid",
        ),
        (
            lambda payload, _: payload["loader"].update({"load_mode": "native"}),
            "loader_mode_invalid",
        ),
        (
            lambda payload, _: payload["source_config"].update({"status": "verified"}),
            "source_identity_invalid",
        ),
        (lambda payload, _: payload.update({"unexpected": True}), "manifest_schema_invalid"),
    ],
)
def test_manifest_and_contract_failures_are_explicit(tmp_path: Path, mutate, expected: str) -> None:
    assert _error(_copy_fixture(tmp_path, mutate)) == expected


def test_stale_alias_is_not_resolved() -> None:
    with pytest.raises(MODULE.ColdLoadError) as caught:
        MODULE.resolve_fixture("latest")
    assert caught.value.code == "stale_alias"


def test_invalid_companion_state_is_checked_after_its_digest(tmp_path: Path) -> None:
    assert _error(_copy_fixture(tmp_path, _invalid_normalizer)) == "normalizer_invalid"
