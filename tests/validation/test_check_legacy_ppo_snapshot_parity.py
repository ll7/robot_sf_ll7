"""Tests for legacy PPO snapshot parity inventory and smoke checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from gymnasium import spaces

from robot_sf.models import registry as model_registry
from robot_sf.models.registry import load_registry
from scripts.validation import check_legacy_ppo_snapshot_parity as checker


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _durable_entry(
    model_id: str, local_path: Path, *, sha256: str, kind: str = "single_file"
) -> dict:
    """Build a durable legacy registry entry pointing at a real local file."""
    entry: dict = {
        "model_id": model_id,
        "local_path": str(local_path),
        "tags": ["ppo", "legacy"],
        "github_release": {
            "repo": "ll7/robot_sf_ll7",
            "tag": "artifact/legacy-models-2026-07-registry-v1",
            "version": "v1",
            "asset_name": f"{model_id}.zip",
            "url": (
                f"https://github.com/ll7/robot_sf_ll7/releases/download/"
                f"artifact/legacy-models-2026-07-registry-v1/{model_id}.zip"
            ),
            "sha256": sha256,
            "size_bytes": local_path.stat().st_size,
        },
        "benchmark_promotion": {
            "claim_boundary": "legacy_non_track",
            "non_benchmark_reason": "legacy non-benchmark checkpoint",
        },
    }
    return entry


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

    rows = checker.build_inventory(
        repo_root=tmp_path,
        registry_path=registry_path,
        durable_checkpoints=(),
    )

    target = next(row for row in rows if row.identifier == model_id)
    assert target.status == "unsupported_missing_durable_pointer"
    assert "sha256" in target.reason


def test_inventory_marks_all_durable_legacy_checkpoints_supported_and_verified() -> None:
    """Every Phase-A durable legacy checkpoint must resolve, byte-match, and be durable."""
    repo_root = Path(__file__).resolve().parents[2]
    rows = checker.build_inventory(
        repo_root=repo_root,
        registry_path=repo_root / "model" / "registry.yaml",
    )
    by_id = {row.identifier: row for row in rows}

    assert {cp.model_id for cp in checker.DURABLE_LEGACY_CHECKPOINTS}.issubset(by_id)
    for cp in checker.DURABLE_LEGACY_CHECKPOINTS:
        row = by_id[cp.model_id]
        assert row.status == "supported", (cp.model_id, row)
        assert row.checksum_status == "verified", (cp.model_id, row)
        assert row.durable_uri.startswith("https://github.com/"), (cp.model_id, row)


def test_durable_legacy_entries_declare_legacy_non_track_claim_boundary() -> None:
    """Durable legacy registry entries must declare the legacy_non_track boundary."""
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_registry(repo_root / "model" / "registry.yaml")

    for cp in checker.DURABLE_LEGACY_CHECKPOINTS:
        promotion = registry[cp.model_id].get("benchmark_promotion")
        assert promotion is not None, cp.model_id
        assert promotion.get("claim_boundary") == "legacy_non_track", cp.model_id
        assert promotion.get("non_benchmark_reason"), cp.model_id


@pytest.mark.parametrize(
    ("version", "expected_reason"),
    [(None, "missing immutable version pin"), ("latest", "must be immutable")],
)
def test_durable_legacy_inventory_rejects_unpinned_release_versions(
    tmp_path: Path, version: str | None, expected_reason: str
) -> None:
    """Legacy release entries must pin a non-moving immutable version."""
    source = tmp_path / "source.zip"
    source.write_bytes(b"checkpoint")
    model_id = "legacy_ppo_synthetic_version_pin"
    entry = _durable_entry(model_id, source, sha256=_sha256(source))
    release = entry["github_release"]
    if version is None:
        release.pop("version")
    else:
        release["version"] = version
    registry_path = tmp_path / "registry.yaml"
    _write_registry(registry_path, [entry])

    rows = checker.build_inventory(
        repo_root=tmp_path,
        registry_path=registry_path,
        supported_model_ids=(),
        durable_checkpoints=(
            checker.DurableLegacyCheckpoint(model_id, ("source.zip",), "single_file"),
        ),
    )

    assert len(rows) == 1
    assert rows[0].status == "unsupported_missing_durable_pointer"
    assert expected_reason in rows[0].reason


def test_durable_legacy_recorded_checksums_match_in_tree_sha256() -> None:
    """Recorded durable checksums must equal an independent in-tree SHA-256 recomputation."""
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_registry(repo_root / "model" / "registry.yaml")

    for cp in checker.DURABLE_LEGACY_CHECKPOINTS:
        release = registry[cp.model_id]["github_release"]
        if cp.kind == "single_file":
            assert len(cp.source_paths) == 1, cp.model_id
            in_tree = repo_root / cp.source_paths[0]
            if not in_tree.is_file():
                # Phase B cutover (issue #6268): the in-tree binary is replaced
                # by a registry/release-backed stub, so there is no in-tree
                # source to recompute. Resolution byte-matching is covered by
                # test_durable_cutover_single_file_resolves_release_artifact.
                continue
            observed = _sha256(in_tree)
            assert observed == release["sha256"], cp.model_id
        elif cp.kind == "multi_file_bundle":
            per_file = release["per_file_sha256"]
            for rel in cp.source_paths:
                key = Path(rel).name
                observed = _sha256(repo_root / rel)
                assert observed == per_file[key], (cp.model_id, key)
        else:  # pragma: no cover - guard
            raise AssertionError(f"unexpected kind {cp.kind}")


def test_durable_legacy_entries_preserve_ga3c_checkpoint_prefix_contract() -> None:
    """Phase-A entries cache single files but retain GA3C's usable in-tree prefix."""
    repo_root = Path(__file__).resolve().parents[2]
    registry_path = repo_root / "model" / "registry.yaml"
    registry = load_registry(registry_path)

    for cp in checker.DURABLE_LEGACY_CHECKPOINTS:
        entry = registry[cp.model_id]
        release = entry["github_release"]
        if cp.kind == "multi_file_bundle":
            assert entry["local_path"] == "model/ga3c_cadrl/IROS18/network_01900000.meta"
        else:
            assert entry["local_path"] == (
                f"output/model_cache/{cp.model_id}/{release['asset_name']}"
            )


def test_release_hydration_uses_cache_and_verifies_downloaded_single_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hydration must use an isolated resolver cache, not an in-tree source path."""
    source = tmp_path / "source.zip"
    source.write_bytes(b"checkpoint")
    model_id = "legacy_ppo_synthetic_hydration"
    entry = _durable_entry(model_id, source, sha256=_sha256(source))
    entry["local_path"] = f"output/model_cache/{model_id}/{model_id}.zip"
    registry_path = tmp_path / "registry.yaml"
    _write_registry(registry_path, [entry])
    cache_dir = tmp_path / "cache"
    hydrated = cache_dir / model_id / f"{model_id}.zip"
    hydrated.parent.mkdir(parents=True)
    hydrated.write_bytes(source.read_bytes())
    calls: list[dict] = []

    def fake_resolve(*args, **kwargs):
        calls.append(kwargs)
        return hydrated

    monkeypatch.setattr(checker, "resolve_model_path", fake_resolve)
    status, detail = checker._verify_durable_checkpoint(
        checker.DurableLegacyCheckpoint(model_id, ("source.zip",), "single_file"),
        entry=entry,
        repo_root=tmp_path,
        registry_path=registry_path,
        verify_release_hydration=True,
        cache_dir=cache_dir,
    )

    assert status == "verified", detail
    assert calls == [
        {"registry_path": registry_path, "allow_download": True, "cache_dir": cache_dir}
    ]


def test_release_hydration_does_not_reuse_worktree_local_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit hydration cache must win over an existing worktree local_path."""
    repo_root = tmp_path / "repo"
    source = repo_root / "model" / "source.zip"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"published-checkpoint")
    model_id = "legacy_ppo_synthetic_isolated_hydration"
    entry = _durable_entry(model_id, source, sha256=_sha256(source))
    entry["local_path"] = f"output/model_cache/{model_id}/{model_id}.zip"
    registry_path = repo_root / "model" / "registry.yaml"
    _write_registry(registry_path, [entry])

    worktree_cached = repo_root / entry["local_path"]
    worktree_cached.parent.mkdir(parents=True)
    worktree_cached.write_bytes(source.read_bytes())
    cache_dir = tmp_path / "isolated-cache"
    downloads: list[Path] = []

    def fake_stream_download(_url: str, target: Path, *, expected_sha256: str = "") -> None:
        downloads.append(target)
        target.write_bytes(source.read_bytes())
        assert expected_sha256 == _sha256(source)

    monkeypatch.setattr(model_registry, "_stream_download_url", fake_stream_download)
    status, detail = checker._verify_durable_checkpoint(
        checker.DurableLegacyCheckpoint(model_id, ("model/source.zip",), "single_file"),
        entry=entry,
        repo_root=repo_root,
        registry_path=registry_path,
        verify_release_hydration=True,
        cache_dir=cache_dir,
    )

    expected_hydrated = cache_dir / model_id / f"{model_id}.zip"
    assert status == "verified", detail
    assert downloads == [expected_hydrated]
    assert expected_hydrated.read_bytes() == source.read_bytes()


def test_durable_cutover_single_file_resolves_release_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cut-over single file (Phase B) byte-matches the durable release artifact."""
    model_id = "legacy_ppo_synthetic_cutover"
    source = tmp_path / "published.zip"
    source.write_bytes(b"published-checkpoint")
    entry = _durable_entry(model_id, source, sha256=_sha256(source))
    entry["local_path"] = f"output/model_cache/{model_id}/{model_id}.zip"
    registry_path = tmp_path / "registry.yaml"
    _write_registry(registry_path, [entry])

    # The in-tree binary is removed (cut over to a stub); the release artifact
    # is hydrated into an isolated cache.
    cache_dir = tmp_path / "release-cache"
    hydrated = cache_dir / model_id / f"{model_id}.zip"
    hydrated.parent.mkdir(parents=True)
    hydrated.write_bytes(source.read_bytes())

    def fake_hydrate(*args, **kwargs):
        return hydrated

    monkeypatch.setattr(checker, "_resolve_single_file_release_hydration", fake_hydrate)
    checkpoint = checker.DurableLegacyCheckpoint(
        model_id=model_id,
        source_paths=("model/run_023.zip",),
        kind="single_file",
        cutover=True,
    )
    status, detail = checker._verify_durable_checkpoint_sources(
        checkpoint,
        entry=entry,
        repo_root=tmp_path,
        registry_path=registry_path,
        cache_dir=cache_dir,
    )
    assert status == "verified", detail
    assert "byte-match" in detail


def test_durable_cutover_single_file_fails_closed_on_unresolvable_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cut-over single file with an unresolvable release fails closed."""
    model_id = "legacy_ppo_synthetic_cutover_unresolvable"
    source = tmp_path / "published.zip"
    source.write_bytes(b"published-checkpoint")
    entry = _durable_entry(model_id, source, sha256=_sha256(source))
    registry_path = tmp_path / "registry.yaml"
    _write_registry(registry_path, [entry])

    def fake_hydrate(*args, **kwargs):
        raise RuntimeError("release asset unavailable")

    monkeypatch.setattr(checker, "_resolve_single_file_release_hydration", fake_hydrate)
    checkpoint = checker.DurableLegacyCheckpoint(
        model_id=model_id,
        source_paths=("model/run_023.zip",),
        kind="single_file",
        cutover=True,
    )
    status, detail = checker._verify_durable_checkpoint_sources(
        checkpoint,
        entry=entry,
        repo_root=tmp_path,
        registry_path=registry_path,
        cache_dir=None,
    )
    assert status == "unresolved"
    assert "release resolution failed" in detail


def test_verify_durable_checkpoint_detects_checksum_mismatch(tmp_path: Path) -> None:
    """A wrong recorded checksum must flip a durable row to unsupported_checksum_mismatch."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"checkpoint-bytes-that-do-not-match")
    real_sha = _sha256(model_path)
    wrong_sha = "0" * 64
    assert real_sha != wrong_sha
    model_id = "legacy_ppo_synthetic_mismatch"
    entry = _durable_entry(model_id, model_path, sha256=wrong_sha)
    registry_path = tmp_path / "registry.yaml"
    _write_registry(registry_path, [entry])

    checkpoint = checker.DurableLegacyCheckpoint(
        model_id=model_id,
        source_paths=(str(model_path),),
        kind="single_file",
    )
    status, detail = checker._verify_durable_checkpoint(
        checkpoint,
        entry=entry,
        repo_root=tmp_path,
        registry_path=registry_path,
    )
    assert status == "checksum_mismatch"
    assert wrong_sha in detail


def test_inventory_reports_missing_durable_source_without_crashing(tmp_path: Path) -> None:
    """A missing durable source should be a structured fail-closed inventory row."""
    model_id = "legacy_ppo_synthetic_missing_source"
    source = tmp_path / "source.zip"
    source.write_bytes(b"checkpoint")
    entry = _durable_entry(model_id, source, sha256=_sha256(source))
    source.unlink()
    registry_path = tmp_path / "registry.yaml"
    _write_registry(registry_path, [entry])

    rows = checker.build_inventory(
        repo_root=tmp_path,
        registry_path=registry_path,
        supported_model_ids=(),
        durable_checkpoints=(
            checker.DurableLegacyCheckpoint(model_id, ("source.zip",), "single_file"),
        ),
    )

    assert len(rows) == 1
    assert rows[0].status == "unsupported_checksum_mismatch"
    assert rows[0].checksum_status == "missing_component"
    assert "not a regular file" in rows[0].checksum_detail


def test_durable_ga3c_bundle_rejects_an_incomplete_manifest() -> None:
    """The GA3C release manifest cannot silently omit a declared checkpoint component."""
    repo_root = Path(__file__).resolve().parents[2]
    registry_path = repo_root / "model" / "registry.yaml"
    entry = load_registry(registry_path)["ga3c_cadrl_iros18"]
    release = dict(entry["github_release"])
    release["bundle_files"] = release["bundle_files"][:-1]
    entry = {**entry, "github_release": release}
    checkpoint = next(
        item for item in checker.DURABLE_LEGACY_CHECKPOINTS if item.model_id == "ga3c_cadrl_iros18"
    )

    status, detail = checker._verify_durable_checkpoint(
        checkpoint,
        entry=entry,
        repo_root=repo_root,
        registry_path=registry_path,
    )

    assert status == "bundle_manifest_mismatch"
    assert "exactly match" in detail


def test_durable_ga3c_bundle_resolves_its_in_tree_checkpoint_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GA3C bundle must retain its usable TensorFlow checkpoint resolver path."""
    repo_root = Path(__file__).resolve().parents[2]
    registry_path = repo_root / "model" / "registry.yaml"
    entry = load_registry(registry_path)["ga3c_cadrl_iros18"]
    checkpoint = next(
        item for item in checker.DURABLE_LEGACY_CHECKPOINTS if item.model_id == "ga3c_cadrl_iros18"
    )
    original_resolve = checker.resolve_model_path
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def tracked_resolve(*args: object, **kwargs: object) -> Path:
        calls.append((args, kwargs))
        return original_resolve(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(checker, "resolve_model_path", tracked_resolve)
    status, detail = checker._verify_durable_checkpoint(
        checkpoint,
        entry=entry,
        repo_root=repo_root,
        registry_path=registry_path,
    )

    assert status == "verified", detail
    assert "resolver returned the in-tree checkpoint path" in detail
    assert calls == [
        (
            ("ga3c_cadrl_iros18",),
            {"registry_path": registry_path, "allow_download": False, "cache_dir": None},
        )
    ]


def test_unsupported_root_local_guard_still_classifies_synthetic_entries(
    tmp_path: Path,
) -> None:
    """The extended UNSUPPORTED_ROOT_LOCAL guard still classifies injected snapshots."""
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "debug.zip").write_text("debug checkpoint", encoding="utf-8")
    registry_path = tmp_path / "registry.yaml"
    _write_registry(
        registry_path,
        [_entry(model_id) for model_id in checker.SUPPORTED_LEGACY_PPO_MODEL_IDS],
    )

    rows = checker.build_inventory(
        repo_root=tmp_path,
        registry_path=registry_path,
        durable_checkpoints=(),
        unsupported_root_local={"model/debug.zip": "synthetic unsupported snapshot"},
    )

    debug_row = next(row for row in rows if row.identifier == "model/debug.zip")
    assert debug_row.status == "unsupported_local_only"
    assert debug_row.source == "root_local_file"
    assert "synthetic unsupported snapshot" in debug_row.reason


def test_no_root_local_legacy_zips_remain_in_unsupported_guard() -> None:
    """Phase A flipped the four root-local PPO snapshots out of the unsupported guard."""
    previously_unsupported = {
        "model/run_023.zip",
        "model/run_043.zip",
        "model/ppo_model_retrained_10m_2024-09-17.zip",
        "model/ppo_model_retrained_10m_2025-02-01.zip",
    }
    assert checker.UNSUPPORTED_ROOT_LOCAL_PPO_SNAPSHOTS == {}
    durable_sources = {
        source for cp in checker.DURABLE_LEGACY_CHECKPOINTS for source in cp.source_paths
    }
    # Every previously-unsupported root-local PPO snapshot is now a durable source.
    assert previously_unsupported.issubset(durable_sources)


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
    durable_ids = {cp.model_id for cp in checker.DURABLE_LEGACY_CHECKPOINTS}
    durable_rows = [row for row in payload["inventory"] if row["identifier"] in durable_ids]
    assert len(durable_rows) == len(durable_ids)
    assert all(row["durable_uri"].startswith("https://github.com/") for row in durable_rows)


def test_cli_release_hydration_requires_an_empty_isolated_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Hydration proof must not reuse an existing worktree or prior release cache."""
    repo_root = Path(__file__).resolve().parents[2]

    with pytest.raises(SystemExit) as missing_cache:
        checker.main(["--repo-root", str(repo_root), "--verify-release-hydration"])
    assert missing_cache.value.code == 2

    nonempty_cache = tmp_path / "nonempty-cache"
    nonempty_cache.mkdir()
    (nonempty_cache / "already-hydrated.zip").write_bytes(b"cached")
    with pytest.raises(SystemExit) as reused_cache:
        checker.main(
            [
                "--repo-root",
                str(repo_root),
                "--verify-release-hydration",
                "--cache-dir",
                str(nonempty_cache),
            ]
        )
    assert reused_cache.value.code == 2

    empty_cache = tmp_path / "empty-cache"
    captured: dict[str, object] = {}

    def fake_build_inventory(**kwargs: object) -> tuple[checker.SnapshotRow, ...]:
        captured.update(kwargs)
        return ()

    monkeypatch.setattr(checker, "build_inventory", fake_build_inventory)
    assert (
        checker.main(
            [
                "--repo-root",
                str(repo_root),
                "--verify-release-hydration",
                "--cache-dir",
                str(empty_cache),
                "--json",
            ]
        )
        == 0
    )
    assert captured["verify_release_hydration"] is True
    assert captured["cache_dir"] == empty_cache.resolve()


def test_cli_inventory_honors_explicit_repo_root_outside_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An explicit repo root must anchor GA3C's relative resolver path."""
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)

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
    assert payload["blocking_rows"] == []
    ga3c_row = next(row for row in payload["inventory"] if row["identifier"] == "ga3c_cadrl_iros18")
    assert ga3c_row["checksum_status"] == "verified"


def test_repo_root_resolution_normalizes_downloaded_relative_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A downloaded relative cache path remains usable after repo-root resolution."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    registry_path = repo_root / "model" / "registry.yaml"
    registry_path.parent.mkdir()
    registry_path.write_text("version: 1\nmodels: []\n", encoding="utf-8")
    caller_root = tmp_path / "caller"
    caller_root.mkdir()
    monkeypatch.chdir(caller_root)

    monkeypatch.setattr(
        checker,
        "resolve_model_path",
        lambda *args, **kwargs: Path("output/model_cache/model.zip"),
    )

    resolved = checker._resolve_model_path_from_repo_root(
        "synthetic-model",
        repo_root=repo_root,
        registry_path=registry_path,
        allow_download=True,
    )

    assert resolved == repo_root / "output/model_cache/model.zip"
    assert resolved.is_absolute()


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
            return (
                np.array([0.1, 0.5], dtype=np.float32),
                1.25,
                np.bool_(False),
                np.bool_(False),
                {"ok": True},
            )

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
