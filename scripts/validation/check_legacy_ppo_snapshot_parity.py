#!/usr/bin/env python3
"""Inventory and optionally smoke-test legacy PPO snapshots against Gymnasium.

The default mode is intentionally cheap: it verifies that legacy PPO checkpoints
that should remain supported are represented by durable registry entries, it
records root-local debug snapshots as explicitly unsupported, and byte-matches
the in-tree source files against their recorded SHA-256 values. Pass
``--verify-release-hydration`` to resolve every durable legacy checkpoint from
its GitHub Release into an isolated cache and verify the hydrated bytes. Pass
``--smoke-model-id`` for a hydrated/downloadable checkpoint smoke that loads the
model, predicts one action, and executes one current ``make_robot_env`` step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from collections.abc import Mapping
from contextlib import chdir
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from robot_sf.models.registry import (
    _download_from_github_release,
    load_registry,
    resolve_model_path,
)

SUPPORTED_LEGACY_PPO_MODEL_IDS = (
    "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
    "ppo_expert_br06_v2_15m_all_maps_20260302T152332",
    "ppo_expert_br06_v2_15m_all_maps_20260303T074433",
)

# Root-local debug checkpoints that are STILL not durable. Phase A of issue #6268
# published the previous four entries (run_023, run_043, and the two retrained
# zips) as durable registry artifacts, so this guard is now empty for those. The
# mechanism is retained intentionally: any root-local snapshot that lacks durable
# registry provenance belongs here, and the inventory will report it as
# unsupported_local_only.
UNSUPPORTED_ROOT_LOCAL_PPO_SNAPSHOTS: dict[str, str] = {}


@dataclass(frozen=True)
class DurableLegacyCheckpoint:
    """One Phase-A durable legacy checkpoint declared in this validator.

    The checkpoint resolves through the registry by ``model_id``. ``source_paths``
    are the in-tree files whose bytes must match the recorded checksums.

    - ``single_file``: one in-tree file; its SHA-256 must equal
      ``github_release.sha256`` (the published asset is a byte copy).
    - ``multi_file_bundle``: several in-tree files published as one coherent
      bundle asset. ``github_release.per_file_sha256`` records each component
      checksum; every component must byte-match. The optional release-hydration
      check also reads the downloaded archive without extracting it.
    """

    model_id: str
    source_paths: tuple[str, ...]
    kind: str
    display_path: str = ""


# Phase A of issue #6268: legacy checkpoints now published as durable GitHub
# release artifacts (tag artifact/legacy-models-2026-07-registry-v1) with
# immutable SHA-256 checksums and registry entries. The four PPO debug/retrained
# zips below previously lived in UNSUPPORTED_ROOT_LOCAL_PPO_SNAPSHOTS; the
# pedestrian zips and the ga3c_cadrl triplet had no durable classification.
# Phase A flips all of them to supported/durable. Nothing is deleted, moved, or
# renamed. Single-file registry local paths point at the resolver's ignored release
# cache; GA3C retains its existing in-tree .meta resolver path because SA-CADRL
# requires the adjacent TensorFlow checkpoint files.
DURABLE_LEGACY_CHECKPOINTS: tuple[DurableLegacyCheckpoint, ...] = (
    DurableLegacyCheckpoint(
        model_id="legacy_ppo_run_023",
        source_paths=("model/run_023.zip",),
        kind="single_file",
    ),
    DurableLegacyCheckpoint(
        model_id="legacy_ppo_run_043",
        source_paths=("model/run_043.zip",),
        kind="single_file",
    ),
    DurableLegacyCheckpoint(
        model_id="legacy_ppo_retrained_10m_2024_09_17",
        source_paths=("model/ppo_model_retrained_10m_2024-09-17.zip",),
        kind="single_file",
    ),
    DurableLegacyCheckpoint(
        model_id="legacy_ppo_retrained_10m_2025_02_01",
        source_paths=("model/ppo_model_retrained_10m_2025-02-01.zip",),
        kind="single_file",
    ),
    DurableLegacyCheckpoint(
        model_id="legacy_ppo_pedestrian_ped_01",
        source_paths=("model/pedestrian/ppo_ped_01.zip",),
        kind="single_file",
    ),
    DurableLegacyCheckpoint(
        model_id="legacy_ppo_pedestrian_ped_02",
        source_paths=("model/pedestrian/ppo_ped_02.zip",),
        kind="single_file",
    ),
    DurableLegacyCheckpoint(
        model_id="legacy_ppo_pedestrian_headon",
        source_paths=("model/pedestrian/ppo_headon.zip",),
        kind="single_file",
    ),
    DurableLegacyCheckpoint(
        model_id="legacy_ppo_pedestrian_intersection",
        source_paths=("model/pedestrian/ppo_intersection.zip",),
        kind="single_file",
    ),
    DurableLegacyCheckpoint(
        model_id="legacy_ppo_pedestrian_corner",
        source_paths=("model/pedestrian/ppo_corner.zip",),
        kind="single_file",
    ),
    DurableLegacyCheckpoint(
        model_id="ga3c_cadrl_iros18",
        source_paths=(
            "model/ga3c_cadrl/IROS18/network_01900000.data-00000-of-00001",
            "model/ga3c_cadrl/IROS18/network_01900000.index",
            "model/ga3c_cadrl/IROS18/network_01900000.meta",
        ),
        kind="multi_file_bundle",
    ),
)


@dataclass(frozen=True)
class SnapshotRow:
    """One legacy snapshot support-status row."""

    identifier: str
    status: str
    source: str
    local_path: str
    durable_uri: str
    reason: str
    checksum_status: str = ""
    checksum_detail: str = ""


@dataclass(frozen=True)
class SmokeReport:
    """One optional model-load and Gymnasium step smoke result."""

    model_id: str
    status: str
    model_path: str
    observation_space: str
    action_space: str
    action_shape: tuple[int, ...]
    reward_type: str
    terminated_type: str
    truncated_type: str
    info_keys: tuple[str, ...]


def _release_uri(entry: Mapping[str, Any]) -> str:
    release = entry.get("github_release")
    if not isinstance(release, Mapping):
        return ""
    url = str(release.get("url") or "").strip()
    if url:
        return url
    repo = str(release.get("repo") or "").strip()
    tag = str(release.get("tag") or "").strip()
    asset_name = str(release.get("asset_name") or "").strip()
    if repo and tag and asset_name:
        return f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"
    return ""


_MOVING_RELEASE_VERSION_ALIASES = frozenset({"best", "best-success", "current", "latest"})


def _durable_release_reason(
    entry: Mapping[str, Any], *, require_immutable_version: bool = False
) -> str:
    """Return why a registry entry lacks the required durable-release metadata."""
    release = entry.get("github_release")
    if not isinstance(release, Mapping):
        return "missing github_release pointer"
    missing = [
        field
        for field in ("asset_name", "sha256", "size_bytes")
        if str(release.get(field) or "").strip() == ""
    ]
    if missing:
        return f"github_release missing {', '.join(missing)}"
    if not _release_uri(entry):
        return "github_release needs url or repo/tag/asset_name"
    if require_immutable_version:
        version = str(release.get("version") or "").strip()
        if not version:
            return "github_release missing immutable version pin"
        if version.casefold() in _MOVING_RELEASE_VERSION_ALIASES:
            return f"github_release version must be immutable, not {version!r}"
    return ""


def _sha256(path: Path) -> str:
    """Return the lowercase-hex SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_model_path_from_repo_root(
    model_id: str,
    *,
    repo_root: Path,
    registry_path: Path,
    allow_download: bool,
    cache_dir: Path | None = None,
) -> Path:
    """Resolve a model while anchoring registry-relative paths at ``repo_root``.

    ``resolve_model_path`` intentionally interprets a relative ``local_path`` from
    the current working directory. This validator exposes ``--repo-root``, so it
    must temporarily use that selected checkout rather than the caller's directory.
    """
    with chdir(repo_root):
        return resolve_model_path(
            model_id,
            registry_path=registry_path,
            allow_download=allow_download,
            cache_dir=cache_dir,
        )


def _resolve_single_file_release_hydration(
    model_id: str,
    *,
    repo_root: Path,
    registry_path: Path,
    cache_dir: Path | None,
) -> Path:
    """Resolve one release asset without reusing the worktree's default local cache.

    ``resolve_model_path`` checks an entry's relative ``local_path`` before its
    explicit ``cache_dir``. When the caller requests an isolated hydration cache,
    anchor that relative lookup at the isolated cache as well; otherwise a
    previously hydrated worktree could make the release proof pass without
    exercising the requested cache.
    """
    resolution_root = repo_root if cache_dir is None else cache_dir.resolve()
    resolution_root.mkdir(parents=True, exist_ok=True)
    with chdir(resolution_root):
        resolved = resolve_model_path(
            model_id,
            registry_path=registry_path,
            allow_download=True,
            cache_dir=cache_dir,
        )
    if cache_dir is not None and not resolved.resolve().is_relative_to(cache_dir.resolve()):
        raise ValueError(
            f"release hydration resolved outside the requested cache {cache_dir}: {resolved}"
        )
    return resolved


def _verify_single_file(entry: Mapping[str, Any], *, resolved: Path) -> tuple[str, str]:
    """Byte-match a single-file durable checkpoint against its recorded SHA-256."""
    release = entry.get("github_release")
    if not isinstance(release, Mapping):
        return "missing_release", "entry has no github_release mapping"
    expected = str(release.get("sha256") or "").strip().lower()
    if not expected:
        return "missing_sha256", "github_release.sha256 is empty"
    observed = _sha256(resolved)
    if observed != expected:
        return (
            "checksum_mismatch",
            f"observed {observed} != recorded {expected} for {resolved}",
        )
    return "verified", f"single-file byte-match for {resolved}"


def _verify_multi_file_bundle_sources(
    entry: Mapping[str, Any], *, repo_root: Path
) -> tuple[str, str]:
    """Byte-match each in-tree component of a multi-file bundle checkpoint."""
    release = entry.get("github_release")
    if not isinstance(release, Mapping):
        return "missing_release", "entry has no github_release mapping"
    per_file = release.get("per_file_sha256")
    if not isinstance(per_file, Mapping) or not per_file:
        return "missing_per_file_sha256", "github_release.per_file_sha256 is absent"
    bundle_files = release.get("bundle_files")
    if not isinstance(bundle_files, list) or not bundle_files:
        return "missing_bundle_files", "github_release.bundle_files is absent"
    details: list[str] = []
    for rel in bundle_files:
        component = repo_root / str(rel)
        if not component.exists():
            return "missing_component", f"bundle component absent in-tree: {rel}"
        key = Path(str(rel)).name
        expected = str(per_file.get(key) or "").strip().lower()
        if not expected:
            return "missing_component_sha256", f"no per_file_sha256 entry for {key}"
        observed = _sha256(component)
        if observed != expected:
            return (
                "checksum_mismatch",
                f"component {key}: observed {observed} != recorded {expected}",
            )
        details.append(f"{key}=ok")
    return "verified", f"multi-file bundle byte-match ({'; '.join(details)})"


def _validate_bundle_manifest(
    entry: Mapping[str, Any], *, expected_sources: tuple[str, ...]
) -> tuple[str, str]:
    """Require the registry bundle metadata to cover the declared checkpoint exactly."""
    release = entry.get("github_release")
    if not isinstance(release, Mapping):
        return "missing_release", "entry has no github_release mapping"
    bundle_files = release.get("bundle_files")
    if not isinstance(bundle_files, list):
        return "missing_bundle_files", "github_release.bundle_files is absent"
    configured_sources = tuple(str(path) for path in bundle_files)
    if len(configured_sources) != len(set(configured_sources)):
        return "duplicate_component", "github_release.bundle_files contains duplicate paths"
    if set(configured_sources) != set(expected_sources):
        return (
            "bundle_manifest_mismatch",
            "github_release.bundle_files must exactly match the declared checkpoint components",
        )
    per_file = release.get("per_file_sha256")
    if not isinstance(per_file, Mapping):
        return "missing_per_file_sha256", "github_release.per_file_sha256 is absent"
    expected_names = {Path(path).name for path in expected_sources}
    configured_names = {str(name) for name in per_file}
    if configured_names != expected_names:
        return (
            "bundle_manifest_mismatch",
            "github_release.per_file_sha256 must exactly match the declared checkpoint components",
        )
    return "verified", "bundle manifest covers every declared checkpoint component"


def _verify_multi_file_bundle_resolution(
    checkpoint: DurableLegacyCheckpoint,
    *,
    entry: Mapping[str, Any],
    repo_root: Path,
    registry_path: Path,
    cache_dir: Path | None,
) -> tuple[str, str]:
    """Verify that a multi-file checkpoint still resolves to its in-tree local path.

    Phase A intentionally leaves GA3C-CADRL's TensorFlow ``.meta`` path in-tree: callers need
    the adjacent ``.data`` and ``.index`` files, which an unextracted release archive cannot
    provide. This check proves that the registry resolver preserves that contract before release
    hydration verifies the archive itself.
    """
    raw_local_path = entry.get("local_path")
    if not raw_local_path:
        return "missing_local_path", "multi-file checkpoint has no registry local_path"

    expected = Path(str(raw_local_path))
    if not expected.is_absolute():
        expected = repo_root / expected
    expected = expected.resolve()
    declared_sources = {(repo_root / source).resolve() for source in checkpoint.source_paths}
    if expected not in declared_sources:
        return (
            "resolver_path_mismatch",
            "registry local_path must name one declared multi-file checkpoint component",
        )

    try:
        resolved = _resolve_model_path_from_repo_root(
            checkpoint.model_id,
            repo_root=repo_root,
            registry_path=registry_path,
            allow_download=False,
            cache_dir=cache_dir,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return "unresolved", f"registry resolution failed: {exc}"
    if not resolved.exists():
        return "unresolved", f"registry resolution returned a missing path: {resolved}"
    if resolved.resolve() != expected:
        return (
            "resolver_path_mismatch",
            f"resolver returned {resolved}, expected the in-tree checkpoint path {expected}",
        )
    return "verified", f"resolver returned the in-tree checkpoint path {resolved}"


def _verify_hydrated_multi_file_bundle(
    entry: Mapping[str, Any], *, archive_path: Path
) -> tuple[str, str]:
    """Verify a release-hydrated GA3C archive and each expected member checksum."""
    release = entry.get("github_release")
    if not isinstance(release, Mapping):
        return "missing_release", "entry has no github_release mapping"
    expected_archive_sha = str(release.get("sha256") or "").strip().lower()
    if not expected_archive_sha:
        return "missing_sha256", "github_release.sha256 is empty"
    observed_archive_sha = _sha256(archive_path)
    if observed_archive_sha != expected_archive_sha:
        return (
            "checksum_mismatch",
            f"archive observed {observed_archive_sha} != recorded {expected_archive_sha}",
        )
    per_file = release.get("per_file_sha256")
    if not isinstance(per_file, Mapping) or not per_file:
        return "missing_per_file_sha256", "github_release.per_file_sha256 is absent"

    expected = {str(name): str(value).strip().lower() for name, value in per_file.items()}
    status, observed = _archive_component_digests(archive_path, expected_names=set(expected))
    if status:
        return status, str(observed)
    assert isinstance(observed, dict)
    missing = sorted(set(expected) - set(observed))
    if missing:
        return "missing_component", f"hydrated archive missing: {', '.join(missing)}"
    mismatches = [name for name, digest in observed.items() if digest != expected[name]]
    if mismatches:
        return "checksum_mismatch", f"hydrated archive components mismatch: {', '.join(mismatches)}"
    return "verified", f"hydrated bundle byte-match ({'; '.join(sorted(observed))})"


def _archive_component_digests(
    archive_path: Path, *, expected_names: set[str]
) -> tuple[str, dict[str, str] | str]:
    """Read expected regular-file digests from a gzip tar archive without extracting it."""
    observed: dict[str, str] = {}
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if name not in expected_names:
                    continue
                if name in observed:
                    return "duplicate_component", f"archive contains duplicate component {name}"
                handle = archive.extractfile(member)
                if handle is None:
                    return "unreadable_component", f"cannot read archive component {name}"
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                observed[name] = digest.hexdigest()
    except (OSError, tarfile.TarError) as exc:
        return "invalid_bundle", f"cannot read hydrated archive {archive_path}: {exc}"
    return "", observed


def _verify_durable_checkpoint_sources(
    checkpoint: DurableLegacyCheckpoint,
    *,
    entry: Mapping[str, Any],
    repo_root: Path,
    registry_path: Path,
    cache_dir: Path | None,
) -> tuple[str, str]:
    """Verify a durable checkpoint's in-tree bytes and resolver contract."""
    if checkpoint.kind == "single_file":
        return _verify_single_file(entry, resolved=repo_root / checkpoint.source_paths[0])
    if checkpoint.kind != "multi_file_bundle":
        return "unknown_kind", f"unsupported checkpoint kind: {checkpoint.kind}"

    resolution_status, resolution_detail = _verify_multi_file_bundle_resolution(
        checkpoint,
        entry=entry,
        repo_root=repo_root,
        registry_path=registry_path,
        cache_dir=cache_dir,
    )
    if resolution_status != "verified":
        return resolution_status, resolution_detail
    manifest_status, manifest_detail = _validate_bundle_manifest(
        entry, expected_sources=checkpoint.source_paths
    )
    if manifest_status != "verified":
        return manifest_status, manifest_detail
    source_status, source_detail = _verify_multi_file_bundle_sources(entry, repo_root=repo_root)
    if source_status != "verified":
        return source_status, source_detail
    return "verified", f"{resolution_detail}; {source_detail}"


def _verify_durable_checkpoint(
    checkpoint: DurableLegacyCheckpoint,
    *,
    entry: Mapping[str, Any],
    repo_root: Path,
    registry_path: Path,
    verify_release_hydration: bool = False,
    cache_dir: Path | None = None,
) -> tuple[str, str]:
    """Byte-match a durable legacy checkpoint and optionally prove release hydration.

    Returns a ``(checksum_status, detail)`` tuple. ``checksum_status`` is
    ``"verified"`` when the in-tree source bytes match the recorded checksums,
    and when requested, the release-hydrated cache bytes match too.
    """
    source_status, source_detail = _verify_durable_checkpoint_sources(
        checkpoint,
        entry=entry,
        repo_root=repo_root,
        registry_path=registry_path,
        cache_dir=cache_dir,
    )
    if source_status != "verified" or not verify_release_hydration:
        return source_status, source_detail

    try:
        if checkpoint.kind == "multi_file_bundle":
            # Phase A deliberately retains GA3C's in-tree ``.meta`` local_path so
            # SA-CADRL receives a usable checkpoint prefix. Download the published
            # bundle directly for provenance verification without changing that
            # runtime resolver contract.
            resolved = _download_from_github_release(
                dict(entry), cache_dir=cache_dir, allow_download=True
            )
        else:
            resolved = _resolve_single_file_release_hydration(
                checkpoint.model_id,
                repo_root=repo_root,
                registry_path=registry_path,
                cache_dir=cache_dir,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return "unresolved", f"release resolution failed: {exc}"
    if checkpoint.kind == "single_file":
        status, detail = _verify_single_file(entry, resolved=resolved)
    else:
        status, detail = _verify_hydrated_multi_file_bundle(entry, archive_path=resolved)
    if status != "verified":
        return status, detail
    return "verified", f"source verified; {source_detail}; {detail}"


def build_inventory(
    *,
    repo_root: Path,
    registry_path: Path,
    supported_model_ids: tuple[str, ...] = SUPPORTED_LEGACY_PPO_MODEL_IDS,
    durable_checkpoints: tuple[DurableLegacyCheckpoint, ...] = DURABLE_LEGACY_CHECKPOINTS,
    unsupported_root_local: Mapping[str, str] = UNSUPPORTED_ROOT_LOCAL_PPO_SNAPSHOTS,
    verify_release_hydration: bool = False,
    cache_dir: Path | None = None,
) -> tuple[SnapshotRow, ...]:
    """Return support-status rows for legacy PPO snapshots."""
    registry = load_registry(registry_path)
    rows: list[SnapshotRow] = []
    for model_id in supported_model_ids:
        entry = registry.get(model_id)
        if entry is None:
            rows.append(
                SnapshotRow(
                    identifier=model_id,
                    status="missing_registry_entry",
                    source="model_registry",
                    local_path="",
                    durable_uri="",
                    reason="supported legacy checkpoint is absent from model/registry.yaml",
                )
            )
            continue
        reason = _durable_release_reason(entry)
        rows.append(
            SnapshotRow(
                identifier=model_id,
                status="supported" if not reason else "unsupported_missing_durable_pointer",
                source="model_registry",
                local_path=str(entry.get("local_path") or ""),
                durable_uri=_release_uri(entry),
                reason=reason or "durable GitHub release pointer with checksum",
            )
        )

    for checkpoint in durable_checkpoints:
        entry = registry.get(checkpoint.model_id)
        if entry is None:
            rows.append(
                SnapshotRow(
                    identifier=checkpoint.model_id,
                    status="missing_registry_entry",
                    source="model_registry",
                    local_path="",
                    durable_uri="",
                    reason=(
                        f"durable legacy checkpoint {checkpoint.model_id} is absent from "
                        "model/registry.yaml"
                    ),
                )
            )
            continue
        release_reason = _durable_release_reason(entry, require_immutable_version=True)
        durable_uri = _release_uri(entry)
        if release_reason:
            rows.append(
                SnapshotRow(
                    identifier=checkpoint.model_id,
                    status="unsupported_missing_durable_pointer",
                    source="model_registry",
                    local_path=str(entry.get("local_path") or ""),
                    durable_uri=durable_uri,
                    reason=release_reason,
                )
            )
            continue
        checksum_status, checksum_detail = _verify_durable_checkpoint(
            checkpoint,
            entry=entry,
            repo_root=repo_root,
            registry_path=registry_path,
            verify_release_hydration=verify_release_hydration,
            cache_dir=cache_dir,
        )
        if checksum_status == "verified":
            status = "supported"
            proof = (
                "release cache byte-match"
                if verify_release_hydration
                else "in-tree source byte-match"
            )
            reason = f"durable GitHub release pointer; {proof} against recorded checksum"
        else:
            status = "unsupported_checksum_mismatch"
            reason = f"{checksum_status}: {checksum_detail}"
        display_path = checkpoint.display_path or str(entry.get("local_path") or "")
        rows.append(
            SnapshotRow(
                identifier=checkpoint.model_id,
                status=status,
                source="model_registry",
                local_path=display_path,
                durable_uri=durable_uri,
                reason=reason,
                checksum_status=checksum_status,
                checksum_detail=checksum_detail,
            )
        )

    for rel_path, reason in unsupported_root_local.items():
        rows.append(
            SnapshotRow(
                identifier=rel_path,
                status="unsupported_local_only",
                source="root_local_file"
                if (repo_root / rel_path).exists()
                else "root_local_missing",
                local_path=rel_path,
                durable_uri="",
                reason=reason,
            )
        )
    return tuple(rows)


def _load_ppo_model(model_path: Path):
    """Load a Stable-Baselines3 PPO checkpoint for the opt-in smoke path."""
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("stable_baselines3 is required for --smoke-model-id") from exc
    return PPO.load(str(model_path), env=None, device="cpu", print_system_info=False)


def _make_smoke_env(seed: int):
    """Create the current Gymnasium robot env for the opt-in smoke path."""
    from robot_sf.gym_env.environment_factory import make_robot_env
    from robot_sf.gym_env.unified_config import RobotSimulationConfig

    return make_robot_env(config=RobotSimulationConfig(map_id="uni_campus_big"), seed=seed)


def run_model_step_smoke(
    *,
    model_id: str,
    repo_root: Path,
    registry_path: Path,
    allow_download: bool,
    seed: int,
) -> SmokeReport:
    """Load a PPO checkpoint and execute one current Gymnasium robot-env step."""

    model_path = _resolve_model_path_from_repo_root(
        model_id,
        repo_root=repo_root,
        registry_path=registry_path,
        allow_download=allow_download,
    )
    env = _make_smoke_env(seed)
    try:
        model = _load_ppo_model(model_path)
        obs, _reset_info = env.reset(seed=seed)
        raw_action, _state = model.predict(obs, deterministic=True)
        action = np.asarray(raw_action, dtype=getattr(env.action_space, "dtype", np.float32))
        if not env.action_space.contains(action):
            raise ValueError(
                f"Model action is outside current env action_space: action={action!r}, "
                f"space={env.action_space!r}"
            )
        step_obs, reward, terminated, truncated, info = env.step(action)
        if not env.observation_space.contains(step_obs):
            raise ValueError("Step observation is outside current env observation_space")
        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(
            truncated, (bool, np.bool_)
        ):
            raise TypeError("Gymnasium step must return bool terminated/truncated flags")
        if not isinstance(info, Mapping):
            raise TypeError("Gymnasium step info must be a mapping")
        return SmokeReport(
            model_id=model_id,
            status="ok",
            model_path=str(model_path),
            observation_space=type(env.observation_space).__name__,
            action_space=type(env.action_space).__name__,
            action_shape=tuple(action.shape),
            reward_type=type(reward).__name__,
            terminated_type=type(terminated).__name__,
            truncated_type=type(truncated).__name__,
            info_keys=tuple(sorted(str(key) for key in info.keys())),
        )
    finally:
        env.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--registry-path", type=Path, default=Path("model/registry.yaml"))
    parser.add_argument(
        "--smoke-model-id",
        action="append",
        default=[],
        help="Registry model id to load and step once. May be repeated.",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--verify-release-hydration",
        action="store_true",
        help="Download each durable legacy release asset into --cache-dir and verify its bytes.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Ignored model-cache root used by --verify-release-hydration.",
    )
    parser.add_argument("--seed", type=int, default=3469)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the inventory and optional smoke checks."""
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    registry_path = args.registry_path
    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path
    cache_dir = args.cache_dir.resolve() if args.cache_dir is not None else None

    inventory = build_inventory(
        repo_root=repo_root,
        registry_path=registry_path,
        verify_release_hydration=bool(args.verify_release_hydration),
        cache_dir=cache_dir,
    )
    smoke_reports = [
        run_model_step_smoke(
            model_id=model_id,
            repo_root=repo_root,
            registry_path=registry_path,
            allow_download=bool(args.allow_download),
            seed=args.seed,
        )
        for model_id in args.smoke_model_id
    ]
    durable_ids = {checkpoint.model_id for checkpoint in DURABLE_LEGACY_CHECKPOINTS}
    blocking_rows = [
        row
        for row in inventory
        if (row.identifier in SUPPORTED_LEGACY_PPO_MODEL_IDS or row.identifier in durable_ids)
        and row.status != "supported"
    ]
    payload = {
        "schema": "legacy_ppo_snapshot_parity.v1",
        "status": "failed" if blocking_rows else "ok",
        "inventory": [asdict(row) for row in inventory],
        "smoke": [asdict(report) for report in smoke_reports],
        "blocking_rows": [asdict(row) for row in blocking_rows],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "legacy PPO snapshot parity: "
            f"status={payload['status']} supported={len(SUPPORTED_LEGACY_PPO_MODEL_IDS)} "
            f"durable={len(DURABLE_LEGACY_CHECKPOINTS)} smoke={len(smoke_reports)} "
            f"blocking={len(blocking_rows)}"
        )
        for row in inventory:
            extra = f" [{row.checksum_status}]" if row.checksum_status else ""
            print(f"- {row.status}: {row.identifier}{extra} ({row.reason})")
    return 2 if blocking_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
