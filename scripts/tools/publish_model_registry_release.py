#!/usr/bin/env python3
"""Stage and publish model-registry artifacts as GitHub release assets."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256
from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.models import registry as model_registry

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", type=Path, default=Path("model/registry.yaml"))
    parser.add_argument("--repo", default="ll7/robot_sf_ll7")
    parser.add_argument(
        "--tag",
        required=True,
        help="GitHub release tag, for example artifact/models-2026-05-registry-v1.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Limit publication to one model id. May be repeated. Defaults to all W&B-backed rows.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Directory for staged model assets, metadata, manifest, and checksums.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Use registry download metadata to hydrate missing local model files before staging.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("output/model_cache"),
        help="Cache directory used when --download-missing needs to hydrate an artifact.",
    )
    parser.add_argument(
        "--execute-upload",
        action="store_true",
        help="Run gh release upload after staging. Default is dry-run plan only.",
    )
    parser.add_argument(
        "--create-release",
        action="store_true",
        help="Create the release before upload. Use with --execute-upload for a new tag.",
    )
    parser.add_argument(
        "--update-registry",
        action="store_true",
        help="Write github_release pointers into the registry after staging/upload planning.",
    )
    parser.add_argument(
        "--allow-registry-update-without-upload",
        action="store_true",
        help=(
            "Allow --update-registry without --execute-upload for offline fixture generation. "
            "Default is fail-closed so unpublished release pointers are not written."
        ),
    )
    parser.add_argument(
        "--registry-output",
        type=Path,
        default=None,
        help="Optional output path for the updated registry. Defaults to --registry-path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the publication plan JSON.",
    )
    return parser


def _load_registry_data(path: Path) -> dict[str, Any]:
    """Load the raw registry YAML document."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in registry: {path}")
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError(f"Registry is missing a models list: {path}")
    return payload


def _has_wandb_reference(entry: dict[str, Any]) -> bool:
    """Return whether a registry entry currently references W&B as a model source."""
    return bool(
        entry.get("wandb_artifact_path")
        or entry.get("wandb_run_path")
        or (entry.get("wandb_entity") and entry.get("wandb_project") and entry.get("wandb_run_id"))
    )


def _safe_component(value: str) -> str:
    """Make a stable release-asset component from a model id or filename."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip(".-")
    if not cleaned:
        raise ValueError(f"Could not build a safe release asset component from: {value!r}")
    return cleaned


def _asset_name_for(entry: dict[str, Any], source_path: Path) -> str:
    """Return the model release asset name for one registry entry."""
    model_id = _safe_component(str(entry["model_id"]))
    file_name = _safe_component(str(entry.get("wandb_file") or source_path.name))
    return f"{model_id}-{file_name}"


def _source_path_for(
    entry: dict[str, Any],
    *,
    registry_path: Path,
    download_missing: bool,
    cache_dir: Path,
) -> tuple[Path | None, str | None]:
    """Resolve the local source model path, optionally hydrating from registry metadata."""
    local_path = entry.get("local_path")
    if isinstance(local_path, str) and local_path.strip():
        path = Path(local_path)
        if not path.is_absolute():
            path = get_repository_root() / path
        if path.exists():
            return path, None

    if not download_missing:
        return None, "missing local_path; rerun with --download-missing to hydrate from registry"

    try:
        path = model_registry.resolve_model_path(
            str(entry["model_id"]),
            registry_path=registry_path,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        return None, f"download failed: {exc}"
    return path, None


def _copy_model_asset(source_path: Path, target_path: Path) -> None:
    """Copy one model artifact into the release staging directory."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def _model_metadata(
    entry: dict[str, Any], *, asset_name: str, sha256: str, size_bytes: int
) -> dict:
    """Build a small per-model metadata payload for release publication."""
    keep_keys = [
        "model_id",
        "display_name",
        "config_path",
        "commit",
        "tags",
        "notes",
        "wandb_artifact_path",
        "wandb_run_path",
        "wandb_run_id",
        "wandb_entity",
        "wandb_project",
        "wandb_file",
    ]
    metadata = {key: entry.get(key) for key in keep_keys if key in entry}
    metadata["release_asset"] = {
        "asset_name": asset_name,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }
    return metadata


def _release_url(repo: str, tag: str, asset_name: str) -> str:
    """Return the public GitHub release asset URL."""
    return f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"


def _write_text(path: Path, body: str) -> None:
    """Write UTF-8 text, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _stage_assets(args: argparse.Namespace, registry_data: dict[str, Any]) -> dict[str, Any]:
    """Stage model assets and return the publication plan."""
    repo_root = get_repository_root()
    staging_dir = args.staging_dir or repo_root / "output" / "model_registry_release" / args.tag
    staging_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = set(args.model_id)
    models = [
        entry
        for entry in registry_data["models"]
        if isinstance(entry, dict)
        and (not selected_ids or entry.get("model_id") in selected_ids)
        and (_has_wandb_reference(entry) or selected_ids)
    ]
    if selected_ids:
        found = {str(entry.get("model_id")) for entry in models}
        missing = sorted(selected_ids - found)
        if missing:
            raise KeyError(f"Requested model ids are not in the registry: {missing}")

    now = datetime.now(UTC).isoformat(timespec="seconds")
    published: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    checksum_lines: list[str] = []
    upload_assets: list[str] = []
    registry_updates: dict[str, dict[str, Any]] = {}

    for entry in models:
        model_id = str(entry["model_id"])
        source_path, error = _source_path_for(
            entry,
            registry_path=args.registry_path,
            download_missing=bool(args.download_missing),
            cache_dir=args.cache_dir,
        )
        if source_path is None:
            skipped.append({"model_id": model_id, "reason": str(error)})
            continue

        asset_name = _asset_name_for(entry, source_path)
        staged_model = staging_dir / asset_name
        _copy_model_asset(source_path, staged_model)
        sha256 = _sha256(staged_model)
        size_bytes = staged_model.stat().st_size
        metadata_name = f"{_safe_component(model_id)}-metadata.json"
        metadata_path = staging_dir / metadata_name
        metadata = _model_metadata(
            entry,
            asset_name=asset_name,
            sha256=sha256,
            size_bytes=size_bytes,
        )
        _write_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")

        checksum_lines.append(f"{sha256}  {asset_name}")
        checksum_lines.append(f"{_sha256(metadata_path)}  {metadata_name}")
        upload_assets.extend([str(staged_model), str(metadata_path)])
        release_pointer = {
            "repo": args.repo,
            "tag": args.tag,
            "asset_name": asset_name,
            "url": _release_url(args.repo, args.tag, asset_name),
            "sha256": sha256,
            "size_bytes": size_bytes,
            "metadata_asset": metadata_name,
            "published_at_utc": now,
        }
        registry_updates[model_id] = release_pointer
        published.append(
            {
                "model_id": model_id,
                "source_path": str(source_path),
                "asset_name": asset_name,
                "metadata_asset": metadata_name,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "release_asset_url": release_pointer["url"],
            }
        )

    manifest = {
        "schema_version": "robot-sf-model-registry-release.v1",
        "created_at_utc": now,
        "repo": args.repo,
        "tag": args.tag,
        "registry_path": str(args.registry_path),
        "models": published,
        "skipped": skipped,
    }
    manifest_path = staging_dir / "manifest.json"
    checksums_path = staging_dir / "SHA256SUMS"
    notes_path = staging_dir / "release_notes.md"
    _write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksum_lines.append(f"{_sha256(manifest_path)}  manifest.json")
    _write_text(checksums_path, "\n".join(checksum_lines) + ("\n" if checksum_lines else ""))
    _write_text(
        notes_path,
        "\n".join(
            [
                "# Robot SF model registry artifacts",
                "",
                f"Registry source: `{args.registry_path}`",
                f"Published models: {len(published)}",
                "",
                "This release contains curated model artifacts copied from the model registry, "
                "per-model metadata JSON, `manifest.json`, and `SHA256SUMS`.",
                "",
            ]
        ),
    )
    upload_assets.extend([str(manifest_path), str(checksums_path)])

    upload_command = [
        "gh",
        "release",
        "upload",
        args.tag,
        *upload_assets,
        "--repo",
        args.repo,
        "--clobber",
    ]
    create_command = [
        "gh",
        "release",
        "create",
        args.tag,
        "--repo",
        args.repo,
        "--title",
        args.tag,
        "--notes-file",
        str(notes_path),
    ]
    return {
        "staging_dir": str(staging_dir),
        "manifest_path": str(manifest_path),
        "checksums_path": str(checksums_path),
        "release_notes_path": str(notes_path),
        "published": published,
        "skipped": skipped,
        "registry_updates": registry_updates,
        "create_release_command": create_command,
        "upload_command": upload_command,
    }


def _apply_registry_updates(
    registry_data: dict[str, Any],
    updates: dict[str, dict[str, Any]],
    *,
    output_path: Path,
) -> None:
    """Write GitHub release pointers back into the model registry."""
    for entry in registry_data["models"]:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("model_id"))
        if model_id in updates:
            entry["github_release"] = updates[model_id]
            entry["public_artifact_source"] = "github_release"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(registry_data, sort_keys=False), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the model registry publication workflow."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if (
        args.update_registry
        and not args.execute_upload
        and not args.allow_registry_update_without_upload
    ):
        parser.error("--update-registry requires --execute-upload or explicit offline override.")
    args.registry_path = args.registry_path.resolve()
    args.cache_dir = args.cache_dir.resolve()
    registry_data = _load_registry_data(args.registry_path)
    plan = _stage_assets(args, registry_data)

    if args.execute_upload:
        if args.create_release:
            subprocess.run(plan["create_release_command"], check=True)
        subprocess.run(plan["upload_command"], check=True)

    if args.update_registry:
        output_path = (args.registry_output or args.registry_path).resolve()
        _apply_registry_updates(registry_data, plan["registry_updates"], output_path=output_path)
        plan["updated_registry_path"] = str(output_path)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
