#!/usr/bin/env python3
"""Prepare public GitHub release pointers for model registry entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REGISTRY_PATH = Path("model/registry.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML as a mapping."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write YAML using the repository's registry style."""
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, width=88)


def _load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the release manifest."""
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected manifest mapping in {path}")
    models = manifest.get("models")
    if not isinstance(models, list):
        raise ValueError(f"Manifest {path} must contain a models list")
    return manifest


def _load_sha256s(path: Path) -> dict[str, str]:
    """Load a SHA256SUMS file keyed by asset name."""
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        digest, asset_name = stripped.split(maxsplit=1)
        checksums[asset_name.lstrip("*")] = digest
    return checksums


def _manifest_model_index(
    manifest: dict[str, Any],
    *,
    sha256s: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Return manifest rows indexed by model id, with checksum consistency checked."""
    indexed: dict[str, dict[str, Any]] = {}
    for item in manifest["models"]:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid manifest model row: {item!r}")
        model_id = str(item.get("model_id", "") or "").strip()
        asset_name = str(item.get("asset_name", "") or "").strip()
        sha256 = str(item.get("sha256", "") or "").strip().lower()
        if not model_id or not asset_name or not sha256:
            raise ValueError(f"Manifest row missing model_id, asset_name, or sha256: {item!r}")
        checksum_sha256 = sha256s.get(asset_name)
        if checksum_sha256 is not None and checksum_sha256.lower() != sha256:
            raise ValueError(
                f"Checksum mismatch for {asset_name}: manifest={sha256} "
                f"SHA256SUMS={checksum_sha256}"
            )
        indexed[model_id] = item
    return indexed


def _cache_file_name(entry: dict[str, Any], manifest_row: dict[str, Any]) -> str:
    """Return the cache filename expected by resolve_model_path."""
    if entry.get("wandb_file"):
        return str(entry["wandb_file"])
    asset_name = str(manifest_row["asset_name"])
    model_id = str(manifest_row["model_id"])
    prefix = f"{model_id}-"
    return asset_name.removeprefix(prefix) if asset_name.startswith(prefix) else asset_name


def _public_pointer(
    entry: dict[str, Any],
    manifest_row: dict[str, Any],
    *,
    repository: str,
    tag: str,
) -> dict[str, Any]:
    """Build the github_release pointer for a registry entry."""
    pointer: dict[str, Any] = {
        "repository": repository,
        "tag": tag,
        "asset_name": manifest_row["asset_name"],
        "file_name": _cache_file_name(entry, manifest_row),
        "sha256": str(manifest_row["sha256"]).lower(),
        "size_bytes": manifest_row["size_bytes"],
    }
    metadata_asset = manifest_row.get("metadata_asset")
    if metadata_asset:
        pointer["metadata_asset_name"] = metadata_asset
    return pointer


def _pointer_yaml_lines(pointer: dict[str, Any]) -> list[str]:
    """Return a github_release block formatted like model/registry.yaml."""
    lines = ["    github_release:"]
    for key, value in pointer.items():
        lines.append(f"      {key}: {value}")
    return lines


def _insert_pointers_preserving_registry_text(  # noqa: C901
    *,
    registry_path: Path,
    manifest_models: dict[str, dict[str, Any]],
    pointers: dict[str, dict[str, Any]],
) -> None:
    """Insert public pointers without reserializing the whole YAML registry."""
    original = registry_path.read_text(encoding="utf-8")
    lines = original.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        output.append(line)
        if line.startswith("  - model_id: "):
            model_id = line.split(": ", 1)[1]
            pointer = pointers.get(model_id)
            if pointer is not None:
                scan_index = index + 1
                insert_at = None
                existing_pointer_start = None
                existing_pointer_end = None
                while scan_index < len(lines) and not lines[scan_index].startswith(
                    "  - model_id: "
                ):
                    if lines[scan_index].startswith("    github_release:"):
                        existing_pointer_start = scan_index
                        existing_pointer_end = scan_index + 1
                        while existing_pointer_end < len(lines) and lines[
                            existing_pointer_end
                        ].startswith("      "):
                            existing_pointer_end += 1
                    if lines[scan_index].startswith("    tags:") or lines[scan_index].startswith(
                        "    notes:"
                    ):
                        insert_at = scan_index
                        break
                    scan_index += 1
                if insert_at is None:
                    insert_at = scan_index
                while index + 1 < insert_at:
                    index += 1
                    if existing_pointer_start is not None and (
                        existing_pointer_start <= index < existing_pointer_end
                    ):
                        continue
                    output.append(lines[index])
                output.extend(_pointer_yaml_lines(pointer))
                index = insert_at - 1
            manifest_models.pop(model_id, None)
        index += 1
    registry_path.write_text("\n".join(output) + "\n", encoding="utf-8")


def apply_manifest(
    *,
    registry_path: Path,
    manifest_path: Path,
    sha256s_path: Path,
    output_path: Path | None,
    write: bool,
) -> dict[str, Any]:
    """Apply release pointers from a manifest to matching registry entries."""
    registry = _load_yaml(registry_path)
    manifest = _load_manifest(manifest_path)
    manifest_models = _manifest_model_index(manifest, sha256s=_load_sha256s(sha256s_path))
    repository = str(manifest.get("repo", "") or "").strip()
    tag = str(manifest.get("tag", "") or "").strip()
    if not repository or not tag:
        raise ValueError("Manifest must include repo and tag")

    models = registry.get("models")
    if not isinstance(models, list):
        raise ValueError(f"Registry {registry_path} must contain a models list")

    updated: list[str] = []
    missing_from_registry = set(manifest_models)
    pointers: dict[str, dict[str, Any]] = {}
    for entry in models:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("model_id", "") or "").strip()
        manifest_row = manifest_models.get(model_id)
        if manifest_row is None:
            continue
        entry["github_release"] = _public_pointer(
            entry,
            manifest_row,
            repository=repository,
            tag=tag,
        )
        pointers[model_id] = entry["github_release"]
        updated.append(model_id)
        missing_from_registry.discard(model_id)

    result = {
        "registry_path": str(registry_path),
        "manifest_path": str(manifest_path),
        "updated": updated,
        "missing_from_registry": sorted(missing_from_registry),
    }
    destination = output_path if output_path is not None else registry_path
    if output_path is not None:
        _write_yaml(destination, registry)
    elif write:
        _insert_pointers_preserving_registry_text(
            registry_path=registry_path,
            manifest_models=dict(manifest_models),
            pointers=pointers,
        )
    return result


def inventory(*, registry_path: Path, manifest_path: Path | None) -> dict[str, Any]:
    """Return a compact publication inventory for registry review."""
    registry = _load_yaml(registry_path)
    models = registry.get("models")
    if not isinstance(models, list):
        raise ValueError(f"Registry {registry_path} must contain a models list")
    manifest_ids: set[str] = set()
    if manifest_path is not None:
        manifest = _load_manifest(manifest_path)
        manifest_ids = {
            str(item.get("model_id"))
            for item in manifest["models"]
            if isinstance(item, dict) and item.get("model_id")
        }
    rows = []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("model_id", "") or "")
        rows.append(
            {
                "model_id": model_id,
                "has_github_release": isinstance(entry.get("github_release"), dict),
                "has_wandb_pointer": bool(
                    entry.get("wandb_artifact_path")
                    or entry.get("wandb_run_path")
                    or (
                        entry.get("wandb_entity")
                        and entry.get("wandb_project")
                        and entry.get("wandb_run_id")
                    )
                ),
                "local_only": bool(entry.get("local_only")),
                "in_manifest": model_id in manifest_ids if manifest_path is not None else None,
            }
        )
    return {"registry_path": str(registry_path), "models": rows}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    apply_parser = subparsers.add_parser(
        "apply-manifest",
        help="Apply github_release pointers from a release manifest.",
    )
    apply_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    apply_parser.add_argument("--manifest", type=Path, required=True)
    apply_parser.add_argument("--sha256s", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path)
    apply_parser.add_argument(
        "--write",
        action="store_true",
        help="Update the registry in place. Without --write, only JSON summary is emitted.",
    )

    inventory_parser = subparsers.add_parser(
        "inventory",
        help="Summarize registry publication state.",
    )
    inventory_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    inventory_parser.add_argument("--manifest", type=Path)
    return parser


def main() -> None:
    """Run the CLI."""
    args = build_parser().parse_args()
    if args.command == "apply-manifest":
        result = apply_manifest(
            registry_path=args.registry,
            manifest_path=args.manifest,
            sha256s_path=args.sha256s,
            output_path=args.output,
            write=args.write,
        )
    else:
        result = inventory(registry_path=args.registry, manifest_path=args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
