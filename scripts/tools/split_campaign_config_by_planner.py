#!/usr/bin/env python3
"""Split a camera-ready campaign YAML into deterministic per-arm configs (issue #5251).

Long benchmark campaigns can keep model state for every planner in one process. This CPU-only
tool emits one config per enabled planner arm (or per ``planner_group``) so existing campaign
launchers can schedule bounded jobs without changing seeds, metrics, or scenario settings.

Each child retains the parent payload, changes only ``name`` and ``planners``, and adds
``split_provenance``. A manifest records the parent and child SHA-256 digests. No campaign is
run by this tool.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_NAME = "split_campaign_config_by_planner.py"


class CampaignSplitError(ValueError):
    """Raised when a campaign cannot be split without changing its contract."""


@dataclass(frozen=True)
class SplitChild:
    """A written child config and the enabled planner keys it contains."""

    output_path: Path
    planner_keys: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class SplitResult:
    """Details of one deterministic config split operation."""

    children: tuple[SplitChild, ...]
    manifest_path: Path
    disabled_planner_keys: tuple[str, ...]


def _sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest for raw file bytes."""

    return hashlib.sha256(payload).hexdigest()


def _source_reference(config_path: Path) -> str:
    """Return a portable repository-relative source reference when available."""

    resolved = config_path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(config_path)


def _planner_key(entry: Any, index: int) -> str:
    """Extract a non-empty planner key from one raw YAML entry."""

    if not isinstance(entry, dict):
        raise CampaignSplitError(f"Planner entry {index} must be a mapping")
    key = str(entry.get("key") or entry.get("algo") or "").strip()
    if not key:
        raise CampaignSplitError(f"Planner entry {index} requires a non-empty key or algo")
    if "/" in key or "\\" in key:
        raise CampaignSplitError(f"Planner key '{key}' cannot contain a path separator")
    return key


def _safe_output_component(value: str, *, label: str) -> str:
    """Return a filename-safe config-derived component or fail before emission."""

    component = value.strip()
    if not component or component in {".", ".."} or "/" in component or "\\" in component:
        raise CampaignSplitError(f"{label} cannot contain a path separator or traversal component")
    return component


def _collect_enabled_planners(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Return enabled planner entries and disabled planner keys in input order."""

    planners = payload.get("planners")
    if not isinstance(planners, list) or not planners:
        raise CampaignSplitError("Campaign config requires a non-empty 'planners' list")

    enabled: list[dict[str, Any]] = []
    disabled_keys: list[str] = []
    enabled_keys: set[str] = set()
    for index, entry in enumerate(planners):
        key = _planner_key(entry, index)
        if bool(entry.get("enabled", True)):
            if key in enabled_keys:
                raise CampaignSplitError(f"Enabled planner key appears more than once: {key}")
            enabled_keys.add(key)
            enabled.append(entry)
        else:
            disabled_keys.append(key)

    if not enabled:
        raise CampaignSplitError("Campaign config has no enabled planner arms to split")
    return enabled, tuple(disabled_keys)


def _split_groups(
    planners: list[dict[str, Any]], *, group_by: str | None
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Build ordered one-arm or planner-group output units."""

    if group_by is None:
        return [(_planner_key(planner, index), [planner]) for index, planner in enumerate(planners)]
    if group_by != "planner_group":
        raise CampaignSplitError(f"Unsupported group field: {group_by}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, planner in enumerate(planners):
        group = str(planner.get("planner_group", "experimental")).strip()
        if not group:
            raise CampaignSplitError(
                f"Enabled planner '{_planner_key(planner, index)}' has an empty planner_group"
            )
        if "/" in group or "\\" in group:
            raise CampaignSplitError(f"Planner group '{group}' cannot contain a path separator")
        grouped.setdefault(group, []).append(planner)
    return list(grouped.items())


def _child_payload(
    parent: dict[str, Any],
    planners: list[dict[str, Any]],
    *,
    child_name: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Return a deep-copied parent with the only allowed split changes applied."""

    child = copy.deepcopy(parent)
    child["name"] = child_name
    child["planners"] = copy.deepcopy(planners)
    child["split_provenance"] = provenance
    return child


def _validate_union(
    parent_planners: list[dict[str, Any]], children: list[list[dict[str, Any]]]
) -> None:
    """Fail before writing unless generated children cover exactly the enabled parent arms."""

    parent_keys = {_planner_key(planner, index) for index, planner in enumerate(parent_planners)}
    child_keys = {
        _planner_key(planner, index) for child in children for index, planner in enumerate(child)
    }
    if child_keys != parent_keys:
        raise CampaignSplitError(
            "Refusing to write split configs: child planner-key union does not match enabled parent "
            f"planner keys (parent={sorted(parent_keys)}, children={sorted(child_keys)})"
        )


def split_campaign_config(
    config_path: Path, out_dir: Path, *, group_by: str | None = None
) -> SplitResult:
    """Write deterministic campaign children and a provenance manifest.

    The invariant check completes before ``out_dir`` is created, so invalid inputs never create
    partial output. ``arm_index`` is zero-based and follows parent planner order.
    """

    if not config_path.is_file():
        raise FileNotFoundError(f"Campaign config not found: {config_path}")
    source_bytes = config_path.read_bytes()
    try:
        parent = yaml.safe_load(source_bytes) or {}
    except yaml.YAMLError as exc:
        raise CampaignSplitError(f"Campaign config is not valid YAML: {exc}") from exc
    if not isinstance(parent, dict):
        raise CampaignSplitError("Campaign config must be a top-level mapping")

    enabled, disabled_keys = _collect_enabled_planners(parent)
    groups = _split_groups(enabled, group_by=group_by)
    _validate_union(enabled, [planners for _, planners in groups])
    parent_name = _safe_output_component(
        str(parent.get("name") or config_path.stem), label="Campaign config name"
    )
    if "split_provenance" in parent:
        raise CampaignSplitError(
            "Campaign config already has split_provenance; refusing to overwrite it"
        )

    split_mode = "per_group" if group_by else "per_planner"
    source_config = _source_reference(config_path)
    source_sha256 = _sha256_bytes(source_bytes)
    child_specs: list[tuple[Path, tuple[str, ...], bytes]] = []
    for arm_index, (arm_key, child_planners) in enumerate(groups):
        suffix = "group" if group_by else "arm"
        child_name = f"{parent_name}__{suffix}_{arm_key}"
        output_path = out_dir / f"{child_name}.yaml"
        provenance = {
            "source_config": source_config,
            "source_sha256": source_sha256,
            "split_mode": split_mode,
            "arm_key": arm_key,
            "arm_index": arm_index,
            "arm_total": len(groups),
            "tool": TOOL_NAME,
        }
        child = _child_payload(
            parent,
            child_planners,
            child_name=child_name,
            provenance=provenance,
        )
        serialized = yaml.safe_dump(child, sort_keys=False, allow_unicode=True).encode("utf-8")
        keys = tuple(_planner_key(planner, index) for index, planner in enumerate(child_planners))
        child_specs.append((output_path, keys, serialized))

    output_names = [path.name for path, _, _ in child_specs]
    if len(output_names) != len(set(output_names)):
        raise CampaignSplitError("Split output filenames would collide")
    if out_dir.exists() and not out_dir.is_dir():
        raise CampaignSplitError(f"Output path is not a directory: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    children: list[SplitChild] = []
    for output_path, planner_keys, serialized in child_specs:
        output_path.write_bytes(serialized)
        children.append(
            SplitChild(
                output_path=output_path,
                planner_keys=planner_keys,
                sha256=_sha256_bytes(serialized),
            )
        )

    manifest = {
        "source_config": source_config,
        "source_sha256": source_sha256,
        "split_mode": split_mode,
        "children": [
            {
                "filename": child.output_path.name,
                "sha256": child.sha256,
                "planner_keys": list(child.planner_keys),
            }
            for child in children
        ],
    }
    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return SplitResult(tuple(children), manifest_path, disabled_keys)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for config splitting."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Parent camera-ready YAML config."
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for child configs.")
    parser.add_argument(
        "--group-by",
        choices=("planner_group",),
        default=None,
        help="Emit one child per planner_group instead of one child per enabled planner arm.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the campaign splitter command-line interface."""

    args = build_arg_parser().parse_args(argv)
    try:
        result = split_campaign_config(args.config, args.out_dir, group_by=args.group_by)
    except (CampaignSplitError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for key in result.disabled_planner_keys:
        print(f"Skipping disabled planner arm: {key}", file=sys.stderr)
    print(f"Wrote {len(result.children)} split config(s): {result.manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
