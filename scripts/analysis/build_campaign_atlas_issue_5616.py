"""CLI entry point for the campaign atlas and ensemble context views (#5616).

Builds the campaign atlas plus optional per-cell event-aligned ensemble context
views from a campaign inventory file and (optionally) a frozen selection
manifest, then writes the artifact-catalog manifest. Supports a
``--check-determinism`` mode that rebuilds into a temp directory and asserts
byte-identical / hash-stable output.

Usage:
    uv run python scripts/analysis/build_campaign_atlas_issue_5616.py \
        --inventory campaign_inventory.jsonl \
        --out-dir output/campaign_atlas_<id> \
        --campaign-id <id> \
        --ensemble-anchor near_miss_start \
        --render-html
    uv run python scripts/analysis/build_campaign_atlas_issue_5616.py \
        --inventory campaign_inventory.jsonl --out-dir <dir> --check-determinism
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

from robot_sf.benchmark.campaign_atlas import (
    AtlasConfig,
    EpisodeInventoryRow,
    PredicateInterval,
    TrajectoryPoint,
    build_campaign_atlas,
)

_INVENTORY_REQUIRED = ("episode_id", "planner", "scenario_id", "scenario_family", "seed", "outcome")


def _row_from_record(record: dict[str, Any]) -> EpisodeInventoryRow:
    """Parse one campaign inventory JSON record into an ``EpisodeInventoryRow``."""
    missing = [key for key in _INVENTORY_REQUIRED if key not in record]
    if missing:
        raise ValueError(f"inventory record missing required keys: {', '.join(missing)}")
    trajectory = tuple(
        TrajectoryPoint(
            t=float(point["t"]),
            x=float(point["x"]),
            y=float(point["y"]),
        )
        for point in record.get("trajectory", [])
    )
    event_anchors = {str(k): float(v) for k, v in record.get("event_anchors", {}).items()}
    predicate_timeline = tuple(
        PredicateInterval(
            t_start=float(interval["t_start"]),
            t_end=float(interval["t_end"]),
            label=str(interval["label"]),
        )
        for interval in record.get("predicate_timeline", [])
    )
    return EpisodeInventoryRow(
        episode_id=str(record["episode_id"]),
        planner=str(record["planner"]),
        scenario_id=str(record["scenario_id"]),
        scenario_family=str(record["scenario_family"]),
        seed=int(record["seed"]),
        outcome=str(record["outcome"]),
        release_arm_id=str(record["release_arm_id"])
        if record.get("release_arm_id") not in (None, "")
        else None,
        metrics=dict(record.get("metrics", {})),
        trajectory=trajectory,
        event_anchors=event_anchors,
        predicate_timeline=predicate_timeline,
    )


def load_inventory(path: Path) -> list[EpisodeInventoryRow]:
    """Load campaign inventory rows from a JSON or JSONL file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        records = payload if isinstance(payload, list) else payload.get("episodes", [])
    if not isinstance(records, list):
        raise ValueError("inventory file must contain a JSON array or JSONL records")
    return [_row_from_record(record) for record in records]


def load_selection_manifest(path: Path) -> tuple[list[str], str]:
    """Return ``(exemplar_episode_ids, manifest_hash)`` from a selection manifest.

    The manifest may be a ``SelectionManifest`` JSON (exemplar-selection.v1) or a
    plain list of episode ids.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        selected = payload.get("selected", [])
        if not isinstance(selected, list):
            raise ValueError("selection manifest 'selected' must be a list")
        malformed = [
            item
            for item in selected
            if not isinstance(item, dict)
            or not isinstance(item.get("episode_id"), str)
            or not item["episode_id"].strip()
        ]
        if malformed:
            raise ValueError(
                f"selection manifest has {len(malformed)} malformed 'selected' entries"
            )
        episode_ids = [item["episode_id"] for item in selected]
        manifest_hash = str(payload.get("source_sha256", "") or _sha_text(path))
    else:
        episode_ids = [str(item) for item in payload]
        manifest_hash = _sha_text(path)
    return episode_ids, manifest_hash


def _sha_text(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    import hashlib

    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _build(
    out_dir: Path,
    rows: list[EpisodeInventoryRow],
    args: argparse.Namespace,
    *,
    command: str = "build_campaign_atlas_issue_5616.py",
) -> None:
    """Run the atlas build into *out_dir*."""
    exemplar_ids: list[str] = []
    selection_hash = ""
    if args.selection_manifest:
        exemplar_ids, selection_hash = load_selection_manifest(Path(args.selection_manifest))

    config = AtlasConfig(
        campaign_id=args.campaign_id,
        min_cell_size=args.min_cell_size,
        metric_definitions=dict(args.metric_definitions or {}),
        eligible_scenario_families=tuple(args.eligible_scenario_families)
        if args.eligible_scenario_families
        else None,
        eligible_planners=tuple(args.eligible_planners) if args.eligible_planners else None,
    )
    build_campaign_atlas(
        rows,
        out_dir=out_dir,
        config=config,
        exemplar_episode_ids=exemplar_ids,
        selection_manifest_hash=selection_hash,
        ensemble_anchor=args.ensemble_anchor,
        render_html=args.render_html,
        command=command,
        commit=args.commit or "unknown",
        source_inventory=Path(args.inventory),
    )


def _hash_tree(directory: Path) -> str:
    """Return a stable hash across all files in *directory* (sorted by path)."""
    import hashlib

    h = hashlib.sha256()
    for file in sorted(Path(directory).rglob("*"), key=str):
        if file.is_file():
            h.update(str(file.relative_to(directory)).encode("utf-8"))
            h.update(file.read_bytes())
    return h.hexdigest()


def check_determinism(
    rows: list[EpisodeInventoryRow],
    args: argparse.Namespace,
    *,
    command: str = "build_campaign_atlas_issue_5616.py",
) -> bool:
    """Rebuild into two temp dirs and assert hash-stable output.

    Returns:
        ``True`` when both builds produced identical artifact trees.
    """
    with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
        first = Path(first_tmp) / "a"
        second = Path(second_tmp) / "b"
        _build(first, rows, args, command=command)
        _build(second, rows, args, command=command)
        return _hash_tree(first) == _hash_tree(second)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the campaign atlas CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Build a campaign atlas and event-aligned ensemble context views."
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        required=True,
        help="Campaign inventory JSON/JSONL file (episode rows).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=False,
        default=None,
        help="Output directory for artifacts.",
    )
    parser.add_argument("--campaign-id", type=str, default="campaign", help="Campaign identifier.")
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=None,
        help="Frozen selection manifest JSON (exemplar-selection.v1 or episode-id list).",
    )
    parser.add_argument(
        "--ensemble-anchor",
        type=str,
        default=None,
        help="Named event anchor to align ensemble views on (e.g. 'near_miss_start').",
    )
    parser.add_argument(
        "--render-html",
        action="store_true",
        help="Also render the optional Altair/Vega-Lite HTML exploration atlas.",
    )
    parser.add_argument(
        "--min-cell-size", type=int, default=1, help="Minimum episodes for a cell to be eligible."
    )
    parser.add_argument(
        "--eligible-scenario-families",
        type=str,
        nargs="*",
        default=None,
        help="Restrict the atlas to these scenario families.",
    )
    parser.add_argument(
        "--eligible-planners",
        type=str,
        nargs="*",
        default=None,
        help="Restrict the atlas to these planners.",
    )
    parser.add_argument(
        "--metric-definitions",
        type=json.loads,
        default=None,
        help="JSON object mapping metric name to definition string.",
    )
    parser.add_argument("--commit", type=str, default="", help="Generation commit hash.")
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="Rebuild into two temp dirs and assert hash-stable output, then exit.",
    )
    return parser


def _generation_command(argv: list[str] | None) -> str:
    """Return the shell-escaped invocation used to generate the catalog."""
    command_args = sys.argv[1:] if argv is None else argv
    return shlex.join(["build_campaign_atlas_issue_5616.py", *command_args])


def main(argv: list[str] | None = None) -> int:
    """Run the campaign atlas CLI and return a shell-friendly exit code."""
    args = build_arg_parser().parse_args(argv)
    command = _generation_command(argv)
    rows = load_inventory(Path(args.inventory))

    if args.check_determinism:
        ok = check_determinism(rows, args, command=command)
        print(f"determinism check: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 2

    if args.out_dir is None:
        print("--out-dir is required unless --check-determinism is given", file=sys.stderr)
        return 2

    _build(Path(args.out_dir), rows, args, command=command)
    print(f"campaign atlas written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
