#!/usr/bin/env python3
"""Collect launch-packet oracle candidate traces for imitation dataset staging."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.training.oracle_imitation_launch_packet import (
    load_launch_packet,
    validate_launch_packet,
)
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.policy_search_common import summarize_policy_search_records
from scripts.validation.run_policy_search_candidate import (
    _group_scenarios_by_config_overrides,
    _load_records,
    _prepare_scenarios_for_inline_run,
    _write_records,
    load_candidate_definition,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = Path("configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml")
DEFAULT_REGISTRY = Path("docs/context/policy_search/candidate_registry.yaml")
EPISODE_ID_RE = re.compile(r"^(?P<split>[a-z_]+)__(?P<scenario>.+)__seed(?P<seed>\d+)$")


def _git_hash() -> str | None:
    """Return the current Git SHA when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _scenario_id(scenario: dict[str, Any]) -> str:
    """Return the stable scenario id used by policy-search manifests."""
    return str(
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
    )


def _parse_episode_id(episode_id: str, *, split: str) -> tuple[str, int]:
    """Parse one launch-packet episode id into `(scenario_id, seed)`.

    Args:
        episode_id: Launch-packet episode id shaped like `train__scenario__seed201`.
        split: Expected split prefix.

    Returns:
        Scenario id and integer seed.

    Raises:
        ValueError: If the id does not match the packet contract.
    """
    match = EPISODE_ID_RE.match(episode_id)
    if match is None:
        raise ValueError(f"invalid launch-packet episode id: {episode_id!r}")
    actual_split = match.group("split")
    if actual_split != split:
        raise ValueError(
            f"episode id split mismatch for {episode_id!r}: expected {split!r}, got {actual_split!r}"
        )
    return match.group("scenario"), int(match.group("seed"))


def build_split_scenarios(
    packet: dict[str, Any], *, split: str, repo_root: Path
) -> list[dict[str, Any]]:
    """Build exact scenario rows from a launch packet split.

    The packet's `scenario_source` can include more scenarios than one split uses.  This helper
    follows `episode_ids_by_split` so train/validation/evaluation rows stay explicit.

    Args:
        packet: Parsed launch packet.
        split: Split name to materialize.
        repo_root: Repository root for resolving scenario paths.

    Returns:
        Scenario mappings with a single `seeds` entry per packet episode.
    """
    episode_ids_by_split = packet.get("episode_ids_by_split")
    if not isinstance(episode_ids_by_split, dict) or split not in episode_ids_by_split:
        raise ValueError(f"launch packet is missing episode_ids_by_split.{split}")
    raw_episode_ids = episode_ids_by_split[split]
    if not isinstance(raw_episode_ids, list) or not raw_episode_ids:
        raise ValueError(f"episode_ids_by_split.{split} must be a non-empty list")

    scenario_source = packet.get("scenario_source")
    if not isinstance(scenario_source, str) or not scenario_source.strip():
        raise ValueError("launch packet scenario_source must be a non-empty path")
    scenario_path = Path(scenario_source)
    if not scenario_path.is_absolute():
        scenario_path = repo_root / scenario_path
    scenario_path = scenario_path.resolve()

    source_scenarios = load_scenarios(scenario_path)
    scenarios_by_id = {
        _scenario_id(dict(scenario)): dict(scenario) for scenario in source_scenarios
    }

    split_scenarios: list[dict[str, Any]] = []
    for raw_episode_id in raw_episode_ids:
        if not isinstance(raw_episode_id, str) or not raw_episode_id.strip():
            raise ValueError(f"episode_ids_by_split.{split} contains a non-string episode id")
        scenario_id, seed = _parse_episode_id(raw_episode_id, split=split)
        if scenario_id not in scenarios_by_id:
            raise ValueError(
                f"episode {raw_episode_id!r} references scenario {scenario_id!r}, "
                f"but {scenario_path} does not provide it"
            )
        scenario = deepcopy(scenarios_by_id[scenario_id])
        scenario["seeds"] = [seed]
        metadata = scenario.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["oracle_imitation_episode_id"] = raw_episode_id
            metadata["oracle_imitation_split"] = split
        split_scenarios.append(scenario)

    return _prepare_scenarios_for_inline_run(split_scenarios, scenario_root=scenario_path.parent)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def collect_candidate_traces(  # noqa: PLR0913
    *,
    packet_path: Path,
    candidate_registry: Path,
    split: str,
    output_dir: Path,
    horizon: int,
    dt: float,
    workers: int,
    benchmark_profile: str,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Run packet-selected scenarios through the source candidate and write trace artifacts."""
    repo_root = REPO_ROOT
    packet_path = (
        (repo_root / packet_path).resolve() if not packet_path.is_absolute() else packet_path
    )
    registry_path = (
        (repo_root / candidate_registry).resolve()
        if not candidate_registry.is_absolute()
        else candidate_registry
    )
    output_dir = output_dir.resolve()

    validation_report = validate_launch_packet(packet_path, repo_root=repo_root)
    packet = load_launch_packet(packet_path)
    source_candidate = str(packet["source_candidate"])
    scenarios = build_split_scenarios(packet, split=split, repo_root=repo_root)
    entry, candidate_payload, candidate_config, candidate_config_path = load_candidate_definition(
        registry_path,
        source_candidate,
    )
    default_algo = str(candidate_payload.get("algo", "")).strip().lower()
    if not default_algo:
        raise ValueError(f"candidate {source_candidate!r} is missing algo")

    grouped = _group_scenarios_by_config_overrides(
        scenarios,
        candidate_payload=candidate_payload,
        candidate_config=candidate_config,
        default_algo=default_algo,
        config_anchor=candidate_config_path.parent,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    group_summaries: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []

    for group_name, group in sorted(grouped.items()):
        group_dir = output_dir / "groups" / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        algo_config_path = group_dir / "algo.yaml"
        jsonl_path = group_dir / "episodes.jsonl"
        algo_config_path.write_text(
            yaml.safe_dump(group["config"], sort_keys=False),
            encoding="utf-8",
        )
        if jsonl_path.exists():
            jsonl_path.unlink()
        batch_summary = run_map_batch(
            group["scenarios"],
            jsonl_path,
            schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
            algo=str(group["algo"]),
            algo_config_path=str(algo_config_path),
            horizon=int(horizon),
            dt=float(dt),
            workers=int(workers),
            resume=False,
            benchmark_profile=benchmark_profile,
        )
        records = _load_records(jsonl_path) if jsonl_path.exists() else []
        all_records.extend(records)
        group_summaries[group_name] = {
            "algo": group["algo"],
            "algo_config_path": str(algo_config_path),
            "jsonl_path": str(jsonl_path),
            "scenario_count": len(group["scenarios"]),
            "batch_summary": batch_summary,
            "summary": summarize_policy_search_records(records),
        }
        if int(batch_summary.get("failed_jobs", 0) or 0) > 0:
            failures.extend(batch_summary.get("failures", []))

    combined_jsonl = output_dir / f"{split}__{source_candidate}__combined.jsonl"
    _write_records(combined_jsonl, all_records)
    summary = summarize_policy_search_records(all_records)
    expected_episodes = len(scenarios)
    if len(all_records) != expected_episodes:
        failures.append(
            {
                "error": "episode_count_mismatch",
                "expected": expected_episodes,
                "actual": len(all_records),
            }
        )

    manifest = {
        "schema_version": "oracle-imitation-candidate-traces.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_hash": _git_hash(),
        "launch_packet": str(packet_path),
        "launch_packet_validation": validation_report,
        "candidate_registry": str(registry_path),
        "candidate": source_candidate,
        "candidate_entry": entry,
        "candidate_config_path": str(candidate_config_path),
        "split": split,
        "scenario_count": len(scenarios),
        "horizon": int(horizon),
        "dt": float(dt),
        "workers": int(workers),
        "benchmark_profile": benchmark_profile,
        "combined_jsonl": str(combined_jsonl),
        "summary": summary,
        "group_summaries": group_summaries,
        "failures": failures,
        "dataset_boundary": (
            "This trace collection is not the final imitation NPZ dataset. It proves the "
            "launch-packet source candidate can run on the selected split and preserves JSONL "
            "episode evidence for downstream dataset materialization."
        ),
    }
    manifest_path = output_dir / "oracle_candidate_trace_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)

    if failures and not allow_partial:
        raise RuntimeError(
            f"oracle candidate trace collection failed closed with {len(failures)} failure(s); "
            f"see {manifest_path}"
        )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--candidate-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--split", choices=("train", "validation", "evaluation"), default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--benchmark-profile", default="testing")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    manifest = collect_candidate_traces(
        packet_path=args.config,
        candidate_registry=args.candidate_registry,
        split=args.split,
        output_dir=args.output_dir,
        horizon=args.horizon,
        dt=args.dt,
        workers=args.workers,
        benchmark_profile=args.benchmark_profile,
        allow_partial=bool(args.allow_partial),
    )
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"manifest={manifest['manifest_path']}")
        print(f"combined_jsonl={manifest['combined_jsonl']}")
        print(f"episodes={manifest['summary'].get('episodes', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
