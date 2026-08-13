#!/usr/bin/env python3
"""Export one release-pinned trace dossier from existing campaign artifacts.

This command is deliberately read-only over the campaign result store.  It resolves one
``(scenario_id, seed, planner_id, release_manifest)`` tuple, converts an existing JSONL episode
record when necessary, and writes a renderer-neutral ``simulation_trace_export.v1`` trace with a
provenance manifest and checksums.  It does not run a simulation or admit benchmark evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

import yaml

from robot_sf.analysis_workbench.simulation_trace_export import (
    SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
    SimulationTraceExportValidationError,
    simulation_trace_export_from_dict,
)
from robot_sf.benchmark.candidate_trace_resolution import resolve_episode_source
from robot_sf.benchmark.classic_interactions_loader import (
    iter_episode_seeds,
    load_classic_matrix,
    select_scenario,
)
from robot_sf.benchmark.release_protocol import (
    build_release_provenance,
    load_release_manifest,
    validate_release_manifest,
)
from robot_sf.evidence.writers import sha256_file, write_json, write_sha256sums
from scripts.tools.build_simulation_trace_export import (
    build_simulation_trace_export_with_receipt,
)

TRACE_DOSSIER_MANIFEST_SCHEMA_VERSION = "trace_dossier_export_manifest.v1"
TRACE_IDENTITY_RECEIPT_SCHEMA_VERSION = "simulation_trace_export.identity_receipt.v1"


class TraceDossierExportError(ValueError):
    """Raised when a release-pinned trace cannot be exported defensibly."""


def _repo_root() -> Path:
    """Return the current worktree root."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def _git_commit() -> str:
    """Return the exact exporter source commit."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo_relative(path: Path, *, root: Path) -> str:
    """Return a stable repository-relative path when the source is in this worktree."""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _resolve_seed_policy(manifest: Any, scenario: Mapping[str, Any]) -> tuple[int, ...]:
    """Resolve the release manifest's seed policy without inventing a fallback set."""
    policy = manifest.seed_policy
    mode = str(policy.get("mode", "")).strip().lower()
    if mode == "fixed-list":
        seeds = policy.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise TraceDossierExportError("release seed policy fixed-list has no seeds")
        return tuple(int(seed) for seed in seeds)
    if mode == "seed-set":
        seed_set = str(policy.get("seed_set", "")).strip()
        seed_sets_path_raw = policy.get("seed_sets_path")
        if not seed_set or not isinstance(seed_sets_path_raw, str) or not seed_sets_path_raw:
            raise TraceDossierExportError("release seed policy seed-set is incomplete")
        seed_sets_path = (manifest.path.parent / seed_sets_path_raw).resolve()
        payload = yaml.safe_load(seed_sets_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get(seed_set), list):
            raise TraceDossierExportError(
                f"release seed set {seed_set!r} is missing from {seed_sets_path}"
            )
        return tuple(int(seed) for seed in payload[seed_set])
    if mode == "scenario-default":
        return tuple(iter_episode_seeds(dict(scenario)))
    raise TraceDossierExportError(f"unsupported release seed policy mode: {mode!r}")


def _validate_release_tuple(
    manifest: Any,
    *,
    scenario_id: str,
    planner_id: str,
    seed: int,
) -> dict[str, Any]:
    """Validate tuple membership against the release's canonical inputs."""
    validation = validate_release_manifest(manifest)
    if validation["status"] != "valid":
        problems = "; ".join(str(problem) for problem in validation["problems"])
        raise TraceDossierExportError(f"release manifest is invalid: {problems}")
    if planner_id not in manifest.planner_keys:
        raise TraceDossierExportError(
            f"planner {planner_id!r} is not in release planner roster {manifest.planner_keys!r}"
        )
    scenarios = load_classic_matrix(str(manifest.scenario_matrix_path))
    try:
        scenario = select_scenario(scenarios, scenario_id)
    except ValueError as exc:
        raise TraceDossierExportError(str(exc)) from exc
    seeds = _resolve_seed_policy(manifest, scenario)
    if seed not in seeds:
        raise TraceDossierExportError(f"seed {seed} is not in release seed policy {list(seeds)!r}")
    return {
        "scenario_name": str(scenario.get("name", scenario_id)),
        "scenario_matrix": str(manifest.scenario_matrix_path),
        "scenario_matrix_sha256": manifest.scenario_matrix_sha256,
        "resolved_seeds": list(seeds),
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a trace without the review marker used by generic evidence JSON."""
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _trace_identity_matches(
    source: Mapping[str, Any],
    *,
    scenario_id: str,
    planner_id: str,
    seed: int,
    episode_id: str,
) -> bool:
    """Return whether a source trace carries the requested episode identity."""
    try:
        source_seed = int(source.get("seed", -1))
    except (TypeError, ValueError):
        return False
    return (
        source.get("scenario_id") == scenario_id
        and source.get("planner_id") == planner_id
        and source_seed == seed
        and str(source.get("episode_id")) == episode_id
    )


def _trace_source_payload(
    source_path: Path,
    *,
    scenario_id: str,
    planner_id: str,
    seed: int,
    episode_id: str,
    source_sha256: str,
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load an existing typed trace or convert an existing JSONL episode source."""
    if source_path.suffix.lower() == ".json":
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TraceDossierExportError(f"source JSON is unreadable: {exc}") from exc
        if isinstance(payload, dict) and payload.get("schema_version") == (
            SIMULATION_TRACE_EXPORT_SCHEMA_VERSION
        ):
            try:
                simulation_trace_export_from_dict(payload, source=source_path)
            except SimulationTraceExportValidationError as exc:
                raise TraceDossierExportError(f"source trace schema mismatch: {exc}") from exc
            source = payload.get("source")
            if not isinstance(source, dict) or not _trace_identity_matches(
                source,
                scenario_id=scenario_id,
                planner_id=planner_id,
                seed=seed,
                episode_id=episode_id,
            ):
                raise TraceDossierExportError(
                    "source trace identity does not match requested scenario/planner/seed/episode"
                )
            receipt = {
                "schema_version": TRACE_IDENTITY_RECEIPT_SCHEMA_VERSION,
                "source_sha256": source_sha256,
                "transformation": "identity",
                "provenance": dict(provenance),
            }
            return payload, receipt

    try:
        return build_simulation_trace_export_with_receipt(
            source_path,
            planner_id=planner_id,
            scenario_id=scenario_id,
            source_signature=source_sha256,
            provenance=provenance,
        )
    except (OSError, ValueError, SimulationTraceExportValidationError) as exc:
        raise TraceDossierExportError(f"source trace conversion failed: {exc}") from exc


def export_trace_dossier(
    *,
    scenario_id: str,
    planner_id: str,
    seed: int,
    release_manifest_path: Path,
    campaign_store_dir: Path,
    output_dir: Path,
    trace_search_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Export one existing campaign episode as a release-pinned trace dossier."""
    root = _repo_root()
    manifest = load_release_manifest(release_manifest_path)
    release_tuple = _validate_release_tuple(
        manifest,
        scenario_id=scenario_id,
        planner_id=planner_id,
        seed=seed,
    )
    resolution = resolve_episode_source(
        scenario_id=scenario_id,
        planner_id=planner_id,
        seed=seed,
        campaign_store_dir=campaign_store_dir,
        trace_search_roots=trace_search_roots,
    )
    if resolution.get("resolution_status") != "resolved":
        raise TraceDossierExportError(
            f"{resolution.get('resolution_status')}: {resolution.get('reason_code')}"
        )
    source_path = Path(str(resolution["source_path"]))
    source_sha256 = str(resolution["source_sha256"])
    release_provenance = build_release_provenance(
        manifest,
        campaign_root=campaign_store_dir,
        invoked_command="scripts/tools/export_trace_dossier.py",
    )
    provenance = {
        **release_provenance,
        "exporter_commit": _git_commit(),
        "campaign_id": resolution.get("campaign_id"),
        "campaign_row_reference": resolution.get("campaign_row_reference"),
        "episode_id": resolution.get("episode_id"),
        "source_artifact_uri": resolution.get("artifact_uri"),
        "source_artifact_sha256": resolution.get("artifact_sha256"),
        "source_sha256": source_sha256,
    }
    payload, receipt = _trace_source_payload(
        source_path,
        scenario_id=scenario_id,
        planner_id=planner_id,
        seed=seed,
        episode_id=str(resolution.get("episode_id")),
        source_sha256=source_sha256,
        provenance=provenance,
    )
    source_identity = payload.get("source")
    if not isinstance(source_identity, dict) or not _trace_identity_matches(
        source_identity,
        scenario_id=scenario_id,
        planner_id=planner_id,
        seed=seed,
        episode_id=str(resolution.get("episode_id")),
    ):
        raise TraceDossierExportError("converted trace identity does not match campaign tuple")
    trace = simulation_trace_export_from_dict(payload, source=source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    known_outputs = {"trace.json", "normalization_receipt.json", "manifest.json", "SHA256SUMS"}
    unexpected_outputs = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in known_outputs
    )
    if unexpected_outputs:
        raise TraceDossierExportError(
            "output directory contains unexpected files: " + ", ".join(unexpected_outputs)
        )
    trace_path = output_dir / "trace.json"
    trace_path.write_bytes(_canonical_json_bytes(trace.to_dict()))
    trace_sha256 = sha256_file(trace_path)

    receipt_path = output_dir / "normalization_receipt.json"
    write_json(receipt_path, receipt)
    receipt_sha256 = sha256_file(receipt_path)

    manifest_payload: dict[str, Any] = {
        "schema_version": TRACE_DOSSIER_MANIFEST_SCHEMA_VERSION,
        "claim_boundary": (
            "one existing release-pinned trace artifact for renderer and analysis use; "
            "no benchmark, statistical, safety, or paper-facing claim"
        ),
        "release": {
            "release_id": manifest.release_id,
            "release_tag": manifest.release_tag,
            "manifest_path": _repo_relative(manifest.path, root=root),
            "manifest_sha256": sha256_file(manifest.path),
            "scenario_matrix": _repo_relative(manifest.scenario_matrix_path, root=root),
            "scenario_matrix_sha256": manifest.scenario_matrix_sha256,
            "resolved_seeds": release_tuple["resolved_seeds"],
        },
        "identity": {
            "scenario_id": scenario_id,
            "planner_id": planner_id,
            "seed": seed,
            "scenario_name": release_tuple["scenario_name"],
            "episode_id": resolution.get("episode_id"),
            "exporter_commit": _git_commit(),
        },
        "source": {
            "campaign_id": resolution.get("campaign_id"),
            "campaign_row_reference": resolution.get("campaign_row_reference"),
            "artifact_uri": resolution.get("artifact_uri"),
            "artifact_sha256": resolution.get("artifact_sha256"),
            "source_path": _repo_relative(source_path, root=root),
            "source_sha256": source_sha256,
        },
        "artifacts": {
            "trace": {
                "path": trace_path.name,
                "schema_version": SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
                "sha256": trace_sha256,
            },
            "normalization_receipt": {
                "path": receipt_path.name,
                "sha256": receipt_sha256,
            },
            "checksums": {"path": "SHA256SUMS"},
        },
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest_payload)
    write_sha256sums(output_dir)
    return manifest_payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the export CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--planner", dest="planner_id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--campaign-store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trace-search-root",
        type=Path,
        action="append",
        default=[],
        help="Optional local root used to resolve repository-relative or URI artifact names.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the trace dossier export."""
    args = _parse_args(argv)
    try:
        export_trace_dossier(
            scenario_id=args.scenario_id,
            planner_id=args.planner_id,
            seed=args.seed,
            release_manifest_path=args.release_manifest,
            campaign_store_dir=args.campaign_store,
            output_dir=args.output_dir,
            trace_search_roots=tuple(args.trace_search_root),
        )
    except (OSError, TraceDossierExportError, ValueError) as exc:
        print(f"trace dossier export failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote trace dossier to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TraceDossierExportError", "export_trace_dossier", "main"]
