"""Compare #9671 trace episodes with the frozen 0.0.7 release rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import tarfile
from pathlib import Path
from typing import Any

import yaml

SOURCE_SHA = "07f7e8d43084de748915e1b1eb8b2a1603357c6e"
ARCHIVE_SHA256 = "684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f"
TRACE_KEYS = ("record_forces", "record_planner_decision_trace", "record_simulation_step_trace")
CONFIG_SHA256 = {
    "headon_group": "fe3ee17279a74b1ef541f68f746bc265e7b32bb29a711e862081f9f5ee1e277e",
    "doorway": "bae73148b125af5652726a12dced5d64a3e108e9783894293da2d44e7f513fb0",
}
EFFECTIVE_HASH = {"headon_group": "c013f96aae21ada2", "doorway": "fa98965443316b1a"}
EXPECTED_TUPLES = {
    (planner, scenario, seed)
    for planner in ("goal", "social_force", "orca")
    for scenario in ("classic_head_on_corridor_medium", "classic_group_crossing_medium")
    for seed in (22, 23, 24)
} | {("ppo", "classic_doorway_medium", seed) for seed in (113, 114)}


def _rows(lines: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for line in lines if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["scenario_params"]["algo"]), str(row["scenario_id"]), int(row["seed"])


def _release_rows(archive: Path, planners: set[str]) -> dict[tuple[str, str, int], dict[str, Any]]:
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not member.name.endswith("/episodes.jsonl") or "/payload/runs/" not in member.name:
                continue
            planner = member.name.split("/payload/runs/", 1)[1].split("__", 1)[0]
            if planner not in planners:
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read release member {member.name}")
            for row in _rows(stream):
                key = (planner, str(row["scenario_id"]), int(row["seed"]))
                if key in indexed:
                    raise ValueError(f"duplicate release tuple {key}")
                indexed[key] = row
    return indexed


def _config_tuples(config: dict[str, Any]) -> set[tuple[str, str, int]]:
    return {
        (str(planner["key"]), str(scenario), int(seed))
        for planner in config["planners"]
        for scenario in config["scenario_candidates"]
        for seed in config["seed_policy"]["seeds"]
    }


def _validate_bindings(
    config_paths: dict[str, Path],
    manifest_paths: dict[str, Path],
    required: set[tuple[str, str, int]],
    config_sha256: dict[str, str],
    effective_hash: dict[str, str],
) -> dict[str, Any]:
    """Bind separately versioned trace configs to frozen-source run manifests."""
    if set(config_paths) != {"headon_group", "doorway"} or set(manifest_paths) != set(config_paths):
        raise ValueError("both diagnostic configs and campaign manifests are required")
    bound: dict[str, Any] = {}
    tuples: set[tuple[str, str, int]] = set()
    for name, path in sorted(config_paths.items()):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != config_sha256[name]:
            raise ValueError(f"{name} diagnostic config SHA-256 mismatch")
        config = yaml.safe_load(raw)
        if (
            config.get("horizon") != 600
            or config.get("dt") != 0.1
            or config.get("kinematics_matrix") != ["differential_drive"]
            or any(config.get(flag) is not True for flag in TRACE_KEYS)
        ):
            raise ValueError(f"{name} diagnostic config changes execution contract")
        selected = _config_tuples(config)
        if tuples & selected:
            raise ValueError("diagnostic config tuple sets overlap")
        tuples |= selected
        manifest = json.loads(manifest_paths[name].read_text())
        if (
            manifest.get("git", {}).get("commit") != SOURCE_SHA
            or manifest.get("config_hash") != effective_hash[name]
            or manifest.get("scenario_matrix") != config["scenario_matrix"]
            or set(manifest.get("seed_policy", {}).get("resolved_seeds", []))
            != set(config["seed_policy"]["seeds"])
        ):
            raise ValueError(f"{name} campaign manifest identity mismatch")
        bound[name] = {
            "config_path": str(path),
            "config_sha256": digest,
            "campaign_manifest_path": str(manifest_paths[name]),
            "campaign_manifest_sha256": hashlib.sha256(
                manifest_paths[name].read_bytes()
            ).hexdigest(),
            "effective_config_hash": effective_hash[name],
        }
    if tuples != required:
        raise ValueError("diagnostic configs do not declare exactly the requested tuple inventory")
    return bound


def _vector2(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
            for x in value
        )
    )


def _validate_steps(key: tuple[str, str, int], trace: dict[str, Any]) -> None:
    """Require robot, pedestrian, and available force state at each recorded step."""
    if not isinstance(trace.get("steps"), list) or not trace["steps"]:
        raise ValueError(f"trace tuple {key} has no per-step state")
    for index, step in enumerate(trace["steps"]):
        robot = step.get("robot") if isinstance(step, dict) else None
        pedestrians = step.get("pedestrians") if isinstance(step, dict) else None
        if (
            not isinstance(robot, dict)
            or not _vector2(robot.get("position"))
            or not _vector2(robot.get("velocity"))
            or not isinstance(robot.get("heading"), (int, float))
            or not math.isfinite(robot["heading"])
        ):
            raise ValueError(f"trace tuple {key} step {index} lacks robot state")
        if not isinstance(pedestrians, list) or any(
            not isinstance(ped, dict)
            or not _vector2(ped.get("position"))
            or not _vector2(ped.get("velocity"))
            for ped in pedestrians
        ):
            raise ValueError(f"trace tuple {key} step {index} lacks pedestrian state")
        planner = step.get("planner") or {}
        ammv = planner.get("ammv") or {}
        forces = ammv.get("pedestrian_force_vectors")
        if pedestrians and (
            not isinstance(forces, list)
            or len(forces) != len(pedestrians)
            or not all(_vector2(force) for force in forces)
        ):
            raise ValueError(f"trace tuple {key} step {index} lacks pedestrian forces")


def _compare_row(
    key: tuple[str, str, int],
    row: dict[str, Any],
    frozen: dict[str, Any] | None,
    reference: dict[str, Any],
) -> dict[str, Any]:
    metadata = row.get("algorithm_metadata") or {}
    trace = metadata.get("simulation_step_trace") or {}
    if row.get("git_hash") != SOURCE_SHA:
        raise ValueError(f"trace tuple {key} has wrong source commit")
    _validate_steps(key, trace)
    params = row.get("scenario_params") or {}
    if any(params.get(flag) is not True for flag in TRACE_KEYS):
        raise ValueError(f"trace tuple {key} lacks requested recording flags")
    frozen_params = copy.deepcopy(reference["scenario_params"])
    rerun_params = copy.deepcopy(params)
    for flag in TRACE_KEYS:
        frozen_params.pop(flag, None)
        rerun_params.pop(flag, None)
    frozen_params["simulation_config"]["route_spawn_seed"] = key[2]
    if frozen_params != rerun_params:
        raise ValueError(f"trace tuple {key} changes release scenario/planner parameters")
    return {
        "planner": key[0],
        "scenario": key[1],
        "seed": key[2],
        "trace_status": row.get("status"),
        "release_status": frozen.get("status") if frozen else None,
        "comparison": (
            "no_release_row"
            if frozen is None
            else "match"
            if row.get("status") == frozen.get("status")
            else "mismatch"
        ),
    }


def check(
    archive: Path,
    traces: list[Path],
    config_paths: dict[str, Path],
    manifest_paths: dict[str, Path],
    *,
    expected: set[tuple[str, str, int]] | None = None,
    expected_archive_sha256: str | None = ARCHIVE_SHA256,
    expected_config_sha256: dict[str, str] | None = None,
    expected_effective_hash: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return every outcome comparison, including absent release rows."""
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    if expected_archive_sha256 is not None and archive_sha256 != expected_archive_sha256:
        raise ValueError("release archive SHA-256 does not match frozen 0.0.7 identity")
    required = EXPECTED_TUPLES if expected is None else expected
    bindings = _validate_bindings(
        config_paths,
        manifest_paths,
        required,
        CONFIG_SHA256 if expected_config_sha256 is None else expected_config_sha256,
        EFFECTIVE_HASH if expected_effective_hash is None else expected_effective_hash,
    )
    trace_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    digests: dict[str, str] = {}
    for path in traces:
        data = path.read_bytes()
        digests[str(path)] = hashlib.sha256(data).hexdigest()
        for row in _rows(data.splitlines()):
            key = _key(row)
            if key in trace_rows:
                raise ValueError(f"duplicate trace tuple {key}")
            trace_rows[key] = row
    if set(trace_rows) != required:
        raise ValueError(
            f"trace tuple inventory mismatch: missing={sorted(required - set(trace_rows))}; "
            f"extra={sorted(set(trace_rows) - required)}"
        )
    release = _release_rows(archive, {key[0] for key in trace_rows})
    comparisons = []
    for key, row in sorted(trace_rows.items()):
        reference = release.get(key)
        if reference is None:
            reference = next(
                (
                    item
                    for (planner, scenario, _seed), item in release.items()
                    if (planner, scenario) == key[:2]
                ),
                None,
            )
        if reference is None:
            raise ValueError(f"no release scenario/planner parameter reference for {key}")
        comparisons.append(_compare_row(key, row, release.get(key), reference))
    counts = {
        status: sum(row["comparison"] == status for row in comparisons)
        for status in ("match", "mismatch", "no_release_row")
    }
    return {
        "schema_version": "issue_9671_trace_release_comparison.v1",
        "source_commit": SOURCE_SHA,
        "release_archive_sha256": archive_sha256,
        "diagnostic_inputs": bindings,
        "trace_inputs_sha256": digests,
        "comparisons": comparisons,
        "comparison_counts": counts,
    }


def main() -> int:
    """Validate the diagnostic traces and write a deterministic comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-archive", required=True, type=Path)
    parser.add_argument("--traces", required=True, type=Path, nargs="+")
    parser.add_argument("--headon-config", required=True, type=Path)
    parser.add_argument("--doorway-config", required=True, type=Path)
    parser.add_argument("--headon-manifest", required=True, type=Path)
    parser.add_argument("--doorway-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = check(
        args.release_archive,
        args.traces,
        {"headon_group": args.headon_config, "doorway": args.doorway_config},
        {"headon_group": args.headon_manifest, "doorway": args.doorway_manifest},
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 2 if report["comparison_counts"]["mismatch"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
