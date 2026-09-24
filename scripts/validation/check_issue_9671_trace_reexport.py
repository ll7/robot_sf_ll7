"""Compare #9671 trace episodes with the frozen 0.0.7 release rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shlex
import tarfile
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.fallback_policy import (
    resolve_execution_mode,
    runtime_fallback_or_degraded_marker,
)
from robot_sf.benchmark.utils import _config_hash

SOURCE_SHA = "07f7e8d43084de748915e1b1eb8b2a1603357c6e"
FROZEN_PUBLIC_ROOT = Path("/home/luttkule/git/robot_sf_ll7.worktrees/issue-9671-frozen-execution")
ARCHIVE_SHA256 = "684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f"
TRACE_KEYS = ("record_forces", "record_planner_decision_trace", "record_simulation_step_trace")
CONFIG_SHA256 = {
    "headon_group": "fe3ee17279a74b1ef541f68f746bc265e7b32bb29a711e862081f9f5ee1e277e",
    "doorway": "bae73148b125af5652726a12dced5d64a3e108e9783894293da2d44e7f513fb0",
}
EFFECTIVE_HASH = {"headon_group": "b195d55f16871ba2", "doorway": "a30c4555ce8a3f0a"}
DIAGNOSTIC_CONFIG = {
    "headon_group": "output/benchmarks/issue9671/inputs/issue_9671_trace_headon_group_v007.yaml",
    "doorway": "output/benchmarks/issue9671/inputs/issue_9671_trace_doorway_v007.yaml",
}
CAMPAIGN_ID = {
    "headon_group": "issue9671_trace_headon_group_v007_07f7e8d_20260924",
    "doorway": "issue9671_trace_doorway_v007_07f7e8d_20260924",
}
FROZEN_INPUT_SHA256 = {
    "robot_sf/benchmark/schemas/episode.schema.v1.json": "3c04b755fbab4764a429c034af3897eaf0db30360b81aebb390f83722105e224",
    "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml": "03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c",
    "configs/algos/social_force_terminal_goal_v1.yaml": "bcd785fff7753dd31bcb899b1bab0cfec8d0c706b053ff28cd2988ca58761afb",
    "configs/baselines/ppo_issue_791_eval_aligned_large_capacity_cpu.yaml": "51ccfbf4400a306b355e2c3f0f46eda3489d5ce3bc85beaa023a6a1da9c9fb41",
}
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


def _invocation_args(value: Any) -> list[str]:
    """Drop only the Python/script prefix from the frozen runner's invocation."""
    if not isinstance(value, str):
        raise ValueError("producer invocation is missing")
    tokens = shlex.split(value)
    for index, token in enumerate(tokens):
        if token.endswith("scripts/tools/run_camera_ready_benchmark.py"):
            return tokens[index + 1 :]
    raise ValueError("producer invocation does not use the frozen benchmark runner")


def _argument(args: list[str], flag: str) -> str:
    if args.count(flag) != 1 or args.index(flag) + 1 >= len(args):
        raise ValueError(f"producer invocation lacks unique {flag}")
    return args[args.index(flag) + 1]


def _input_matches(entry: Any, relative_path: str) -> bool:
    if not isinstance(entry, dict) or entry.get("artifact_status") != "available":
        return False
    path = entry.get("path")
    return (
        isinstance(path, str)
        and (path == relative_path or path.endswith("/" + relative_path))
        and entry.get("sha256") == FROZEN_INPUT_SHA256[relative_path]
    )


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
        invocation = _invocation_args(manifest.get("invoked_command"))
        if (
            manifest.get("git", {}).get("commit") != SOURCE_SHA
            or manifest.get("config_hash") != effective_hash[name]
            or manifest.get("campaign_id") != CAMPAIGN_ID[name]
            or not manifest.get("scenario_matrix_hash")
            or manifest.get("scenario_matrix") != config["scenario_matrix"]
            or set(manifest.get("seed_policy", {}).get("resolved_seeds", []))
            != set(config["seed_policy"]["seeds"])
            or _argument(invocation, "--config")
            != str(FROZEN_PUBLIC_ROOT / DIAGNOSTIC_CONFIG[name])
            or _argument(invocation, "--campaign-id") != manifest["campaign_id"]
            or _argument(invocation, "--mode") != "run"
        ):
            raise ValueError(f"{name} campaign manifest identity mismatch")
        bound[name] = {
            "campaign_id": manifest.get("campaign_id"),
            "scenario_matrix_hash": manifest.get("scenario_matrix_hash"),
            "tuples": sorted(selected),
            "invocation_args": invocation,
            "scenario_matrix_path": config["scenario_matrix"],
            "planner_configs": {
                planner["key"]: planner.get("algo_config") for planner in config["planners"]
            },
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


def _validate_producer_inputs(path: Path, sidecar: dict[str, Any], binding: dict[str, Any]) -> None:
    campaign = sidecar.get("campaign_identity") or {}
    inputs = sidecar.get("inputs") or {}
    if not _input_matches(
        inputs.get("schema_path"), "robot_sf/benchmark/schemas/episode.schema.v1.json"
    ) or not _input_matches(inputs.get("scenario_matrix"), binding["scenario_matrix_path"]):
        raise ValueError(f"{path} producer input hashes do not match frozen source")
    algo_config = binding["planner_configs"].get(campaign["algorithm"])
    if algo_config is None:
        entry = inputs.get("algo_config") or {}
        if entry.get("artifact_status") != "not_provided" or entry.get("sha256") is not None:
            raise ValueError(f"{path} producer algorithm input is unexpected")
    elif not _input_matches(inputs.get("algo_config"), algo_config):
        raise ValueError(f"{path} producer algorithm input hash does not match frozen source")


def _validate_producer_rows(
    path: Path, rows: list[dict[str, Any]], producer_rows: list[dict[str, Any]], planner: str
) -> None:
    if len(producer_rows) != len(rows):
        raise ValueError(f"{path} producer manifest row count mismatch")
    for index, (row, producer) in enumerate(zip(rows, producer_rows, strict=True)):
        row_hash = _config_hash(row["scenario_params"])
        if (
            producer.get("jsonl_line") != index
            or producer.get("episode_id") != row.get("episode_id")
            or producer.get("scenario_id") != row.get("scenario_id")
            or producer.get("seed") != row.get("seed")
            or producer.get("repo_commit") != SOURCE_SHA
            or producer.get("config_hash") != row_hash
            or row.get("config_hash") != row_hash
            or (row.get("result_provenance") or {}).get("config_hash") != row_hash
            or _key(row)[0] != planner
        ):
            raise ValueError(f"{path} producer manifest row {index} identity mismatch")


def _validate_producer_manifest(
    path: Path, data: bytes, rows: list[dict[str, Any]], bindings: dict[str, Any]
) -> str:
    """Bind each JSONL byte stream and row to its frozen runner sidecar and campaign."""
    sidecar_path = path.with_name(path.name + ".provenance.json")
    sidecar = json.loads(sidecar_path.read_text())
    raw_artifacts = sidecar.get("raw_artifacts") or []
    if not any(
        artifact.get("kind") == "episodes_jsonl"
        and artifact.get("sha256") == hashlib.sha256(data).hexdigest()
        for artifact in raw_artifacts
    ):
        raise ValueError(f"{path} producer manifest does not bind JSONL bytes")
    campaign = sidecar.get("campaign_identity") or {}
    matched = []
    for name, binding in bindings.items():
        campaign_root = Path(binding["campaign_manifest_path"]).parent
        if (
            path.parent.parent == campaign_root / "runs"
            # The producer's hash covers its selected planner arm, while the
            # campaign hash covers the full matrix. Exact frozen input bytes
            # are checked below, so these two hashes must not be equated.
            and isinstance(campaign.get("scenario_matrix_hash"), str)
            and bool(campaign["scenario_matrix_hash"])
            and all(_key(row) in binding["tuples"] for row in rows)
        ):
            matched.append(name)
    if len(matched) != 1:
        raise ValueError(f"{path} producer manifest has no unique diagnostic campaign binding")
    binding = bindings[matched[0]]
    if sidecar.get("run", {}).get("repo_commit") != SOURCE_SHA:
        raise ValueError(f"{path} producer manifest has wrong source commit")
    if _invocation_args(sidecar.get("run", {}).get("invocation")) != binding["invocation_args"]:
        raise ValueError(f"{path} producer invocation does not match diagnostic campaign")
    planner = path.parent.name.split("__", 1)[0]
    if campaign.get("algorithm") != planner:
        raise ValueError(f"{path} producer manifest planner mismatch")
    _validate_producer_inputs(path, sidecar, binding)
    _validate_producer_rows(path, rows, sidecar.get("rows") or [], planner)
    return hashlib.sha256(sidecar_path.read_bytes()).hexdigest()


def _vector2(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
            for x in value
        )
    )


def _validate_steps(key: tuple[str, str, int], row: dict[str, Any], trace: dict[str, Any]) -> None:
    """Require robot, pedestrian, and available force state at each recorded step."""
    if not isinstance(trace.get("steps"), list) or not trace["steps"]:
        raise ValueError(f"trace tuple {key} has no per-step state")
    count = row.get("steps")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(trace["steps"]):
        raise ValueError(f"trace tuple {key} step count does not match episode")
    if trace.get("schema_version") != "simulation-step-trace.v1":
        raise ValueError(f"trace tuple {key} has wrong step-trace schema")
    dt = trace.get("dt")
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not math.isfinite(dt)
        or dt <= 0
        or not math.isclose(dt, row["scenario_params"]["run_dt"], abs_tol=1e-12)
    ):
        raise ValueError(f"trace tuple {key} has wrong step-trace timestep")
    for index, step in enumerate(trace["steps"]):
        if (
            not isinstance(step, dict)
            or isinstance(step.get("step"), bool)
            or not isinstance(step.get("step"), int)
            or step.get("step") != index
            or isinstance(step.get("time_s"), bool)
            or not isinstance(step.get("time_s"), (int, float))
            or not math.isfinite(step["time_s"])
            or not math.isclose(step["time_s"], (index + 1) * dt, abs_tol=1e-9)
        ):
            raise ValueError(f"trace tuple {key} step {index} has noncontiguous index/time")
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
    mode = resolve_execution_mode(metadata)
    runtime = {
        field: metadata[field]
        for field in (
            "execution_mode",
            "planner_kinematics",
            "adapter_impact",
            "planner_runtime",
            "planner_diagnostics",
            "foresight_prediction",
        )
        if field in metadata
    }
    marker = runtime_fallback_or_degraded_marker(runtime)
    if mode not in {"native", "adapter"} or marker is not None:
        raise ValueError(f"trace tuple {key} has inadmissible runtime mode {mode}: {marker}")
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
    _validate_steps(key, row, trace)
    trace_exposure = (row.get("interaction_exposure") or {}).get("interaction_exposure_steps")
    release_exposure = (
        (frozen.get("interaction_exposure") or {}).get("interaction_exposure_steps")
        if frozen
        else None
    )
    return {
        "planner": key[0],
        "scenario": key[1],
        "seed": key[2],
        "trace_status": row.get("status"),
        "release_status": frozen.get("status") if frozen else None,
        "trace_interaction_exposure_steps": trace_exposure,
        "release_interaction_exposure_steps": release_exposure,
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
    producer_digests: dict[str, str] = {}
    for path in traces:
        data = path.read_bytes()
        digests[str(path)] = hashlib.sha256(data).hexdigest()
        rows = _rows(data.splitlines())
        producer_digests[str(path.with_name(path.name + ".provenance.json"))] = (
            _validate_producer_manifest(path, data, rows, bindings)
        )
        for row in rows:
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
        "producer_manifests_sha256": producer_digests,
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
