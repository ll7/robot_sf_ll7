#!/usr/bin/env python3
"""Empty-world sweep: the 0.0.8 roster on every scenario with all pedestrians removed (#9978).

The sweep loads the 0.0.8 campaign template and its scenario matrix (and, with
``--suite width``, the 90-cell doorway width slice) exactly as the release runner does
(``load_campaign_config`` + ``load_scenarios``), removes every pedestrian through the existing
``make_actor_free_scenario`` diagnostic hook, keeps everything else identical, and executes the
result through the normal camera-ready campaign path (``run_campaign``). No simulator loop is
implemented here.

Hard rules enforced in code:

* Only development seeds are allowed. Any seed outside 1001-1030 aborts the run; the evaluation
  holdout 111-140 is therefore unreachable. The default is 1001-1002.
* The code under test is named with ``--head-sha``. The checkout must contain exactly that tree
  (apart from this script and its test), otherwise the run aborts.

Outputs (under ``--output-dir``): ``episodes.jsonl`` (one row per episode, all arms), the
per-arm x scenario ``summary.csv`` / ``summary.md``, ``README.md`` (trace format), and the raw
campaign directory with step traces.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

DEV_SEED_MIN = 1001
DEV_SEED_MAX = 1030
DEFAULT_SEEDS = (1001, 1002)
DEFAULT_MAX_WORKERS = 24
# Mirrors map_runner._validate_behavior_sanity: behaviours that need single_pedestrians.
PED_DEPENDENT_BEHAVIORS = frozenset({"wait", "join", "leave", "follow", "lead", "accompany"})

REPO_ROOT = Path(__file__).resolve().parents[2]

SUITES: dict[str, str] = {
    "main": "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml",
    "width": "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_three_width_doorway_v1.yaml",
}

# Files that may differ between the tested head and the checkout (the sweep itself).
SWEEP_OWN_PATHS = (
    "scripts/validation/run_empty_world_sweep.py",
    "tests/validation/test_run_empty_world_sweep.py",
)


class SeedGuardError(ValueError):
    """Raised when a seed outside the development range is requested."""


def assert_dev_seeds(seeds: Any) -> list[int]:
    """Return ``seeds`` as ints or abort when any seed is outside 1001-1030.

    Returns:
        The validated seed list.

    Raises:
        SeedGuardError: On an empty list, a non-integer, or any seed outside the dev range.
    """
    values = list(seeds)
    if not values:
        raise SeedGuardError("empty seed list")
    checked: list[int] = []
    for raw in values:
        if isinstance(raw, bool) or not isinstance(raw, (int, str)):
            raise SeedGuardError(f"seed {raw!r} is not an integer")
        try:
            seed = int(raw)
        except ValueError as exc:
            raise SeedGuardError(f"seed {raw!r} is not an integer") from exc
        if not DEV_SEED_MIN <= seed <= DEV_SEED_MAX:
            raise SeedGuardError(
                f"seed {seed} is outside the development range {DEV_SEED_MIN}-{DEV_SEED_MAX} "
                "(retired 111-140 and the sealed 0.0.8 seeds remain held out); aborting"
            )
        checked.append(seed)
    return checked


def remove_pedestrians(scenario: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    """Return a copy of ``scenario`` with every pedestrian removed and dev seeds set.

    Uses the repository's own actor-free diagnostic variant (crowd density 0, no
    single_pedestrians, no social groups, plus the runtime hook that also clears pedestrians
    authored in the map file). Everything else is kept.

    Returns:
        The actor-free scenario mapping.
    """
    from robot_sf.scenario_certification.feasibility_diagnostics import (
        make_actor_free_scenario,
    )

    seeds = assert_dev_seeds(seeds)
    out = make_actor_free_scenario(copy.deepcopy(scenario))
    out["seeds"] = list(seeds)
    metadata = dict(out.get("metadata") or {})
    metadata["diagnostic_variant"] = "empty_world_sweep_9978"
    behavior = str(metadata.get("behavior") or "").strip().lower()
    if behavior in PED_DEPENDENT_BEHAVIORS:
        # The map runner skips these scenarios when no single_pedestrians remain
        # (`_validate_behavior_sanity`); the label describes a pedestrian that is gone.
        metadata["empty_world_original_behavior"] = metadata["behavior"]
        metadata["behavior"] = "none"
    out["metadata"] = metadata
    return out


def pedestrian_residue(scenario: dict[str, Any]) -> list[str]:
    """List the pedestrian-related settings that still allow pedestrians in ``scenario``.

    Returns:
        Human-readable findings; empty when the scenario is pedestrian free.
    """
    problems: list[str] = []
    sim = scenario.get("simulation_config") or {}
    if float(sim.get("ped_density", 0.0) or 0.0) != 0.0:
        problems.append("ped_density != 0")
    for key in ("single_pedestrians", "social_groups", "pedestrian_flows", "ped_routes"):
        if scenario.get(key):
            problems.append(f"{key} present")
        if sim.get(key):
            problems.append(f"simulation_config.{key} present")
    if scenario.get("_diagnostic_remove_pedestrian_actors") is not True:
        problems.append("runtime map-pedestrian removal flag missing")
    return problems


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def verify_head(head_sha: str) -> str:
    """Confirm the checkout holds the tree of ``head_sha`` (plus this sweep's own files).

    Returns:
        The full commit sha of the tested head.

    Raises:
        RuntimeError: If the tree differs from the tested head or the worktree is dirty.
    """
    full = _git("rev-parse", "--verify", f"{head_sha}^{{commit}}")
    excludes = [f":(exclude){p}" for p in SWEEP_OWN_PATHS]
    diff = _git("diff", "--name-only", full, "HEAD", "--", ".", *excludes)
    if diff:
        raise RuntimeError(f"checkout differs from tested head {full}: {diff.splitlines()[:5]}")
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", ".", *excludes)
    if dirty:
        raise RuntimeError(f"worktree has uncommitted changes: {dirty.splitlines()[:5]}")
    return full


def build_derived_inputs(  # noqa: C901
    suite: str,
    *,
    seeds: list[int],
    arms: list[str] | None,
    scenarios_filter: list[str] | None,
    workers: int,
    out_dir: Path,
    step_trace: bool,
) -> tuple[Path, list[dict[str, Any]]]:
    """Write the derived scenario matrix and campaign config for ``suite``.

    Returns:
        The derived campaign config path and the actor-free scenario list.
    """
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
    from robot_sf.training.scenario_loader import load_scenarios

    seeds = assert_dev_seeds(seeds)
    out_dir = out_dir.resolve()
    template_path = REPO_ROOT / SUITES[suite]
    if not template_path.is_file():
        raise FileNotFoundError(
            f"{template_path} is missing; check out a head that carries the 0.0.8 {suite} inputs"
        )
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    matrix_rel = payload["scenario_matrix"]
    matrix_path = (REPO_ROOT / matrix_rel).resolve()
    loaded = load_scenarios(matrix_path, base_dir=matrix_path.parent)

    scenarios: list[dict[str, Any]] = []
    for scenario in loaded:
        item = dict(scenario)
        map_file = item.get("map_file")
        if isinstance(map_file, str) and not Path(map_file).is_absolute():
            candidates = [
                (matrix_path.parent / map_file).resolve(),
                (REPO_ROOT / map_file).resolve(),
            ]
            candidate = next((p for p in candidates if p.is_file()), None)
            if candidate is None:
                raise FileNotFoundError(f"unresolved source map: {map_file}")
            item["map_file"] = candidate.as_posix()
        if scenarios_filter and item["name"] not in scenarios_filter:
            continue
        cleaned = remove_pedestrians(item, seeds)
        residue = pedestrian_residue(cleaned)
        if residue:
            raise RuntimeError(f"{cleaned['name']}: pedestrian residue {residue}")
        scenarios.append(cleaned)
    if not scenarios:
        raise RuntimeError("no scenarios selected")

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_out = out_dir / f"scenarios_{suite}_empty_world.yaml"
    matrix_out.write_text(yaml.safe_dump({"scenarios": scenarios}, sort_keys=False), "utf-8")

    for slot in ("release_tag", "doi"):
        if isinstance(payload.get(slot), str) and "{{" in payload[slot]:
            payload[slot] = "empty-world-sweep-9978-unpublished"
    payload["name"] = f"{payload['name']}_empty_world_sweep_9978"
    payload["scenario_matrix"] = str(matrix_out)
    payload["seed_policy"] = {"mode": "fixed-list", "seeds": list(seeds)}
    payload["workers"] = int(workers)
    payload["resume"] = False
    payload["stop_on_failure"] = False
    payload["export_publication_bundle"] = False
    payload["paper_facing"] = False  # dev seeds are not admissible paper evidence
    payload["record_simulation_step_trace"] = bool(step_trace)
    if arms:
        known = {p["key"] for p in payload["planners"]}
        missing = set(arms) - known
        if missing:
            raise RuntimeError(f"unknown arms {sorted(missing)}")
        payload["planners"] = [p for p in payload["planners"] if p["key"] in arms]
    cfg_out = out_dir / f"campaign_{suite}_empty_world.yaml"
    cfg_out.write_text(yaml.safe_dump(payload, sort_keys=False), "utf-8")

    cfg = load_campaign_config(cfg_out)  # same loader the release runner uses
    from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios

    resolved = _load_campaign_scenarios(cfg)
    verify_actor_free(resolved, REPO_ROOT / "scoped_scenarios.json")
    seen = {int(s) for sc in resolved for s in sc.get("seeds", [])}
    assert_dev_seeds(sorted(seen))
    return cfg_out, scenarios


def verify_actor_free(scenarios: list[dict[str, Any]], matrix_path: Path) -> list[dict[str, Any]]:
    """Build each scenario's runtime config and confirm it holds no pedestrian actor.

    Returns:
        One census record per scenario.

    Raises:
        RuntimeError: If any scenario still has a pedestrian actor source.
    """
    from robot_sf.scenario_certification.v1 import scenario_actor_source_census
    from robot_sf.training.scenario_loader import build_robot_config_from_scenario

    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        config = build_robot_config_from_scenario(scenario, scenario_path=matrix_path)
        census = scenario_actor_source_census(config)
        records.append({"scenario": scenario["name"], "verified_empty": census["verified_empty"]})
        if census["verified_empty"] is not True:
            raise RuntimeError(f"{scenario['name']}: runtime config still has actors: {census}")
    return records


def find_episode_files(campaign_root: Path) -> list[Path]:
    """Return every per-arm episodes JSONL below ``campaign_root``."""
    return sorted(campaign_root.rglob("episodes.jsonl"))


def load_rows(campaign_root: Path) -> list[dict[str, Any]]:
    """Load all episode rows, tagging the arm from the run directory when absent.

    Returns:
        Episode rows with the seed guard applied.
    """
    rows: list[dict[str, Any]] = []
    for path in find_episode_files(campaign_root):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("_source_file", str(path.relative_to(campaign_root)))
            rows.append(row)
    if rows:
        assert_dev_seeds(sorted({r["seed"] for r in rows}))
    return rows


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _arm_key(row: dict[str, Any]) -> str | None:
    """Return the roster key (run directory ``<key>__<kinematics>``), not the shared algo name."""
    if row.get("_sweep_arm"):
        return row["_sweep_arm"]
    source = row.get("_source_file")
    if isinstance(source, str) and "__" in Path(source).parent.name:
        return Path(source).parent.name.split("__")[0]
    return row.get("algo")


def _trace_of(row: dict[str, Any]) -> dict[str, Any] | None:
    trace = (row.get("algorithm_metadata") or {}).get("simulation_step_trace")
    return trace if isinstance(trace, dict) else None


def _trace_geometry(trace: dict[str, Any]) -> tuple[float | None, int]:
    """Return (path length in metres, max pedestrians seen in any step) from a step trace."""
    length = 0.0
    prev = ((trace.get("reset") or {}).get("robot") or {}).get("position")
    max_peds = 0
    for step in trace.get("steps") or []:
        pos = (step.get("robot") or {}).get("position")
        if pos is not None:
            if prev is not None:
                length += math.dist(prev, pos)
            prev = pos
        max_peds = max(max_peds, len(step.get("pedestrians") or []))
    return (length if prev is not None else None), max_peds


def classify_outcome(row: dict[str, Any]) -> str:
    """Return success / collision / timeout / other for an episode row."""
    if row.get("_sweep_execution_status") in {"missing_episode", "execution_failed"}:
        return row["_sweep_execution_status"]
    outcome = row.get("outcome") or {}
    metrics = row.get("metrics") or {}
    if metrics.get("success") or outcome.get("route_complete"):
        return "success"
    if outcome.get("collision_event"):
        return "collision"
    if outcome.get("timeout_event"):
        return "timeout"
    return f"other:{row.get('termination_reason')}"


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    """Reduce an episode record to the sweep's columns.

    Returns:
        A flat mapping used for the summary tables.
    """
    m = row.get("metrics") or {}
    trace = _trace_of(row)
    # SocNavBench definitions: ratio = path length / straight-line start-to-goal displacement.
    path_len = _num(m.get("socnavbench_path_length"))
    ratio = _num(m.get("socnavbench_path_length_ratio"))
    straight = path_len / ratio if path_len is not None and ratio else None
    max_peds = None
    trace_len = None
    if trace is not None:
        trace_len, max_peds = _trace_geometry(trace)
    return {
        "arm": _arm_key(row),
        "scenario": row.get("scenario_id"),
        "seed": int(row["seed"]),
        "episode_id": row.get("episode_id"),
        "success": bool(m.get("success")),
        "outcome": classify_outcome(row),
        "termination_reason": row.get("termination_reason"),
        "steps": row.get("steps"),
        "horizon": row.get("horizon"),
        "min_clearance": _num(m.get("clearing_distance_min")),
        "collisions": m.get("total_collision_count", m.get("collisions")),
        "path_length": path_len,
        "straight_line": straight,
        "path_over_straight": ratio,
        "path_length_from_trace": trace_len,
        "max_pedestrians_in_trace": max_peds,
        "has_step_trace": trace is not None,
        "status": row.get("status"),
        "kinematics": _kinematics(row),
        "execution_status": row.get("_sweep_execution_status", "written"),
        "trace_complete": _trace_complete(row),
        "execution_error": row.get("_sweep_execution_error"),
        "source_file": row.get("_source_file"),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate flattened rows per arm x scenario.

    Returns:
        One summary row per arm x scenario.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in map(flatten, rows):
        groups.setdefault((str(row["arm"]), str(row["scenario"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (arm, scenario), items in sorted(groups.items()):
        reasons = sorted(
            {f"{i['execution_status']}:{i['outcome']}({i['termination_reason']})" for i in items}
        )
        clearances = [i["min_clearance"] for i in items if i["min_clearance"] is not None]
        ratios = [i["path_over_straight"] for i in items if i["path_over_straight"] is not None]
        out.append(
            {
                "arm": arm,
                "scenario": scenario,
                "episodes": len(items),
                "successes": sum(1 for i in items if i["success"]),
                "termination_reasons": "|".join(reasons),
                "max_pedestrians_in_trace": max(
                    (i["max_pedestrians_in_trace"] or 0) for i in items
                ),
                "steps_mean": sum(i["steps"] or 0 for i in items) / len(items),
                "min_clearance": min(clearances) if clearances else None,
                "path_over_straight_mean": sum(ratios) / len(ratios) if ratios else None,
            }
        )
    return out


def _kinematics(row: dict[str, Any]) -> str:
    """Read the runtime robot type or the arm directory suffix.

    Returns:
        The kinematics key of this episode slot.
    """
    source = row.get("_source_file")
    if isinstance(source, str) and "__" in Path(source).parent.name:
        return Path(source).parent.name.split("__", 1)[1]
    return str(
        row.get("_sweep_kinematics")
        or (row.get("scenario_params") or {}).get("robot_config", {}).get("type")
        or "differential_drive"
    )


def _trace_complete(row: dict[str, Any]) -> bool:
    """Check that requested geometry covers reset and every recorded step.

    Returns:
        True for a finite, complete simulation trace, including empty actor lists.
    """
    trace = _trace_of(row)
    if trace is None or trace.get("schema_version") not in {
        "simulation-step-trace.v1",
        "simulation-step-trace.v2",
    }:
        return False
    steps = trace.get("steps")
    dt = _num(trace.get("dt"))
    reset = ((trace.get("reset") or {}).get("robot") or {}).get("position")

    def pair(value: Any) -> bool:
        return (
            isinstance(value, list) and len(value) == 2 and all(_num(v) is not None for v in value)
        )

    if (
        not isinstance(steps, list)
        or not steps
        or len(steps) != row.get("steps")
        or not dt
        or dt <= 0
        or not pair(reset)
    ):
        return False
    return all(
        isinstance(step, dict)
        and pair((step.get("robot") or {}).get("position"))
        and isinstance(step.get("pedestrians"), list)
        for step in steps
    )


def reconcile_execution(
    cfg: Any,
    scenarios: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    root: Path,
    result: dict[str, Any],
    *,
    step_trace: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep written rows and materialize every failed or missing expected slot.

    Returns:
        Rows for all expected slots (plus duplicates/extras), and execution axes.
        Episode outcomes such as timeout are diagnostic results, not execution failures.
    """
    expected = {
        (planner.key, kin, str(sc.get("id") or sc.get("scenario_id") or sc["name"]), seed)
        for planner in cfg.planners
        for kin in cfg.kinematics_matrix
        for sc in scenarios
        for seed in sc["seeds"]
    }
    failed = {}
    for file in sorted(root.glob("runs/*/summary.json")):
        arm, sep, kin = file.parent.name.partition("__")
        if not sep:
            continue
        summary = json.loads(file.read_text())
        failures = summary.get("failures", [])
        if not isinstance(failures, list):
            raise ValueError(f"malformed failures in {file}")
        for failure in failures:
            failed[(arm, kin, str(failure["scenario_id"]), failure["seed"])] = failure.get(
                "error", "runner failure"
            )
    observed = {}
    annotated = []
    duplicates, unexpected, incomplete_trace = [], [], []
    for raw in rows:
        row = dict(raw)
        slot = (_arm_key(row), _kinematics(row), str(row.get("scenario_id")), row["seed"])
        observed[slot] = observed.get(slot, 0) + 1
        row["_sweep_execution_status"] = "written"
        if slot not in expected:
            unexpected.append(slot)
            row["_sweep_execution_status"] = "unexpected_episode"
        elif observed[slot] > 1:
            duplicates.append(slot)
            row["_sweep_execution_status"] = "duplicate_episode"
        elif step_trace and not _trace_complete(row):
            incomplete_trace.append(slot)
            row["_sweep_execution_status"] = "incomplete_trace"
        annotated.append(row)
    missing = []
    for arm, kin, scenario, seed in sorted(expected - observed.keys()):
        slot = (arm, kin, scenario, seed)
        status = (
            "execution_failed"
            if slot in failed or result.get("execution_error")
            else "missing_episode"
        )
        missing.append(slot)
        annotated.append(
            {
                "_sweep_arm": arm,
                "_sweep_kinematics": kin,
                "scenario_id": scenario,
                "seed": seed,
                "status": status,
                "termination_reason": status,
                "_sweep_execution_status": status,
                "_sweep_execution_error": failed.get(slot) or result.get("execution_error"),
            }
        )
    axes = {
        key: result.get(key)
        for key in (
            "campaign_execution_status",
            "status",
            "exit_code",
            "unexpected_failed_runs",
            "total_episodes",
            "non_success_runs",
            "accepted_unavailable_runs",
            "execution_error",
        )
    }
    runner_ok = (
        result.get("campaign_execution_status") == "completed"
        and result.get("exit_code") == 0
        and result.get("unexpected_failed_runs") == 0
    )
    axes.update(
        expected_slots=len(expected),
        written_rows=len(rows),
        failed_slots=[list(slot) for slot in sorted(failed)],
        missing_slots=[list(slot) for slot in missing],
        duplicate_slots=[list(slot) for slot in duplicates],
        unexpected_slots=[list(slot) for slot in unexpected],
        incomplete_trace_slots=[list(slot) for slot in incomplete_trace],
        trace_requested=step_trace,
    )
    axes["complete"] = (
        bool(expected)
        and runner_ok
        and not (missing or failed or duplicates or unexpected or incomplete_trace)
        and result.get("total_episodes") == len(rows)
    )
    return annotated, axes


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], suite: str, meta: dict) -> None:
    """Write episodes.jsonl, summary.csv, summary.md and per-arm counts."""
    episodes = out_dir / f"episodes_{suite}.jsonl"
    with episodes.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(flatten(row), sort_keys=True, default=str) + "\n")
    summary = summarize(rows)
    if summary:
        with (out_dir / f"summary_{suite}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    arm_counts: dict[str, list[int]] = {}
    for item in summary:
        c = arm_counts.setdefault(item["arm"], [0, 0])
        c[0] += item["successes"]
        c[1] += item["episodes"]
    lines = [f"# Empty-world sweep: {suite}", "", f"`{json.dumps(meta, sort_keys=True)}`", ""]
    lines += ["| arm | successes | episodes |", "| --- | ---: | ---: |"]
    lines += [f"| {a} | {s} | {n} |" for a, (s, n) in sorted(arm_counts.items())]
    lines += ["", "## Failing cells", ""]
    lines += ["| arm | scenario | succ/eps | termination | steps | min clearance | path/straight |"]
    lines += ["| --- | --- | ---: | --- | ---: | ---: | ---: |"]
    for it in summary:
        if it["successes"] == it["episodes"]:
            continue
        mc = "" if it["min_clearance"] is None else f"{it['min_clearance']:.2f}"
        pr = "" if it["path_over_straight_mean"] is None else f"{it['path_over_straight_mean']:.2f}"
        lines.append(
            f"| {it['arm']} | {it['scenario']} | {it['successes']}/{it['episodes']} | "
            f"{it['termination_reasons']} | {it['steps_mean']:.0f} | {mc} | {pr} |"
        )
    (out_dir / f"summary_{suite}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def describe_trace(campaign_root: Path) -> str:
    """Describe where step traces live for the README.

    Returns:
        A short markdown paragraph.
    """
    for path in find_episode_files(campaign_root):
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            continue
        line = lines[0]
        trace = _trace_of(json.loads(line))
        if trace is None:
            continue
        step = (trace.get("steps") or [{}])[0]
        return (
            f"Inline in each raw episode row (`campaigns/<id>/runs/<arm>__differential_drive/"
            f"episodes.jsonl`) at `algorithm_metadata.simulation_step_trace`, schema "
            f"`{trace.get('schema_version')}`: keys {sorted(trace)}; `dt`={trace.get('dt')}, "
            f"`initial_goal_distance_m` = straight-line start-to-goal distance, `reset` = initial "
            f"state, `steps` = one entry per step with keys {sorted(step)} "
            f"(`robot.position/heading/velocity`, `pedestrians` (empty here), "
            f"`planner.selected_action/applied_environment_action`, `rl.terminated/truncated`)."
        )
    return "No step traces were found in the episode rows."


def main(argv: list[str] | None = None) -> int:
    """Run the sweep.

    Returns:
        Process exit code: 1 for failed/incomplete execution or requested traces;
        episode outcomes alone do not cause a nonzero exit. CLI flags are unchanged.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-sha", required=True, help="Commit under test (checkout must match)")
    parser.add_argument("--suite", choices=[*SUITES, "both"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--arms", nargs="*", default=None, help="Restrict to these arm keys")
    parser.add_argument("--scenarios", nargs="*", default=None, help="Restrict scenario names")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--summarize-only", action="store_true", help="Rewrite tables from existing campaigns"
    )
    parser.add_argument("--check-only", action="store_true", help="Derive inputs, verify, stop")
    parser.add_argument("--no-step-trace", action="store_true")
    parser.add_argument("--arm-isolation", choices=("in_process", "subprocess"), default=None)
    args = parser.parse_args(argv)

    seeds = assert_dev_seeds(args.seeds)  # abort before anything else happens
    full_sha = verify_head(args.head_sha)
    sys.path.insert(0, str(REPO_ROOT))
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    from robot_sf.benchmark.camera_ready_campaign import (
        load_campaign_config,
        run_campaign,
    )

    suites = list(SUITES) if args.suite == "both" else [args.suite]
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    trace_notes: list[str] = []
    incomplete_execution = False
    for suite in suites:
        cfg_path, scenarios = build_derived_inputs(
            suite,
            seeds=seeds,
            arms=args.arms,
            scenarios_filter=args.scenarios,
            workers=args.workers,
            out_dir=out_dir,
            step_trace=not args.no_step_trace,
        )
        cfg = load_campaign_config(cfg_path)
        if args.summarize_only:
            root = out_dir / "campaigns" / f"empty_world_{suite}"
            receipt = out_dir / f"execution_{suite}.json"
            summary_path = root / "reports" / "campaign_summary.json"
            result = (
                json.loads(receipt.read_text())
                if receipt.is_file()
                else (json.loads(summary_path.read_text()) if summary_path.is_file() else {})
            )
            rows, axes = reconcile_execution(
                cfg, scenarios, load_rows(root), root, result, step_trace=not args.no_step_trace
            )
            incomplete_execution |= not axes["complete"]
            meta = {"suite": suite, "head_sha": full_sha, **axes}
            write_outputs(out_dir, rows, suite, meta)
            receipt.write_text(json.dumps(meta, indent=2) + "\n")
            continue
        if args.check_only:
            print(f"{suite}: derived inputs verified actor-free ({cfg_path})")
            continue
        cfg = load_campaign_config(cfg_path)
        t0 = time.time()
        try:
            result = run_campaign(
                cfg,
                output_root=out_dir / "campaigns",
                label=f"empty_world_{suite}",
                campaign_id=f"empty_world_{suite}",
                skip_publication_bundle=True,
                invoked_command=" ".join(sys.argv),
                arm_isolation=args.arm_isolation,
            )
        except (RuntimeError, OSError, ValueError) as error:
            result = {
                "campaign_execution_status": "failed",
                "status": "failed",
                "exit_code": 1,
                "unexpected_failed_runs": 1,
                "total_episodes": None,
                "execution_error": f"{type(error).__name__} during campaign execution",
            }
        campaign_root = Path(
            result.get("campaign_root") or out_dir / "campaigns" / f"empty_world_{suite}"
        )
        rows, axes = reconcile_execution(
            cfg,
            scenarios,
            load_rows(campaign_root),
            campaign_root,
            result,
            step_trace=not args.no_step_trace,
        )
        incomplete_execution |= not axes["complete"]
        meta = {
            "suite": suite,
            "head_sha": full_sha,
            "seeds": seeds,
            "workers": args.workers,
            "runtime_s": round(time.time() - t0, 1),
            "episodes": len(rows),
            **axes,
        }
        write_outputs(out_dir, rows, suite, meta)
        (out_dir / f"execution_{suite}.json").write_text(json.dumps(meta, indent=2) + "\n")
        trace_notes.append(f"### {suite}\n\n{describe_trace(campaign_root)}\n")
    if args.check_only or args.summarize_only:
        return int(incomplete_execution)
    readme = (
        "# Empty-world sweep output (#9978)\n\n"
        f"Head under test: `{full_sha}`. Seeds: {seeds}. Total runtime: "
        f"{time.time() - started:.0f} s.\n\n"
        "Files: `episodes_<suite>.jsonl` (one campaign episode row per line), "
        "`summary_<suite>.csv/.md` (arm x scenario), `campaigns/` (raw campaign roots).\n\n"
        "## Step-trace format\n\n" + "\n".join(trace_notes)
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return int(incomplete_execution)


if __name__ == "__main__":
    raise SystemExit(main())
