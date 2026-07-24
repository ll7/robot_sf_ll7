"""Adapter: convert re-exported PPO episode rows (job-13483 doorway re-execution) into
the ``trace_series.json`` / ``metadata.json`` bundle schema consumed by
``scripts/repro/butterfly_hinge_figure_proto.py`` (via
``robot_sf.benchmark.trace_scene_figure.load_episode`` and that script's own
``EpisodeTrace`` loader).

STATUS: adapter for Stage 5-6 of the butterfly worked-example pipeline (see
``butterfly_hinge_figure_proto.py``'s own module docstring for the larger context);
this module only performs the schema conversion plus a small provenance-sidecar
augmentation step -- it renders nothing itself.

Source data
-----------
``output/benchmarks/doorway_butterfly_trace_reexport/job-13483/doorway_butterfly_ppo_a307/
runs/ppo__differential_drive/episodes.jsonl`` -- one row per episode (``paper_eval_s30``
seed set, seeds 111-140), produced by a pinned-commit re-execution (exec commit
``a307ef276d701f8d14dead1aa0513f44ee97c0b0``, Slurm job 13483, config
``configs/benchmarks/doorway_butterfly_trace_reexport.yaml``) whose outcomes reproduce
the release for 28/30 seeds -- see
``output/benchmarks/doorway_butterfly_trace_reexport/job-13483/PER_SEED_DIFF.md`` for the
full release-vs-rerun diff this adapter's provenance augmentation quotes from.

Each row's per-episode step trace lives at
``row["algorithm_metadata"]["simulation_step_trace"]`` (schema
``simulation-step-trace.v1``: ``{dt, initial_goal_distance_m, schema_version, steps}``).
Each entry of ``steps`` already has EXACTLY the shape one ``trace_series.json``
``frames[]`` entry needs (``{pedestrians, planner, rl, robot, step, time_s}`` -- verified
against the reference bundle below field-for-field), so ``steps`` is copied verbatim into
``frames``; no per-field remapping happens there.

Target schema
-------------
Matched against the reference bundle
``docs/context/evidence/issue_4891_head_on_corridor_exemplars_2026-07/orca/
classic_head_on_corridor_medium_seed24_best/{trace_series.json,metadata.json}`` -- the
shape actually enforced by ``robot_sf.benchmark.trace_scene_figure.load_episode`` /
``_require_metadata`` / ``_parse_derived_rows`` / ``_parse_frames``, and read by
``butterfly_hinge_figure_proto.load_episode`` + ``compute_trace_metrics``:

``trace_series.json``:
  ``schema_version``  This adapter's own container-format tag, NOT the issue-4891 tag --
                       this is a different generation pipeline (re-export, not a
                       campaign-selection bundle).
  ``metadata``         Same dict as ``metadata.json``, minus the top-level
                       ``review_marker`` key that sits beside it there (see below --
                       mirrors how the reference bundle duplicates that field).
  ``frames``           ``row["algorithm_metadata"]["simulation_step_trace"]["steps"]``,
                       copied verbatim.
  ``derived_rows``     One row per frame. REQUIRED by
                       ``trace_scene_figure._parse_derived_rows``: ``step``, ``time_s``,
                       ``robot_x_m``, ``robot_y_m``, ``robot_heading_rad``,
                       ``executed_speed_m_s``, ``min_robot_ped_distance_m``,
                       ``nearest_pedestrian_id``. Plus the reference bundle's
                       non-required-but-present columns (``commanded_*``,
                       ``executed_v{x,y}_m_s``, ``pedestrian_count``,
                       ``pedestrian_positions_json``) -- all straightforward per-step
                       derivations from the same frame, not invented.
  ``review_marker``    ``"AI-GENERATED"`` -- this bundle is machine-generated; no
                       ``NEEDS-REVIEW`` suffix, because the one non-verbatim derivation
                       (nearest-pedestrian distance/id) uses the *identical* algorithm
                       ``butterfly_trace_to_video_proto.compute_trace_metrics`` uses on
                       the emitted ``frames``, so the render script independently
                       re-derives the same numbers rather than trusting a copied value.

``metadata.json``: the same dict as ``trace_series.json["metadata"]``, plus a top-level
  ``review_marker`` key (mirrors the reference bundle's own duplication of that field).

Fields OMITTED vs. the issue-4891 reference (no source in this row, or not applicable --
NOT silently dropped; see ``_build_metadata`` for where each decision is made):

- ``selection_metric`` / ``selection_metric_value`` / ``selection_mode``: the issue-4891
  bundles were picked from a campaign by a best/worst/median path-efficiency rule; these
  two seeds (113, 114) were picked directly (113 = success repro, 114 = collision repro),
  not by that selection rule. Neither field is read by any loader in this repo (grepped
  ``trace_scene_figure.py`` and ``butterfly_hinge_figure_proto.py``), so omitting them
  loses no required information.
- ``issue``: the issue-4891 bundles cite the GitHub issue that requested them; this
  re-export has no single equivalent issue, so the field is omitted.
- ``claim_boundary``: intentionally left out of ``metadata.json`` / ``trace_series.json``
  to avoid a second, possibly-drifting copy of the release-vs-rerun bounding language.
  The authoritative bounding statements live in the provenance sidecar (see
  ``augment_provenance_sidecar`` below) -- the artifact actually attached to the
  rendered figure.
- ``README.md`` / ``SHA256SUMS`` / ``*.csv`` sidecar files present in the issue-4891
  bundle directories: none of them are read by ``trace_scene_figure.load_episode`` or
  ``butterfly_hinge_figure_proto.load_episode`` (grepped both modules), so they are
  bookkeeping for that issue's own bundle-collection process, not part of the schema
  this adapter needs to reproduce.

Fields DERIVED (not verbatim-copied, not invented -- see inline comments at each call
site for the exact source field(s)):

- ``derived_rows[].min_robot_ped_distance_m`` / ``nearest_pedestrian_id``: nearest
  robot-pedestrian center-to-center distance per step, computed with the identical
  algorithm ``butterfly_trace_to_video_proto.compute_trace_metrics`` uses (strict
  less-than nearest-neighbor scan over ``frame["pedestrians"]`` in list order) -- so
  this adapter's own numbers agree with what the render script independently
  re-derives from the ``frames`` this adapter emits, by construction.
- ``metadata.summary.global_min_robot_ped_distance_m`` / ``global_min_distance_step``:
  global min / argmin of that same per-step series.
- ``metadata.episode_status``: ``row["status"]`` verbatim -- already one of
  ``"success"`` / ``"collision"`` / ``"failure"``, the exact strings
  ``butterfly_hinge_figure_proto.main``'s outcome-kind branch checks for.

Usage::

    uv run python scripts/repro/butterfly_reexport_to_trace_series.py build-bundle \\
        --episodes-jsonl output/benchmarks/doorway_butterfly_trace_reexport/job-13483/doorway_butterfly_ppo_a307/runs/ppo__differential_drive/episodes.jsonl \\
        --seed 113 \\
        --out-dir output/butterfly_hinge_doorway/bundles/seed113

    uv run python scripts/repro/butterfly_reexport_to_trace_series.py augment-sidecar \\
        --provenance-json output/butterfly_hinge_doorway/butterfly_hinge_provenance.json \\
        --episodes-jsonl output/benchmarks/doorway_butterfly_trace_reexport/job-13483/doorway_butterfly_ppo_a307/runs/ppo__differential_drive/episodes.jsonl \\
        --seed-b 114
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pinned-commit re-execution facts (job-13483). These are documented, verified
# facts about how the source episodes.jsonl was produced -- not re-derived here,
# just carried through into the bundle metadata / provenance sidecar so the
# emitted artifacts are self-describing without a second lookup.
# ---------------------------------------------------------------------------

EXEC_COMMIT: str = "a307ef276d701f8d14dead1aa0513f44ee97c0b0"
SLURM_JOB: str = "13483"
CONFIG_PATH: str = "configs/benchmarks/doorway_butterfly_trace_reexport.yaml"
SEED_SET_NAME: str = "paper_eval_s30"

#: Bounding statements, author-ruled 2026-07-16 (see
#: output/benchmarks/doorway_butterfly_trace_reexport/job-13483/PER_SEED_DIFF.md,
#: "Usage rules for the figure" section) -- quoted verbatim into the provenance
#: sidecar so the figure's evidentiary scope is explicit wherever the sidecar travels.
BOUNDING_STATEMENT_I: str = (
    "traces from a pinned-commit re-execution (job 13483) whose outcomes reproduce the "
    "release for both episodes; per-step detail is not the release execution"
)
BOUNDING_STATEMENT_II: str = (
    "re-execution outcome fidelity 28/30 vs the release cell; seeds 128 and 130 flipped "
    "success-to-collision in the rerun and are excluded from any claim; the release "
    "cell's 15/15 split is release-surface evidence and is deliberately NOT annotated on "
    "this figure (author ruling 2026-07-16, option b)"
)

#: Seed-114 near-miss counts. The rerun number is read from episodes.jsonl at
#: augment-sidecar time (not hardcoded); the release number is NOT re-derivable in this
#: worktree (the release bundle is a local Mac artifact outside this checkout -- see
#: PER_SEED_DIFF.md's own provenance note), so it is carried here as a literal, cited
#: fact rather than fabricated or silently omitted.
RELEASE_SEED_114_NEAR_MISSES: int = 62
RELEASE_SEED_114_NEAR_MISSES_SOURCE: str = (
    "output/benchmarks/doorway_butterfly_trace_reexport/job-13483/PER_SEED_DIFF.md "
    '("seed 114: near 62->78, ..."); release bundle itself is not present in this '
    "worktree, so this number is carried as a cited literal, not independently "
    "recomputed here"
)
FLIPPED_SEEDS: tuple[int, ...] = (128, 130)
OUTCOME_FIDELITY: str = "28/30"


@dataclass(frozen=True)
class ReexportArmSpec:
    """Pinned provenance for one worked-example re-export arm.

    Carries the fail-closed expectations ``build_bundle`` checks each episode row
    against. Campaign/job labels are deliberately not stored here: those identifiers
    must come from the source row's ``result_provenance`` rather than being invented by
    the selected arm. Keeping the fail-closed identity pins here lets one adapter serve
    both #5756 worked-example pairs from the same renderer.
    """

    key: str
    planner: str
    scenario_id: str
    execution_commit: str


#: The pinned doorway PPO butterfly arm (job 13483). Outcomes reproduce the release
#: for 28/30 seeds; seeds 128/130 flipped and are excluded from any release claim.
DOORWAY_ARM = ReexportArmSpec(
    key="doorway_ppo",
    planner="ppo",
    scenario_id="classic_doorway_medium",
    execution_commit=EXEC_COMMIT,
)

#: The pinned double-bottleneck goal arm (job 13487). 30/30 outcome-faithful vs release.
BOTTLENECK_GOAL_ARM = ReexportArmSpec(
    key="double_bottleneck_goal",
    planner="goal",
    scenario_id="classic_realworld_double_bottleneck_high",
    execution_commit=EXEC_COMMIT,
)

#: The pinned double-bottleneck PPO arm (job 13488). 30/30 outcome-faithful vs release.
BOTTLENECK_PPO_ARM = ReexportArmSpec(
    key="double_bottleneck_ppo",
    planner="ppo",
    scenario_id="classic_realworld_double_bottleneck_high",
    execution_commit=EXEC_COMMIT,
)

ARMS: dict[str, ReexportArmSpec] = {
    DOORWAY_ARM.key: DOORWAY_ARM,
    BOTTLENECK_GOAL_ARM.key: BOTTLENECK_GOAL_ARM,
    BOTTLENECK_PPO_ARM.key: BOTTLENECK_PPO_ARM,
}

# Backward-compatible doorway aliases retained for existing callers and tests. New
# multi-arm code should use ``DOORWAY_ARM`` directly.
CAMPAIGN_ID: str = "doorway_butterfly_ppo_a307"
EXPECTED_ALGO: str = DOORWAY_ARM.planner
EXPECTED_SCENARIO_ID: str = DOORWAY_ARM.scenario_id


# ---------------------------------------------------------------------------
# build-bundle
# ---------------------------------------------------------------------------


def _load_row(episodes_jsonl: Path, seed: int) -> dict[str, Any]:
    """Find the one row in ``episodes_jsonl`` with ``row["seed"] == seed``.

    Returns:
        The matching row.

    Raises:
        ValueError: if zero or more than one row matches (both would indicate the
            input file does not have the expected one-row-per-episode shape).
    """
    matches = []
    with episodes_jsonl.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("seed") == seed:
                matches.append(row)
    if not matches:
        raise ValueError(f"no row with seed={seed} found in {episodes_jsonl}")
    if len(matches) > 1:
        raise ValueError(f"{len(matches)} rows with seed={seed} found in {episodes_jsonl}")
    return matches[0]


def _validate_source_row(row: dict[str, Any], arm: ReexportArmSpec) -> dict[str, Any]:
    """Fail closed when an episode does not match its pinned re-export arm.

    Every arm must carry row-level provenance. This prevents a file containing an
    arbitrary matching planner/scenario/seed row from inheriting a trusted-looking arm
    identity. The validated source object is returned so the emitted bundle can carry
    the actual provenance rather than hard-coded campaign or job labels.

    Returns:
        A copy of the validated ``result_provenance`` object.
    """
    expected = {
        "git_hash": arm.execution_commit,
        "algo": arm.planner,
        "scenario_id": arm.scenario_id,
    }
    mismatches = [
        f"{key}={row.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if row.get(key) != value
    ]
    result_provenance = row.get("result_provenance")
    if not isinstance(result_provenance, dict):
        mismatches.append("result_provenance is missing or not an object")
    else:
        provenance_expectations = {
            "repo_commit": arm.execution_commit,
            "scenario_id": arm.scenario_id,
            "seed": row.get("seed"),
        }
        if row.get("config_hash") is not None:
            provenance_expectations["config_hash"] = row.get("config_hash")
        mismatches.extend(
            f"result_provenance.{key}={result_provenance.get(key)!r} (expected {value!r})"
            for key, value in provenance_expectations.items()
            if result_provenance.get(key) != value
        )
    if mismatches:
        arm_label = "doorway" if arm == DOORWAY_ARM else arm.key
        raise ValueError(
            f"episode row does not match pinned {arm_label} re-export provenance: "
            + "; ".join(mismatches)
        )
    return dict(result_provenance)


def _nearest_pedestrian(frame: dict[str, Any]) -> tuple[float, int | None]:
    """Center-to-center distance and id of the nearest pedestrian in one frame.

    Uses the exact same strict-less-than nearest-neighbor scan (in
    ``frame["pedestrians"]`` list order, so ties resolve identically) as
    ``scripts/repro/butterfly_trace_to_video_proto.compute_trace_metrics`` -- the
    function that will independently re-derive this same value from the ``frames``
    this adapter emits, so the two stay in agreement by construction.

    Returns:
        ``(best_distance, best_id)``; ``best_id`` is ``None`` if the frame has no
        pedestrians (``best_distance`` is then ``inf``, which the caller must not
        write into ``derived_rows`` -- see ``_build_derived_rows``).
    """
    rx, ry = frame["robot"]["position"]
    best_dist = math.inf
    best_id: int | None = None
    for ped in frame.get("pedestrians", []):
        px, py = ped["position"]
        dist = math.hypot(rx - px, ry - py)
        if dist < best_dist:
            best_dist = dist
            best_id = ped.get("id")
    return best_dist, best_id


def _build_derived_rows(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the ``derived_rows`` array required by
    ``trace_scene_figure._parse_derived_rows``, one row per frame.

    Returns:
        List of derived-row dicts, in frame order.

    Raises:
        ValueError: if any frame has zero pedestrians (nearest-pedestrian distance
            would be non-finite, which the downstream schema validator rejects) --
            a genuine data limitation this adapter refuses to paper over with a
            fabricated distance.
    """
    rows: list[dict[str, Any]] = []
    for frame in frames:
        dist, ped_id = _nearest_pedestrian(frame)
        if not math.isfinite(dist):
            raise ValueError(
                f"frame step={frame.get('step')} has zero pedestrians -- cannot compute "
                "a finite nearest-pedestrian distance for derived_rows"
            )
        rx, ry = frame["robot"]["position"]
        vx, vy = frame["robot"]["velocity"]
        action = frame.get("planner", {}).get("selected_action", {})
        peds = frame.get("pedestrians", [])
        rows.append(
            {
                "step": frame["step"],
                "time_s": frame["time_s"],
                "robot_x_m": rx,
                "robot_y_m": ry,
                "robot_heading_rad": frame["robot"]["heading"],
                "executed_speed_m_s": math.hypot(vx, vy),
                "executed_vx_m_s": vx,
                "executed_vy_m_s": vy,
                "commanded_linear_velocity_m_s": action.get("linear_velocity"),
                "commanded_angular_velocity_rad_s": action.get("angular_velocity"),
                "min_robot_ped_distance_m": dist,
                "nearest_pedestrian_id": str(ped_id) if ped_id is not None else None,
                "pedestrian_count": len(peds),
                "pedestrian_positions_json": json.dumps(
                    [
                        {"id": str(p["id"]), "x_m": p["position"][0], "y_m": p["position"][1]}
                        for p in peds
                    ]
                ),
            }
        )
    return rows


def _build_metadata(
    row: dict[str, Any],
    derived_rows: list[dict[str, Any]],
    arm: ReexportArmSpec,
    result_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build the ``metadata`` dict shared by ``trace_series.json["metadata"]`` and
    ``metadata.json`` (minus the ``review_marker`` key ``metadata.json`` adds on top --
    see the module docstring's field-omission notes for what is deliberately absent
    here vs. the issue-4891 reference).

    Returns:
        Metadata dict satisfying ``trace_scene_figure._require_metadata``.
    """
    dists = [r["min_robot_ped_distance_m"] for r in derived_rows]
    global_min_idx = min(range(len(dists)), key=lambda i: dists[i])
    metadata = {
        "episode_id": row["episode_id"],
        # episode_status: row["status"] verbatim -- already one of
        # "success"/"collision"/"failure", the exact strings
        # butterfly_hinge_figure_proto.main's outcome-kind branch checks for.
        "episode_status": row["status"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": row["git_hash"],
        "planner": row["algo"],
        "result_provenance": result_provenance,
        "scenario_id": row["scenario_id"],
        "schema_version": "butterfly-reexport-exemplar-trace.v1",
        "seed": row["seed"],
        "source_arm": arm.key,
        "summary": {
            "episode_status": row["status"],
            "global_min_distance_step": derived_rows[global_min_idx]["step"],
            "global_min_robot_ped_distance_m": dists[global_min_idx],
            "planner": row["algo"],
            "scenario_id": row["scenario_id"],
            "seed": row["seed"],
            "step_count": len(derived_rows),
            "termination_reason": row["termination_reason"],
        },
    }
    # Preserve source-provided campaign/job locators when present, but never invent
    # them from the selected arm. Older benchmark rows omit these fields, in which
    # case the bundle remains honest and simply carries no unverified locator.
    campaign_id = result_provenance.get("campaign_id")
    if isinstance(campaign_id, str) and campaign_id.strip():
        metadata["campaign_id"] = campaign_id
    campaign_job = result_provenance.get("slurm_job_id", result_provenance.get("campaign_job"))
    if isinstance(campaign_job, (str, int)) and not isinstance(campaign_job, bool):
        metadata["campaign_job"] = str(campaign_job)
    return metadata


def build_bundle(
    episodes_jsonl: Path, seed: int, out_dir: Path, *, arm: ReexportArmSpec = DOORWAY_ARM
) -> dict[str, Any]:
    """Convert one episode row into a ``trace_series.json`` + ``metadata.json`` bundle.

    Args:
        episodes_jsonl: One-row-per-episode rerun output for ``arm``.
        seed: The episode seed to isolate.
        out_dir: Destination bundle directory.
        arm: The pinned re-export arm whose provenance the row must match.

    Returns:
        Small summary dict (paths written, step count, outcome) for CLI reporting.
    """
    row = _load_row(episodes_jsonl, seed)
    result_provenance = _validate_source_row(row, arm)
    sst = row["algorithm_metadata"]["simulation_step_trace"]
    if sst.get("schema_version") != "simulation-step-trace.v1":
        raise ValueError(
            f"unexpected simulation_step_trace schema_version: {sst.get('schema_version')!r}"
        )
    frames = sst["steps"]  # already {pedestrians, planner, rl, robot, step, time_s} -- verbatim
    if not isinstance(frames, list) or not frames:
        raise ValueError("simulation_step_trace.steps must be a non-empty array")
    derived_rows = _build_derived_rows(frames)
    metadata = _build_metadata(row, derived_rows, arm, result_provenance)

    trace_series = {
        "schema_version": "butterfly-reexport-trace-series.v1",
        "metadata": metadata,
        "frames": frames,
        "derived_rows": derived_rows,
        "review_marker": "AI-GENERATED",
    }
    metadata_json = dict(metadata)
    metadata_json["review_marker"] = "AI-GENERATED"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trace_series.json").write_text(json.dumps(trace_series, indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(metadata_json, indent=2), encoding="utf-8")

    return {
        "seed": seed,
        "episode_id": row["episode_id"],
        "scenario_id": row["scenario_id"],
        "planner": row["algo"],
        "episode_status": row["status"],
        "n_steps": len(frames),
        "global_min_distance_step": metadata["summary"]["global_min_distance_step"],
        "global_min_robot_ped_distance_m": metadata["summary"]["global_min_robot_ped_distance_m"],
        "out_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# augment-sidecar
# ---------------------------------------------------------------------------


def augment_provenance_sidecar(
    provenance_json: Path, episodes_jsonl: Path, seed_b: int
) -> dict[str, Any]:
    """Add the release-vs-rerun bounding context to an existing
    ``butterfly_hinge_provenance.json`` (written by
    ``butterfly_hinge_figure_proto.build_provenance_sidecar``), under a new top-level
    ``release_reexport_provenance`` key. Does not touch any other key in the file.

    Returns:
        The full, updated provenance dict (also written back to ``provenance_json``).
    """
    provenance = json.loads(provenance_json.read_text(encoding="utf-8"))
    row_b = _load_row(episodes_jsonl, seed_b)
    rerun_near_misses_b = row_b["metrics"]["near_misses"]

    provenance["release_reexport_provenance"] = {
        "execution_commit": EXEC_COMMIT,
        "slurm_job": SLURM_JOB,
        "config": CONFIG_PATH,
        "seed_set": SEED_SET_NAME,
        "bounding_statement_i": BOUNDING_STATEMENT_I,
        "bounding_statement_ii": BOUNDING_STATEMENT_II,
        "outcome_fidelity_vs_release": OUTCOME_FIDELITY,
        "flipped_seeds_excluded_from_claim": list(FLIPPED_SEEDS),
        f"seed_{seed_b}_near_misses": {
            "rerun_execution": rerun_near_misses_b,
            "rerun_source": str(episodes_jsonl),
            "release_execution": RELEASE_SEED_114_NEAR_MISSES,
            "release_source": RELEASE_SEED_114_NEAR_MISSES_SOURCE,
            "note": (
                "any near-miss count shown or captioned for this seed must be the "
                "rerun execution's number (matches the traces this figure is rendered "
                "from), not the release row's number"
            ),
        },
        "source_diff_doc": (
            "output/benchmarks/doorway_butterfly_trace_reexport/job-13483/PER_SEED_DIFF.md"
        ),
    }
    provenance_json.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    return provenance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser(
        "build-bundle", help="Convert one episode row into a trace_series.json bundle dir."
    )
    build_p.add_argument("--episodes-jsonl", type=Path, required=True)
    build_p.add_argument("--seed", type=int, required=True)
    build_p.add_argument("--out-dir", type=Path, required=True)
    build_p.add_argument(
        "--arm",
        choices=tuple(ARMS),
        default=DOORWAY_ARM.key,
        help="Pinned re-export arm whose provenance the row must match.",
    )

    aug_p = sub.add_parser(
        "augment-sidecar",
        help="Add release-vs-rerun bounding context to an existing provenance sidecar.",
    )
    aug_p.add_argument("--provenance-json", type=Path, required=True)
    aug_p.add_argument("--episodes-jsonl", type=Path, required=True)
    aug_p.add_argument(
        "--seed-b", type=int, required=True, help="Seed of the collision (B) episode."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    if args.command == "build-bundle":
        summary = build_bundle(args.episodes_jsonl, args.seed, args.out_dir, arm=ARMS[args.arm])
        print(json.dumps(summary, indent=2))
    elif args.command == "augment-sidecar":
        provenance = augment_provenance_sidecar(
            args.provenance_json, args.episodes_jsonl, args.seed_b
        )
        print(json.dumps(provenance["release_reexport_provenance"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
