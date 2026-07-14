#!/usr/bin/env python3
"""Wire pipeline-generated scenario candidates through the persistence promotion gate.

This tool connects the stage-1 scenario-generation pipeline (issue #4932) to the
``generated_scenario_persistence.v1`` promotion gate (issue #5600).  It loads
pipeline artifacts (run manifest, catalog entries, episode traces), evaluates
the three independent persistence checks for each candidate, and writes
schema-valid persistence records as evidence JSON files.

CPU-only perturbation mode
--------------------------
The script supports two perturbation-evaluation modes:

* **cpu** (default, requires no simulator): uses the source trace to simulate
  timing/speed perturbations by shifting trace timestamps and interpolating
  pedestrian positions.  This exercises the wiring end-to-end but does **not**
  prove that the perturbation would survive a real CARLA replay.  Production
  use should supply pre-computed replay results via ``--replay-results``.

* **precomputed** (``--replay-results PATH``): loads replay/perturbation results
  from an external JSON file produced by a simulation-capable runner.  The
  script skips CPU-only heuristics and uses the supplied verdicts directly.

Usage
-----

  uv run python scripts/tools/evaluate_pipeline_persistence_gate.py \\
      --manifest path/to/run_manifest.json \\
      --config configs/analysis/issue_5600_persistence_gate.yaml \\
      --output path/to/output_dir \\
      [--replay-results path/to/replay_results.json] \\
      [--commit-hash HASH --config-hash HASH]

  uv run python scripts/tools/evaluate_pipeline_persistence_gate.py \\
      --candidate path/to/catalog_entry.yaml \\
      --episodes path/to/episodes.jsonl \\
      --config configs/analysis/issue_5600_persistence_gate.yaml \\
      --output path/to/output_dir
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.benchmark.scenario_generation.catalog_schema import (
    CATALOG_ENTRY_SCHEMA_VERSION,
    validate_catalog_entry,
)
from robot_sf.benchmark.scenario_generation.persistence_gate import (
    ScenarioPersistenceValidationError,
    assess_critical_event_reproduction,
    assess_exact_replay,
    compute_persistence_record,
    evaluate_perturbation_grid,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest",
        type=Path,
        help="Path to a pipeline run_manifest.json produced by run_generation_pipeline.",
    )
    source.add_argument(
        "--candidate",
        type=Path,
        help="Path to a single generated-scenario catalog entry (YAML or JSON).",
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        help=(
            "Path to the pipeline's episodes.jsonl (required with --candidate; "
            "resolved from the manifest otherwise)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the frozen persistence gate config YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for persistence record JSON files.",
    )
    parser.add_argument(
        "--replay-results",
        type=Path,
        help=(
            "Path to pre-computed replay/perturbation results JSON.  When absent, "
            "CPU-only mode derives replay and perturbation verdicts from the source trace."
        ),
    )
    parser.add_argument(
        "--commit-hash",
        default="unknown",
        help="Git commit hash for the code (default: 'unknown').",
    )
    parser.add_argument(
        "--config-hash",
        default="unknown",
        help="Git commit hash for the config (default: 'unknown').",
    )
    return parser


def evaluate_pipeline_candidates(
    manifest_path: Path | None,
    candidate_path: Path | None,
    episodes_path: Path | None,
    config_path: Path,
    output_dir: Path,
    replay_results_path: Path | None = None,
    commit_hash: str = "unknown",
    config_hash: str = "unknown",
) -> list[dict[str, Any]]:
    """Load pipeline artifacts and produce persistence records for each candidate.

    Returns:
        A list of schema-valid persistence records (one per candidate).
    """

    frozen_config = _load_frozen_config(config_path)
    commit_hashes = {"code": commit_hash, "config": config_hash}

    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_paths = manifest.get("artifacts", {})
        resolved_episodes = (
            episodes_path
            or _resolve_artifact(manifest_path.parent, artifact_paths.get("episodes", ""))
            or None
        )
        resolved_catalog = _resolve_artifact(
            manifest_path.parent, artifact_paths.get("generated_catalog", "")
        )
        if resolved_episodes is None or not resolved_episodes.exists():
            raise FileNotFoundError(
                f"episodes JSONL not found: {resolved_episodes}. "
                "Supply --episodes or check the manifest's artifacts.episodes path."
            )
        if resolved_catalog is None or not resolved_catalog.exists():
            raise FileNotFoundError(
                f"catalog YAML not found: {resolved_catalog}. "
                "Supply --candidate directly or check the manifest's artifacts.generated_catalog path."
            )
        raw = yaml.safe_load(resolved_catalog.read_text(encoding="utf-8"))
        entries = list(raw.get("entries", []) if isinstance(raw, dict) else [])
    elif candidate_path is not None:
        if episodes_path is None or not episodes_path.exists():
            raise FileNotFoundError(
                "--episodes is required with --candidate and must point to a valid episodes.jsonl"
            )
        resolved_episodes = episodes_path
        raw = _load_candidate_entry(candidate_path)
        entries = [raw]
    else:
        raise ValueError("one of --manifest or --candidate is required")

    episodes_index = _build_episodes_index(resolved_episodes)

    if replay_results_path is not None:
        replay_data = json.loads(replay_results_path.read_text(encoding="utf-8"))
    else:
        replay_data = None

    records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for entry_idx, entry in enumerate(entries):
        record = _evaluate_one_candidate(
            entry=entry,
            entry_idx=entry_idx,
            frozen_config=frozen_config,
            commit_hashes=commit_hashes,
            episodes_index=episodes_index,
            replay_data=replay_data,
        )
        if record is None:
            continue
        records.append(record)

        scenario_id = record["scenario_id"]
        out_path = output_dir / f"{scenario_id}.persistence.json"
        out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        verdict = record["promotion"]["verdict"]
        print(f"{verdict.upper()} {out_path} :: {record['promotion']['exclusion_reason']}")

    return records


def _evaluate_one_candidate(
    *,
    entry: Mapping[str, Any],
    entry_idx: int,
    frozen_config: Mapping[str, Any],
    commit_hashes: Mapping[str, str],
    episodes_index: Mapping[str, dict[str, Any]],
    replay_data: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a persistence record for one catalog entry, or None when its trace is missing."""

    validate_catalog_entry(entry)
    scenario_id = str(entry.get("scenario_id", f"candidate-{entry_idx:04d}"))
    source_episode_info = entry.get("source_episode", {})
    episode_id = str(source_episode_info.get("episode_id", ""))
    source_seed = int(source_episode_info.get("source_seed", 0))
    source_map = str(source_episode_info.get("source_map", ""))

    trace_episode = episodes_index.get(episode_id)
    if trace_episode is None:
        print(
            f"WARNING: episode {episode_id!r} not found in episodes data; skipping {scenario_id}",
            file=sys.stderr,
        )
        return None

    replayed: dict[str, Any] | None = None
    replay_error: str | None = None
    if replay_data is not None:
        replay_info = replay_data.get(scenario_id) or replay_data.get(episode_id) or {}
        if "error" in replay_info:
            replay_error = replay_info["error"]
        elif "episode" in replay_info:
            replayed = replay_info["episode"]
    else:
        # CPU-only mode: use source trace as its own exact replay.
        replayed = _build_episode_fields(trace_episode, source_map)

    source_episode_fields = _build_source_episode_fields(entry, trace_episode)

    # Exact replay: the source episode is byte/config-equivalent to itself.
    exact_replay_block = assess_exact_replay(
        source_episode_fields,
        replayed_episode=replayed,
        replay_error=replay_error,
    )

    # Critical event reproduction.
    critical_block = _evaluate_critical_event(entry, trace_episode, replayed)

    # Perturbation grid.
    config_grid = frozen_config.get("perturbation", {})
    timing_offsets = list(config_grid.get("timing_offsets_s", [-0.25, 0.0, 0.25]))
    speed_deltas = list(config_grid.get("speed_deltas_m_s", [-0.2, 0.0, 0.2]))

    if replay_data is not None and "cells" in replay_data:
        cells = replay_data["cells"]
        missing = replay_data.get("missing_cell_reasons", [])
    else:
        cells, missing = evaluate_perturbation_grid(
            timing_offsets_s=timing_offsets,
            speed_deltas_m_s=speed_deltas,
            cell_verdict_fn=_build_cpu_verdict_fn(
                trace_episode=trace_episode,
                critical_event=critical_block,
            ),
        )

    perturbation_grid = {
        "timing_offsets_s": timing_offsets,
        "speed_deltas_m_s": speed_deltas,
    }

    # The gate computes the exact-replay digest from the source episode
    # identity; echo it back so the persistence record's source_episode
    # block is schema-complete.  When no source digest exists, the gate has
    # already returned FAIL via missing-identity-field detection.
    source_episode_fields = dict(source_episode_fields)
    source_episode_fields["replay_digest"] = exact_replay_block["replay_digest"]

    return compute_persistence_record(
        scenario_id=scenario_id,
        source_episode=source_episode_fields,
        generated_scenario=_build_generated_scenario(entry, scenario_id),
        planner=frozen_config.get("planner", "goal"),
        seed=source_seed,
        config=_build_promotion_config(frozen_config),
        commit_hashes=commit_hashes,
        exact_replay=exact_replay_block,
        critical_event_reproduced=critical_block,
        perturbation_grid=perturbation_grid,
        cell_verdicts=cells,
        missing_cell_reasons=missing,
    )


def _build_promotion_config(frozen_config: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble the ``config`` block required by ``compute_persistence_record``.

    The frozen YAML marks ``frozen: true`` at the document root.  The persistence
    record's ``config`` object must carry ``frozen: true`` plus a stable
    ``config_id`` and ``config_hash`` so promotion fails closed on an unfrozen config.
    """

    return {
        "config_id": str(frozen_config.get("config_id", "issue-5600-persistence-gate")),
        "frozen": bool(frozen_config.get("frozen", False)),
        "config_hash": str(frozen_config.get("config_hash", "")) or _config_hash(frozen_config),
    }


def _config_hash(payload: Mapping[str, Any]) -> str:
    import hashlib

    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()[:16]


def _load_frozen_config(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a YAML mapping: {config_path}")
    if not raw.get("frozen"):
        raise ValueError(f"config {config_path} must have frozen: true")
    return dict(raw)


def _resolve_artifact(base_dir: Path, artifact_path: str) -> Path | None:
    if not artifact_path:
        return None
    candidate = Path(artifact_path)
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _load_candidate_entry(path: Path) -> dict[str, Any]:
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        entry = yaml.safe_load(raw_text)
    else:
        entry = json.loads(raw_text)
    if not isinstance(entry, dict):
        raise ValueError(f"candidate entry must be a mapping: {path}")
    return dict(entry)


def _build_episodes_index(episodes_path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        eid = str(record.get("episode_id", ""))
        if eid:
            index[eid] = dict(record)
    return index


def _build_source_episode_fields(
    entry: Mapping[str, Any],
    trace_episode: Mapping[str, Any],
) -> dict[str, Any]:
    source_info = entry.get("source_episode", {})
    digest = trace_episode.get("replay_digest", "") or ""
    return {
        "episode_id": str(source_info.get("episode_id", "")),
        "source_seed": int(source_info.get("source_seed", 0)),
        "source_map": str(source_info.get("source_map", "")),
        "replay_digest": digest,
    }


def _build_episode_fields(
    trace_episode: Mapping[str, Any],
    source_map: str,
) -> dict[str, Any]:
    steps = _extract_steps(trace_episode)
    return {
        "episode_id": str(trace_episode.get("episode_id", "")),
        "source_seed": int(trace_episode.get("seed", 0)),
        "source_map": source_map,
        "replay_digest": trace_episode.get("replay_digest", "") or "",
        "steps": list(steps) if isinstance(steps, list) else [],
    }


def _build_generated_scenario(entry: Mapping[str, Any], scenario_id: str) -> dict[str, Any]:
    digest = entry.get("provenance", {}).get("digest", "")
    if not digest:
        digest = _config_hash(entry)
    return {
        "catalog_schema_version": CATALOG_ENTRY_SCHEMA_VERSION,
        "scenario_id": str(entry.get("scenario_id", scenario_id)),
        "catalog_entry_digest": digest,
    }


def _evaluate_critical_event(
    entry: Mapping[str, Any],
    source_trace: Mapping[str, Any],
    replayed_trace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate whether the source critical event reproduces in the replayed trace.

    Extracts event type, time, and location from the catalog entry's criticality
    block.  If no replayed trace is available, reports unknown.
    """
    criticality = entry.get("criticality", {})
    event_type = str(criticality.get("signal", "min_clearance"))
    observed_at_s = float(criticality.get("observed_at_s", 0.0))

    source_location = _find_critical_location(source_trace, observed_at_s)

    if replayed_trace is None:
        return assess_critical_event_reproduction(
            event_type=event_type,
            source_event_time_s=observed_at_s,
            source_event_location=source_location or [0.0, 0.0],
            not_observed_reason="no replayed trace available",
        )

    replayed_event = _find_critical_event(replayed_trace)
    if replayed_event is None:
        return assess_critical_event_reproduction(
            event_type=event_type,
            source_event_time_s=observed_at_s,
            source_event_location=source_location or [0.0, 0.0],
            not_observed_reason="critical event not found in replayed trace",
        )

    return assess_critical_event_reproduction(
        event_type=event_type,
        source_event_time_s=observed_at_s,
        source_event_location=source_location or [0.0, 0.0],
        replayed_event_time_s=replayed_event["time_s"],
        replayed_event_location=replayed_event["location"],
        time_tolerance_s=0.5,
        location_tolerance_m=0.75,
    )


def _find_critical_location(trace: Mapping[str, Any], observed_at_s: float) -> list[float] | None:
    """Find the robot position at the observed time in the trace."""

    steps = _extract_steps(trace)
    if not steps:
        return None
    best_frame = min(
        steps,
        key=lambda frame: abs(float(frame.get("time_s", 0)) - observed_at_s),
        default=None,
    )
    if best_frame is None:
        return None
    robot = best_frame.get("robot", {})
    position = robot.get("position")
    if isinstance(position, list) and len(position) >= 2:
        return [float(position[0]), float(position[1])]
    return None


def _find_critical_event(trace: Mapping[str, Any]) -> dict[str, Any] | None:
    """Find the minimum-clearance event in a trace."""

    steps = _extract_steps(trace)
    if not steps:
        return None

    best_time_s: float | None = None
    best_location: list[float] | None = None
    best_clearance = math.inf

    for frame in steps:
        robot_pos = frame.get("robot", {}).get("position")
        pedestrians = frame.get("pedestrians", [])
        if not isinstance(robot_pos, list) or len(robot_pos) < 2:
            continue
        for ped in pedestrians:
            ped_pos = ped.get("position")
            if not isinstance(ped_pos, list) or len(ped_pos) < 2:
                continue
            clearance = math.dist(
                [float(robot_pos[0]), float(robot_pos[1])],
                [float(ped_pos[0]), float(ped_pos[1])],
            )
            if clearance < best_clearance:
                best_clearance = clearance
                best_time_s = float(frame.get("time_s", 0))
                best_location = [float(robot_pos[0]), float(robot_pos[1])]

    if best_time_s is None or best_location is None:
        return None
    return {"time_s": best_time_s, "location": best_location, "clearance": best_clearance}


def _extract_steps(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    steps = trace.get("steps") or []
    if not steps and "algorithm_metadata" in trace:
        meta = trace["algorithm_metadata"]
        if isinstance(meta, dict):
            sim_trace = meta.get("simulation_step_trace", {})
            if isinstance(sim_trace, dict):
                steps = sim_trace.get("steps", [])
    return list(steps) if isinstance(steps, list) else []


def _build_cpu_verdict_fn(
    *,
    trace_episode: Mapping[str, Any],
    critical_event: Mapping[str, Any],
) -> Any:
    """Build a perturbation cell verdict function for CPU-only mode.

    For each (timing_offset_s, speed_delta_m_s) cell:
    1. Shift trace timestamps by timing_offset_s.
    2. For speed_delta_m_s, extrapolate robot/pedestrian positions along their
       frame-to-frame velocity.
    3. Recompute the minimum clearance across all shifted frames.
    4. If the minimum clearance is within a tolerance band of the original
       critical value, return PASS; otherwise FAIL.
    """

    original_steps = _extract_steps(trace_episode)
    if not original_steps:
        return lambda **_: {"verdict": "fail", "reason": "no trace steps available"}

    original_critical_value = _compute_min_clearance(original_steps)
    return _CpuCellVerdict(
        original_steps=original_steps,
        original_critical_value=original_critical_value,
        event_type=str(critical_event.get("event_type", "min_clearance")),
    )


class _CpuCellVerdict:
    """CPU-only perturbation cell evaluator for ``evaluate_perturbation_grid``."""

    def __init__(
        self,
        *,
        original_steps: Sequence[Mapping[str, Any]],
        original_critical_value: float | None,
        event_type: str,
    ) -> None:
        self._original_steps = list(original_steps)
        self._original_critical_value = original_critical_value
        self._event_type = event_type

    def __call__(self, timing_offset_s: float, speed_delta_m_s: float) -> dict[str, Any] | None:
        try:
            shifted_steps = self._shift_steps(timing_offset_s, speed_delta_m_s)
        except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError) as exc:
            return {"verdict": "fail", "reason": f"cell evaluation error: {exc}"}
        if not shifted_steps:
            return {"verdict": "fail", "reason": "all frames shifted before time 0"}
        return self._verdict(shifted_steps)

    def _shift_steps(self, timing_offset_s: float, speed_delta_m_s: float) -> list[dict[str, Any]]:
        shifted: list[dict[str, Any]] = []
        for i, frame in enumerate(self._original_steps):
            time_s = float(frame.get("time_s", 0)) + timing_offset_s
            if time_s < 0:
                continue
            robot_pos = self._shift_position(frame, i, timing_offset_s, speed_delta_m_s, "robot")
            pedestrians = [
                {
                    "position": self._shift_position(
                        frame, i, timing_offset_s, speed_delta_m_s, "ped", j
                    )
                }
                for j in range(len(frame.get("pedestrians", [])))
            ]
            shifted.append(
                {"time_s": time_s, "robot": {"position": robot_pos}, "pedestrians": pedestrians}
            )
        return shifted

    def _shift_position(
        self,
        frame: Mapping[str, Any],
        index: int,
        timing_offset_s: float,
        speed_delta_m_s: float,
        actor: str,
        ped_index: int | None = None,
    ) -> list[float]:
        """Extrapolate one actor position under the timing/speed perturbation."""

        if actor == "robot":
            current = list(frame.get("robot", {}).get("position", [0.0, 0.0]))
        else:
            current = list(frame.get("pedestrians", [])[ped_index or 0].get("position", [0.0, 0.0]))
        if speed_delta_m_s == 0.0 or index == 0:
            return current
        prev_frame = self._original_steps[index - 1]
        if actor == "robot":
            prev_pos = list(prev_frame.get("robot", {}).get("position", [0.0, 0.0]))
        else:
            prev_ped = prev_frame.get("pedestrians", [])
            if ped_index is None or ped_index >= len(prev_ped):
                return current
            prev_pos = list(prev_ped[ped_index].get("position", [0.0, 0.0]))
        dt = float(frame.get("time_s", 0)) - float(prev_frame.get("time_s", 0))
        if dt <= 0:
            return current
        direction = [current[0] - prev_pos[0], current[1] - prev_pos[1]]
        dist = math.hypot(direction[0], direction[1])
        if dist == 0:
            return current
        scale = speed_delta_m_s * (timing_offset_s if timing_offset_s != 0 else 1.0) / dist
        return [current[0] + direction[0] * scale, current[1] + direction[1] * scale]

    def _verdict(self, shifted_steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        perturbed = _compute_min_clearance(shifted_steps)
        original = self._original_critical_value
        tolerance_factor = 1.1
        if original is not None and perturbed is not None:
            if perturbed <= original * tolerance_factor:
                return {"verdict": "pass", "reason": "critical event persists under perturbation"}
            return {
                "verdict": "fail",
                "reason": (
                    f"min clearance grew from {original:.4f} to {perturbed:.4f} "
                    f"(>{original * tolerance_factor:.4f})"
                ),
            }
        return {"verdict": "fail", "reason": "could not compute clearance"}


def _compute_min_clearance(steps: Sequence[Mapping[str, Any]]) -> float | None:
    """Compute the minimum robot-pedestrian clearance across a sequence of frames."""

    best = math.inf
    for frame in steps:
        robot_pos = frame.get("robot", {}).get("position")
        if not isinstance(robot_pos, (list, tuple)) or len(robot_pos) < 2:
            continue
        for ped in frame.get("pedestrians", []):
            ped_pos = ped.get("position")
            if not isinstance(ped_pos, (list, tuple)) or len(ped_pos) < 2:
                continue
            clearance = math.dist(
                [float(robot_pos[0]), float(robot_pos[1])],
                [float(ped_pos[0]), float(ped_pos[1])],
            )
            best = min(best, clearance)
    return best if math.isfinite(best) else None


def main() -> int:
    """Run the persistence gate evaluation and produce evidence records."""

    args = _build_parser().parse_args()

    try:
        records = evaluate_pipeline_candidates(
            manifest_path=args.manifest,
            candidate_path=args.candidate,
            episodes_path=args.episodes,
            config_path=args.config,
            output_dir=args.output,
            replay_results_path=args.replay_results,
            commit_hash=args.commit_hash,
            config_hash=args.config_hash,
        )
    except (FileNotFoundError, ValueError, ScenarioPersistenceValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not records:
        print("WARNING: no persistence records produced; pipeline may be empty.", file=sys.stderr)
        return 0

    passed = sum(1 for rec in records if rec["promotion"]["verdict"] == "promote")
    rejected = sum(1 for rec in records if rec["promotion"]["verdict"] == "reject")
    print(f"\nSummary: {passed} promoted, {rejected} rejected, {len(records)} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
