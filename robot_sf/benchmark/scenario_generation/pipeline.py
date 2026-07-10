"""CPU-only stage 1--3 data-driven scenario-generation pipeline."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.benchmark.scenario_generation.catalog_writer import (
    deduplicate_catalog_entries,
    write_generated_catalog,
)
from robot_sf.benchmark.scenario_generation.random_sampler import (
    SampledEpisode,
    sample_episode_jobs,
)
from robot_sf.benchmark.scenario_generation.replay_validation import assess_replay_status
from robot_sf.benchmark.scenario_generation.segment_extraction import extract_critical_segment
from robot_sf.training.scenario_loader import load_scenarios

CONFIG_SCHEMA_VERSION = "data-driven-scenario-generation.v1"
RUN_SCHEMA_VERSION = "scenario-generation-run.v1"
CLAIM_BOUNDARY = "generated scenario hypotheses only"
_EPISODE_SCHEMA = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")


def load_generation_config(path: Path) -> dict[str, Any]:
    """Load and fail closed on the small stage-1 configuration contract.

    Returns:
        The validated YAML configuration as a mutable mapping.
    """

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("scenario-generation config must be a mapping")
    config = dict(payload)
    _validate_generation_contract(config)
    return config


def _validate_generation_contract(config: Mapping[str, Any]) -> None:
    """Validate top-level identity plus each nested configuration section."""

    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    if config.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError(f"claim_boundary must be {CLAIM_BOUNDARY!r}")
    _required_string(config, "source_scenarios")
    _required_int(config, "seed")
    if _required_int(config, "episode_budget") <= 0:
        raise ValueError("episode_budget must be > 0")
    _validate_sampler_config(config.get("sampler"))
    _validate_runner_config(config.get("runner"))
    _validate_dedup_config(config.get("deduplication"))


def _validate_sampler_config(raw_sampler: object) -> None:
    """Validate the conservative Monte Carlo sampler contract."""

    sampler = raw_sampler
    if not isinstance(sampler, Mapping) or sampler.get("type") != "monte_carlo":
        raise ValueError("sampler.type must be 'monte_carlo'")
    if sampler.get("obstacle_policy") != "disabled_for_mvp":
        raise ValueError("sampler.obstacle_policy must be 'disabled_for_mvp'")


def _validate_runner_config(raw_runner: object) -> None:
    """Validate the bounded existing-runner configuration."""

    runner = raw_runner
    if not isinstance(runner, Mapping):
        raise ValueError("runner must be a mapping")
    if not isinstance(runner.get("horizon"), int) or runner["horizon"] <= 0:
        raise ValueError("runner.horizon must be a positive integer")
    if not isinstance(runner.get("algo"), str) or not runner["algo"].strip():
        raise ValueError("runner.algo must be a non-empty string")


def _validate_dedup_config(raw_dedup: object) -> None:
    """Validate deterministic feature-distance deduplication controls."""

    dedup = raw_dedup
    if not isinstance(dedup, Mapping):
        raise ValueError("deduplication must be a mapping")
    threshold = dedup.get("distance_threshold")
    if not isinstance(threshold, int | float) or isinstance(threshold, bool):
        raise ValueError("deduplication.distance_threshold must be numeric")


def run_generation_pipeline(
    config_path: Path,
    *,
    output_root: Path | None = None,
    scenario_loader: Callable[[Path], Sequence[Mapping[str, Any]]] = load_scenarios,
    batch_runner: Callable[..., Mapping[str, Any]] = run_map_batch,
    replay_config_builder: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run seeded sampling, trace distillation, deduplication, and load validation.

    Returns:
        The persisted run manifest payload.
    """

    config = load_generation_config(config_path)
    output_root = output_root or Path(_required_string(config, "output_root"))
    _prepare_output_root(output_root)
    source_path = Path(_required_string(config, "source_scenarios"))
    source_scenarios = list(scenario_loader(source_path))
    sampler = dict(config["sampler"])
    sampled = sample_episode_jobs(
        source_scenarios,
        seed=int(config["seed"]),
        episode_budget=int(config["episode_budget"]),
        episode_seed_min=int(sampler.get("episode_seed_min", 1)),
        episode_seed_max=int(sampler.get("episode_seed_max", 2_147_483_647)),
    )
    sampled_matrix_path = output_root / "sampled_scenarios.yaml"
    sampled_matrix_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "generated-scenario-sample-matrix.v1",
                "claim_boundary": CLAIM_BOUNDARY,
                "scenarios": [sample.scenario for sample in sampled],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    episodes_path = output_root / "episodes.jsonl"
    runner_config = dict(config["runner"])
    summary = dict(
        batch_runner(
            [sample.scenario for sample in sampled],
            episodes_path,
            _EPISODE_SCHEMA,
            scenario_path=source_path,
            horizon=int(runner_config["horizon"]),
            dt=runner_config.get("dt"),
            record_forces=False,
            algo=str(runner_config["algo"]),
            algo_config_path=runner_config.get("algo_config"),
            benchmark_profile="experimental",
            workers=1,
            resume=False,
            record_simulation_step_trace=True,
        )
    )
    if int(summary.get("failed_jobs", 0)) or int(summary.get("written", 0)) != len(sampled):
        raise RuntimeError(
            "map runner did not produce every sampled episode: "
            f"written={summary.get('written')} expected={len(sampled)} "
            f"failures={summary.get('failures')}"
        )

    records = _load_jsonl(episodes_path)
    records_by_name = {str(record["scenario_id"]): record for record in records}
    source_by_name = {_source_scenario_name(scenario): scenario for scenario in source_scenarios}
    extraction = dict(config.get("extraction") or {})
    entries: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for sample in sampled:
        materialized_name = str(sample.scenario["name"])
        if materialized_name not in records_by_name:
            raise RuntimeError(f"missing map-runner record for {materialized_name}")
        record = records_by_name[materialized_name]
        trace_episode = _distiller_episode(record, sample)
        entry = extract_critical_segment(
            trace_episode,
            pre_margin_s=float(extraction.get("pre_margin_s", 1.0)),
            post_margin_s=float(extraction.get("post_margin_s", 1.0)),
        )
        replay_kwargs: dict[str, Any] = {}
        if replay_config_builder is not None:
            replay_kwargs["config_builder"] = replay_config_builder
        entry = assess_replay_status(
            entry,
            source_scenario=source_by_name[sample.source_scenario_name],
            scenario_path=source_path,
            **replay_kwargs,
        )
        entries.append(entry)
        outcomes.append(_outcome_record(record, sample, trace_episode["steps"]))

    outcomes_path = output_root / "episode_outcomes.jsonl"
    _write_jsonl(outcomes_path, outcomes)
    kept, dropped = deduplicate_catalog_entries(
        entries,
        distance_threshold=float(config["deduplication"]["distance_threshold"]),
    )
    run_manifest_path = output_root / "run_manifest.json"
    catalog_path, provenance_path = write_generated_catalog(
        output_root,
        kept,
        dropped_duplicates=dropped,
        run_manifest_path=run_manifest_path,
    )
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "seed": int(config["seed"]),
        "sampler": "monte_carlo.v1",
        "source_scenarios": source_path.as_posix(),
        "episode_count": len(sampled),
        "claim_boundary": CLAIM_BOUNDARY,
        "sampled_parameters": [sample.manifest_record() for sample in sampled],
        "sampling_policies": {
            "robot_start_goal_policy": sampler.get("robot_start_goal_policy"),
            "pedestrian_policy": sampler.get("pedestrian_policy"),
            "obstacle_policy": sampler.get("obstacle_policy"),
        },
        "runner_summary": summary,
        "catalog": {
            "candidate_count": len(entries),
            "kept_count": len(kept),
            "dropped_duplicate_count": len(dropped),
        },
        "artifacts": {
            "sampled_scenarios": sampled_matrix_path.as_posix(),
            "episodes": episodes_path.as_posix(),
            "episode_outcomes": outcomes_path.as_posix(),
            "generated_catalog": catalog_path.as_posix(),
            "catalog_provenance": provenance_path.as_posix(),
        },
        "governance": {
            "required_manual_review": True,
            "benchmark_evidence": False,
            "release_matrix_inclusion": False,
        },
    }
    run_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _prepare_output_root(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output_root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _source_scenario_name(scenario: Mapping[str, Any]) -> str:
    return _required_string(scenario, "name")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(record)
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _distiller_episode(record: Mapping[str, Any], sample: SampledEpisode) -> dict[str, Any]:
    metadata = record.get("algorithm_metadata")
    trace = metadata.get("simulation_step_trace") if isinstance(metadata, Mapping) else None
    if not isinstance(trace, Mapping) or trace.get("schema_version") != "simulation-step-trace.v1":
        raise ValueError("episode record is missing simulation-step-trace.v1")
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("episode record has no simulation trace steps")
    return {
        "episode_id": _required_string(record, "episode_id"),
        "seed": int(record["seed"]),
        "source_map": sample.source_map,
        "steps": steps,
    }


def _criticality_series(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for step in steps:
        robot = step["robot"]["position"]
        peds = step["pedestrians"]
        clearances = [math.dist(robot, pedestrian["position"]) for pedestrian in peds]
        series.append(
            {
                "time_s": float(step["time_s"]),
                "min_clearance_m": min(clearances) if clearances else None,
            }
        )
    return series


def _outcome_record(
    record: Mapping[str, Any],
    sample: SampledEpisode,
    steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "scenario-generation-episode-outcome.v1",
        "sample": sample.manifest_record(),
        "episode_id": record["episode_id"],
        "status": record.get("status"),
        "termination_reason": record.get("termination_reason"),
        "metrics": record.get("metrics", {}),
        "criticality_time_series": _criticality_series(steps),
        "benchmark_evidence": False,
    }


__all__ = [
    "CLAIM_BOUNDARY",
    "CONFIG_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "load_generation_config",
    "run_generation_pipeline",
]
