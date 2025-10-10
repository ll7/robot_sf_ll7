"""Orchestration logic for executing episode jobs and adaptive sampling.

Implemented incrementally in tasks T026-T029, T027 (parallel), T028 (adaptive iteration),
T029 (full run orchestration skeleton).
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.benchmark.errors import AggregationMetadataError

from .aggregation import aggregate_metrics
from .effects import compute_effect_sizes
from .io_utils import append_episode_record, write_manifest
from .planning import expand_episode_jobs, load_scenario_matrix, plan_scenarios
from .precision import evaluate_precision
from .replay import ReplayCapture  # T021 optional replay capture
from .visuals import generate_visual_artifacts  # new visual artifact integration

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

# Import new visualization functions for real plots/videos from episode data
try:
    from robot_sf.benchmark.visualization import (
        VisualizationError,
        generate_benchmark_plots,
        generate_benchmark_videos,
        validate_visual_artifacts,
    )

    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

# -----------------------------
# Manifest dataclass & helpers
# -----------------------------


@dataclass
class BenchmarkManifest:
    output_root: Path
    git_hash: str
    scenario_matrix_hash: str
    config: object
    episodes_path: str
    created_at: float = field(default_factory=time.time)
    executed_jobs: int = 0
    skipped_jobs: int = 0
    notes: str = "skeleton_t029"
    runtime_sec: float = 0.0
    episodes_per_second: float = 0.0
    workers: int = 1
    scaling_efficiency: dict = field(default_factory=dict)


def _ensure_algo_metadata(
    record: dict[str, Any],
    *,
    algo: str | None,
    episode_id: str | None,
    logger_ctx=None,
) -> dict[str, Any]:
    """Mirror the algorithm identifier into scenario_params and validate payloads."""

    log = logger_ctx or logger
    algo_value = algo.strip() if isinstance(algo, str) else ""
    if not algo_value:
        raise AggregationMetadataError(
            "Episode missing algorithm identifier required for aggregation.",
            episode_id=str(episode_id) if episode_id is not None else None,
            missing_fields=("algo", "scenario_params.algo"),
            advice="Ensure the benchmark configuration sets `algo` before writing episodes.",
        )

    scenario_params = record.get("scenario_params")
    if scenario_params is None:
        scenario_params = {}
        record["scenario_params"] = scenario_params
    elif not isinstance(scenario_params, dict):
        raise AggregationMetadataError(
            "scenario_params must be a mapping to inject algorithm metadata.",
            episode_id=str(episode_id) if episode_id is not None else None,
            missing_fields=("scenario_params", "scenario_params.algo"),
            advice="Regenerate the episode with structured scenario parameters.",
        )

    existing_algo = scenario_params.get("algo")
    log = log.bind(episode_id=episode_id, algo=algo_value)
    if existing_algo is None:
        scenario_params["algo"] = algo_value
        log.bind(event="episode_metadata_injection").debug(
            "Mirrored algorithm metadata into scenario_params",
        )
    elif str(existing_algo) != algo_value:
        scenario_params["algo"] = algo_value
        log.bind(event="episode_metadata_mismatch", previous=str(existing_algo)).warning(
            "Corrected mismatched algorithm metadata for episode",
        )

    record["algo"] = algo_value
    return record


def _compute_git_hash(root: Path) -> str:
    """Best‑effort retrieval of current git HEAD short hash.

    Falls back to 'unknown' if repository metadata is inaccessible. Separated to keep
    orchestration function lean (polish phase refactor for C901).
    """
    git_hash = "unknown"
    try:  # pragma: no cover - environment dependent
        head_ref = root / ".git" / "HEAD"
        if head_ref.exists():
            content = head_ref.read_text(encoding="utf-8").strip()
            if content.startswith("ref:"):
                ref_path = content.split(" ", 1)[1].strip()
                ref_file = root / ".git" / ref_path
                if ref_file.exists():
                    git_hash = ref_file.read_text(encoding="utf-8").strip()[:12]
            else:
                git_hash = content[:12]
    except OSError as exc:
        # Filesystem access errors -> return unknown but log for diagnostics
        logger.debug("_compute_git_hash fs access error: %s", exc)
    except (RuntimeError, TypeError):  # pragma: no cover - defensive
        # Unexpected but plausible runtime/type errors -> log at debug and continue
        logger.debug("_compute_git_hash unexpected error")
    return git_hash


def _prepare_output_dirs(cfg):
    root = Path(cfg.output_root)
    episodes_dir = root / "episodes"
    aggregates_dir = root / "aggregates"
    reports_dir = root / "reports"
    plots_dir = root / "plots"
    for d in (episodes_dir, aggregates_dir, reports_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)
    return root, episodes_dir, aggregates_dir, reports_dir, plots_dir


def _init_manifest(
    root: Path,
    episodes_path: Path,
    cfg,
    scenario_matrix_hash: str,
) -> BenchmarkManifest:
    return BenchmarkManifest(
        output_root=root,
        git_hash=_compute_git_hash(root),
        scenario_matrix_hash=scenario_matrix_hash,
        config=cfg,
        episodes_path=str(episodes_path),
    )


def _update_scaling_efficiency(manifest: BenchmarkManifest, cfg):
    """Update runtime, throughput and synthetic parallel efficiency stats in manifest."""
    now = time.time()
    manifest.runtime_sec = max(0.0, now - manifest.created_at)
    manifest.workers = int(getattr(cfg, "workers", 1) or 1)
    if manifest.runtime_sec > 0:
        manifest.episodes_per_second = manifest.executed_jobs / manifest.runtime_sec
    ideal_rate = manifest.workers * manifest.episodes_per_second if manifest.workers > 0 else 0
    efficiency = 0.0
    if ideal_rate > 0:
        efficiency = manifest.episodes_per_second / ideal_rate
    manifest.scaling_efficiency = {
        "runtime_sec": manifest.runtime_sec,
        "executed_jobs": manifest.executed_jobs,
        "skipped_jobs": manifest.skipped_jobs,
        "episodes_per_second": manifest.episodes_per_second,
        "workers": manifest.workers,
        "parallel_efficiency_placeholder": efficiency,
    }
    return manifest.scaling_efficiency


def _write_iteration_artifacts(root: Path, groups, effects, precision_report):
    _write_json(root / "aggregates" / "summary.json", _serialize_groups(groups))
    _write_json(root / "reports" / "effect_sizes.json", _serialize_effects(effects))
    _write_json(
        root / "reports" / "statistical_sufficiency.json",
        _serialize_precision(precision_report),
    )


def _episode_id_from_job(job) -> str:
    """Deterministically derive an episode_id from a job.

    Contract (early phase): scenario_id + '-' + seed. Horizon intentionally excluded
    to keep reproducibility with initial tests; may evolve later when multi‑horizon
    episodes are introduced.
    """
    return f"{job.scenario_id}-{job.seed}"


def _scan_existing_episode_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Lightweight JSON parse (json imported at module top level)
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed episode record line in {}", path)
                    continue
                ep_id = rec.get("episode_id")
                if isinstance(ep_id, str):
                    ids.add(ep_id)
    except OSError as exc:  # pragma: no cover - unlikely on normal FS
        logger.warning("Failed reading existing episodes file {}: {}", path, exc)
    return ids


def _make_episode_record(job, cfg) -> dict[str, Any]:  # minimal synthetic execution
    """Produce a placeholder EpisodeRecord structure matching contract fixtures.

    Real implementation (post T026) will execute simulation & compute metrics. For now
    we populate required structural fields with synthetic values to allow downstream
    aggregation/effects tasks to proceed under TDD.
    """
    episode_id = _episode_id_from_job(job)
    now = time.time()
    algo_value = getattr(cfg, "algo", None)
    record: dict[str, Any] = {
        "episode_id": episode_id,
        "scenario_id": job.scenario_id,
        "seed": job.seed,
        "archetype": job.archetype,
        "density": job.density,
        "status": "success",
        "metrics": {
            # Placeholder metric values (kept deterministic)
            "collision_rate": 0.0,
            "success_rate": 1.0,
            "time_to_goal": 10.0,
            "path_efficiency": 0.9,
            "average_speed": 1.0,
            "snqi": 0.75,
        },
        "steps": min(job.horizon, 120),  # bounded placeholder
        # Basic timing placeholders updated by caller for sequential path
        "wall_time_sec": 0.0,
        "created_at": now,
    }
    record["scenario_params"] = {
        "archetype": job.archetype,
        "density": job.density,
        "max_episode_steps": job.horizon,
        "scenario_id": job.scenario_id,
    }
    # Optional: attach placeholder replay steps when capture enabled (T021)
    if getattr(cfg, "capture_replay", False):
        # Build a tiny deterministic trajectory: straight line x increasing, heading fixed.
        horizon = min(job.horizon, 20)
        cap = ReplayCapture(episode_id=episode_id, scenario_id=job.scenario_id)
        ped_positions_series: list[list[tuple[float, float]]] = []
        actions_series: list[tuple[float, float]] = []
        for i in range(horizon):
            t_rel = float(i) * 0.1
            # Simple oscillating lateral motion for pedestrians & dummy action vector
            ped_positions = (
                [(float(i) * 0.05, 0.2), (float(i) * 0.05, -0.2)]
                if i % 2 == 0
                else [(float(i) * 0.05, 0.25)]
            )
            action = (0.05, 0.0)
            ped_positions_series.append(ped_positions)
            actions_series.append(action)
            cap.record(
                t=t_rel,
                x=float(i) * 0.05,
                y=0.0,
                heading=0.0,
                speed=0.5,
                ped_positions=ped_positions,
                action=action,
            )
        finalized = cap.finalize().steps
        record["replay_steps"] = [(s.t, s.x, s.y, s.heading) for s in finalized]
        record["replay_peds"] = ped_positions_series
        record["replay_actions"] = actions_series
    _ensure_algo_metadata(
        record,
        algo=algo_value,
        episode_id=episode_id,
    )
    return record


def _partition_jobs(existing_ids: set[str], job_iter: Iterable[object]) -> tuple[list[object], int]:
    run_list: list[object] = []
    skip_count = 0
    for jb in job_iter:
        if _episode_id_from_job(jb) in existing_ids:
            skip_count += 1
        else:
            run_list.append(jb)
    return run_list, skip_count


def _execute_seq(
    job_list: list[object],
    existing_ids: set[str],
    episodes_path: Path,
    cfg,
    manifest,
) -> Iterator[dict]:
    for jb in job_list:
        start = time.time()
        rec = _make_episode_record(jb, cfg)
        end = time.time()
        rec["wall_time_sec"] = end - start
        append_episode_record(episodes_path, rec)
        existing_ids.add(rec["episode_id"])
        if hasattr(manifest, "executed_jobs"):
            manifest.executed_jobs += 1
        yield rec


def _worker_job_wrapper(job, algo):  # top-level for pickling on spawn
    class _TempCfg:
        def __init__(self, a):
            self.algo = a

    start = time.time()
    rec = _make_episode_record(job, _TempCfg(algo))
    rec["wall_time_sec"] = time.time() - start
    return rec


def _execute_parallel(
    job_list: list[object],
    existing_ids: set[str],
    episodes_path: Path,
    cfg,
    manifest,
    workers: int,
) -> Iterator[dict]:
    logger.debug("Executing {} jobs in parallel with {} workers", len(job_list), workers)
    algo = getattr(cfg, "algo", "unknown")
    results_map: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(_worker_job_wrapper, j, algo): j for j in job_list}
        for fut in as_completed(future_map):
            rec = fut.result()
            results_map[rec["episode_id"]] = rec
    # Deterministic ordering for append
    for jb in job_list:
        ep_id = _episode_id_from_job(jb)
        rec = results_map[ep_id]
        append_episode_record(episodes_path, rec)
        existing_ids.add(ep_id)
        if hasattr(manifest, "executed_jobs"):
            manifest.executed_jobs += 1
        yield rec


def run_episode_jobs(jobs: Iterable[object], cfg, manifest) -> Iterator[dict]:  # T026/T027
    """Execute episode jobs with resume + optional parallel workers.

    T026 (completed earlier): sequential execution + resume scan.
    T027 extension:
      - If cfg.workers > 1 use a process pool to compute episode records in parallel.
      - Parent process performs file appends (avoids concurrent writes).
      - Update manifest counters: executed_jobs, skipped_jobs.
    """
    episodes_path = Path(manifest.episodes_path)
    existing_ids = _scan_existing_episode_ids(episodes_path)
    logger.debug("Found {} existing episode records (resume)", len(existing_ids))
    to_run, skipped = _partition_jobs(existing_ids, list(jobs))
    if hasattr(manifest, "skipped_jobs"):
        manifest.skipped_jobs += skipped
    workers = int(getattr(cfg, "workers", 1) or 1)
    if workers <= 1 or len(to_run) <= 1:
        yield from _execute_seq(to_run, existing_ids, episodes_path, cfg, manifest)
    else:
        yield from _execute_parallel(to_run, existing_ids, episodes_path, cfg, manifest, workers)


def adaptive_sampling_iteration(current_records, cfg, scenarios, manifest):  # T028
    """Decide whether additional episode jobs are required (placeholder T028).

    Minimal implementation for contract phase:
      - Count existing episodes per scenario from current_records.
      - If counts >= cfg.max_episodes (or no scenarios needing more) -> return (True, []).
      - Else create up to cfg.batch_size new synthetic jobs per iteration (evenly per scenario needing more, but simplified here: all remaining for first scenario).

    Future iterations (T034) will incorporate precision evaluation. Seeds are derived
    by extending scenario.planned_seeds with deterministic incremental integers if
    needed (placeholder logic) to avoid blocking on full seeding strategy.
    """
    # Touch manifest to avoid unused param lint (future: record iteration stats)
    _ = manifest
    # Gather counts
    per_scenario: dict[str, int] = {}
    for r in current_records:
        sid = r.get("scenario_id")
        if sid is not None:
            per_scenario[sid] = per_scenario.get(sid, 0) + 1

    # Identify scenarios needing more episodes
    needs: list[object] = []
    max_eps = int(getattr(cfg, "max_episodes", 0) or 0)
    batch_size = int(getattr(cfg, "batch_size", 1) or 1)
    for sc in scenarios:
        count = per_scenario.get(sc.scenario_id, 0)
        if count < max_eps:
            needs.append(sc)

    if not needs:
        return True, []

    # Generate new jobs for first needing scenario (simple contract satisfaction)
    target_sc = needs[0]
    existing = per_scenario.get(target_sc.scenario_id, 0)
    remaining = max_eps - existing
    to_create = min(batch_size, remaining)

    # Derive seeds: reuse planned_seeds then extend with increasing integers
    seeds: list[int] = list(getattr(target_sc, "planned_seeds", []))
    # Ensure enough seeds
    while len(seeds) < existing + to_create:
        seeds.append(len(seeds))  # deterministic extension

    # Build lightweight job objects (mirroring EpisodeJob subset) without relying on full dataclass
    jobs = []
    horizon = getattr(cfg, "horizon_override", None) or 100
    start_index = existing
    for i in range(to_create):
        seed = seeds[start_index + i]
        job_id = f"{target_sc.scenario_id}:{seed}:{horizon}"  # simple deterministic id
        job = type("EpisodeJobLite", (), {})()
        job.job_id = job_id
        job.scenario_id = target_sc.scenario_id
        job.seed = seed
        job.archetype = getattr(target_sc, "archetype", "unknown")
        job.density = getattr(target_sc, "density", "unknown")
        job.horizon = horizon
        jobs.append(job)

    done_flag = False  # more iterations likely needed until max reached
    # If after adding this batch we would reach or exceed max for all scenarios mark done next time
    if existing + to_create >= max_eps and len(needs) == 1:
        # After these jobs scenario will be full; check others already full.
        done_flag = all(per_scenario.get(sc.scenario_id, 0) >= max_eps for sc in scenarios)

    return done_flag, jobs


def run_full_benchmark(cfg):  # T029 + T034 integration (refactored in polish phase)  # noqa: C901
    """Execute classic benchmark with adaptive precision loop.

    Refactored to reduce cyclomatic complexity (extracting helpers for setup, manifest
    initialization, scaling efficiency instrumentation, artifact writes). Public
    semantics preserved for existing tests.
    """
    # Output & planning
    root, episodes_dir, _aggregates_dir, _reports_dir, _plots_dir = _prepare_output_dirs(cfg)
    episodes_path = episodes_dir / "episodes.jsonl"
    raw = load_scenario_matrix(cfg.scenario_matrix_path)
    matrix_bytes = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    scenario_matrix_hash = hashlib.sha1(matrix_bytes).hexdigest()[:12]
    rng = random.Random(int(getattr(cfg, "master_seed", 123)))
    scenarios = plan_scenarios(raw, cfg, rng=rng)
    jobs = expand_episode_jobs(scenarios, cfg)

    # Manifest & initial execution
    manifest = _init_manifest(root, episodes_path, cfg, scenario_matrix_hash)
    all_records = list(run_episode_jobs(jobs, cfg, manifest))
    scenarios_list = list(scenarios)
    max_episodes = int(getattr(cfg, "max_episodes", 0) or 0)

    # Adaptive loop (iteration guard for smoke / tiny budgets)
    iteration_count = 0
    while True:
        groups = aggregate_metrics(all_records, cfg)
        effects = compute_effect_sizes(groups, cfg)
        precision_report = evaluate_precision(groups, cfg)

        # Instrumentation & artifact persistence
        scaling = _update_scaling_efficiency(manifest, cfg)
        try:  # attach for downstream JSON serialization if model allows attribute
            precision_report.scaling_efficiency = scaling  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            # precision_report may be a plain dict or a lightweight namespace; ignore
            # absence of attribute or wrong type but do not swallow unrelated errors.
            pass
        _write_iteration_artifacts(root, groups, effects, precision_report)

        # Exit conditions
        if precision_report.final_pass:
            logger.info("Precision criteria met; stopping adaptive loop")
            break
        if max_episodes and sum(g.count for g in groups) >= max_episodes * len(scenarios_list):
            logger.info("Reached max episodes budget; stopping adaptive loop")
            break

        # Additional sampling
        done_flag, new_jobs = adaptive_sampling_iteration(
            all_records,
            cfg,
            scenarios_list,
            manifest,
        )
        if not new_jobs:
            if done_flag:
                logger.info("Adaptive iteration indicated done; no new jobs.")
            break
        new_records = list(run_episode_jobs(new_jobs, cfg, manifest))
        all_records.extend(new_records)
        iteration_count += 1
        # Safety: In smoke mode with very small episode budgets we break after first iteration
        # to prevent runaway loops in early scaffolding stages.
        if getattr(cfg, "smoke", False) and max_episodes <= 2 and iteration_count >= 1:
            logger.info("Early exit guard (smoke small-budget) triggered after first iteration")
            break

    # Finalize & persist manifest
    _update_scaling_efficiency(manifest, cfg)
    manifest.scaling_efficiency.setdefault("finalized", True)
    write_manifest(manifest, str(root / "manifest.json"))

    # Visual artifacts (plots + videos) generation (post adaptive loop single pass)
    try:
        generate_visual_artifacts(root, cfg, groups, all_records)

        # Also generate real visualizations using new visualization module
        # Skip generating heavy real visualizations when running in smoke mode
        # (smoke mode intentionally keeps runtime small for tests).
        if _VISUALIZATION_AVAILABLE and not getattr(cfg, "smoke", False):
            logger.info("Generating additional real visualizations from episode data")
            try:
                plots_dir = root / "plots"
                videos_dir = root / "videos"
                plots_dir.mkdir(exist_ok=True)
                videos_dir.mkdir(exist_ok=True)

                # Generate real plots from episode data
                plot_artifacts = generate_benchmark_plots(all_records, str(plots_dir))
                logger.info("Generated {} real plots", len(plot_artifacts))

                # Generate real videos from episode data
                video_artifacts = generate_benchmark_videos(all_records, str(videos_dir))
                logger.info("Generated {} real videos", len(video_artifacts))

                # Validate all generated artifacts
                all_artifacts = plot_artifacts + video_artifacts
                validation = validate_visual_artifacts(all_artifacts)
                if validation.passed:
                    logger.info("All real visualizations validated successfully")
                else:
                    logger.warning(
                        "Some visualizations failed validation: {} failed artifacts",
                        len(validation.failed_artifacts),
                    )

            except (VisualizationError, FileNotFoundError) as vis_exc:
                logger.warning("Real visualization generation failed (non-fatal): {}", vis_exc)

    except (VisualizationError, FileNotFoundError) as exc:
        logger.warning("Visual artifact generation failed (non-fatal): {}", exc)

    return manifest


def _write_json(path: Path, obj):  # helper
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        tmp.replace(path)
    except (OSError, TypeError) as exc:
        logger.warning("Failed writing JSON artifact {}: {}", path, exc)


def _serialize_groups(groups):
    out = []
    for g in groups:
        out.append(
            {
                "archetype": g.archetype,
                "density": g.density,
                "count": g.count,
                "metrics": {
                    k: {
                        "mean": m.mean,
                        "median": m.median,
                        "p95": m.p95,
                        "mean_ci": m.mean_ci,
                        "median_ci": m.median_ci,
                    }
                    for k, m in g.metrics.items()
                },
            },
        )
    return out


def _serialize_effects(effects):
    out = []
    for rep in effects:
        out.append(
            {
                "archetype": rep.archetype,
                "comparisons": [
                    {
                        "metric": c.metric,
                        "density_low": c.density_low,
                        "density_high": c.density_high,
                        "diff": c.diff,
                        "standardized": c.standardized,
                    }
                    for c in rep.comparisons
                ],
            },
        )
    return out


def _serialize_precision(report):
    return {
        "final_pass": report.final_pass,
        "evaluations": [
            {
                "scenario_id": ev.scenario_id,
                "archetype": ev.archetype,
                "density": ev.density,
                "episodes": ev.episodes,
                "all_pass": ev.all_pass,
                "metric_status": [
                    {
                        "metric": ms.metric,
                        "half_width": ms.half_width,
                        "target": ms.target,
                        "passed": ms.passed,
                    }
                    for ms in ev.metric_status
                ],
            }
            for ev in report.evaluations
        ],
    }
