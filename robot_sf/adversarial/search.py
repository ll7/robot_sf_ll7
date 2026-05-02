"""Programmable adversarial scenario search runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from robot_sf.adversarial.attribution import (
    FailureAttribution,
    attribution_from_episode_record,
    attribution_from_error,
)
from robot_sf.adversarial.bundle import (
    write_candidate_inputs,
    write_json,
    write_search_manifest,
    write_trajectory_csv,
)
from robot_sf.adversarial.certification import (
    CertificationStatus,
    candidate_allowed,
    certify_candidate,
    failed_status,
)
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    SearchConfig,
    SearchRunResult,
)
from robot_sf.adversarial.io import read_first_jsonl_record
from robot_sf.adversarial.objectives import get_objective
from robot_sf.adversarial.samplers import CandidateSampler, RandomCandidateSampler
from robot_sf.benchmark.runner import run_batch

CandidateEvaluator = Callable[[SearchConfig, CandidateSpec, Path, Path], CandidateEvaluation]
CandidateCertifier = Callable[[CandidateSpec, Path, bool], CertificationStatus]

DEFAULT_SCHEMA_PATH = (
    Path(__file__).parent.parent / "benchmark" / "schemas" / "episode.schema.v1.json"
)


def _default_evaluator(
    config: SearchConfig,
    candidate: CandidateSpec,
    scenario_yaml_path: Path,
    candidate_dir: Path,
) -> CandidateEvaluation:
    """Evaluate one candidate through the existing benchmark batch runner."""
    episode_path = candidate_dir / "episode_records.jsonl"
    snqi_weights = config.load_optional_json(config.snqi_weights_path)
    snqi_baseline = config.load_optional_json(config.snqi_baseline_path)
    summary = run_batch(
        scenario_yaml_path,
        out_path=episode_path,
        schema_path=DEFAULT_SCHEMA_PATH,
        horizon=config.horizon or 100,
        dt=config.dt or 0.1,
        record_forces=config.record_forces,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
        algo=config.policy,
        algo_config_path=str(config.algo_config_path) if config.algo_config_path else None,
        benchmark_profile=config.benchmark_profile,
        workers=config.workers,
        resume=False,
    )
    if len(summary.get("failures", [])) > 0:
        raise RuntimeError(f"candidate evaluation failed: {summary.get('failures')}")
    record = read_first_jsonl_record(episode_path)
    trajectory_path = write_trajectory_csv(candidate_dir / "trajectory.csv", record)
    attribution = attribution_from_episode_record(record or {})
    write_json(candidate_dir / "failure_attribution.json", attribution.to_json())
    return CandidateEvaluation(
        candidate=candidate,
        certification_status=failed_status("certification not assigned by evaluator"),
        objective_value=None,
        failure_attribution=attribution,
        episode_record_path=episode_path,
        trajectory_csv_path=trajectory_path,
        scenario_yaml_path=scenario_yaml_path,
        bundle_path=candidate_dir,
    )


def _default_certifier(
    candidate: CandidateSpec,
    scenario_yaml_path: Path,
    require_certification: bool,
) -> CertificationStatus:
    """Default scenario-certification adapter."""
    return certify_candidate(
        candidate,
        scenario_yaml_path=scenario_yaml_path,
        require_certification=require_certification,
    )


def _invalid_evaluation(
    *,
    candidate: CandidateSpec,
    certification_status: CertificationStatus,
    scenario_yaml_path: Path | None,
    bundle_path: Path | None,
    reason: str,
) -> CandidateEvaluation:
    """Build an evaluation payload for a rejected candidate."""
    return CandidateEvaluation(
        candidate=candidate,
        certification_status=certification_status,
        objective_value=None,
        failure_attribution=FailureAttribution(
            status="not_evaluated",
            primary_failure="invalid_candidate",
            reasons=[reason],
            details={},
        ),
        episode_record_path=None,
        trajectory_csv_path=None,
        scenario_yaml_path=scenario_yaml_path,
        bundle_path=bundle_path,
        error=reason,
    )


def run_adversarial_search(
    config: SearchConfig,
    *,
    evaluator: CandidateEvaluator | None = None,
    certifier: CandidateCertifier | None = None,
    sampler: CandidateSampler | None = None,
) -> SearchRunResult:
    """Run a bounded adversarial scenario search.

    Candidate sampling, certification, evaluation, and manifest ordering are intentionally
    sequential so a fixed sampler seed produces stable candidate directories and replayable
    manifests. ``SearchConfig.workers`` is forwarded to the benchmark runner for each individual
    candidate evaluation; it is not candidate-level parallelism.

    Args:
        config: Search configuration and contracts.
        evaluator: Optional injected evaluator for tests or experiment harnesses.
        certifier: Optional injected certification adapter.
        sampler: Optional injected sampler or optimizer adapter.

    Returns:
        SearchRunResult containing the manifest path and best candidate.
    """
    config.validate()
    objective = get_objective(config.objective)
    active_evaluator = evaluator or _default_evaluator
    active_certifier = certifier or _default_certifier
    active_sampler = sampler or RandomCandidateSampler(config.search_space, seed=config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    evaluations: list[CandidateEvaluation] = []
    num_invalid = 0
    num_failed = 0
    best: CandidateEvaluation | None = None

    for index in range(config.budget):
        candidate = active_sampler.sample()
        candidate_dir = config.output_dir / f"candidate_{index:04d}"
        validation_errors = config.search_space.validate_candidate(candidate)
        if validation_errors:
            num_invalid += 1
            evaluation = _invalid_evaluation(
                candidate=candidate,
                certification_status=failed_status(
                    "search-space validation failed",
                    details={"errors": validation_errors},
                ),
                scenario_yaml_path=None,
                bundle_path=None,
                reason="; ".join(validation_errors),
            )
            evaluations.append(evaluation)
            _observe_candidate(active_sampler, evaluation)
            continue

        scenario_yaml_path, _route_path = write_candidate_inputs(
            config=config,
            candidate=candidate,
            candidate_dir=candidate_dir,
            index=index,
        )
        certification_status = active_certifier(
            candidate,
            scenario_yaml_path,
            config.require_certification,
        )
        if not candidate_allowed(
            certification_status,
            require_certification=config.require_certification,
        ):
            num_invalid += 1
            evaluation = _invalid_evaluation(
                candidate=candidate,
                certification_status=certification_status,
                scenario_yaml_path=scenario_yaml_path,
                bundle_path=candidate_dir,
                reason=certification_status.reason,
            )
            write_json(
                candidate_dir / "failure_attribution.json", evaluation.failure_attribution.to_json()
            )
            evaluations.append(evaluation)
            _observe_candidate(active_sampler, evaluation)
            continue

        try:
            evaluation = active_evaluator(config, candidate, scenario_yaml_path, candidate_dir)
            evaluation = replace(evaluation, certification_status=certification_status)
            score = objective(evaluation)
            evaluation = evaluation.with_objective(score)
        except Exception as exc:
            num_failed += 1
            error = repr(exc)
            attribution = attribution_from_error(error)
            write_json(candidate_dir / "failure_attribution.json", attribution.to_json())
            evaluation = CandidateEvaluation(
                candidate=candidate,
                certification_status=certification_status,
                objective_value=None,
                failure_attribution=attribution,
                episode_record_path=candidate_dir / "episode_records.jsonl",
                trajectory_csv_path=None,
                scenario_yaml_path=scenario_yaml_path,
                bundle_path=candidate_dir,
                error=error,
            )
        evaluations.append(evaluation)
        _observe_candidate(active_sampler, evaluation)
        if evaluation.objective_value is not None and (
            best is None
            or best.objective_value is None
            or evaluation.objective_value > best.objective_value
        ):
            best = evaluation

    manifest_path = write_search_manifest(
        config=config,
        manifest_path=config.output_dir / "manifest.json",
        evaluations=evaluations,
        best=best,
        num_invalid_candidates=num_invalid,
        num_failed_evaluations=num_failed,
    )
    return SearchRunResult(
        manifest_path=manifest_path,
        best_candidate=best,
        best_bundle_path=best.bundle_path if best else None,
        num_candidates=len(evaluations),
        num_valid_candidates=len(evaluations) - num_invalid,
        num_invalid_candidates=num_invalid,
        num_failed_evaluations=num_failed,
    )


def _observe_candidate(sampler: CandidateSampler, evaluation: CandidateEvaluation) -> None:
    """Notify feedback-capable samplers about one completed candidate."""
    observe = getattr(sampler, "observe", None)
    if callable(observe):
        observe(evaluation)
