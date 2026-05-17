"""Seed-sensitivity replay helpers for adversarial counterexamples."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.adversarial.bundle import write_candidate_inputs, write_json
from robot_sf.adversarial.certification import (
    CertificationStatus,
    candidate_allowed,
    certify_candidate,
)
from robot_sf.adversarial.config import CandidateEvaluation, CandidateSpec, SearchConfig
from robot_sf.adversarial.io import read_first_jsonl_record
from robot_sf.adversarial.objectives import get_objective

CandidateEvaluator = Callable[[SearchConfig, CandidateSpec, Path, Path], CandidateEvaluation]
CandidateCertifier = Callable[[CandidateSpec, Path, bool], CertificationStatus]
MAX_ABS_SPEED_DELTA_MPS = 1.0
MAX_ABS_TIMING_DELTA_S = 5.0


@dataclass(frozen=True)
class SeedSensitivityPerturbation:
    """Timing and speed deltas applied around a fixed adversarial candidate."""

    label: str | None = None
    pedestrian_speed_delta_mps: float = 0.0
    pedestrian_delay_delta_s: float = 0.0
    spawn_time_delta_s: float = 0.0

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible perturbation payload."""
        return {
            "label": self.label,
            "pedestrian_speed_delta_mps": float(self.pedestrian_speed_delta_mps),
            "pedestrian_delay_delta_s": float(self.pedestrian_delay_delta_s),
            "spawn_time_delta_s": float(self.spawn_time_delta_s),
        }


@dataclass(frozen=True)
class SeedSensitivityReplay:
    """One seed-perturbation replay result in a sensitivity summary."""

    seed: int
    perturbation: SeedSensitivityPerturbation
    status: str
    outcome: str
    reason: str | None
    objective_value: float | None
    bundle_path: Path | None
    episode_record_path: Path | None
    trajectory_csv_path: Path | None
    started_at: str | None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible replay payload."""
        return {
            "seed": int(self.seed),
            "perturbation": self.perturbation.to_json(),
            "status": self.status,
            "outcome": self.outcome,
            "reason": self.reason,
            "objective_value": self.objective_value,
            "bundle_path": self.bundle_path.as_posix() if self.bundle_path else None,
            "episode_record_path": self.episode_record_path.as_posix()
            if self.episode_record_path
            else None,
            "trajectory_csv_path": self.trajectory_csv_path.as_posix()
            if self.trajectory_csv_path
            else None,
            "started_at": self.started_at,
        }


@dataclass(frozen=True)
class SeedSensitivitySummary:
    """Compact stability summary for one adversarial candidate across replay seeds."""

    schema_version: str
    candidate: CandidateSpec
    seeds: tuple[int, ...]
    classification: str
    failure_persistence_rate: float
    objective_score_spread: float | None
    min_persistence_rate: float
    num_fail_closed_exclusions: int
    perturbations: tuple[SeedSensitivityPerturbation, ...]
    replays: tuple[SeedSensitivityReplay, ...]
    summary_path: Path

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible seed-sensitivity payload."""
        return {
            "schema_version": self.schema_version,
            "candidate": self.candidate.to_json(),
            "seeds": list(self.seeds),
            "classification": self.classification,
            "failure_persistence_rate": self.failure_persistence_rate,
            "objective_score_spread": self.objective_score_spread,
            "min_persistence_rate": self.min_persistence_rate,
            "num_fail_closed_exclusions": self.num_fail_closed_exclusions,
            "perturbations": [perturbation.to_json() for perturbation in self.perturbations],
            "replays": [replay.to_json() for replay in self.replays],
        }


def run_seed_sensitivity(
    config: SearchConfig,
    *,
    candidate: CandidateSpec,
    seeds: Iterable[int],
    output_dir: Path,
    evaluator: CandidateEvaluator,
    certifier: CandidateCertifier | None = None,
    min_persistence_rate: float = 0.5,
    perturbations: Iterable[SeedSensitivityPerturbation] | None = None,
) -> SeedSensitivitySummary:
    """Replay one adversarial candidate over explicit scenario seeds.

    The explorer treats seed changes as controlled perturbations of an already
    valid candidate, so seed values are not constrained by the original search
    space's sampled seed range. Certification still runs for every perturbation
    and rejects fail closed when the configured certification policy disallows a
    replay. Optional perturbations apply bounded timing/speed deltas around the
    fixed candidate in deterministic ``seed x perturbation`` order. The replay
    loop is intentionally single-process: evaluators and environment setup may
    reset process-global RNG state for each seed.
    """
    config.validate()
    replay_seeds = tuple(int(seed) for seed in seeds)
    if not replay_seeds:
        raise ValueError("seeds must contain at least one replay seed")
    if not 0.0 <= min_persistence_rate <= 1.0:
        raise ValueError("min_persistence_rate must be between 0 and 1")
    replay_perturbations = _normalize_perturbations(
        perturbations,
        candidate=candidate,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    objective = get_objective(config.objective)
    active_certifier = certifier or _default_certifier

    replays: list[SeedSensitivityReplay] = []
    replay_index = 0
    for seed in replay_seeds:
        for perturbation_index, perturbation in enumerate(replay_perturbations):
            replay_started_at = datetime.now(UTC).isoformat()
            replay_candidate = _apply_perturbation(candidate, seed=seed, perturbation=perturbation)
            replay_dir = _replay_dir(
                output_dir,
                index=replay_index,
                seed=seed,
                perturbation_index=perturbation_index,
                include_perturbation=len(replay_perturbations) > 1,
            )
            scenario_yaml_path, _route_path = write_candidate_inputs(
                config=config,
                candidate=replay_candidate,
                candidate_dir=replay_dir,
                index=replay_index,
            )
            certification_status = active_certifier(
                replay_candidate,
                scenario_yaml_path,
                config.require_certification,
            )
            if not candidate_allowed(
                certification_status,
                require_certification=config.require_certification,
            ):
                replays.append(
                    SeedSensitivityReplay(
                        seed=seed,
                        perturbation=perturbation,
                        status=certification_status.status,
                        outcome="fail_closed_exclusion",
                        reason=certification_status.reason,
                        objective_value=None,
                        bundle_path=replay_dir,
                        episode_record_path=None,
                        trajectory_csv_path=None,
                        started_at=replay_started_at,
                    )
                )
                replay_index += 1
                continue

            evaluation = evaluator(config, replay_candidate, scenario_yaml_path, replay_dir)
            evaluation = replace(evaluation, certification_status=certification_status)
            score = objective(evaluation)
            evaluation = evaluation.with_objective(score)
            replays.append(
                _replay_from_evaluation(
                    evaluation,
                    perturbation=perturbation,
                    started_at=replay_started_at,
                )
            )
            replay_index += 1

    summary = _summarize(
        candidate=candidate,
        seeds=replay_seeds,
        perturbations=replay_perturbations,
        replays=tuple(replays),
        min_persistence_rate=min_persistence_rate,
        summary_path=output_dir / "seed_sensitivity_summary.json",
    )
    write_json(summary.summary_path, summary.to_json())
    return summary


def _normalize_perturbations(
    perturbations: Iterable[SeedSensitivityPerturbation] | None,
    *,
    candidate: CandidateSpec,
) -> tuple[SeedSensitivityPerturbation, ...]:
    """Return validated perturbations, defaulting to one no-op replay."""
    normalized = tuple(perturbations or (SeedSensitivityPerturbation(),))
    if not normalized:
        raise ValueError("perturbations must contain at least one entry")
    for perturbation in normalized:
        _validate_perturbation(perturbation, candidate=candidate)
    return normalized


def _validate_perturbation(
    perturbation: SeedSensitivityPerturbation,
    *,
    candidate: CandidateSpec,
) -> None:
    """Validate one timing/speed perturbation against the bounded opt-in surface."""
    values = {
        "pedestrian_speed_delta_mps": perturbation.pedestrian_speed_delta_mps,
        "pedestrian_delay_delta_s": perturbation.pedestrian_delay_delta_s,
        "spawn_time_delta_s": perturbation.spawn_time_delta_s,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    if abs(float(perturbation.pedestrian_speed_delta_mps)) > MAX_ABS_SPEED_DELTA_MPS:
        raise ValueError(
            "pedestrian_speed_delta_mps must be between "
            f"{-MAX_ABS_SPEED_DELTA_MPS} and {MAX_ABS_SPEED_DELTA_MPS}"
        )
    if abs(float(perturbation.pedestrian_delay_delta_s)) > MAX_ABS_TIMING_DELTA_S:
        raise ValueError(
            "pedestrian_delay_delta_s must be between "
            f"{-MAX_ABS_TIMING_DELTA_S} and {MAX_ABS_TIMING_DELTA_S}"
        )
    if abs(float(perturbation.spawn_time_delta_s)) > MAX_ABS_TIMING_DELTA_S:
        raise ValueError(
            f"spawn_time_delta_s must be between {-MAX_ABS_TIMING_DELTA_S} "
            f"and {MAX_ABS_TIMING_DELTA_S}"
        )

    perturbed = _apply_perturbation(
        candidate, seed=candidate.scenario_seed, perturbation=perturbation
    )
    if perturbed.pedestrian_speed_mps <= 0.0:
        raise ValueError("perturbed pedestrian_speed_mps must stay positive")
    if perturbed.pedestrian_delay_s < 0.0:
        raise ValueError("perturbed pedestrian_delay_s must stay non-negative")
    if perturbed.spawn_time_s < 0.0:
        raise ValueError("perturbed spawn_time_s must stay non-negative")


def _apply_perturbation(
    candidate: CandidateSpec,
    *,
    seed: int,
    perturbation: SeedSensitivityPerturbation,
) -> CandidateSpec:
    """Apply one perturbation to a candidate while replacing the replay seed."""
    return replace(
        candidate,
        scenario_seed=int(seed),
        spawn_time_s=float(candidate.spawn_time_s) + float(perturbation.spawn_time_delta_s),
        pedestrian_speed_mps=float(candidate.pedestrian_speed_mps)
        + float(perturbation.pedestrian_speed_delta_mps),
        pedestrian_delay_s=float(candidate.pedestrian_delay_s)
        + float(perturbation.pedestrian_delay_delta_s),
    )


def _replay_dir(
    output_dir: Path,
    *,
    index: int,
    seed: int,
    perturbation_index: int,
    include_perturbation: bool,
) -> Path:
    """Return a deterministic replay directory name."""
    if not include_perturbation:
        return output_dir / f"replay_{index:04d}_seed_{seed}"
    return output_dir / f"replay_{index:04d}_seed_{seed}_perturb_{perturbation_index:02d}"


def _default_certifier(
    candidate: CandidateSpec,
    scenario_yaml_path: Path,
    require_certification: bool,
) -> CertificationStatus:
    """Run the default adversarial scenario-certification adapter."""
    return certify_candidate(
        candidate,
        scenario_yaml_path=scenario_yaml_path,
        require_certification=require_certification,
    )


def _replay_from_evaluation(
    evaluation: CandidateEvaluation,
    *,
    perturbation: SeedSensitivityPerturbation,
    started_at: str | None = None,
) -> SeedSensitivityReplay:
    """Convert a candidate evaluation into a seed-sensitivity replay row."""
    record = read_first_jsonl_record(evaluation.episode_record_path)
    fallback_status = (
        evaluation.failure_attribution.status
        if evaluation.failure_attribution is not None
        else "evaluated"
    )
    record_status = record.get("status") if record else None
    return SeedSensitivityReplay(
        seed=int(evaluation.candidate.scenario_seed),
        perturbation=perturbation,
        status=str(record_status) if record_status else fallback_status,
        outcome=_outcome_from_evaluation(evaluation, record),
        reason=_reason_from_evaluation(evaluation),
        objective_value=evaluation.objective_value,
        bundle_path=evaluation.bundle_path,
        episode_record_path=evaluation.episode_record_path,
        trajectory_csv_path=evaluation.trajectory_csv_path,
        started_at=started_at,
    )


def _outcome_from_evaluation(
    evaluation: CandidateEvaluation,
    record: dict[str, Any] | None,
) -> str:
    """Return the categorical outcome used for stability accounting."""
    if (
        evaluation.failure_attribution is not None
        and evaluation.failure_attribution.primary_failure
    ):
        return evaluation.failure_attribution.primary_failure
    if record is None:
        return "unknown"
    raw_outcome = record.get("outcome")
    outcome = raw_outcome if isinstance(raw_outcome, dict) else {}
    if bool(outcome.get("collision") or outcome.get("collision_event")):
        return "collision"
    if bool(outcome.get("timeout") or outcome.get("timeout_event")):
        return "timeout"
    if bool(outcome.get("route_complete")):
        return "success"
    return str(record.get("termination_reason", record.get("status", "unknown")))


def _reason_from_evaluation(evaluation: CandidateEvaluation) -> str | None:
    """Return a compact reason string for replay summaries."""
    if evaluation.error:
        return evaluation.error
    if evaluation.failure_attribution is None:
        return None
    if evaluation.failure_attribution.reasons:
        return "; ".join(evaluation.failure_attribution.reasons)
    return None


def _summarize(
    *,
    candidate: CandidateSpec,
    seeds: tuple[int, ...],
    perturbations: tuple[SeedSensitivityPerturbation, ...],
    replays: tuple[SeedSensitivityReplay, ...],
    min_persistence_rate: float,
    summary_path: Path,
) -> SeedSensitivitySummary:
    """Build aggregate persistence and objective-spread values."""
    evaluated = [replay for replay in replays if replay.outcome != "fail_closed_exclusion"]
    failures = [replay for replay in evaluated if _is_failure_outcome(replay.outcome)]
    failure_persistence_rate = len(failures) / len(evaluated) if evaluated else 0.0
    classification = _classification(
        failure_persistence_rate=failure_persistence_rate,
        has_failure=bool(failures),
        min_persistence_rate=min_persistence_rate,
    )
    scores = [replay.objective_value for replay in evaluated if replay.objective_value is not None]
    objective_score_spread = max(scores) - min(scores) if scores else None
    return SeedSensitivitySummary(
        schema_version="adversarial-seed-sensitivity.v1",
        candidate=candidate,
        seeds=seeds,
        classification=classification,
        failure_persistence_rate=float(failure_persistence_rate),
        objective_score_spread=float(objective_score_spread)
        if objective_score_spread is not None
        else None,
        min_persistence_rate=float(min_persistence_rate),
        num_fail_closed_exclusions=sum(
            1 for replay in replays if replay.outcome == "fail_closed_exclusion"
        ),
        perturbations=perturbations,
        replays=replays,
        summary_path=summary_path,
    )


def _is_failure_outcome(outcome: str) -> bool:
    """Return whether an outcome counts as a persisted counterexample failure."""
    return outcome in {"collision", "timeout", "incomplete", "evaluation_error"}


def _classification(
    *,
    failure_persistence_rate: float,
    has_failure: bool,
    min_persistence_rate: float,
) -> str:
    """Classify the replay grid as stable, brittle, or non-failing."""
    if not has_failure:
        return "no_failure"
    if failure_persistence_rate >= min_persistence_rate:
        return "stable_failure"
    return "brittle_failure"
