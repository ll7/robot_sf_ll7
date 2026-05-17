"""Seed-sensitivity replay helpers for adversarial counterexamples."""

from __future__ import annotations

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


@dataclass(frozen=True)
class SeedSensitivityReplay:
    """One seed-perturbation replay result in a sensitivity summary."""

    seed: int
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
) -> SeedSensitivitySummary:
    """Replay one adversarial candidate over explicit scenario seeds.

    The explorer treats seed changes as controlled perturbations of an already
    valid candidate, so seed values are not constrained by the original search
    space's sampled seed range. Certification still runs for every perturbation
    and rejects fail closed when the configured certification policy disallows a
    replay. The replay loop is intentionally single-process: evaluators and
    environment setup may reset process-global RNG state for each seed.
    """
    config.validate()
    replay_seeds = tuple(int(seed) for seed in seeds)
    if not replay_seeds:
        raise ValueError("seeds must contain at least one replay seed")
    if not 0.0 <= min_persistence_rate <= 1.0:
        raise ValueError("min_persistence_rate must be between 0 and 1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    objective = get_objective(config.objective)
    active_certifier = certifier or _default_certifier

    replays: list[SeedSensitivityReplay] = []
    for index, seed in enumerate(replay_seeds):
        replay_started_at = datetime.now(UTC).isoformat()
        replay_candidate = replace(candidate, scenario_seed=seed)
        replay_dir = output_dir / f"replay_{index:04d}_seed_{seed}"
        scenario_yaml_path, _route_path = write_candidate_inputs(
            config=config,
            candidate=replay_candidate,
            candidate_dir=replay_dir,
            index=index,
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
            continue

        evaluation = evaluator(config, replay_candidate, scenario_yaml_path, replay_dir)
        evaluation = replace(evaluation, certification_status=certification_status)
        score = objective(evaluation)
        evaluation = evaluation.with_objective(score)
        replays.append(_replay_from_evaluation(evaluation, started_at=replay_started_at))

    summary = _summarize(
        candidate=candidate,
        seeds=replay_seeds,
        replays=tuple(replays),
        min_persistence_rate=min_persistence_rate,
        summary_path=output_dir / "seed_sensitivity_summary.json",
    )
    write_json(summary.summary_path, summary.to_json())
    return summary


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
    outcome = record.get("outcome") if isinstance(record.get("outcome"), dict) else {}
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
