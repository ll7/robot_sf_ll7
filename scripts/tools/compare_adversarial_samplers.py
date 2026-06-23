"""Compare adversarial candidate samplers on a bounded search config."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.adversarial.attribution import attribution_from_episode_record
from robot_sf.adversarial.bundle import write_trajectory_csv
from robot_sf.adversarial.certification import passed_status
from robot_sf.adversarial.config import CandidateEvaluation, SearchConfig
from robot_sf.adversarial.samplers import (
    CandidateSampler,
    CoordinateRefinementSampler,
    OptunaCandidateSampler,
    RandomCandidateSampler,
)
from robot_sf.adversarial.search import run_adversarial_search

if TYPE_CHECKING:
    from collections.abc import Sequence

    from robot_sf.adversarial.config import CandidateSpec, SearchSpaceConfig


@dataclass(frozen=True)
class SamplerComparisonRow:
    """One sampler result row for the comparison report."""

    sampler: str
    budget: int
    seed: int
    manifest_path: str
    best_bundle_path: str | None
    best_objective_value: float | None
    best_valid_objective: float | None
    num_candidates: int
    num_valid_candidates: int
    num_invalid_candidates: int
    num_failed_evaluations: int
    invalid_candidate_rate: float
    first_failure_iteration: int | None
    certified_valid_failure_count: int
    replayable_valid_failure_count: int
    replay_success_rate: float | None
    fallback_candidate_count: int
    degraded_candidate_count: int
    held_out_family_yield: float | None
    held_out_family_status: str
    caveats: tuple[str, ...]


def build_sampler(name: str, search_space: SearchSpaceConfig, *, seed: int) -> CandidateSampler:
    """Build a named adversarial sampler."""
    key = name.strip().lower()
    if key == "random":
        return RandomCandidateSampler(search_space, seed=seed)
    if key == "coordinate":
        return CoordinateRefinementSampler(search_space, seed=seed)
    if key == "optuna":
        return OptunaCandidateSampler(search_space, seed=seed)
    raise ValueError("sampler must be one of: random, coordinate, optuna")


def run_sampler_comparison(
    *,
    config: SearchConfig,
    sampler_names: Sequence[str],
    synthetic: bool,
    budgets: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[SamplerComparisonRow]:
    """Run the configured search once per sampler and return compact rows."""
    rows: list[SamplerComparisonRow] = []
    active_budgets = tuple(budgets or (config.budget,))
    active_seeds = tuple(seeds or (config.seed,))
    for budget in active_budgets:
        if budget <= 0:
            raise ValueError("budgets must be positive")
        for base_seed in active_seeds:
            for sampler_name in sampler_names:
                run_seed = int(base_seed)
                sampler_output_dir = (
                    config.output_dir
                    / f"budget_{int(budget):04d}"
                    / f"seed_{int(base_seed)}"
                    / sampler_name
                )
                sampler_config = replace(
                    config,
                    budget=int(budget),
                    output_dir=sampler_output_dir,
                    seed=run_seed,
                )
                result = run_adversarial_search(
                    sampler_config,
                    sampler=build_sampler(sampler_name, config.search_space, seed=run_seed),
                    evaluator=_synthetic_evaluator if synthetic else None,
                    certifier=(
                        (lambda _candidate, _path, _required: passed_status("synthetic comparison"))
                        if synthetic
                        else None
                    ),
                )
                rows.append(
                    _comparison_row_from_manifest(
                        sampler=sampler_name,
                        budget=int(budget),
                        seed=run_seed,
                        manifest_path=result.manifest_path,
                        best_bundle_path=result.best_bundle_path,
                        best_objective_value=result.best_objective_value,
                        num_candidates=result.num_candidates,
                        num_valid_candidates=result.num_valid_candidates,
                        num_invalid_candidates=result.num_invalid_candidates,
                        num_failed_evaluations=result.num_failed_evaluations,
                    )
                )
    return rows


def _comparison_row_from_manifest(  # noqa: PLR0913
    *,
    sampler: str,
    budget: int,
    seed: int,
    manifest_path: Path,
    best_bundle_path: Path | None,
    best_objective_value: float | None,
    num_candidates: int,
    num_valid_candidates: int,
    num_invalid_candidates: int,
    num_failed_evaluations: int,
) -> SamplerComparisonRow:
    """Derive conservative package-B diagnostics from one search manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest.get("candidates") if isinstance(manifest, dict) else []
    if not isinstance(candidates, list):
        candidates = []

    certified_valid_failures = [
        (index, item)
        for index, item in enumerate(candidates, start=1)
        if _is_certified_valid_failure(item)
    ]
    replayable_failures = [
        item for _index, item in certified_valid_failures if _has_replay_paths(item)
    ]
    valid_objectives = [
        float(item["objective_value"]) for item in candidates if _is_valid_scored_candidate(item)
    ]
    replay_success_rate = (
        len(replayable_failures) / len(certified_valid_failures)
        if certified_valid_failures
        else None
    )
    caveats = (
        "diagnostic/local nominal report; not paper-facing benchmark evidence",
        "held-out-family yield is not evaluated in package B; narrow archive caveat applies",
        "learned failure proposal #2921 remains stretch/out of scope",
    )
    return SamplerComparisonRow(
        sampler=sampler,
        budget=budget,
        seed=seed,
        manifest_path=manifest_path.as_posix(),
        best_bundle_path=best_bundle_path.as_posix() if best_bundle_path else None,
        best_objective_value=best_objective_value,
        best_valid_objective=max(valid_objectives) if valid_objectives else None,
        num_candidates=num_candidates,
        num_valid_candidates=num_valid_candidates,
        num_invalid_candidates=num_invalid_candidates,
        num_failed_evaluations=num_failed_evaluations,
        invalid_candidate_rate=(num_invalid_candidates / num_candidates if num_candidates else 0.0),
        first_failure_iteration=(
            certified_valid_failures[0][0] if certified_valid_failures else None
        ),
        certified_valid_failure_count=len(certified_valid_failures),
        replayable_valid_failure_count=len(replayable_failures),
        replay_success_rate=replay_success_rate,
        fallback_candidate_count=sum(
            1 for item in candidates if _candidate_mode(item) == "fallback"
        ),
        degraded_candidate_count=sum(
            1 for item in candidates if _candidate_mode(item) == "degraded"
        ),
        held_out_family_yield=None,
        held_out_family_status="not_evaluated_narrow_archive",
        caveats=caveats,
    )


def _is_certified_valid_failure(item: Any) -> bool:
    """Return whether a manifest candidate is a certified, valid behavioral failure."""
    if not isinstance(item, dict):
        return False
    if item.get("error") is not None:
        return False
    certification = item.get("certification_status")
    if not isinstance(certification, dict) or certification.get("status") != "passed":
        return False
    attribution = item.get("failure_attribution")
    if not isinstance(attribution, dict):
        return False
    return attribution.get("primary_failure") in {
        "collision",
        "timeout",
        "near_miss",
        "comfort_violation",
        "incomplete",
    }


def _is_valid_scored_candidate(item: Any) -> bool:
    """Return whether a candidate has a usable objective score and no exclusion status."""
    if not isinstance(item, dict):
        return False
    if item.get("error") is not None or item.get("objective_value") is None:
        return False
    certification = item.get("certification_status")
    if not isinstance(certification, dict) or certification.get("status") != "passed":
        return False
    attribution = item.get("failure_attribution")
    if isinstance(attribution, dict) and attribution.get("primary_failure") in {
        "invalid_candidate",
        "evaluation_error",
    }:
        return False
    return True


def _has_replay_paths(item: dict[str, Any]) -> bool:
    """Return whether manifest paths needed for local replay inspection exist."""
    for key in ("scenario_yaml_path", "episode_record_path", "trajectory_csv_path", "bundle_path"):
        raw_path = item.get(key)
        if not raw_path or not Path(str(raw_path)).exists():
            return False
    return True


def _candidate_mode(item: Any) -> str | None:
    """Extract fallback/degraded mode tags when evaluators report them."""
    if not isinstance(item, dict):
        return None
    attribution = item.get("failure_attribution")
    details = attribution.get("details") if isinstance(attribution, dict) else None
    if not isinstance(details, dict):
        return None
    for key in ("execution_mode", "readiness_status", "availability_status"):
        value = details.get(key)
        if str(value).lower() in {"fallback", "degraded"}:
            return str(value).lower()
    return None


def _synthetic_evaluator(
    config: SearchConfig,
    candidate: CandidateSpec,
    scenario_yaml_path: Path,
    candidate_dir: Path,
) -> CandidateEvaluation:
    """Write a small deterministic episode record for sampler-comparison smoke tests."""
    del config
    target_x = 1.0
    synthetic_snqi = abs(float(candidate.start.x) - target_x) + (
        0.05 * float(candidate.pedestrian_delay_s)
    )
    record = {
        "episode_id": f"synthetic-{candidate.scenario_seed}",
        "seed": int(candidate.scenario_seed),
        "status": "success",
        "steps": 1,
        "termination_reason": "success",
        "outcome": {"route_complete": True, "collision": False, "timeout": False},
        "metrics": {"snqi": float(synthetic_snqi), "success": 1.0},
    }
    episode_path = candidate_dir / "episode_records.jsonl"
    episode_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    trajectory_path = write_trajectory_csv(candidate_dir / "trajectory.csv", record)
    return CandidateEvaluation(
        candidate=candidate,
        certification_status=passed_status("synthetic comparison"),
        objective_value=None,
        failure_attribution=attribution_from_episode_record(record),
        episode_record_path=episode_path,
        trajectory_csv_path=trajectory_path,
        scenario_yaml_path=scenario_yaml_path,
        bundle_path=candidate_dir,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-template",
        type=Path,
        default=Path("configs/scenarios/templates/crossing_ttc.yaml"),
    )
    parser.add_argument(
        "--search-space",
        type=Path,
        default=Path("configs/adversarial/crossing_ttc_space.yaml"),
    )
    parser.add_argument("--policy", default="goal")
    parser.add_argument("--objective", default="worst_case_snqi")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--budget",
        type=int,
        action="append",
        default=None,
        help=(
            "Candidate budget to run; repeat for a budget grid. "
            "Defaults to 8 unless --package-b-budget-grid is set."
        ),
    )
    parser.add_argument(
        "--package-b-budget-grid",
        action="store_true",
        help="Run the issue #3079 package-B fixed budgets: 16, 32, and 64.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="Base seed to run; repeat for repeated-seed budget matching. Defaults to 123.",
    )
    parser.add_argument(
        "--sampler",
        action="append",
        dest="samplers",
        choices=("random", "coordinate", "optuna"),
        default=None,
        help="Sampler to run; repeat to select multiple. Defaults to all three.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a deterministic synthetic evaluator instead of running benchmark episodes.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sampler comparison CLI."""
    args = parse_args(argv)
    config = SearchConfig.from_files(
        policy=args.policy,
        scenario_template=args.scenario_template,
        search_space=args.search_space,
        objective=args.objective,
        output_dir=args.output_dir,
        budget=(args.budget or [8])[0],
        seed=(args.seed or [123])[0],
    )
    budgets = (16, 32, 64) if args.package_b_budget_grid and args.budget is None else args.budget
    seeds = args.seed or [123]
    rows = run_sampler_comparison(
        config=config,
        sampler_names=tuple(args.samplers or ("random", "coordinate", "optuna")),
        synthetic=bool(args.synthetic),
        budgets=budgets,
        seeds=seeds,
    )
    payload = {
        "schema_version": "adversarial-sampler-comparison.v2",
        "report_status": "diagnostic_local_nominal",
        "claim_scope": "not_paper_facing_benchmark_evidence",
        "budget_grid": list(budgets or [config.budget]),
        "seeds": list(seeds),
        "package_b_notes": {
            "learned_failure_proposal_issue_2921": "stretch_out_of_scope",
            "held_out_family_yield": "not_evaluated_narrow_archive",
        },
        "rows": [asdict(r) for r in rows],
    }
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
