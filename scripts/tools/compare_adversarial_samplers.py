"""Compare adversarial candidate samplers on a bounded search config."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

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
    manifest_path: str
    best_bundle_path: str | None
    best_objective_value: float | None
    num_candidates: int
    num_valid_candidates: int
    num_invalid_candidates: int
    num_failed_evaluations: int


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
) -> list[SamplerComparisonRow]:
    """Run the configured search once per sampler and return compact rows."""
    rows: list[SamplerComparisonRow] = []
    for offset, sampler_name in enumerate(sampler_names):
        sampler_output_dir = config.output_dir / sampler_name
        sampler_config = replace(
            config,
            output_dir=sampler_output_dir,
            seed=config.seed + offset,
        )
        result = run_adversarial_search(
            sampler_config,
            sampler=build_sampler(sampler_name, config.search_space, seed=config.seed + offset),
            evaluator=_synthetic_evaluator if synthetic else None,
            certifier=(
                (lambda _candidate, _path, _required: passed_status("synthetic comparison"))
                if synthetic
                else None
            ),
        )
        rows.append(
            SamplerComparisonRow(
                sampler=sampler_name,
                manifest_path=result.manifest_path.as_posix(),
                best_bundle_path=(
                    result.best_bundle_path.as_posix() if result.best_bundle_path else None
                ),
                best_objective_value=result.best_objective_value,
                num_candidates=result.num_candidates,
                num_valid_candidates=result.num_valid_candidates,
                num_invalid_candidates=result.num_invalid_candidates,
                num_failed_evaluations=result.num_failed_evaluations,
            )
        )
    return rows


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
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
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
        budget=args.budget,
        seed=args.seed,
    )
    rows = run_sampler_comparison(
        config=config,
        sampler_names=tuple(args.samplers or ("random", "coordinate", "optuna")),
        synthetic=bool(args.synthetic),
    )
    payload = {
        "schema_version": "adversarial-sampler-comparison.v1",
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
