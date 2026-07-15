"""Compare adversarial candidate samplers on a bounded search config."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.adversarial.attribution import attribution_from_episode_record
from robot_sf.adversarial.bundle import write_trajectory_csv
from robot_sf.adversarial.certification import passed_status
from robot_sf.adversarial.config import CandidateEvaluation, SearchConfig
from robot_sf.adversarial.samplers import (
    CandidateSampler,
    CmaEsCandidateSampler,
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

    objective: str
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
    if key == "cmaes":
        return CmaEsCandidateSampler(search_space, seed=seed)
    raise ValueError("sampler must be one of: random, coordinate, optuna, cmaes")


def run_sampler_comparison(
    *,
    config: SearchConfig,
    sampler_names: Sequence[str],
    synthetic: bool,
    objective_names: Sequence[str] | None = None,
    budgets: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[SamplerComparisonRow]:
    """Run the configured search once per sampler and objective and return compact rows."""
    rows: list[SamplerComparisonRow] = []
    active_objectives = tuple(objective_names or (config.objective,))
    if len(active_objectives) != len(set(active_objectives)):
        raise ValueError("objective_names must not contain duplicates")
    active_budgets = tuple(budgets or (config.budget,))
    active_seeds = tuple(seeds or (config.seed,))
    for objective_name in active_objectives:
        for budget in active_budgets:
            if budget <= 0:
                raise ValueError("budgets must be positive")
            for base_seed in active_seeds:
                for sampler_name in sampler_names:
                    run_seed = int(base_seed)
                    sampler_output_dir = (
                        config.output_dir
                        / objective_name
                        / f"budget_{int(budget):04d}"
                        / f"seed_{int(base_seed)}"
                        / sampler_name
                    )
                    sampler_config = replace(
                        config,
                        objective=objective_name,
                        budget=int(budget),
                        output_dir=sampler_output_dir,
                        seed=run_seed,
                    )
                    result = run_adversarial_search(
                        sampler_config,
                        sampler=build_sampler(sampler_name, config.search_space, seed=run_seed),
                        evaluator=_synthetic_evaluator if synthetic else None,
                        certifier=(
                            (
                                lambda _candidate, _path, _required: passed_status(
                                    "synthetic comparison"
                                )
                            )
                            if synthetic
                            else None
                        ),
                    )
                    rows.append(
                        _comparison_row_from_manifest(
                            objective=objective_name,
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
    objective: str,
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
        item
        for _index, item in certified_valid_failures
        if _has_replay_paths(item, manifest_path=manifest_path)
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
        objective=objective,
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


def _has_replay_paths(item: dict[str, Any], *, manifest_path: Path) -> bool:
    """Return whether manifest paths needed for local replay inspection exist."""
    for key in ("scenario_yaml_path", "episode_record_path", "trajectory_csv_path", "bundle_path"):
        raw_path = item.get(key)
        if not raw_path:
            return False
        path = Path(str(raw_path))
        if path.is_absolute():
            candidates = (path,)
        else:
            candidates = (manifest_path.parent / path, Path.cwd() / path)
        if not any(candidate.exists() for candidate in candidates):
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


def build_comparison_payload(
    *,
    rows: Sequence[SamplerComparisonRow],
    objectives: Sequence[str],
    budgets: Sequence[int],
    seeds: Sequence[int],
    claim_scope: str = "not_paper_facing_benchmark_evidence",
    report_status: str = "diagnostic_local_nominal",
    held_out_status: str = "not_evaluated_narrow_archive",
) -> dict[str, Any]:
    """Build the durable Package-B comparison report payload from result rows.

    The payload preserves the existing report-gate contract: every sampler/budget/seed
    cell appears exactly once, held-out yield stays null, and the claim scope stays
    diagnostic. The resulting mapping can be written directly and validated by
    ``validate_package_b_report``.
    """
    return {
        "schema_version": "adversarial-sampler-comparison.v3",
        "report_status": report_status,
        "claim_scope": claim_scope,
        "objectives": list(objectives),
        "budget_grid": list(budgets),
        "seeds": list(seeds),
        "package_b_notes": {
            "learned_failure_proposal_issue_2921": "stretch_out_of_scope",
            "held_out_family_yield": held_out_status,
        },
        "rows": [asdict(r) for r in rows],
    }


def _resolve_manifest_path(value: Any, *, repo_root: Path, field: str) -> Path:
    """Resolve a required repository-relative manifest path."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Package-B manifest {field} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _manifest_int_tuple(payload: dict[str, Any], key: str) -> tuple[int, ...]:
    """Load a non-empty integer list from a Package-B manifest."""
    value = payload.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"Package-B manifest {key} must be a non-empty integer list")
    return tuple(int(item) for item in value)


def load_package_b_manifest(
    manifest_path: Path,
    *,
    repo_root: Path | None = None,
) -> tuple[
    SearchConfig,
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Load a Package-B manifest and derive the runner configuration.

    Returns:
        A base ``SearchConfig`` scoped to the first objective/budget/seed, the
        list of objectives to compare, the sampler names, the budget grid, and
        the repeated seeds declared by the manifest. All repository-relative
        paths are resolved against ``repo_root`` (or the current working
        directory when it is omitted).

        The compared objectives come from the manifest's top-level ``objectives``
        list when present (enabling multi-objective comparison such as issue
        #5326, where ``temporal_robustness`` is compared against
        ``worst_case_snqi``). When the top-level list is absent, the single
        ``base_config.objective`` is used for backward compatibility with the
        issue #3079 manifest.
    """
    root = (repo_root or Path.cwd()).resolve()
    manifest_path = (
        manifest_path if manifest_path.is_absolute() else root / manifest_path
    ).resolve()
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Package-B manifest payload must be a mapping")
    base_config = payload.get("base_config")
    if not isinstance(base_config, dict):
        raise ValueError("Package-B manifest base_config must be a mapping")
    output_artifacts = payload.get("output_artifacts")
    if not isinstance(output_artifacts, dict):
        raise ValueError("Package-B manifest output_artifacts must be a mapping")
    output_dir = _resolve_manifest_path(
        output_artifacts.get("output_dir"),
        repo_root=root,
        field="output_artifacts.output_dir",
    )

    def _path(key: str) -> Path:
        return _resolve_manifest_path(
            base_config.get(key),
            repo_root=root,
            field=f"base_config.{key}",
        )

    policy = base_config.get("policy")
    objectives = _manifest_objectives(payload, base_config)
    if not isinstance(policy, str) or not policy.strip():
        raise ValueError("Package-B manifest base_config.policy must be a non-empty string")

    budgets = _manifest_int_tuple(payload, "budget_grid")
    seeds = _manifest_int_tuple(payload, "repeated_seeds")
    samplers_raw = payload.get("samplers")
    if (
        not isinstance(samplers_raw, list)
        or not samplers_raw
        or any(not isinstance(item, str) or not item.strip() for item in samplers_raw)
    ):
        raise ValueError("Package-B manifest samplers must be a non-empty string list")
    samplers = tuple(str(item) for item in samplers_raw)

    config = SearchConfig.from_files(
        policy=policy,
        scenario_template=_path("scenario_template"),
        search_space=_path("search_space"),
        objective=objectives[0],
        output_dir=output_dir,
        budget=budgets[0],
        seed=seeds[0],
    )
    return config, objectives, samplers, budgets, seeds


def _manifest_objectives(payload: dict[str, Any], base_config: dict[str, Any]) -> tuple[str, ...]:
    """Resolve the compared objective list from a Package-B manifest.

    Prefers the top-level ``objectives`` list (multi-objective comparison, e.g.
    issue #5326). Falls back to the single ``base_config.objective`` for the
    issue #3079 manifest, which declares only one objective. Duplicate entries
    are rejected so the comparison matrix cannot collapse a cell.
    """
    top_level = payload.get("objectives")
    if isinstance(top_level, list) and top_level:
        if any(not isinstance(item, str) or not item.strip() for item in top_level):
            raise ValueError("Package-B manifest objectives must be a non-empty string list")
        deduped = tuple(str(item) for item in dict.fromkeys(top_level))
        if len(deduped) != len(top_level):
            raise ValueError("Package-B manifest objectives must not contain duplicates")
        return deduped

    single = base_config.get("objective")
    if not isinstance(single, str) or not single.strip():
        raise ValueError(
            "Package-B manifest must declare objectives (top-level list) or base_config.objective"
        )
    return (single,)


def render_durable_comparison_table(
    *,
    report_path: Path | None,
    rows: Sequence[SamplerComparisonRow],
    objectives: Sequence[str],
    budget_grid: Sequence[int],
    seeds: Sequence[int],
) -> str:
    """Render the issue #5326 durable comparison table (exclusions, failures, stop-rule).

    The table is diagnostic-tier only: it never asserts a benchmark claim and
    fails closed when any row shows fallback/degraded execution or is missing
    the required objective columns. The signed ``temporal_robustness`` rows are
    annotated with the per-property violation count read from their
    ``robustness_report.json`` sidecar; baseline objectives carry none.
    """
    required_objectives = set(objectives)
    observed_objectives = {row.objective for row in rows}
    missing_objectives = sorted(required_objectives - observed_objectives)

    degraded_rows = [
        row for row in rows if row.fallback_candidate_count or row.degraded_candidate_count
    ]
    any_degraded = bool(degraded_rows)

    header_cols = [
        "objective",
        "sampler",
        "budget",
        "seed",
        "best_valid_objective",
        "certified_valid_failures",
        "replayable_valid_failures",
        "replay_success_rate",
        "invalid_candidate_rate",
        "signed_property_violations",
        "held_out_family_status",
        "fallback/degraded",
    ]
    lines: list[str] = []
    lines.append("## Issue #5326 durable objective-comparison table (diagnostic tier)\n")
    lines.append(
        "> Claim scope: not paper-facing benchmark evidence. The `--synthetic` CPU path"
        " is reproducible by construction; the `--empirical` CPU path runs the real"
        " `pysocialforce` evaluator and produces certified/replayable failures without"
        " Slurm/GPU. Matched-budget confirmation at paper tier still requires artifact-level"
        " review of certification/replay/independent-seed evidence.\n"
    )
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join("---" for _ in header_cols) + " |")
    for row in rows:
        signed_violations = _read_signed_property_violations(
            bundle_path=Path(row.best_bundle_path) if row.best_bundle_path else None,
            objective=row.objective,
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row.objective,
                    row.sampler,
                    str(row.budget),
                    str(row.seed),
                    _fmt_opt(row.best_valid_objective),
                    str(row.certified_valid_failure_count),
                    str(row.replayable_valid_failure_count),
                    _fmt_opt(row.replay_success_rate),
                    f"{row.invalid_candidate_rate:.3f}",
                    str(signed_violations) if signed_violations is not None else "-",
                    row.held_out_family_status,
                    (
                        f"fb={row.fallback_candidate_count},dg={row.degraded_candidate_count}"
                        if (row.fallback_candidate_count or row.degraded_candidate_count)
                        else "none"
                    ),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("### Stop-rule decision")
    lines.append("")
    if any_degraded:
        decision = (
            "**STOP / fail closed.** One or more comparison rows report"
            " fallback/degraded candidate execution; those rows are excluded from any"
            " success interpretation and cannot serve as matched-budget evidence."
        )
    elif missing_objectives:
        decision = (
            "**NARROW / incomplete.** Required objective(s) missing from the comparison:"
            f" {', '.join(missing_objectives)}. Cannot discriminate objective lift until all"
            " objectives are present under matched budgets."
        )
    else:
        decision = (
            "**DIRECTION NARROWED (diagnostic).** Both objectives compared under matched"
            " CPU-synthetic budgets with no degraded execution. This is a contract/structure"
            " check only; it does not constitute benchmark evidence for the signed-objective"
            " hypothesis (requires artifact-level confirmation of certification/replay/"
            "independent-seed evidence)."
        )
    lines.append(decision)

    lines.append("")
    lines.append("### Exclusions and caveats")
    lines.append("")
    lines.extend(
        (
            "- learned failure proposal #2921: stretch/out of scope",
            "- held-out-family yield: not evaluated (narrow archive caveat)",
            "- paper-facing success claims: forbidden at this tier",
            "- confirmation tier: artifact-level review of certification/replay/independent-seed",
        )
    )
    lines.append(
        f"- report_status: diagnostic_local_nominal; schema"
        f" adversarial-sampler-comparison.v3; budgets={list(budget_grid)};"
        f" seeds={list(seeds)}"
    )
    if report_path is not None:
        lines.append(f"- source report: {report_path.as_posix()}")
    return "\n".join(lines) + "\n"


def _read_signed_property_violations(*, bundle_path: Path | None, objective: str) -> int | None:
    """Return the per-property violation count for a signed-objective row sidecar."""
    if objective != "temporal_robustness" or bundle_path is None:
        return None
    sidecar = bundle_path / "robustness_report.json"
    if not sidecar.exists():
        return None
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    properties = payload.get("properties") if isinstance(payload, dict) else None
    if not isinstance(properties, list):
        return None
    return sum(1 for prop in properties if prop.get("violated"))


def _fmt_opt(value: float | None) -> str:
    """Format an optional float for the markdown table."""
    return f"{value:.4f}" if value is not None else "-"


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
    parser.add_argument(
        "--objective",
        action="append",
        dest="objectives",
        default=None,
        help=(
            "Objective to evaluate; repeat to compare multiple objectives. "
            "Defaults to worst_case_snqi."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root for non-manifest runs; the manifest output_artifacts path takes precedence.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Package-B manifest (adversarial-package-b-comparison.v1) whose budget grid, "
            "repeated seeds, samplers, and output root drive the comparison. Overrides "
            "--package-b-budget-grid, --seed, --sampler, and --output-dir."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve manifest-relative paths.",
    )
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
        choices=("random", "coordinate", "optuna", "cmaes"),
        default=None,
        help="Sampler to run; repeat to select multiple. Defaults to all four.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a deterministic synthetic evaluator instead of running benchmark episodes.",
    )
    parser.add_argument(
        "--empirical",
        action="store_true",
        help=(
            "Run the real CPU benchmark evaluator (pysocialforce) instead of the synthetic "
            "path. Produces certified, replayable, valid failures. This is diagnostic/local "
            "nominal evidence, not paper-facing benchmark evidence."
        ),
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help=(
            "Write the durable issue #5326 comparison table (markdown) with exclusions,"
            " failures, and the stop-rule decision."
        ),
    )
    args = parser.parse_args(argv)
    if args.manifest is None and args.output_dir is None:
        parser.error("--output-dir is required unless --manifest is supplied")
    if args.empirical and args.synthetic:
        parser.error("--empirical and --synthetic are mutually exclusive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sampler comparison CLI."""
    args = parse_args(argv)
    if args.manifest is not None:
        repo_root = args.repo_root.resolve()
        config, objectives, samplers, budgets, seeds = load_package_b_manifest(
            args.manifest,
            repo_root=repo_root,
        )
        output_dir = config.output_dir
        out_json = (
            args.out_json
            if args.out_json is None or args.out_json.is_absolute()
            else repo_root / args.out_json
        )
    else:
        objectives = args.objectives or ["worst_case_snqi"]
        output_dir = args.output_dir
        config = SearchConfig.from_files(
            policy=args.policy,
            scenario_template=args.scenario_template,
            search_space=args.search_space,
            objective=objectives[0],
            output_dir=output_dir,
            budget=(args.budget or [8])[0],
            seed=(args.seed or [123])[0],
        )
        budgets = (
            (16, 32, 64) if args.package_b_budget_grid and args.budget is None else args.budget
        )
        seeds = args.seed or [123]
        samplers = args.samplers or ("random", "coordinate", "optuna", "cmaes")
        out_json = args.out_json

    rows = run_sampler_comparison(
        config=config,
        sampler_names=tuple(samplers),
        objective_names=objectives,
        synthetic=bool(args.synthetic) and not args.empirical,
        budgets=budgets,
        seeds=seeds,
    )
    payload = build_comparison_payload(
        rows=rows,
        objectives=objectives,
        budgets=budgets,
        seeds=seeds,
    )
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.out_md is not None:
        out_md = (
            args.out_md if args.out_md.is_absolute() else (args.repo_root.resolve() / args.out_md)
        )
        out_md.parent.mkdir(parents=True, exist_ok=True)
        table_md = render_durable_comparison_table(
            report_path=out_json,
            rows=rows,
            objectives=objectives,
            budget_grid=budgets,
            seeds=seeds,
        )
        out_md.write_text(table_md, encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
