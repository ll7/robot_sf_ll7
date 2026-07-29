"""Compare adversarial candidate samplers on a bounded search config."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.adversarial.attribution import attribution_from_episode_record
from robot_sf.adversarial.bundle import write_trajectory_csv
from robot_sf.adversarial.certification import passed_status
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    Pose2D,
    SearchConfig,
    WarmStartCandidate,
)
from robot_sf.adversarial.objectives import constraints_first_outcome_projection
from robot_sf.adversarial.samplers import build_sampler
from robot_sf.adversarial.search import run_adversarial_search
from robot_sf.benchmark.issue_5303_search_promotion_preflight import (
    DEFAULT_CONTRACT_PATH,
    preflight_issue_5303_contract,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


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


@dataclass(frozen=True)
class Issue5303DiagnosticContext:
    """Pinned provenance needed for the issue #5303 diagnostic search-stage rows."""

    scenario_family: str
    target_planner_config: Path
    neutral_reference_planner_config: Path
    execution_mode: str
    execution_context_label: str
    execution_commit: str


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
                        sampler=build_sampler(
                            sampler_name,
                            sampler_config.search_space,
                            seed=run_seed,
                            warm_start=sampler_config.warm_start,
                        ),
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
        "severe_intrusion",
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


def _sha256_file(path: Path) -> str:
    """Return a SHA-256 digest for one file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible provenance data."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _first_jsonl_record(path: Path | None, *, manifest_path: Path) -> dict[str, Any]:
    """Read one candidate episode record when its path resolves locally."""
    if path is None:
        return {}
    candidates = (path,) if path.is_absolute() else (manifest_path.parent / path, Path.cwd() / path)
    for candidate in candidates:
        try:
            with candidate.open(encoding="utf-8") as handle:
                first_line = handle.readline()
        except OSError:
            continue
        try:
            parsed = json.loads(first_line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _episode_execution_status(
    record: dict[str, Any], attribution: dict[str, Any]
) -> tuple[str, str, str]:
    """Extract conservative execution status fields from an episode or attribution payload."""
    details = attribution.get("details") if isinstance(attribution.get("details"), dict) else {}
    metadata_declared = "algorithm_metadata" in record or "algorithm" in record
    if "algorithm_metadata" in record:
        raw_algorithm = record.get("algorithm_metadata")
    else:
        raw_algorithm = record.get("algorithm")
    algorithm = raw_algorithm if isinstance(raw_algorithm, dict) else {}
    planner_kinematics = (
        algorithm.get("planner_kinematics")
        if isinstance(algorithm.get("planner_kinematics"), dict)
        else {}
    )

    def _status_value(payload: dict[str, Any], key: str) -> str | None:
        """Return one normalized status value without coercing missing values."""
        value = payload.get(key)
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized or None

    # The episode's algorithm metadata is authoritative when present. Do not let
    # malformed or status-less metadata fall back to derived manifest attribution.
    if metadata_declared:
        execution_mode = next(
            (
                value
                for payload, key in (
                    (algorithm, "execution_mode"),
                    (planner_kinematics, "execution_mode"),
                )
                if (value := _status_value(payload, key)) is not None
            ),
            "unknown",
        )
        metadata_status = _status_value(algorithm, "status")
        if metadata_status is None:
            return execution_mode, "degraded", "not_available"
        if metadata_status != "ok":
            readiness_status = "fallback" if "fallback" in metadata_status else "degraded"
            return execution_mode, readiness_status, "not_available"
        if execution_mode not in {"native", "adapter", "mixed"}:
            return execution_mode, "degraded", "not_available"
        readiness_status = _status_value(algorithm, "readiness_status") or execution_mode
        availability_status = _status_value(algorithm, "availability_status") or "available"
        return execution_mode, readiness_status, availability_status

    execution_mode = next(
        (
            value
            for payload, key in (
                (record, "execution_mode"),
                (details, "execution_mode"),
            )
            if (value := _status_value(payload, key)) is not None
        ),
        "unknown",
    )

    readiness_status = next(
        (
            value
            for payload in (record, algorithm, details)
            if (value := _status_value(payload, "readiness_status")) is not None
        ),
        None,
    )
    availability_status = next(
        (
            value
            for payload in (record, algorithm, details)
            if (value := _status_value(payload, "availability_status")) is not None
        ),
        None,
    )
    if readiness_status is None and execution_mode in {"native", "adapter", "mixed"}:
        readiness_status = execution_mode
    if availability_status is None and readiness_status in {"native", "adapter", "mixed"}:
        availability_status = "available"
    return execution_mode, readiness_status or "unknown", availability_status or "unknown"


def _load_archive_warm_starts(
    archive_path: Path,
    record_ids: Sequence[str],
) -> tuple[WarmStartCandidate, ...]:
    """Load selected archive candidates while preserving their source provenance."""
    payload = json.loads(archive_path.read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError("warm-start archive must contain an entries list")
    by_id = {
        str(entry.get("archive_id")): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("archive_id"), str)
    }
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("warm-start record IDs must be unique")
    warm_starts: list[WarmStartCandidate] = []
    for record_id in record_ids:
        entry = by_id.get(record_id)
        if entry is None:
            raise ValueError(f"warm-start record not found in archive: {record_id}")
        candidate = entry.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError(f"warm-start record has no candidate: {record_id}")
        start = candidate.get("start")
        goal = candidate.get("goal")
        if not isinstance(start, dict) or not isinstance(goal, dict):
            raise ValueError(f"warm-start record has invalid poses: {record_id}")
        scenario = entry.get("scenario_family")
        provenance = entry.get("provenance")
        planner = provenance.get("target_planner") if isinstance(provenance, dict) else None
        if not isinstance(scenario, str) or not scenario.strip():
            raise ValueError(f"warm-start record has no source scenario family: {record_id}")
        if not isinstance(planner, str) or not planner.strip():
            raise ValueError(f"warm-start record has no source planner: {record_id}")
        warm_starts.append(
            WarmStartCandidate(
                candidate=CandidateSpec(
                    start=Pose2D(**start),
                    goal=Pose2D(**goal),
                    spawn_time_s=float(candidate["spawn_time_s"]),
                    pedestrian_speed_mps=float(candidate["pedestrian_speed_mps"]),
                    pedestrian_delay_s=float(candidate["pedestrian_delay_s"]),
                    scenario_seed=int(candidate["scenario_seed"]),
                ),
                scenario=scenario,
                planner=planner,
            )
        )
    return tuple(warm_starts)


def _constraints_first_outcome(record: dict[str, Any]) -> dict[str, Any]:
    """Project one search-stage episode into the frozen constraints-first outcome vector."""
    return constraints_first_outcome_projection(record)


def _git_head() -> str:
    """Return the checked-out commit, or a fail-closed unavailable marker."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    commit = result.stdout.strip()
    return commit if commit else "unavailable"


def build_issue_5303_search_outcome_rows(
    *,
    rows: Sequence[SamplerComparisonRow],
    context: Issue5303DiagnosticContext,
) -> list[dict[str, Any]]:
    """Build complete, non-admitted search-stage rows for the #5303 diagnostic handoff.

    The diagnostic command records every scheduled search attempt and intentionally
    marks all rows as not admitted: it does not substitute for deterministic replay,
    five-seed target/reference confirmation, or a second execution context.  Those
    omissions are represented explicitly instead of being silently dropped from an
    estimand denominator.
    """
    target_hash = _sha256_file(context.target_planner_config)
    reference_hash = _sha256_file(context.neutral_reference_planner_config)
    outcome_rows: list[dict[str, Any]] = []
    for comparison_row in rows:
        manifest_path = Path(comparison_row.manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_config = manifest.get("config") if isinstance(manifest, dict) else {}
        candidates = manifest.get("candidates") if isinstance(manifest, dict) else []
        if not isinstance(manifest_config, dict) or not isinstance(candidates, list):
            raise ValueError(f"search manifest is missing config/candidates: {manifest_path}")
        scenario_template = Path(str(manifest_config.get("scenario_template", "")))
        search_space = Path(str(manifest_config.get("search_space_path", "")))
        if not scenario_template.is_file() or not search_space.is_file():
            raise ValueError(f"search manifest has unresolved frozen inputs: {manifest_path}")
        for candidate_index, raw_candidate in enumerate(candidates):
            item = raw_candidate if isinstance(raw_candidate, dict) else {}
            candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
            attribution = (
                item.get("failure_attribution")
                if isinstance(item.get("failure_attribution"), dict)
                else {}
            )
            episode_path_raw = item.get("episode_record_path")
            episode_path = Path(str(episode_path_raw)) if episode_path_raw else None
            record = _first_jsonl_record(episode_path, manifest_path=manifest_path)
            observed_execution_mode, readiness_status, availability_status = (
                _episode_execution_status(
                    record,
                    attribution,
                )
            )
            row = {
                "schema_version": "issue_5303_search_promotion_outcome_row.v1",
                "row_id": (
                    f"{comparison_row.sampler}:{comparison_row.seed}:{candidate_index:04d}:search"
                ),
                "arm": comparison_row.sampler,
                "method": comparison_row.sampler,
                "search_seed": int(comparison_row.seed),
                "candidate_index": candidate_index,
                "normalized_candidate_config_sha256": _canonical_sha256(candidate),
                "candidate": candidate,
                "scenario_family": context.scenario_family,
                "scenario_config_path": scenario_template.as_posix(),
                "scenario_config_sha256": _sha256_file(scenario_template),
                "search_space_path": search_space.as_posix(),
                "search_space_sha256": _sha256_file(search_space),
                "target_planner_config_path": context.target_planner_config.as_posix(),
                "target_planner_config_sha256": target_hash,
                "neutral_reference_planner_config_path": (
                    context.neutral_reference_planner_config.as_posix()
                ),
                "neutral_reference_planner_config_sha256": reference_hash,
                "execution_stage": "search",
                "execution_seed": candidate.get("scenario_seed"),
                "seed_lineage": {
                    "search_seed": int(comparison_row.seed),
                    "candidate_scenario_seed": candidate.get("scenario_seed"),
                    "deterministic_replay_seed": None,
                    "confirmation_seeds": [],
                    "second_context_seed": None,
                },
                # Availability, readiness, and execution mode are all observed per attempt.
                # Missing or degraded status must remain visible and fail closed in the
                # accounting analyzer rather than being replaced by a command-level success tag.
                "execution_mode": observed_execution_mode,
                "readiness_status": readiness_status,
                "availability_status": availability_status,
                "constraints_first_outcome": _constraints_first_outcome(record),
                "objective": comparison_row.objective,
                "objective_value": item.get("objective_value"),
                "primary_failure_mechanism": attribution.get("primary_failure"),
                "stable_attribution_evidence": "not_collected_diagnostic_only",
                "certification": item.get("certification_status"),
                "recertification_lineage": "issue_6139_frozen_input",
                "deterministic_replay": "not_run_diagnostic_only",
                "confirmation_target": "not_run_diagnostic_only",
                "confirmation_reference": "not_run_diagnostic_only",
                "second_execution_context": "not_run_diagnostic_only",
                "execution_commit": context.execution_commit,
                "execution_context_label": context.execution_context_label,
                "admission_decision": "not_admitted_diagnostic_only",
                "exclusion_reason": "diagnostic_only_no_replay_reference_or_second_context",
            }
            row["immutable_record_sha256"] = _canonical_sha256(row)
            outcome_rows.append(row)
    return outcome_rows


def write_issue_5303_search_outcome_rows(rows: Sequence[dict[str, Any]], output: Path) -> None:
    """Write the #5303 search-stage outcome rows as JSON Lines."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def build_comparison_payload(
    *,
    rows: Sequence[SamplerComparisonRow],
    objectives: Sequence[str],
    budgets: Sequence[int],
    seeds: Sequence[int],
    claim_scope: str = "not_paper_facing_benchmark_evidence",
    report_status: str = "diagnostic_local_nominal",
    held_out_status: str = "not_evaluated_narrow_archive",
    issue_5303_diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the durable Package-B comparison report payload from result rows.

    The payload preserves the existing report-gate contract: every sampler/budget/seed
    cell appears exactly once, held-out yield stays null, and the claim scope stays
    diagnostic. The resulting mapping can be written directly and validated by
    ``validate_package_b_report``.
    """
    payload: dict[str, Any] = {
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
    if issue_5303_diagnostic is not None:
        payload["issue_5303_diagnostic"] = issue_5303_diagnostic
    return payload


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
    issue_5303_diagnostic: bool = False,
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
    if issue_5303_diagnostic:
        lines.append("## Issue #5303 diagnostic search-stage comparison table\n")
        lines.append(
            "> Claim boundary: diagnostic-only execution/accounting probe. It is not a"
            " promotion result: deterministic replay, five-seed target/reference confirmation,"
            " and second-context confirmation are intentionally not collected here.\n"
        )
    else:
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
    if issue_5303_diagnostic:
        decision = (
            "**INCONCLUSIVE (predeclared diagnostic-only).** This output checks the frozen"
            " search-stage command and complete attempt accounting only. It cannot satisfy"
            " the promotion endpoint or authorize transfer; a larger re-preregistration is"
            " required before any `promote` decision."
        )
    elif any_degraded:
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
        "--scenario-family",
        default=None,
        help="Optional frozen scenario-family label recorded in issue-specific outcome rows.",
    )
    parser.add_argument(
        "--search-space",
        type=Path,
        default=Path("configs/adversarial/crossing_ttc_space.yaml"),
    )
    parser.add_argument("--policy", default="goal")
    parser.add_argument(
        "--algo-config",
        type=Path,
        default=None,
        help="Optional target planner configuration passed through to the benchmark runner.",
    )
    parser.add_argument(
        "--reference-algo-config",
        type=Path,
        default=None,
        help=(
            "Frozen neutral-reference planner configuration recorded in issue-specific outcome "
            "rows; the generic search stage does not execute it."
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help=(
            "Frozen issue #5303 contract checked before diagnostic execution; ignored by "
            "generic comparison modes."
        ),
    )
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
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument(
        "--require-certification",
        action="store_true",
        help="Fail closed when a candidate lacks required scenario certification.",
    )
    parser.add_argument(
        "--benchmark-profile",
        default="baseline-safe",
        help="Benchmark readiness profile forwarded to candidate execution.",
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
    parser.add_argument(
        "--outcomes-jsonl",
        type=Path,
        default=None,
        help="Optional complete per-attempt outcome rows for the issue #5303 diagnostic handoff.",
    )
    parser.add_argument(
        "--issue-5303-diagnostic-only",
        action="store_true",
        help=(
            "Run the frozen #5303 search-stage diagnostic only. This mode records every "
            "attempt as not admitted and can never return a promotion result."
        ),
    )
    parser.add_argument(
        "--execution-context-label",
        default=None,
        help="Non-sensitive label for the recorded execution context in issue #5303 rows.",
    )
    parser.add_argument(
        "--warm-start-archive",
        type=Path,
        default=None,
        help="Certified archive containing explicitly selected matched warm-start candidates.",
    )
    parser.add_argument(
        "--warm-start-record",
        action="append",
        default=None,
        help="Certified archive ID to use as a warm start; repeat in frozen archive order.",
    )
    args = parser.parse_args(argv)
    if args.manifest is None and args.output_dir is None:
        parser.error("--output-dir is required unless --manifest is supplied")
    if args.empirical and args.synthetic:
        parser.error("--empirical and --synthetic are mutually exclusive")
    if args.issue_5303_diagnostic_only:
        if args.manifest is not None:
            parser.error("--issue-5303-diagnostic-only cannot be combined with --manifest")
        if args.synthetic:
            parser.error("--issue-5303-diagnostic-only requires non-synthetic execution")
        required = {
            "--algo-config": args.algo_config,
            "--reference-algo-config": args.reference_algo_config,
            "--scenario-family": args.scenario_family,
            "--out-json": args.out_json,
            "--out-md": args.out_md,
            "--outcomes-jsonl": args.outcomes_jsonl,
            "--execution-context-label": args.execution_context_label,
            "--warm-start-archive": args.warm_start_archive,
            "--warm-start-record": args.warm_start_record,
        }
        missing = [flag for flag, value in required.items() if value is None or value == ""]
        if missing:
            parser.error("--issue-5303-diagnostic-only requires " + ", ".join(sorted(missing)))
    return args


def _require_issue_5303_preflight_if_requested(
    args: argparse.Namespace, *, repo_root: Path
) -> None:
    """Fail before diagnostic execution when the frozen contract no longer verifies."""
    if not args.issue_5303_diagnostic_only:
        return
    preflight = preflight_issue_5303_contract(
        args.contract,
        repo_root=repo_root,
    )
    if not preflight.ready:
        detail = "; ".join(preflight.blockers) or "unknown frozen-contract failure"
        raise RuntimeError(f"issue #5303 preflight failed before diagnostic execution: {detail}")
    _require_issue_5303_frozen_bindings(args, repo_root=repo_root)
    contract_path = _resolve_issue_5303_diagnostic_path(args.contract, repo_root=repo_root)
    assert contract_path is not None
    try:
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError("issue #5303 diagnostic authorization could not be verified") from exc
    future_run = contract.get("future_run_declaration") if isinstance(contract, dict) else None
    diagnostic_run = (
        future_run.get("separately_justified_diagnostic_search_run")
        if isinstance(future_run, dict)
        else None
    )
    if not isinstance(diagnostic_run, dict) or diagnostic_run.get("authorized") is not True:
        raise RuntimeError(
            "issue #5303 diagnostic execution is not authorized; the frozen command "
            "is retained for preflight binding proof only"
        )


def _resolve_issue_5303_diagnostic_path(value: Path | None, *, repo_root: Path) -> Path | None:
    """Resolve a diagnostic CLI path for equivalence with the frozen command."""
    if value is None:
        return None
    candidate = value if value.is_absolute() else repo_root / value
    return candidate.resolve()


def _frozen_issue_5303_diagnostic_args(*, repo_root: Path) -> argparse.Namespace:
    """Parse the authoritative contract command without running a search."""
    contract_path = repo_root / DEFAULT_CONTRACT_PATH
    try:
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(
            "issue #5303 frozen diagnostic command could not load the canonical contract"
        ) from exc
    step3 = contract.get("step3_execution") if isinstance(contract, dict) else None
    command = step3.get("diagnostic_search_command") if isinstance(step3, dict) else None
    if not isinstance(command, str) or not command.strip():
        raise RuntimeError("issue #5303 frozen diagnostic command is missing from the contract")
    try:
        command_parts = shlex.split(command)
        script_index = command_parts.index("scripts/tools/compare_adversarial_samplers.py")
    except (ValueError, IndexError) as exc:
        raise RuntimeError("issue #5303 frozen diagnostic command cannot be parsed") from exc
    try:
        return parse_args(command_parts[script_index + 1 :])
    except SystemExit as exc:
        raise RuntimeError("issue #5303 frozen diagnostic command is not runnable") from exc


def _require_issue_5303_frozen_bindings(args: argparse.Namespace, *, repo_root: Path) -> None:
    """Reject diagnostic invocation drift before it can generate new search outcomes."""
    expected = _frozen_issue_5303_diagnostic_args(repo_root=repo_root)
    mismatches: list[str] = []
    for field_name in (
        "policy",
        "scenario_family",
        "objectives",
        "budget",
        "seed",
        "horizon",
        "dt",
        "require_certification",
        "benchmark_profile",
        "samplers",
        "synthetic",
        "empirical",
        "package_b_budget_grid",
        "execution_context_label",
        "warm_start_record",
    ):
        if getattr(args, field_name) != getattr(expected, field_name):
            mismatches.append(field_name)
    for field_name in (
        "repo_root",
        "scenario_template",
        "search_space",
        "algo_config",
        "reference_algo_config",
        "contract",
        "output_dir",
        "manifest",
        "out_json",
        "out_md",
        "outcomes_jsonl",
        "warm_start_archive",
    ):
        actual = _resolve_issue_5303_diagnostic_path(getattr(args, field_name), repo_root=repo_root)
        frozen = _resolve_issue_5303_diagnostic_path(
            getattr(expected, field_name), repo_root=repo_root
        )
        if actual != frozen:
            mismatches.append(field_name)
    if mismatches:
        raise RuntimeError(
            "issue #5303 diagnostic execution has mismatched frozen bindings: "
            + ", ".join(mismatches)
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sampler comparison CLI."""
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    _require_issue_5303_preflight_if_requested(args, repo_root=repo_root)
    if args.manifest is not None:
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
        warm_starts = ()
        if args.warm_start_archive is not None or args.warm_start_record:
            if args.warm_start_archive is None or not args.warm_start_record:
                raise ValueError(
                    "--warm-start-archive and at least one --warm-start-record must be supplied together"
                )
            warm_starts = _load_archive_warm_starts(
                args.warm_start_archive,
                tuple(args.warm_start_record),
            )
        config = SearchConfig.from_files(
            policy=args.policy,
            scenario_template=args.scenario_template,
            search_space=args.search_space,
            objective=objectives[0],
            output_dir=output_dir,
            budget=(args.budget or [8])[0],
            seed=(args.seed or [123])[0],
            algo_config_path=args.algo_config,
            horizon=args.horizon,
            dt=args.dt,
            require_certification=bool(args.require_certification),
            benchmark_profile=str(args.benchmark_profile),
            warm_start=warm_starts,
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
    outcomes_jsonl = (
        args.outcomes_jsonl
        if args.outcomes_jsonl is None or args.outcomes_jsonl.is_absolute()
        else repo_root / args.outcomes_jsonl
    )
    diagnostic_context: Issue5303DiagnosticContext | None = None
    diagnostic_payload: dict[str, Any] | None = None
    if args.issue_5303_diagnostic_only:
        assert args.algo_config is not None
        assert args.reference_algo_config is not None
        assert args.scenario_family is not None
        assert args.execution_context_label is not None
        assert outcomes_jsonl is not None
        if not args.algo_config.is_file() or not args.reference_algo_config.is_file():
            raise FileNotFoundError("issue #5303 planner configuration path does not exist")
        execution_commit = _git_head()
        if execution_commit == "unavailable":
            raise RuntimeError("issue #5303 diagnostic execution requires a resolvable git HEAD")
        diagnostic_context = Issue5303DiagnosticContext(
            scenario_family=str(args.scenario_family),
            target_planner_config=args.algo_config,
            neutral_reference_planner_config=args.reference_algo_config,
            execution_mode="adapter",
            execution_context_label=str(args.execution_context_label),
            execution_commit=execution_commit,
        )
        diagnostic_payload = {
            "mode": "search_stage_diagnostic_only",
            "predeclared_decision": "inconclusive",
            "promotion_eligible": False,
            "outcomes_jsonl": outcomes_jsonl.as_posix(),
            "reason": (
                "three search seeds cannot test the frozen two-sided p<=0.05 promotion gate; "
                "replay, target/reference confirmation, and second-context evidence are not "
                "substituted by this diagnostic command"
            ),
        }

    payload = build_comparison_payload(
        rows=rows,
        objectives=objectives,
        budgets=budgets,
        seeds=seeds,
        claim_scope=(
            "issue_5303_diagnostic_execution_only"
            if diagnostic_payload is not None
            else "not_paper_facing_benchmark_evidence"
        ),
        report_status=(
            "diagnostic_inconclusive"
            if diagnostic_payload is not None
            else "diagnostic_local_nominal"
        ),
        held_out_status=(
            "not_admitted_diagnostic_only"
            if diagnostic_payload is not None
            else "not_evaluated_narrow_archive"
        ),
        issue_5303_diagnostic=diagnostic_payload,
    )
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if diagnostic_context is not None:
        assert outcomes_jsonl is not None
        write_issue_5303_search_outcome_rows(
            build_issue_5303_search_outcome_rows(rows=rows, context=diagnostic_context),
            outcomes_jsonl,
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
            issue_5303_diagnostic=diagnostic_context is not None,
        )
        out_md.write_text(table_md, encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
