"""Programmable adversarial scenario search runner."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from robot_sf.adversarial.attribution import (
    FailureAttribution,
    attribution_from_episode_record,
    attribution_from_error,
)
from robot_sf.adversarial.bundle import (
    compute_effective_scenario_hash,
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
from robot_sf.adversarial.samplers import CandidateSampler, build_sampler
from robot_sf.adversarial.scenario_admissibility import (
    classify_scenario_admissibility,
    validate_scenario_admissibility,
)
from robot_sf.benchmark.fallback_policy import (
    resolve_execution_mode,
    summarize_benchmark_availability,
)
from robot_sf.benchmark.runner import run_batch

CandidateEvaluator = Callable[[SearchConfig, CandidateSpec, Path, Path], CandidateEvaluation]
CandidateCertifier = Callable[[CandidateSpec, Path, bool], CertificationStatus]
ProductionCandidateEvaluator = Callable[[SearchConfig, CandidateSpec, int], CandidateEvaluation]

DEFAULT_SCHEMA_PATH = (
    Path(__file__).parent.parent / "benchmark" / "schemas" / "episode.schema.v1.json"
)


def _mark_missing_provenance(enriched: dict[str, Any], reason: str) -> dict[str, Any]:
    """Mark a run unavailable when its execution provenance is incomplete."""
    preflight = enriched.get("preflight")
    preflight_payload = dict(preflight) if isinstance(preflight, dict) else {}
    preflight_status = str(preflight_payload.get("status", "")).strip().lower()
    if preflight_status not in {"fallback", "skipped"}:
        preflight_payload["status"] = "skipped"
        preflight_payload.setdefault("compatibility_reason", reason)
        enriched["preflight"] = preflight_payload
    return enriched


def _validate_preflight_status(enriched: dict[str, Any]) -> str | None:
    """Reject explicit non-terminal preflight statuses before availability scoring."""
    preflight = enriched.get("preflight")
    if "preflight" in enriched and not isinstance(preflight, dict):
        return "episode preflight metadata was missing or malformed"
    if isinstance(preflight, dict):
        preflight_status = str(preflight.get("status", "")).strip().lower()
        if preflight_status not in {"ok", "fallback", "skipped"}:
            return f"episode preflight status was non-terminal: {preflight_status or 'missing'}"
    return None


def _validate_episode_provenance(
    enriched: dict[str, Any], record: dict[str, Any] | None
) -> str | None:
    """Return a missing-provenance reason and enrich direct-runner summaries."""
    if not isinstance(record, dict) or not record:
        return "episode record was missing or malformed"

    metadata_present = "algorithm_metadata" in record
    metadata = record.get("algorithm_metadata")
    if metadata_present and (not isinstance(metadata, dict) or not metadata):
        return "episode algorithm metadata was missing or malformed"
    if isinstance(metadata, dict) and not isinstance(
        enriched.get("algorithm_metadata_contract"), dict
    ):
        enriched["algorithm_metadata_contract"] = metadata

    effective_metadata = enriched.get("algorithm_metadata_contract")
    if not isinstance(effective_metadata, dict) or not effective_metadata:
        return "episode algorithm metadata contract was missing"

    metadata_statuses = [
        str(payload.get("status", "")).strip().lower()
        for payload in (metadata, effective_metadata)
        if isinstance(payload, dict) and str(payload.get("status", "")).strip()
    ]
    if not metadata_statuses:
        return "episode algorithm metadata status was missing"
    non_ok_statuses = [status for status in metadata_statuses if status != "ok"]
    if non_ok_statuses:
        metadata_status = non_ok_statuses[0]
        preflight = enriched.get("preflight")
        preflight_payload = dict(preflight) if isinstance(preflight, dict) else {}
        preflight_status = str(preflight_payload.get("status", "")).strip().lower()
        if preflight_status not in {"fallback", "skipped"}:
            preflight_payload["status"] = (
                "fallback"
                if "fallback" in metadata_status or "unavailable" in metadata_status
                else "skipped"
            )
            preflight_payload.setdefault(
                "compatibility_reason",
                f"episode algorithm metadata status: {metadata_status}",
            )
            enriched["preflight"] = preflight_payload
    else:
        episode_mode = resolve_execution_mode(metadata)
        summary_mode = resolve_execution_mode(effective_metadata)
        if episode_mode == "unknown" or summary_mode == "unknown":
            return "episode algorithm metadata execution mode was missing or malformed"
        if episode_mode != summary_mode:
            return "episode and summary algorithm metadata execution modes disagree"

    # A direct runner may omit the preflight block entirely, in which case the
    # validated episode metadata is the authoritative provenance source.  When
    # a preflight block is present, only terminal statuses are safe to carry
    # forward; failed/partial/unknown states cannot become available because an
    # episode was written.
    return _validate_preflight_status(enriched)


def _availability_summary(summary: dict[str, Any], record: dict[str, Any] | None) -> dict[str, Any]:
    """Complete a batch summary with per-episode execution provenance.

    The direct benchmark runner returns episode counts and failures but does not
    include the algorithm metadata contract that the map runner places in its
    summary. Use the written episode record as the authoritative fallback, and
    surface non-``ok`` algorithm statuses as a preflight failure so fallback or
    degraded execution cannot be reported as available.
    """
    enriched = dict(summary)
    reason = _validate_episode_provenance(enriched, record)
    if reason:
        return _mark_missing_provenance(enriched, reason)
    return enriched


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
    availability = summarize_benchmark_availability(_availability_summary(summary, record))
    attribution = replace(
        attribution,
        details={
            **attribution.details,
            "execution_mode": availability.execution_mode,
            "readiness_status": availability.readiness_status,
            "availability_status": availability.availability_status,
        },
    )
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


def _effective_hash_for_bundle(scenario_yaml_path: Path, candidate_dir: Path) -> str | None:
    """Bind a candidate bundle to its runtime-effective scenario hash, if readable.

    Returns ``None`` when the written inputs cannot be loaded (fail closed: the
    eligibility receipt records ``effective_hash_unbound`` instead of guessing).
    """
    try:
        scenario_payload = yaml.safe_load(scenario_yaml_path.read_text(encoding="utf-8"))
        route_path = candidate_dir / "route_overrides.yaml"
        route_payload = (
            yaml.safe_load(route_path.read_text(encoding="utf-8")) if route_path.is_file() else {}
        )
        scenario = (scenario_payload or {}).get("scenarios", [{}])[0]
        return compute_effective_scenario_hash(scenario, route_payload or {})
    except Exception:  # noqa: BLE001 - unreadable inputs mean unbound hash, never a crash
        return None


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
    active_sampler = sampler or build_sampler(
        "random",
        config.search_space,
        seed=config.seed,
        warm_start=config.warm_start,
    )

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
            certification_status, _case_id = _attach_scenario_admissibility(
                failed_status(
                    "search-space validation failed", details={"errors": validation_errors}
                ),
                candidate=candidate,
                scenario_yaml_path=None,
            )
            num_invalid += 1
            evaluation = replace(
                _invalid_evaluation(
                    candidate=candidate,
                    certification_status=certification_status,
                    scenario_yaml_path=None,
                    bundle_path=None,
                    reason="; ".join(validation_errors),
                ),
                evaluation_disposition="rejected_by_search_space",
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
        certification_status, admissibility_case_id = _attach_scenario_admissibility(
            certification_status,
            candidate=candidate,
            scenario_yaml_path=scenario_yaml_path,
        )
        if _scenario_admissibility_rejects(
            certification_status,
            expected_case_id=admissibility_case_id,
        ):
            num_invalid += 1
            evaluation = replace(
                _invalid_evaluation(
                    candidate=candidate,
                    certification_status=certification_status,
                    scenario_yaml_path=scenario_yaml_path,
                    bundle_path=candidate_dir,
                    reason="scenario admissibility rejected candidate",
                ),
                evaluation_disposition="rejected_by_admissibility",
            )
            write_json(
                candidate_dir / "failure_attribution.json", evaluation.failure_attribution.to_json()
            )
            evaluations.append(evaluation)
            _observe_candidate(active_sampler, evaluation)
            continue
        if not candidate_allowed(
            certification_status,
            require_certification=config.require_certification,
        ):
            num_invalid += 1
            evaluation = replace(
                _invalid_evaluation(
                    candidate=candidate,
                    certification_status=certification_status,
                    scenario_yaml_path=scenario_yaml_path,
                    bundle_path=candidate_dir,
                    reason=certification_status.reason,
                ),
                evaluation_disposition="rejected_by_certification",
            )
            write_json(
                candidate_dir / "failure_attribution.json", evaluation.failure_attribution.to_json()
            )
            evaluations.append(evaluation)
            _observe_candidate(active_sampler, evaluation)
            continue

        try:
            evaluation = active_evaluator(config, candidate, scenario_yaml_path, candidate_dir)
            evaluation = replace(
                evaluation,
                certification_status=certification_status,
                effective_scenario_hash=_effective_hash_for_bundle(
                    scenario_yaml_path, candidate_dir
                ),
                evaluation_disposition="evaluator_invoked",
            )
            score = objective(evaluation)
            evaluation = evaluation.with_objective(score)
        except Exception as exc:  # noqa: BLE001 - evaluator failure records candidate attribution
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
                evaluation_disposition="evaluator_invoked",
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


def _attach_scenario_admissibility(
    certification_status: CertificationStatus,
    *,
    candidate: CandidateSpec,
    scenario_yaml_path: Path | None,
) -> tuple[CertificationStatus, str]:
    """Bind one conservative admissibility verdict to a materialized search candidate.

    The existing certifier result is reused; this helper never reruns the producer. The
    admissibility adapter independently binds the certificate to the selected scenario bytes
    and runtime-referenced map/route closure. Any absent, mismatched, or unreadable evidence
    remains a retained unknown row in the normal search manifest.
    """
    raw_scenario: bytes | None = None
    scenario_id: str | None = None
    certificate: dict[str, Any] | None = None
    classification_error_type: str | None = None
    try:
        if scenario_yaml_path is not None:
            raw_scenario = scenario_yaml_path.read_bytes()
            scenario_document = yaml.safe_load(raw_scenario.decode("utf-8"))
            scenarios = (
                scenario_document.get("scenarios") if isinstance(scenario_document, dict) else None
            )
            if (
                isinstance(scenarios, list)
                and len(scenarios) == 1
                and isinstance(scenarios[0], dict)
            ):
                row = scenarios[0]
                value = row.get("name") or row.get("scenario_id") or row.get("id")
                if isinstance(value, str) and value.strip():
                    scenario_id = value.strip()
        certificates = certification_status.details.get("certificates")
        if isinstance(certificates, list) and scenario_id is not None:
            matching = [
                item
                for item in certificates
                if isinstance(item, dict) and item.get("scenario_id") == scenario_id
            ]
            if len(matching) == 1:
                certificate = matching[0]

        case_bytes = raw_scenario or json.dumps(
            candidate.to_json(), sort_keys=True, separators=(",", ":"), allow_nan=True
        ).encode("utf-8")
        case_id = f"adversarial-search-{hashlib.sha256(case_bytes).hexdigest()}"
        verdict = classify_scenario_admissibility(
            case_id,
            scenario_artifact_path=scenario_yaml_path,
            scenario_id=scenario_id,
            scenario_certificate=certificate,
        ).to_dict()
    except Exception as exc:  # noqa: BLE001 - retain unknown evidence instead of dropping a row
        classification_error_type = type(exc).__name__
        fallback_bytes = json.dumps(
            candidate.to_json(), sort_keys=True, separators=(",", ":"), allow_nan=True
        ).encode("utf-8")
        case_id = f"adversarial-search-{hashlib.sha256(fallback_bytes).hexdigest()}"
        verdict = {
            "schema_version": "scenario_admissibility.v1",
            "case_id": case_id,
            "scenario_id": scenario_id,
            "verdict": "admissible_feasibility_unknown",
            "target_planner_outcome": "not_evaluated",
            "search_disposition": "retain",
            "reason_codes": ["admissibility_classification_unavailable"],
            "assumptions": {},
            "evidence": {
                "classification_adapter": {
                    "status": "unavailable",
                    "error_type": classification_error_type,
                    "scenario_artifact_sha256": (
                        hashlib.sha256(raw_scenario).hexdigest()
                        if raw_scenario is not None
                        else None
                    ),
                }
            },
        }
    validate_scenario_admissibility(verdict)
    details = dict(certification_status.details)
    details["scenario_admissibility"] = verdict
    if classification_error_type is not None:
        details["scenario_admissibility_status"] = "unavailable"
    return replace(certification_status, details=details), case_id


def _scenario_admissibility_rejects(
    certification_status: CertificationStatus,
    *,
    expected_case_id: str,
) -> bool:
    """Honor only a schema-valid rejection bound to this materialized candidate."""
    verdict = certification_status.details.get("scenario_admissibility")
    if not isinstance(verdict, dict):
        return False
    try:
        validate_scenario_admissibility(verdict)
    except ValueError:
        return False
    return (
        verdict.get("case_id") == expected_case_id and verdict.get("search_disposition") == "reject"
    )


def production_candidate_evaluator(
    *,
    evaluator: CandidateEvaluator | None = None,
    certifier: CandidateCertifier | None = None,
) -> ProductionCandidateEvaluator:
    """Return the canonical per-candidate production pipeline for adversarial search.

    The returned callable runs the same validation -> materialization ->
    certification -> benchmark-evaluation sequence as ``run_adversarial_search`` for
    one candidate, writing its bundle under ``config.output_dir / f"candidate_{index:04d}"``.
    It is the integration point that lets the MAP-Elites quality-diversity archive
    (``robot_sf/adversarial/qd.py``) run against the real adversarial pipeline instead of
    an injected stub: see ``qd.production_qd_evaluator``.

    Args:
        evaluator: Optional injected four-argument evaluator; defaults to the benchmark
            batch runner (``SearchConfig``/scenario/bundle contract).
        certifier: Optional injected certifier; defaults to ``certify_candidate``.

    Returns:
        A callable ``(config, candidate, index) -> CandidateEvaluation``.
    """

    def _evaluate(
        config: SearchConfig, candidate: CandidateSpec, index: int
    ) -> CandidateEvaluation:
        """Run one candidate through the production adversarial pipeline.

        Executes the validate -> materialize -> certify -> benchmark sequence under
        ``config.output_dir / candidate_{index:04d}`` and returns its evaluation.
        """
        candidate_dir = config.output_dir / f"candidate_{index:04d}"
        validation_errors = config.search_space.validate_candidate(candidate)
        if validation_errors:
            return _invalid_evaluation(
                candidate=candidate,
                certification_status=failed_status(
                    "search-space validation failed",
                    details={"errors": validation_errors},
                ),
                scenario_yaml_path=None,
                bundle_path=None,
                reason="; ".join(validation_errors),
            )

        scenario_yaml_path, _route_path = write_candidate_inputs(
            config=config,
            candidate=candidate,
            candidate_dir=candidate_dir,
            index=index,
        )
        active_certifier = certifier or _default_certifier
        certification_status = active_certifier(
            candidate, scenario_yaml_path, config.require_certification
        )
        if not candidate_allowed(
            certification_status, require_certification=config.require_certification
        ):
            evaluation = _invalid_evaluation(
                candidate=candidate,
                certification_status=certification_status,
                scenario_yaml_path=scenario_yaml_path,
                bundle_path=candidate_dir,
                reason=certification_status.reason,
            )
            write_json(
                candidate_dir / "failure_attribution.json",
                evaluation.failure_attribution.to_json(),
            )
            return evaluation

        active_evaluator = evaluator or _default_evaluator
        objective = get_objective(config.objective)
        try:
            evaluation = active_evaluator(config, candidate, scenario_yaml_path, candidate_dir)
            evaluation = replace(
                evaluation,
                certification_status=certification_status,
                effective_scenario_hash=_effective_hash_for_bundle(
                    scenario_yaml_path, candidate_dir
                ),
            )
            score = objective(evaluation)
            return evaluation.with_objective(score)
        except Exception as exc:  # noqa: BLE001 - evaluator failure records candidate attribution
            error = repr(exc)
            attribution = attribution_from_error(error)
            write_json(candidate_dir / "failure_attribution.json", attribution.to_json())
            return CandidateEvaluation(
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

    return _evaluate
