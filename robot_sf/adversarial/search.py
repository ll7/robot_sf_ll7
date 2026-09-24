"""Programmable adversarial scenario search runner."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
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
from robot_sf.adversarial.io import parse_first_jsonl_record, read_first_jsonl_record
from robot_sf.adversarial.objectives import get_objective
from robot_sf.adversarial.samplers import CandidateSampler, build_sampler
from robot_sf.adversarial.scenario_admissibility import (
    classify_scenario_admissibility,
    validate_scenario_admissibility,
)
from robot_sf.benchmark.fallback_policy import (
    resolve_execution_mode,
    runtime_fallback_or_degraded_marker,
    summarize_benchmark_availability,
)
from robot_sf.benchmark.runner import run_batch
from robot_sf.benchmark.termination_reason import (
    TERMINATION_REASONS,
    outcome_contradictions,
    status_from_termination_reason,
)
from robot_sf.benchmark.utils import _config_hash
from robot_sf.scenario_certification.input_identity import scenario_input_identity

CandidateEvaluator = Callable[[SearchConfig, CandidateSpec, Path, Path], CandidateEvaluation]
CandidateCertifier = Callable[[CandidateSpec, Path, bool], CertificationStatus]
ProductionCandidateEvaluator = Callable[[SearchConfig, CandidateSpec, int], CandidateEvaluation]

DEFAULT_SCHEMA_PATH = (
    Path(__file__).parent.parent / "benchmark" / "schemas" / "episode.schema.v1.json"
)
_NATIVE_EXECUTION_MODES = frozenset({"native", "native_command"})


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
    scenario_admissibility: dict[str, Any] | None = None,
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
        scenario_admissibility=scenario_admissibility,
    )


def _candidate_scenario_id(scenario_yaml_path: Path) -> str | None:
    """Read the one materialized scenario ID, returning None for malformed input."""
    try:
        payload = yaml.safe_load(scenario_yaml_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return None
    scenarios = payload.get("scenarios") if isinstance(payload, Mapping) else None
    if not isinstance(scenarios, list) or len(scenarios) != 1:
        return None
    scenario = scenarios[0]
    scenario_id = scenario.get("name") if isinstance(scenario, Mapping) else None
    return scenario_id.strip() if isinstance(scenario_id, str) and scenario_id.strip() else None


def _certificate_for_scenario(
    status: CertificationStatus, scenario_id: str | None
) -> Mapping[str, Any] | None:
    """Select only a certificate whose producer scenario ID matches the candidate."""
    details = status.details if isinstance(status.details, Mapping) else {}
    certificates = details.get("certificates")
    if not isinstance(certificates, list):
        return details if details.get("schema_version") == "scenario_cert.v1" else None
    matches = [
        item
        for item in certificates
        if isinstance(item, Mapping) and item.get("scenario_id") == scenario_id
    ]
    return matches[0] if len(matches) == 1 else None


def _admissibility_payload(
    *,
    index: int,
    scenario_yaml_path: Path,
    certification_status: CertificationStatus,
) -> dict[str, Any]:
    """Classify a materialized candidate for durable stratification and early rejection."""
    scenario_id = _candidate_scenario_id(scenario_yaml_path)
    try:
        verdict = classify_scenario_admissibility(
            f"candidate_{index:04d}",
            scenario_artifact_path=scenario_yaml_path,
            scenario_id=scenario_id,
            scenario_certificate=_certificate_for_scenario(certification_status, scenario_id),
        )
        return verdict.to_dict()
    except Exception as exc:  # noqa: BLE001 - classification failure must remain an explicit unknown.
        payload = classify_scenario_admissibility(f"candidate_{index:04d}").to_dict()
        payload["reason_codes"] = sorted(
            set(payload["reason_codes"] + ["scenario_admissibility_classifier_error"])
        )
        payload["assumptions"] = {"classifier_error_type": type(exc).__name__}
        validate_scenario_admissibility(payload)
        return payload


def _admissibility_rejection_reason(payload: Mapping[str, Any]) -> str:
    """Summarize explicit exclusion reason codes for candidate failure attribution."""
    reason_codes = payload.get("reason_codes")
    reasons = (
        ", ".join(str(reason) for reason in reason_codes) if isinstance(reason_codes, list) else ""
    )
    return f"scenario admissibility rejected candidate: {reasons or 'explicit exclusion'}"


def _post_evaluation_admissibility(
    payload: Mapping[str, Any],
    *,
    config: SearchConfig,
    candidate: CandidateSpec,
    scenario_yaml_path: Path,
    episode_record_path: Path | None,
    failure_attribution: FailureAttribution | None,
    evaluation_error: str | None = None,
) -> dict[str, Any]:
    """Attach a provenance-checked planner observation without changing feasibility."""
    updated = dict(payload)
    evidence = dict(updated.get("evidence", {}))
    observation: dict[str, Any] = {
        "status": "unavailable",
        "reason_code": "target_planner_observation_unavailable",
    }
    target_outcome = "unavailable"
    scenario_identity_snapshot: Mapping[str, Any] | None = None
    if evaluation_error is not None:
        observation["reason_code"] = "target_evaluation_failed"
        observation["evaluation_error"] = evaluation_error
    elif episode_record_path is None:
        observation["reason_code"] = "target_episode_record_path_missing"
    else:
        episode_path = Path(episode_record_path)
        observation["episode_record_path"] = episode_path.as_posix()
        expected_identity = evidence.get("scenario_artifact_identity")
        scenario_id = _candidate_scenario_id(scenario_yaml_path)
        try:
            current_identity = scenario_input_identity(scenario_yaml_path, scenario_id=scenario_id)
        except (OSError, RuntimeError, TypeError, ValueError):
            current_identity = {"status": "unavailable"}
        observation["runtime_input_identity"] = {
            "status": current_identity.get("status"),
            "source_artifact_sha256": current_identity.get("source_artifact_sha256"),
            "effective_input_sha256": current_identity.get("effective_input_sha256"),
        }
        if (
            not isinstance(expected_identity, Mapping)
            or current_identity.get("status") != "available"
            or current_identity.get("source_artifact_sha256") != expected_identity.get("sha256")
            or current_identity.get("effective_input_sha256")
            != expected_identity.get("effective_input_sha256")
        ):
            observation["reason_code"] = (
                "target_scenario_runtime_input_identity_changed_or_unavailable"
            )
            record = None
            should_read_record = False
        else:
            should_read_record = True
            scenario_identity_snapshot = current_identity
            record = None
        try:
            if should_read_record:
                episode_bytes = episode_path.read_bytes()
                observation["episode_records_jsonl_sha256"] = hashlib.sha256(
                    episode_bytes
                ).hexdigest()
                record = parse_first_jsonl_record(episode_bytes, source=episode_path.as_posix())
        except (OSError, RuntimeError, ValueError):
            record = None
            observation["reason_code"] = "target_episode_record_missing_or_malformed"
        if record is None:
            observation.setdefault("reason_code", "target_episode_record_missing_or_malformed")
        else:
            record_metadata = record.get("algorithm_metadata")
            reason_code, route_complete = _target_episode_observation_reason(
                record,
                config=config,
                candidate=candidate,
                scenario_yaml_path=scenario_yaml_path,
                failure_attribution=failure_attribution,
            )
            observation.update(
                {
                    "episode_id": record.get("episode_id"),
                    "scenario_id": record.get("scenario_id"),
                    "planner_id": record.get("algo"),
                    "seed": record.get("seed"),
                    "source_commit": record.get("git_hash"),
                    "scenario_config_hash": record.get("config_hash"),
                    "planner_config_hash": (
                        record_metadata.get("config_hash")
                        if isinstance(record_metadata, Mapping)
                        else None
                    ),
                    "termination_reason": record.get("termination_reason"),
                    "route_complete": route_complete,
                }
            )
            if reason_code is None and isinstance(route_complete, bool):
                target_outcome = "route_completed" if route_complete else "route_incomplete"
                observation.update(status="available", reason_code=None)
            else:
                observation["reason_code"] = reason_code or "target_episode_outcome_unavailable"
    target_outcome = _guard_target_observation_input_stability(
        target_outcome,
        scenario_identity_snapshot=scenario_identity_snapshot,
        scenario_yaml_path=scenario_yaml_path,
        observation=observation,
    )
    evidence["target_planner_observation"] = observation
    updated["evidence"] = evidence
    updated["target_planner_outcome"] = target_outcome
    reason_codes = list(updated.get("reason_codes", []))
    new_reason = (
        "target_planner_outcome_observed"
        if target_outcome in {"route_completed", "route_incomplete"}
        else str(observation["reason_code"])
    )
    if new_reason not in reason_codes:
        reason_codes.append(new_reason)
    updated["reason_codes"] = reason_codes
    validate_scenario_admissibility(updated)
    return updated


def _target_episode_observation_reason(  # noqa: C901 - fail-closed evidence checks stay explicit.
    record: Mapping[str, Any],
    *,
    config: SearchConfig,
    candidate: CandidateSpec,
    scenario_yaml_path: Path,
    failure_attribution: FailureAttribution | None,
) -> tuple[str | None, bool | None]:
    """Require a native, internally consistent episode matching the search candidate."""
    details = failure_attribution.details if failure_attribution is not None else {}
    outcome = record.get("outcome")
    metadata = record.get("algorithm_metadata")
    episode_id = record.get("episode_id")
    scenario_id = _candidate_scenario_id(scenario_yaml_path)
    route_complete = outcome.get("route_complete") if isinstance(outcome, Mapping) else None
    termination = record.get("termination_reason")
    if (
        record.get("version") != "v1"
        or not isinstance(record.get("metrics"), Mapping)
        or not isinstance(record.get("seed"), int)
        or isinstance(record.get("seed"), bool)
    ):
        return "target_episode_schema_fields_missing_or_malformed", None
    if not isinstance(episode_id, str) or not episode_id.strip():
        return "target_episode_identity_missing", None
    if (
        not isinstance(scenario_id, str)
        or record.get("scenario_id") != scenario_id
        or record.get("algo") != config.policy
        or record.get("seed") != candidate.scenario_seed
    ):
        return "target_episode_identity_mismatch", None
    if (
        not isinstance(record.get("git_hash"), str)
        or len(record["git_hash"]) not in {40, 64}
        or any(character not in "0123456789abcdefABCDEF" for character in record["git_hash"])
    ):
        return "target_episode_source_revision_missing_or_malformed", None
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("status") != "ok"
        or resolve_execution_mode(metadata) not in _NATIVE_EXECUTION_MODES
    ):
        return "target_episode_planner_provenance_not_native_or_unavailable", None
    binding_reason = _target_episode_candidate_binding_reason(
        record,
        metadata=metadata,
        config=config,
        candidate=candidate,
        scenario_yaml_path=scenario_yaml_path,
    )
    if binding_reason is not None:
        return binding_reason, None
    if (
        not isinstance(details, Mapping)
        or details.get("availability_status") != "available"
        or details.get("readiness_status") != "native"
        or details.get("execution_mode") != resolve_execution_mode(metadata)
    ):
        return "target_planner_runtime_availability_not_clean", None
    if runtime_fallback_or_degraded_marker(dict(record)) is not None:
        return "target_episode_fallback_or_degraded", None
    if (
        not isinstance(route_complete, bool)
        or not isinstance(outcome.get("collision_event"), bool)
        or not isinstance(outcome.get("timeout_event"), bool)
    ):
        return "target_episode_route_completion_missing_or_malformed", None
    if termination not in TERMINATION_REASONS:
        return "target_episode_termination_reason_missing_or_malformed", None
    if record.get("status") != status_from_termination_reason(termination):
        return "target_episode_status_termination_conflict", None
    integrity = record.get("integrity")
    contradictions = integrity.get("contradictions") if isinstance(integrity, Mapping) else None
    if not isinstance(contradictions, list) or contradictions:
        return "target_episode_integrity_unavailable_or_contradictory", None
    if outcome_contradictions(termination_reason=termination, outcome=outcome):
        return "target_episode_outcome_termination_conflict", None
    if route_complete and termination != "success":
        return "target_episode_outcome_termination_conflict", None
    if not route_complete and termination in {"success", "error"}:
        return "target_episode_outcome_termination_conflict", None
    return None, route_complete


def _guard_target_observation_input_stability(
    target_outcome: str,
    *,
    scenario_identity_snapshot: Mapping[str, Any] | None,
    scenario_yaml_path: Path,
    observation: dict[str, Any],
) -> str:
    """Clear an observed planner outcome if scenario inputs changed during parsing."""
    if (
        target_outcome not in {"route_completed", "route_incomplete"}
        or not scenario_identity_snapshot
    ):
        return target_outcome
    try:
        identity_after_observation = scenario_input_identity(
            scenario_yaml_path,
            scenario_id=_candidate_scenario_id(scenario_yaml_path),
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        identity_after_observation = {"status": "unavailable"}
    identity_stable = (
        identity_after_observation.get("status") == "available"
        and identity_after_observation.get("source_artifact_sha256")
        == scenario_identity_snapshot.get("source_artifact_sha256")
        and identity_after_observation.get("effective_input_sha256")
        == scenario_identity_snapshot.get("effective_input_sha256")
    )
    if identity_stable:
        return target_outcome
    observation.update(
        status="unavailable",
        reason_code="target_scenario_runtime_input_changed_during_observation",
        route_complete=None,
    )
    return "unavailable"


def _target_episode_candidate_binding_reason(
    record: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    config: SearchConfig,
    candidate: CandidateSpec,
    scenario_yaml_path: Path,
) -> str | None:
    """Bind a native episode row to its materialized candidate and planner config."""
    scenario_error, materialized_candidate = _materialized_candidate_provenance(scenario_yaml_path)
    if scenario_error is not None:
        return scenario_error
    if materialized_candidate is None:
        return "target_episode_candidate_provenance_missing"
    expected_candidate = candidate.to_json()
    if not _candidate_payload_matches(materialized_candidate, expected_candidate):
        return "target_episode_candidate_parameters_mismatch"

    scenario_params = record.get("scenario_params")
    recorded_metadata = (
        scenario_params.get("metadata") if isinstance(scenario_params, Mapping) else None
    )
    recorded_candidate = (
        recorded_metadata.get("adversarial_candidate")
        if isinstance(recorded_metadata, Mapping)
        else None
    )
    if not isinstance(scenario_params, Mapping) or not isinstance(recorded_candidate, Mapping):
        return "target_episode_scenario_parameters_missing"
    if not _candidate_payload_matches(recorded_candidate, expected_candidate):
        return "target_episode_record_candidate_parameters_mismatch"
    scenario_id = _candidate_scenario_id(scenario_yaml_path)
    if scenario_params.get("id") != scenario_id or scenario_params.get("algo") != config.policy:
        return "target_episode_record_scenario_parameters_mismatch"
    if record.get("config_hash") != _config_hash(dict(scenario_params)):
        return "target_episode_scenario_config_hash_mismatch"

    recorded_planner_config = metadata.get("config")
    if not isinstance(recorded_planner_config, Mapping):
        return "target_episode_recorded_planner_config_missing"
    expected_planner_config_hash = _config_hash(dict(recorded_planner_config))
    if (
        scenario_params.get("algo_config_hash") != expected_planner_config_hash
        or metadata.get("config_hash") != expected_planner_config_hash
    ):
        return "target_episode_planner_config_hash_mismatch"
    return None


def _materialized_candidate_provenance(
    scenario_yaml_path: Path,
) -> tuple[str | None, Mapping[str, Any] | None]:
    """Read the candidate provenance block from one materialized scenario file."""
    try:
        scenario_payload = yaml.safe_load(scenario_yaml_path.read_bytes().decode("utf-8"))
        scenarios = (
            scenario_payload.get("scenarios") if isinstance(scenario_payload, Mapping) else None
        )
        if not isinstance(scenarios, list) or len(scenarios) != 1:
            return "target_episode_candidate_scenario_missing_or_ambiguous", None
        scenario = scenarios[0]
        scenario_metadata = scenario.get("metadata") if isinstance(scenario, Mapping) else None
        materialized_candidate = (
            scenario_metadata.get("adversarial_candidate")
            if isinstance(scenario_metadata, Mapping)
            else None
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return "target_episode_candidate_scenario_unavailable", None
    return None, materialized_candidate if isinstance(materialized_candidate, Mapping) else None


def _candidate_payload_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Check every declared sampled-candidate field without rejecting added metadata."""
    return all(actual.get(key) == value for key, value in expected.items())


def _store_post_evaluation_admissibility(
    evaluation: CandidateEvaluation,
    *,
    initial_payload: Mapping[str, Any],
    config: SearchConfig,
    candidate: CandidateSpec,
    scenario_yaml_path: Path,
    candidate_dir: Path,
    evaluation_error: str | None = None,
) -> CandidateEvaluation:
    """Save the target observation in both the manifest row and its candidate bundle."""
    episode_path = evaluation.episode_record_path or (candidate_dir / "episode_records.jsonl")
    admissibility = _post_evaluation_admissibility(
        initial_payload,
        config=config,
        candidate=candidate,
        scenario_yaml_path=scenario_yaml_path,
        episode_record_path=episode_path,
        failure_attribution=evaluation.failure_attribution,
        evaluation_error=evaluation_error,
    )
    attribution = evaluation.failure_attribution
    if attribution is not None:
        attribution = replace(
            attribution,
            details={
                **attribution.details,
                "scenario_admissibility": admissibility,
            },
        )
        write_json(candidate_dir / "failure_attribution.json", attribution.to_json())
    return replace(
        evaluation,
        failure_attribution=attribution,
        scenario_admissibility=admissibility,
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
        admissibility = _admissibility_payload(
            index=index,
            scenario_yaml_path=scenario_yaml_path,
            certification_status=certification_status,
        )
        if admissibility["search_disposition"] == "reject":
            num_invalid += 1
            reason = _admissibility_rejection_reason(admissibility)
            evaluation = _invalid_evaluation(
                candidate=candidate,
                certification_status=certification_status,
                scenario_yaml_path=scenario_yaml_path,
                bundle_path=candidate_dir,
                reason=reason,
                scenario_admissibility=admissibility,
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
            evaluation = _invalid_evaluation(
                candidate=candidate,
                certification_status=certification_status,
                scenario_yaml_path=scenario_yaml_path,
                bundle_path=candidate_dir,
                reason=certification_status.reason,
                scenario_admissibility=admissibility,
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
                scenario_admissibility=admissibility,
            )
            score = objective(evaluation)
            evaluation = evaluation.with_objective(score)
            evaluation = _store_post_evaluation_admissibility(
                evaluation,
                initial_payload=admissibility,
                config=config,
                candidate=candidate,
                scenario_yaml_path=scenario_yaml_path,
                candidate_dir=candidate_dir,
            )
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
                scenario_admissibility=admissibility,
            )
            evaluation = _store_post_evaluation_admissibility(
                evaluation,
                initial_payload=admissibility,
                config=config,
                candidate=candidate,
                scenario_yaml_path=scenario_yaml_path,
                candidate_dir=candidate_dir,
                evaluation_error=error,
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
        admissibility = _admissibility_payload(
            index=index,
            scenario_yaml_path=scenario_yaml_path,
            certification_status=certification_status,
        )
        if admissibility["search_disposition"] == "reject":
            reason = _admissibility_rejection_reason(admissibility)
            evaluation = _invalid_evaluation(
                candidate=candidate,
                certification_status=certification_status,
                scenario_yaml_path=scenario_yaml_path,
                bundle_path=candidate_dir,
                reason=reason,
                scenario_admissibility=admissibility,
            )
            write_json(
                candidate_dir / "failure_attribution.json",
                evaluation.failure_attribution.to_json(),
            )
            return evaluation
        if not candidate_allowed(
            certification_status, require_certification=config.require_certification
        ):
            evaluation = _invalid_evaluation(
                candidate=candidate,
                certification_status=certification_status,
                scenario_yaml_path=scenario_yaml_path,
                bundle_path=candidate_dir,
                reason=certification_status.reason,
                scenario_admissibility=admissibility,
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
                scenario_admissibility=admissibility,
            )
            score = objective(evaluation)
            evaluation = evaluation.with_objective(score)
            return _store_post_evaluation_admissibility(
                evaluation,
                initial_payload=admissibility,
                config=config,
                candidate=candidate,
                scenario_yaml_path=scenario_yaml_path,
                candidate_dir=candidate_dir,
            )
        except Exception as exc:  # noqa: BLE001 - evaluator failure records candidate attribution
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
                scenario_admissibility=admissibility,
            )
            return _store_post_evaluation_admissibility(
                evaluation,
                initial_payload=admissibility,
                config=config,
                candidate=candidate,
                scenario_yaml_path=scenario_yaml_path,
                candidate_dir=candidate_dir,
                evaluation_error=error,
            )

    return _evaluate
