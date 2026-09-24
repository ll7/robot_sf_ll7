"""Append-only, replay-verified adversarial challenge corpus (issue #9652).

The corpus stores immutable discovery evidence and append-only planner observations.
Solved status is derived on request; a later success never deletes a challenge or its
original failure evidence. Unknown feasibility and incomplete execution evidence remain
explicit states.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.adversarial.bundle import compute_effective_scenario_hash
from robot_sf.benchmark.termination_reason import outcome_contradictions
from robot_sf.cli_scenarios import validate_scenario_payload

CORPUS_SCHEMA_VERSION = "adversarial-counterexample-corpus.v1"
CASE_SCHEMA_VERSION = "adversarial-counterexample.v1"
ATTEMPT_SCHEMA_VERSION = "adversarial-counterexample-admission-attempt.v1"
EVALUATION_SCHEMA_VERSION = "adversarial-counterexample-planner-evaluation.v1"
SLICE_SCHEMA_VERSION = "adversarial-counterexample-slice.v1"
ISSUE_9645_SUMMARY_SCHEMA = "issue_9645_bounded_pilot_summary.v1"
ISSUE_9645_REPLAY_SCHEMA = "issue_9645_replay_validation_collection.v1"
SEARCH_MANIFEST_SCHEMA = "adversarial-search-manifest.v1"
_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_SCHEMA_PATH = _ROOT / "robot_sf/benchmark/schemas/adversarial-counterexample-corpus.v1.json"


class CorpusError(ValueError):
    """Raised when corpus or source evidence violates the versioned contract."""


def new_corpus() -> dict[str, Any]:
    """Return an empty corpus with stable, append-only collections."""
    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "claim_boundary": (
            "diagnostic challenge memory; not a safety claim, feasibility oracle, "
            "or proof of search-space coverage"
        ),
        "search_runs": [],
        "cases": [],
        "admission_attempts": [],
        "planner_evaluations": [],
    }


def validate_corpus(corpus: Mapping[str, Any]) -> None:
    """Validate corpus shape and stable case/evaluation references."""
    if not isinstance(corpus, Mapping):
        raise CorpusError("corpus root must be an object")
    try:
        schema = json.loads(_CORPUS_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"corpus schema could not be loaded: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(dict(corpus)),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.absolute_path) or "$"
        raise CorpusError(f"invalid corpus at {location}: {first.message}")
    case_ids = [case["case_id"] for case in corpus["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise CorpusError("case_id values must be unique")
    for case in corpus["cases"]:
        if case["case_id"] != f"case-{case['effective_scenario_sha256']}":
            raise CorpusError("case_id must be the full canonical effective-scenario SHA-256")
    case_id_set = set(case_ids)
    for evaluation in corpus["planner_evaluations"]:
        if evaluation["case_id"] not in case_id_set:
            raise CorpusError(
                f"planner evaluation references absent case {evaluation['case_id']!r}"
            )


def load_corpus(path: str | Path, *, create: bool = False) -> dict[str, Any]:
    """Load and validate a corpus JSON file; optionally create an empty one."""
    corpus_path = Path(path)
    if not corpus_path.exists():
        if not create:
            raise CorpusError(f"corpus file does not exist: {corpus_path}")
        corpus = new_corpus()
        save_corpus(corpus_path, corpus)
        return corpus
    try:
        value = json.loads(corpus_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"could not read corpus {corpus_path}: {exc}") from exc
    validate_corpus(value)
    return value


def save_corpus(path: str | Path, corpus: Mapping[str, Any]) -> None:
    """Atomically write a validated corpus with stable JSON formatting."""
    validate_corpus(corpus)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(corpus, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    text += "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            handle.write(text)
            handle.flush()
        Path(temporary_name).replace(destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def import_issue9645_packet(
    payload_root: str | Path,
    corpus: dict[str, Any],
    *,
    corpus_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Import #9645's zero-discovery run and replay-verified #1501 case.

    ``payload_root`` is the ``payload`` directory from the durable #9645 evidence
    bundle. The pilot's zero discoveries are recorded independently of the historical
    case, which came from #1501 and remains marked as historical lineage.
    """
    validate_corpus(corpus)
    payload = Path(payload_root).resolve()
    if (payload / "payload").is_dir():
        payload = payload / "payload"
    root = Path(corpus_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    try:
        pilot = _verify_issue9645_pilot(payload)
    except (CorpusError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        updated, receipt = _record_attempt(
            corpus,
            source_kind="issue_9645_packet",
            source_id="issue_9645_bounded_pilot",
            decision="rejected",
            blockers=[f"pilot_evidence_invalid:{type(exc).__name__}:{exc}"],
            candidate_identity=None,
            near_duplicate_report=_unassessed_near_duplicates(),
        )
        return updated, receipt

    try:
        _persist_pilot_evidence(payload, root, pilot)
    except (CorpusError, OSError) as exc:
        updated, receipt = _record_attempt(
            corpus,
            source_kind="issue_9645_packet",
            source_id="issue_9645_bounded_pilot",
            decision="rejected",
            blockers=[f"pilot_evidence_persistence_failed:{type(exc).__name__}:{exc}"],
            candidate_identity=None,
            near_duplicate_report=_unassessed_near_duplicates(),
        )
        return updated, receipt
    corpus = _append_unique(
        corpus,
        "search_runs",
        pilot,
        key="run_id",
    )
    try:
        case, source_files, observations = _build_issue9645_historical_case(payload)
    except (
        CorpusError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        yaml.YAMLError,
    ) as exc:
        updated, receipt = _record_attempt(
            corpus,
            source_kind="issue_9645_historical_replay",
            source_id="issue_1501/failure_0002",
            decision="rejected",
            blockers=[f"historical_case_invalid:{type(exc).__name__}:{exc}"],
            candidate_identity=None,
            near_duplicate_report=_unassessed_near_duplicates(),
        )
        return updated, receipt

    existing = next(
        (
            item
            for item in corpus["cases"]
            if item["effective_scenario_sha256"] == case["effective_scenario_sha256"]
        ),
        None,
    )
    duplicate_case_id = existing["case_id"] if existing else None
    near_report = _near_duplicate_report(case, corpus["cases"])
    if existing is None:
        try:
            _materialize_case_artifacts(case, source_files, root)
        except (CorpusError, OSError) as exc:
            updated, receipt = _record_attempt(
                corpus,
                source_kind="issue_9645_historical_replay",
                source_id="issue_1501/failure_0002",
                decision="rejected",
                blockers=[f"case_evidence_persistence_failed:{type(exc).__name__}:{exc}"],
                candidate_identity=case["effective_scenario_sha256"],
                near_duplicate_report=near_report,
            )
            return updated, receipt
        corpus["cases"].append(case)
        corpus["cases"].sort(key=lambda item: item["case_id"])
        stored_case_id = case["case_id"]
    else:
        _merge_source_evidence(existing, case["source_evidence"])
        stored_case_id = existing["case_id"]

    for observation in observations:
        observation["case_id"] = stored_case_id
        observation["effective_scenario_sha256"] = case["effective_scenario_sha256"]
        corpus = append_planner_evaluation(corpus, observation)

    corpus, receipt = _record_attempt(
        corpus,
        source_kind="issue_9645_historical_replay",
        source_id="issue_1501/failure_0002",
        decision="admitted" if existing is None else "duplicate",
        blockers=[],
        candidate_identity=case["effective_scenario_sha256"],
        duplicate_case_id=duplicate_case_id,
        near_duplicate_report=near_report,
    )
    receipt["case_id"] = stored_case_id
    receipt["pilot_new_discoveries"] = 0
    receipt["pilot_run_id"] = pilot["run_id"]
    return corpus, receipt


def append_planner_evaluation(
    corpus: dict[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    """Append a planner result only when its case, planner, and input binding are explicit."""
    validate_corpus(corpus)
    item = dict(observation)
    errors = _validate_evaluation(item)
    if errors:
        raise CorpusError("planner evaluation rejected: " + "; ".join(errors))
    case = next((entry for entry in corpus["cases"] if entry["case_id"] == item["case_id"]), None)
    if case is None:
        raise CorpusError(f"unknown case_id: {item['case_id']}")
    if item["effective_scenario_sha256"] != case["effective_scenario_sha256"]:
        raise CorpusError("planner evaluation input hash does not match the corpus case")
    identity = {
        key: item[key]
        for key in (
            "case_id",
            "effective_scenario_sha256",
            "planner_id",
            "planner_config_identity",
            "source_revision",
            "episode_sha256",
            "outcome",
            "termination_reason",
            "metrics",
            "execution_mode",
            "readiness_status",
            "availability_status",
            "fallback_or_degraded",
            "evidence_status",
            "error",
        )
    }
    item["schema_version"] = EVALUATION_SCHEMA_VERSION
    item["evaluation_id"] = hashlib.sha256(_stable_json(identity).encode("utf-8")).hexdigest()
    corpus["planner_evaluations"] = _append_unique(
        corpus, "planner_evaluations", item, key="evaluation_id"
    )["planner_evaluations"]
    validate_corpus(corpus)
    return corpus


def recompute_planner_status(
    corpus: Mapping[str, Any], *, planner_id: str, planner_config_identity: str
) -> dict[str, Any]:
    """Derive solved/unsolved/mixed/unknown status for every case and planner config."""
    validate_corpus(corpus)
    if not planner_id.strip() or not planner_config_identity.strip():
        raise CorpusError("planner_id and planner_config_identity must be non-empty")
    result = []
    for case in sorted(corpus["cases"], key=lambda entry: entry["case_id"]):
        rows = [
            item
            for item in corpus["planner_evaluations"]
            if item["case_id"] == case["case_id"]
            and item["planner_id"] == planner_id
            and item["planner_config_identity"] == planner_config_identity
        ]
        states = [_evaluation_state(row) for row in rows]
        valid_states = [state for state in states if state["status"] in {"solved", "unsolved"}]
        invalid_rows = [state for state in states if state["status"] == "unknown"]
        distinct = {state["status"] for state in valid_states}
        if invalid_rows:
            status = "unknown"
        elif len(distinct) > 1:
            status = "mixed"
        elif distinct:
            status = next(iter(distinct))
        else:
            status = "unknown"
        result.append(
            {
                "case_id": case["case_id"],
                "effective_scenario_sha256": case["effective_scenario_sha256"],
                "planner_id": planner_id,
                "planner_config_identity": planner_config_identity,
                "status": status,
                "observation_count": len(rows),
                "valid_observation_count": len(valid_states),
                "unknown_observation_count": len(invalid_rows),
                "reason_codes": sorted(
                    {reason for state in states for reason in state["reason_codes"]}
                ),
            }
        )
    return {
        "schema_version": "adversarial-counterexample-status-report.v1",
        "planner_id": planner_id,
        "planner_config_identity": planner_config_identity,
        "case_count": len(result),
        "status_counts": {
            name: sum(row["status"] == name for row in result)
            for name in ("solved", "unsolved", "mixed", "unknown")
        },
        "cases": result,
    }


def export_regression_slice(
    corpus: Mapping[str, Any], *, corpus_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Write a deterministic, self-contained replay slice for all admitted cases."""
    validate_corpus(corpus)
    base = Path(corpus_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise CorpusError(f"slice output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    cases = sorted(corpus["cases"], key=lambda entry: entry["case_id"])
    matrix_entries: list[dict[str, Any]] = []
    manifest_cases: list[dict[str, Any]] = []
    try:
        routes_dir = staging / "routes"
        configs_dir = staging / "planner_configs"
        results_dir = staging / "results"
        routes_dir.mkdir()
        configs_dir.mkdir()
        results_dir.mkdir()
        for case in cases:
            case_input = _case_input_paths(case, base)
            scenario_path = case_input["scenario"]
            route_path = case_input["route"]
            scenario_document = _load_yaml_object(scenario_path, "scenario")
            scenarios = scenario_document.get("scenarios")
            if (
                not isinstance(scenarios, list)
                or len(scenarios) != 1
                or not isinstance(scenarios[0], dict)
            ):
                raise CorpusError(f"{case['case_id']}: scenario input must contain one scenario")
            scenario = dict(scenarios[0])
            route_relative = f"routes/{case['case_id']}.yaml"
            config_relative = f"planner_configs/{case['case_id']}.yaml"
            scenario["route_overrides_file"] = route_relative
            scenario["seeds"] = [int(case["scenario_seed"])]
            matrix_entries.append(scenario)
            route_output = staging / route_relative
            shutil.copyfile(route_path, route_output)
            config_snapshot = case["target_planner"].get("configuration_snapshot")
            if not isinstance(config_snapshot, dict):
                raise CorpusError(f"{case['case_id']}: planner configuration snapshot is missing")
            config_output = staging / config_relative
            config_output.write_text(
                yaml.safe_dump(config_snapshot, sort_keys=True, allow_unicode=True),
                encoding="utf-8",
            )
            command_parts = [
                "uv run robot_sf_bench run",
                "--matrix replay_matrix.yaml",
                f"--out results/{case['case_id']}.jsonl",
                f"--algo {shlex.quote(case['target_planner']['planner_id'])}",
                f"--algo-config {shlex.quote(config_relative)}",
                f"--scenario-id {shlex.quote(case['scenario_id'])}",
                "--no-video",
            ]
            manifest_cases.append(
                {
                    "case_id": case["case_id"],
                    "scenario_id": case["scenario_id"],
                    "seed": case["scenario_seed"],
                    "effective_scenario_sha256": case["effective_scenario_sha256"],
                    "planner": case["target_planner"],
                    "scenario_source_sha256": _sha256_file(scenario_path),
                    "route_overrides_sha256": _sha256_file(route_output),
                    "planner_config_path": config_relative,
                    "planner_config_sha256": _sha256_file(config_output),
                    "replay_command": " ".join(command_parts),
                }
            )

        matrix_path = staging / "replay_matrix.yaml"
        matrix_path.write_text(
            yaml.safe_dump({"scenarios": matrix_entries}, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": SLICE_SCHEMA_VERSION,
            "source_corpus_schema_version": corpus["schema_version"],
            "claim_boundary": (
                "replay inputs only; execution must be evaluated under the declared planner, "
                "configuration, simulator revision, and benchmark profile"
            ),
            "case_count": len(manifest_cases),
            "replay_matrix_path": "replay_matrix.yaml",
            "replay_matrix_sha256": _sha256_file(matrix_path),
            "cases": manifest_cases,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.replace(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_issue9645_pilot(payload: Path) -> dict[str, Any]:
    summary = _read_json_object(payload / "summary.json")
    metadata = _read_json_object(payload / "run_metadata.json")
    row_status = _read_json_object(payload / "row_status.json")
    _validate_pilot_summary(summary, metadata)
    _validate_pilot_design(metadata)
    manifests_by_identity = _verify_pilot_manifests(payload, metadata)
    _verify_pilot_candidate_rows(payload, row_status)
    return _pilot_search_run(summary, metadata, manifests_by_identity, payload)


def _validate_pilot_summary(summary: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    if summary.get("schema_version") != ISSUE_9645_SUMMARY_SCHEMA:
        raise CorpusError("unsupported #9645 summary schema")
    if metadata.get("schema_version") != "issue_9645_execution_provenance.v1":
        raise CorpusError("unsupported #9645 execution provenance schema")
    if summary.get("source_revision") != metadata.get("experiment_source_commit"):
        raise CorpusError("#9645 summary and run metadata revisions differ")
    outcomes = summary.get("candidate_outcomes")
    budget = summary.get("pilot_budget")
    if not isinstance(outcomes, dict) or outcomes.get("new_counterexamples_discovered") != 0:
        raise CorpusError("#9645 must explicitly report zero newly discovered counterexamples")
    if outcomes.get("new_counterexamples_admitted_to_regression_corpus") != 0:
        raise CorpusError("#9645 must explicitly report zero newly admitted pilot cases")
    if (
        not isinstance(budget, dict)
        or budget.get("attempted") != 64
        or budget.get("completed") != 64
    ):
        raise CorpusError("#9645 bounded pilot accounting must be 64 attempted and completed")
    if budget.get("failed") != 0 or budget.get("invalid") != 0:
        raise CorpusError("#9645 pilot failed/invalid accounting differs from its durable receipt")
    if summary.get("evaluation_budget_consumed", {}).get("pilot_search_candidates") != 64:
        raise CorpusError("#9645 pilot candidate budget is not bound")


def _validate_pilot_design(metadata: Mapping[str, Any]) -> None:
    if (
        metadata.get("issue") != 9645
        or metadata.get("objective", {}).get("name") != "constraints_first_lexicographic_v1"
    ):
        raise CorpusError("#9645 issue/objective identity is incomplete")
    if metadata.get("pilot_design", {}).get("fixed_scenario_seed") != 123:
        raise CorpusError("#9645 fixed environment seed is missing")
    expected_seeds = {1101, 2202}
    if set(metadata.get("pilot_design", {}).get("sampler_rng_seeds", [])) != expected_seeds:
        raise CorpusError("#9645 sampler seeds do not match its bounded design")
    if set(metadata.get("pilot_design", {}).get("sampler_methods", [])) != {"random", "optuna_tpe"}:
        raise CorpusError("#9645 sampler methods do not match its bounded design")


def _verify_pilot_manifests(
    payload: Path, metadata: Mapping[str, Any]
) -> dict[tuple[str, int], dict[str, Any]]:
    manifest_receipts = metadata.get("manifest_files")
    if not isinstance(manifest_receipts, list) or len(manifest_receipts) != 4:
        raise CorpusError("#9645 must bind all four per-seed/per-sampler manifests")
    manifests_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    source_manifest_paths = sorted((payload / "source_manifests").glob("*.json"))
    if len(source_manifest_paths) != 4:
        raise CorpusError("#9645 packet must contain its four exact search manifests")
    for manifest_path in source_manifest_paths:
        identity, manifest_record = _verify_one_pilot_manifest(manifest_path, manifest_receipts)
        manifests_by_identity[identity] = manifest_record
    if len(manifests_by_identity) != 4:
        raise CorpusError("#9645 manifest sampler/seed identities are duplicated")
    return manifests_by_identity


def _verify_one_pilot_manifest(
    manifest_path: Path, manifest_receipts: Sequence[Mapping[str, Any]]
) -> tuple[tuple[str, int], dict[str, Any]]:
    actual_hash = _sha256_file(manifest_path)
    matches = [entry for entry in manifest_receipts if entry.get("sha256") == actual_hash]
    if len(matches) != 1:
        raise CorpusError(f"#9645 search manifest hash is not bound: {manifest_path.name}")
    manifest = _read_json_object(manifest_path)
    config = manifest.get("config")
    manifest_summary = manifest.get("summary")
    if manifest.get("schema_version") != SEARCH_MANIFEST_SCHEMA or not isinstance(config, dict):
        raise CorpusError(f"#9645 manifest schema/config is invalid: {manifest_path.name}")
    sampler = (
        "optuna_tpe" if "tpe" in manifest_path.name or "optuna" in manifest_path.name else "random"
    )
    seed = int(config.get("seed", -1))
    if (
        config.get("objective") != "constraints_first_lexicographic_v1"
        or config.get("policy") != "goal"
        or config.get("budget") != 16
        or seed not in {1101, 2202}
    ):
        raise CorpusError(f"#9645 manifest search settings differ: {manifest_path.name}")
    if not isinstance(manifest_summary, dict) or any(
        manifest_summary.get(key) != expected
        for key, expected in (
            ("num_candidates", 16),
            ("num_failed_evaluations", 0),
            ("num_invalid_candidates", 0),
        )
    ):
        raise CorpusError(f"#9645 manifest accounting differs: {manifest_path.name}")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 16:
        raise CorpusError(f"#9645 manifest candidate rows are incomplete: {manifest_path.name}")
    if any(not _pilot_candidate_is_success(candidate) for candidate in candidates):
        raise CorpusError(f"#9645 manifest contains a critical or failed row: {manifest_path.name}")
    return (sampler, seed), {
        "path": f"source_manifests/{manifest_path.name}",
        "sha256": actual_hash,
        "candidate_count": len(candidates),
    }


def _pilot_candidate_is_success(candidate: Any) -> bool:
    return (
        isinstance(candidate, dict)
        and not candidate.get("error")
        and candidate.get("objective_value") == 0.0
        and isinstance(candidate.get("failure_attribution"), dict)
        and candidate["failure_attribution"].get("primary_failure") == "success"
    )


def _verify_pilot_candidate_rows(payload: Path, row_status: Mapping[str, Any]) -> None:
    with (payload / "candidate_evaluations.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 64:
        raise CorpusError("#9645 candidate accounting table must contain all 64 rows")
    for row in rows:
        if (
            row.get("row_status") != "successful_evidence"
            or row.get("counts_as_execution_success_evidence") != "True"
            or row.get("execution_mode") != "native"
            or row.get("readiness_status") != "native"
            or row.get("availability_status") != "available"
            or row.get("collision_event") != "False"
            or row.get("route_complete") != "True"
            or row.get("timeout_event") != "False"
            or row.get("near_miss_count") != "0"
            or row.get("objective_value") != "0.0"
        ):
            raise CorpusError("#9645 candidate table has an incomplete or critical row")
    if row_status.get("counts_as_success_evidence_total") != 64:
        raise CorpusError("#9645 row-status receipt disagrees with the candidate table")
    if row_status.get("evidence_tier") != "diagnostic_only":
        raise CorpusError("#9645 diagnostic evidence tier is missing")


def _pilot_search_run(
    summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
    manifests_by_identity: Mapping[tuple[str, int], Mapping[str, Any]],
    payload: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "adversarial-counterexample-search-run.v1",
        "run_id": "issue_9645_bounded_pilot",
        "source_issue": 9645,
        "source_revision": summary["source_revision"],
        "objective": metadata["objective"],
        "search_space": {
            "path": metadata.get("effective_search_space_path"),
            "sha256": metadata.get("effective_search_space_sha256"),
        },
        "planner": summary.get("planner"),
        "fixed_environment_seed": 123,
        "sampler_seeds": [1101, 2202],
        "sampler_methods": ["random", "optuna_tpe"],
        "budget_per_method_seed": 16,
        "attempted_candidates": 64,
        "completed_candidates": 64,
        "failed_candidates": 0,
        "invalid_candidates": 0,
        "new_counterexamples_discovered": 0,
        "new_counterexamples_admitted": 0,
        "all_objective_values": [0.0],
        "terminal_decision": summary.get("terminal_decision"),
        "evidence_tier": "diagnostic_only",
        "source_files": _copy_pilot_evidence(payload),
        "manifest_files": [
            manifests_by_identity[key]
            for key in sorted(manifests_by_identity, key=lambda item: (item[1], item[0]))
        ],
    }


def _copy_pilot_evidence(payload: Path) -> list[dict[str, str]]:
    """Return exact checksums for the minimal #9645 pilot accounting packet."""
    relative_paths = [
        "summary.json",
        "run_metadata.json",
        "candidate_evaluations.csv",
        "row_status.json",
        "replay_validation.json",
        "path_normalization.json",
        "historical_issue_1501_failure_0002.json",
        "source_manifests/random_seed_1101.json",
        "source_manifests/random_seed_2202.json",
        "source_manifests/tpe_seed_1101.json",
        "source_manifests/tpe_seed_2202.json",
    ]
    # The historical case subdirectory is copied with the admitted case below; run-level
    # accounting stays in the evidence inventory and keeps the explicit zero result.
    return [{"path": name, "sha256": _sha256_file(payload / name)} for name in relative_paths]


def _persist_pilot_evidence(payload: Path, corpus_root: Path, pilot: dict[str, Any]) -> None:
    """Copy the accounting rows/manifests into corpus custody and bind their hashes."""
    evidence_dir = corpus_root / "evidence" / "issue_9645_pilot"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [item["path"] for item in pilot["source_files"]]
    for manifest in pilot["manifest_files"]:
        source_paths.append(manifest["path"])
    persisted = []
    for relative in sorted(set(source_paths)):
        source = payload / relative
        destination = evidence_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = _sha256_file(source)
        if destination.exists():
            if _sha256_file(destination) != expected:
                raise CorpusError(f"corpus pilot evidence path conflicts: {relative}")
        else:
            shutil.copyfile(source, destination)
        persisted.append(
            {
                "path": destination.resolve().relative_to(corpus_root.resolve()).as_posix(),
                "sha256": expected,
            }
        )
    pilot["source_files"] = [
        {
            "path": next(
                item["path"] for item in persisted if item["path"].endswith(f"/{source['path']}")
            ),
            "sha256": source["sha256"],
        }
        for source in pilot["source_files"]
    ]
    pilot["manifest_files"] = [
        {
            **manifest,
            "path": next(
                item["path"] for item in persisted if item["path"].endswith(f"/{manifest['path']}")
            ),
        }
        for manifest in pilot["manifest_files"]
    ]
    pilot["evidence_bundle_root"] = "evidence/issue_9645_pilot"


def _build_issue9645_historical_case(
    payload: Path,
) -> tuple[dict[str, Any], dict[str, Path], list[dict[str, Any]]]:
    context = _load_historical_case_context(payload)
    replay = _verify_historical_replay_pair(payload, context)
    case, observations = _materialize_historical_case(context, replay, payload)
    errors = _validate_case_record(case)
    if errors:
        raise CorpusError("case record rejected: " + "; ".join(errors))
    source_files = {**replay["source_files"], "historical_sources": context["historical_sources"]}
    return case, source_files, observations


def _load_historical_case_context(payload: Path) -> dict[str, Any]:
    replay_validation = _read_json_object(payload / "replay_validation.json")
    historical = _read_json_object(payload / "historical_issue_1501_failure_0002.json")
    archived, run_context, result, candidate = _validated_historical_packet(
        replay_validation, historical
    )
    scenario_context = _validated_historical_scenario(payload, candidate)
    archive_hash, archive_path = _validated_historical_archive(historical, archived)
    return {
        "replay_validation": replay_validation,
        "historical": historical,
        "archived": archived,
        "run_context": run_context,
        "result": result,
        "candidate": candidate,
        "archive_hash": archive_hash,
        "archive_path": archive_path,
        "historical_sources": _historical_source_snapshot(historical),
        **scenario_context,
    }


def _validated_historical_packet(
    replay_validation: Mapping[str, Any], historical: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if replay_validation.get("schema_version") != ISSUE_9645_REPLAY_SCHEMA:
        raise CorpusError("unsupported #9645 replay validation schema")
    if historical.get("schema_version") != "issue_1501_archived_failure_replay.v1":
        raise CorpusError("unsupported #1501 archived-case schema")
    archived = historical.get("archived_entry")
    run_context = historical.get("archived_run_context")
    result = replay_validation.get("historical_issue_1501_case")
    if not all(isinstance(value, dict) for value in (archived, run_context, result)):
        raise CorpusError("#1501 archived case, run context, or replay receipt is missing")
    _validate_historical_replay_claim(result)
    _validate_historical_search_identity(archived, run_context)
    candidate = archived.get("candidate")
    if not isinstance(candidate, dict):
        raise CorpusError("#1501 candidate parameters are missing")
    return archived, run_context, result, candidate


def _validate_historical_replay_claim(result: Mapping[str, Any]) -> None:
    if result.get("historical_case_id") != "issue_1501/failure_0002":
        raise CorpusError("#1501 case identity differs from the requested historical failure")
    if result.get("archived_failure_type") != "collision" or result.get("planner") != "goal":
        raise CorpusError("#1501 archive does not identify a goal-planner collision")
    if result.get("replay_count") != 2 or result.get("replays_identical") is not True:
        raise CorpusError("#1501 case does not contain two matching current replays")
    if result.get("dynamic_task_feasibility", "").split("_")[0] != "unknown":
        raise CorpusError("#1501 dynamic feasibility must remain unknown")
    if (
        result.get("current_scenario_certification", {}).get("classification")
        != "hard_but_solvable"
    ):
        raise CorpusError("#1501 static certificate summary is absent")


def _validate_historical_search_identity(
    archived: Mapping[str, Any], run_context: Mapping[str, Any]
) -> None:
    if run_context.get("commit") != "4bd5fb412d2bf023d26f674833dd379aedf8c17e":
        raise CorpusError("#1501 historical source revision is not pinned")
    if (
        archived.get("source_candidate_index") != 8
        or archived.get("source_manifest") != "crossing_ttc/goal/optuna/manifest.json"
    ):
        raise CorpusError("#1501 archived candidate source identity is incomplete")
    if run_context.get("seed") != 42 or run_context.get("budget_per_sampler") != 32:
        raise CorpusError("#1501 search seed/budget provenance is incomplete")


def _validated_historical_scenario(payload: Path, candidate: Mapping[str, Any]) -> dict[str, Any]:
    historical_dir = payload / "historical_issue_1501_failure_0002"
    scenario_path = historical_dir / "scenario.yaml"
    route_path = historical_dir / "route_overrides.yaml"
    scenario_doc = _load_yaml_object(scenario_path, "scenario")
    scenario_rows = scenario_doc.get("scenarios")
    if (
        not isinstance(scenario_rows, list)
        or len(scenario_rows) != 1
        or not isinstance(scenario_rows[0], dict)
    ):
        raise CorpusError("#1501 scenario input must contain one scenario")
    scenario = scenario_rows[0]
    route_doc = _load_yaml_object(route_path, "route overrides")
    scenario_id = str(scenario.get("name") or scenario.get("id") or "").strip()
    seed = _scenario_seed(scenario)
    if seed != 320 or seed != candidate.get("scenario_seed"):
        raise CorpusError("#1501 scenario seed does not match archived candidate parameters")
    metadata_candidate = (scenario.get("metadata") or {}).get("adversarial_candidate")
    if not _candidate_parameters_match(metadata_candidate, candidate):
        raise CorpusError("#1501 materialized scenario differs from archived candidate parameters")
    if scenario.get("route_overrides_file") != "route_overrides.yaml":
        raise CorpusError("#1501 scenario route binding is missing or unexpected")
    scenario_errors = _validate_scenario_structure(scenario_path)
    if scenario_errors:
        raise CorpusError("#1501 scenario is structurally invalid: " + "; ".join(scenario_errors))
    return {
        "historical_dir": historical_dir,
        "scenario_path": scenario_path,
        "route_path": route_path,
        "scenario": scenario,
        "route_doc": route_doc,
        "scenario_id": scenario_id,
        "seed": seed,
        "identity_hash": compute_effective_scenario_hash(scenario, route_doc),
        "scenario_hash": _sha256_file(scenario_path),
        "route_hash": _sha256_file(route_path),
    }


def _candidate_parameters_match(materialized: Any, archived: Mapping[str, Any]) -> bool:
    return isinstance(materialized, dict) and all(
        materialized.get(key) == value for key, value in archived.items()
    )


def _validated_historical_archive(
    historical: Mapping[str, Any], archived: Mapping[str, Any]
) -> tuple[str, Path]:
    archive_hash = str(historical.get("source_archive_sha256") or "")
    archive_path = _ROOT / str(historical.get("source_archive") or "")
    if not archive_path.is_file() or _sha256_file(archive_path) != archive_hash:
        raise CorpusError("#1501 tracked archive is unavailable or its checksum differs")
    archive_data = _read_json_object(archive_path)
    archive_entry = next(
        (
            item
            for item in archive_data.get("entries", [])
            if isinstance(item, dict) and item.get("archive_id") == "failure_0002"
        ),
        None,
    )
    if archive_entry != archived:
        raise CorpusError("#9645 archived entry differs from the tracked #1501 archive")
    return archive_hash, archive_path


def _verify_historical_replay_pair(payload: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    path_normalization = _read_json_object(payload / "path_normalization.json")
    if path_normalization.get("schema_version") != "evidence_path_normalization.v1":
        raise CorpusError("#9645 path-normalization receipt is missing or unsupported")
    normalization_rows = {
        item.get("path"): item
        for item in path_normalization.get("records", [])
        if isinstance(item, dict)
    }
    result = context["result"]
    expected_hashes = result.get("replay_files_sha256")
    if not isinstance(expected_hashes, list) or len(expected_hashes) != 2:
        raise CorpusError("#1501 replay receipt must bind both replay artifact hashes")
    source_files: dict[str, Any] = {
        "scenario": context["scenario_path"],
        "route": context["route_path"],
        "historical_packet": payload / "historical_issue_1501_failure_0002.json",
        "replay_validation": payload / "replay_validation.json",
        "path_normalization": payload / "path_normalization.json",
        "source_archive": context["archive_path"],
        "source_run_report": _ROOT / "docs/context/issue_1501_adversarial_smoke_run.md",
    }
    replay_rows = [
        _verify_one_historical_replay(
            index, context, normalization_rows, expected_hashes[index - 1]
        )
        for index in (1, 2)
    ]
    projections = [row["projection"] for row in replay_rows]
    if projections[0] != projections[1]:
        raise CorpusError("#1501 current replay records disagree on selected event/metric identity")
    _validate_historical_summary(result, projections[0])
    source_files.update(
        {key: value for row in replay_rows for key, value in row["source_files"].items()}
    )
    target = replay_rows[0]["episode"].get("algorithm_metadata")
    if not isinstance(target, dict) or not target.get("config_hash"):
        raise CorpusError("#1501 planner configuration identity is missing")
    return {
        "source_files": source_files,
        "projection": projections[0],
        "expected_hashes": expected_hashes,
        "normalization_rows": normalization_rows,
        "target_config_identity": target["config_hash"],
        "target_config": target.get("config", {}),
        "identity_hash": context["identity_hash"],
    }


def _verify_one_historical_replay(
    index: int,
    context: Mapping[str, Any],
    normalization_rows: Mapping[str, Any],
    expected_source_hash: str,
) -> dict[str, Any]:
    relative = f"historical_issue_1501_failure_0002/replay_{index}"
    replay_file = context["historical_dir"] / f"replay_{index}.jsonl"
    provenance_file = context["historical_dir"] / f"replay_{index}.provenance.json"
    normalized_replay = normalization_rows.get(f"{relative}.jsonl")
    normalized_provenance = normalization_rows.get(f"{relative}.provenance.json")
    if not isinstance(normalized_replay, dict) or not isinstance(normalized_provenance, dict):
        raise CorpusError(f"#1501 replay {index} normalization is not recorded")
    _validate_replay_normalization(
        index,
        normalized_replay,
        normalized_provenance,
        replay_file,
        provenance_file,
        expected_source_hash,
    )
    episode = _read_single_jsonl_record(replay_file)
    provenance = _read_json_object(provenance_file)
    projection = _historical_replay_projection(episode)
    _validate_replay_provenance(
        provenance,
        episode,
        replay_file,
        context["scenario_path"],
        expected_revision=str(context["result"].get("regeneration_commit")),
        expected_source_artifact_sha256=expected_source_hash,
    )
    if projection["outcome"].get("collision_event") is not True:
        raise CorpusError(f"#1501 replay {index} did not reproduce the archived collision event")
    return {
        "episode": episode,
        "projection": projection,
        "source_files": {
            f"replay_{index}": replay_file,
            f"replay_{index}_provenance": provenance_file,
        },
    }


def _validate_replay_normalization(
    index: int,
    normalized_replay: Mapping[str, Any],
    normalized_provenance: Mapping[str, Any],
    replay_file: Path,
    provenance_file: Path,
    expected_source_hash: str,
) -> None:
    replay_ok = (
        normalized_replay.get("source_sha256_before_path_normalization") == expected_source_hash
        and normalized_replay.get("normalized_sha256") == _sha256_file(replay_file)
        and normalized_replay.get("field_rewrites") == {"scenario_params.route_overrides_file": 1}
    )
    provenance_ok = (
        _is_sha256(normalized_provenance.get("source_sha256_before_path_normalization"))
        and normalized_provenance.get("normalized_sha256") == _sha256_file(provenance_file)
        and normalized_provenance.get("field_rewrites") == {"run.invocation": 1}
    )
    if not replay_ok or not provenance_ok:
        raise CorpusError(f"#1501 replay {index} normalized artifact binding differs")


def _validate_historical_summary(result: Mapping[str, Any], projection: Mapping[str, Any]) -> None:
    if result.get("current_replay_outcome", {}).get("outcome") != projection["outcome"]:
        raise CorpusError("#1501 durable outcome summary differs from replay records")
    reported_metrics = result.get("current_replay_metrics")
    aliases = {
        "success": ("success",),
        "collisions": ("collisions", "total_collision_count"),
        "total_collision_count": ("total_collision_count",),
        "ped_collision_count": ("ped_collision_count",),
        "near_misses": ("near_misses",),
        "min_clearance": ("min_clearance", "min_clearance_m"),
    }
    if not isinstance(reported_metrics, dict) or any(
        not any(reported_metrics.get(alias) == value for alias in aliases[key])
        for key, value in projection["metrics"].items()
    ):
        raise CorpusError("#1501 selected metric summary differs from replay records")


def _materialize_historical_case(
    context: Mapping[str, Any], replay: Mapping[str, Any], payload: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = _historical_search_source(context)
    replay_receipt = _historical_replay_receipt(context, replay)
    case = _historical_case_record(context, replay, source, replay_receipt, payload)
    observation = _historical_failure_observation(replay, replay_receipt)
    observations = [
        {**observation, "episode_sha256": _sha256_file(replay["source_files"]["replay_1"])},
        {**observation, "episode_sha256": _sha256_file(replay["source_files"]["replay_2"])},
    ]
    return case, observations


def _historical_search_source(context: Mapping[str, Any]) -> dict[str, Any]:
    historical = context["historical"]
    archived = context["archived"]
    run_context = context["run_context"]
    return {
        "kind": "historical_adversarial_search_archive",
        "origin_issue": 1501,
        "origin_case_id": "issue_1501/failure_0002",
        "historical_source_revision": run_context["commit"],
        "historical_manifest": {
            "declared_path": archived["source_manifest"],
            "status": "not_archived",
            "sha256": None,
        },
        "search_settings": {
            "sampler": "optuna_tpe",
            "objective": "worst_case_snqi",
            "objective_direction": "maximize",
            "budget_per_sampler": 32,
            "seed": 42,
            "candidate_index": archived["source_candidate_index"],
            "search_space_path": "configs/adversarial/crossing_ttc_space.yaml",
            "search_space_sha256": "e90353f9653173cc351117bfc874c1e7d5933d32f1f892f1b264d8148c767f34",
            "scenario_template_path": "configs/scenarios/templates/crossing_ttc.yaml",
            "scenario_template_sha256": "4718aa6ee78d8013f251e20f1c78173ac4052a79e63ec80e19dd2347866b4d8f",
            "objective_source_path": "robot_sf/adversarial/objectives.py",
            "objective_source_sha256": "9c2752b64d57af36f281f59592716d50ca61d9b99dadb76fcfdf9d06096bf0c8",
            "launcher_path": "SLURM/Auxme/adversarial_smoke_1501.sl",
        },
        "archive": {
            "path": historical["source_archive"],
            "sha256": context["archive_hash"],
            "archive_id": archived["archive_id"],
            "objective_value": archived["objective_value"],
            "objective_recomputed": False,
        },
        "source_run_report": {
            "path": "docs/context/issue_1501_adversarial_smoke_run.md",
            "source_candidate_count": run_context["historical_source_candidate_count"],
            "valid_failures": run_context["historical_valid_failures"],
            "valid_non_failures": run_context["historical_valid_non_failures"],
            "invalid_candidates": run_context["historical_invalid_candidates"],
        },
    }


def _historical_replay_receipt(
    context: Mapping[str, Any], replay: Mapping[str, Any]
) -> dict[str, Any]:
    result = context["result"]
    projection = replay["projection"]
    normalization = replay["normalization_rows"]
    expected_hashes = replay["expected_hashes"]
    source_files = replay["source_files"]
    replay_artifacts = []
    for index in (1, 2):
        normalized_provenance = normalization[
            f"historical_issue_1501_failure_0002/replay_{index}.provenance.json"
        ]
        replay_artifacts.append(
            {
                "path": f"source_evidence/replay_{index}.jsonl",
                "normalized_bundle_sha256": _sha256_file(source_files[f"replay_{index}"]),
                "source_artifact_sha256_before_path_normalization": expected_hashes[index - 1],
                "path_normalization": "scenario_params.route_overrides_file",
                "provenance_path": f"source_evidence/replay_{index}.provenance.json",
                "provenance_sha256_before_path_normalization": normalized_provenance[
                    "source_sha256_before_path_normalization"
                ],
                "provenance_sha256_normalized": _sha256_file(
                    source_files[f"replay_{index}_provenance"]
                ),
                "provenance_path_normalization": "run.invocation",
            }
        )
    return {
        "verification_status": "repeated_current_revision_match",
        "historical_origin_match": "not_verifiable_original_raw_episode_absent",
        "historical_original_raw_episode_available": False,
        "target_revision": str(result["regeneration_commit"]),
        "replay_revision": str(result["regeneration_commit"]),
        "target_and_replay_revision_match": True,
        "replay_count": 2,
        "replay_signature_sha256_reported": result.get("replay_signature_sha256"),
        "selected_projection_sha256": hashlib.sha256(
            _stable_json(projection).encode("utf-8")
        ).hexdigest(),
        "selected_projection": projection,
        "replay_artifacts": replay_artifacts,
        "limitations": list(result.get("limitations", [])),
    }


def _historical_failure_observation(
    replay: Mapping[str, Any], replay_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    projection = replay["projection"]
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "case_id": "pending",
        "effective_scenario_sha256": replay["identity_hash"],
        "planner_id": "goal",
        "planner_config_identity": replay["target_config_identity"],
        "source_revision": replay["projection"]["source_revision"],
        "episode_sha256": "pending",
        "execution_mode": "native",
        "readiness_status": "native",
        "availability_status": "available",
        "fallback_or_degraded": False,
        "evidence_status": "complete",
        "error": None,
        "outcome": projection["outcome"],
        "termination_reason": projection["termination_reason"],
        "metrics": projection["metrics"],
        "evidence_role": "replay_verified_discovery_failure",
        "replay_verification_status": replay_receipt["verification_status"],
    }


def _historical_case_record(
    context: Mapping[str, Any],
    replay: Mapping[str, Any],
    source: Mapping[str, Any],
    replay_receipt: Mapping[str, Any],
    payload: Path,
) -> dict[str, Any]:
    archived = context["archived"]
    scenario = context["scenario"]
    identity_hash = context["identity_hash"]
    scenario_hash = context["scenario_hash"]
    route_hash = context["route_hash"]
    config_identity = replay["target_config_identity"]
    feasibility = {
        "verdict": "admissible_feasibility_unknown",
        "dynamic_task_status": "unknown",
        "static_certificate": {
            "schema_version": "scenario_cert.v1",
            "classification": "hard_but_solvable",
            "benchmark_eligibility": "eligible",
            "reasons": ["low_static_clearance_margin"],
            "claim_boundary": "static route certificate only; dynamic task feasibility remains unknown",
        },
    }
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": f"case-{identity_hash}",
        "effective_scenario_sha256": identity_hash,
        "scenario_id": context["scenario_id"],
        "scenario_seed": context["seed"],
        "scenario_template": {
            "scenario_id": str(scenario.get("metadata", {}).get("archetype", "unknown")),
            "map_id": scenario.get("map_id"),
        },
        "inputs": {
            "scenario_path": "pending",
            "scenario_sha256": scenario_hash,
            "route_overrides_path": "pending",
            "route_overrides_sha256": route_hash,
        },
        "structural_validation": {
            "status": "valid",
            "validator": "robot_sf.cli_scenarios.validate_scenario_payload",
            "schema_path": "robot_sf/benchmark/schemas/scenarios.schema.json",
            "schema_sha256": _sha256_file(
                _ROOT / "robot_sf/benchmark/schemas/scenarios.schema.json"
            ),
            "scenario_sha256": scenario_hash,
            "route_overrides_sha256": route_hash,
            "error_count": 0,
        },
        "discovery": {
            "round_id": "issue_1501_crossing_ttc_smoke",
            "discovery_issue": 1501,
            "imported_by_issue": 9652,
            "origin_case_id": "issue_1501/failure_0002",
            "candidate_parameters": context["candidate"],
            "objective": {
                "name": "worst_case_snqi",
                "direction": "maximize",
                "archived_value": archived["objective_value"],
                "value_recomputed": False,
            },
            "failure_attribution": archived["failure_attribution"],
            "search_source": source,
        },
        "admissibility": feasibility,
        "target_planner": {
            "planner_id": "goal",
            "config_identity": config_identity,
            "configuration_snapshot": replay["target_config"],
            "configuration_snapshot_source": "episode.algorithm_metadata.config",
        },
        "replay_receipt": replay_receipt,
        "source_evidence": {
            "source_archive_path": context["historical"]["source_archive"],
            "source_archive_sha256": context["archive_hash"],
            "historical_packet_sha256": _sha256_file(
                payload / "historical_issue_1501_failure_0002.json"
            ),
            "replay_validation_sha256": _sha256_file(payload / "replay_validation.json"),
            "historical_raw_episode_status": "not_archived",
            "source_run_manifest_status": "not_archived; exact settings are pinned by launch receipt/source revision",
            "corpus_files": [],
        },
        "mechanism_group": archived.get("cluster_key"),
        "near_duplicate_report": _unassessed_near_duplicates(),
    }


def _historical_source_snapshot(historical: Mapping[str, Any]) -> dict[str, bytes]:
    """Pin the #1501 objective, search space, template, and launcher source bytes."""
    commit = str(historical.get("archived_run_context", {}).get("commit"))
    sources = {
        "objective_source.py": (
            "robot_sf/adversarial/objectives.py",
            "9c2752b64d57af36f281f59592716d50ca61d9b99dadb76fcfdf9d06096bf0c8",
        ),
        "search_space.yaml": (
            "configs/adversarial/crossing_ttc_space.yaml",
            "e90353f9653173cc351117bfc874c1e7d5933d32f1f892f1b264d8148c767f34",
        ),
        "scenario_template.yaml": (
            "configs/scenarios/templates/crossing_ttc.yaml",
            "4718aa6ee78d8013f251e20f1c78173ac4052a79e63ec80e19dd2347866b4d8f",
        ),
        "launcher.sl": ("SLURM/Auxme/adversarial_smoke_1501.sl", None),
    }
    result: dict[str, bytes] = {}
    for name, (relative, expected_sha) in sources.items():
        try:
            content = subprocess.run(
                ["git", "show", f"{commit}:{relative}"],
                cwd=_ROOT,
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise CorpusError(f"historical source snapshot unavailable: {relative}: {exc}") from exc
        digest = hashlib.sha256(content).hexdigest()
        if expected_sha is not None and digest != expected_sha:
            raise CorpusError(f"historical source snapshot hash differs: {relative}")
        result[name] = content
    return result


def _validate_replay_provenance(
    provenance: Mapping[str, Any],
    episode: Mapping[str, Any],
    replay_file: Path,
    scenario_file: Path,
    *,
    expected_revision: str,
    expected_source_artifact_sha256: str,
) -> None:
    if provenance.get("schema_version") != "benchmark_result_provenance.v1":
        raise CorpusError("historical replay provenance schema is invalid")
    run = provenance.get("run")
    inputs = provenance.get("inputs")
    identity = provenance.get("campaign_identity")
    rows = provenance.get("rows")
    raw = provenance.get("raw_artifacts")
    if not all(isinstance(value, dict) for value in (run, inputs, identity)):
        raise CorpusError("historical replay provenance is incomplete")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise CorpusError("historical replay must have exactly one provenance row")
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        raise CorpusError("historical replay raw artifact binding is incomplete")
    row = rows[0]
    _validate_provenance_execution_identity(run, row, identity, episode, expected_revision)
    _validate_provenance_input_identity(
        inputs, raw[0], episode, scenario_file, expected_source_artifact_sha256
    )
    _validate_native_replay_execution(episode)


def _validate_provenance_execution_identity(
    run: Mapping[str, Any],
    row: Mapping[str, Any],
    identity: Mapping[str, Any],
    episode: Mapping[str, Any],
    expected_revision: str,
) -> None:
    current_revision = run.get("repo_commit")
    if current_revision != expected_revision or row.get("repo_commit") != expected_revision:
        raise CorpusError("historical target execution revision does not match replay revision")
    if (
        episode.get("git_hash") != expected_revision
        or episode.get("provenance", {}).get("git_hash") != expected_revision
    ):
        raise CorpusError("historical replay episode revision differs from its receipt")
    if identity.get("algorithm") != "goal" or episode.get("algo") != "goal":
        raise CorpusError("historical replay planner identity differs")
    if row.get("episode_id") != episode.get("episode_id") or row.get("scenario_id") != episode.get(
        "scenario_id"
    ):
        raise CorpusError("historical replay episode identity differs from provenance")
    if row.get("seed") != episode.get("seed") or episode.get("seed") != 320:
        raise CorpusError("historical replay seed differs from provenance")


def _validate_provenance_input_identity(
    inputs: Mapping[str, Any],
    raw_artifact: Mapping[str, Any],
    episode: Mapping[str, Any],
    scenario_file: Path,
    expected_source_artifact_sha256: str,
) -> None:
    if raw_artifact.get("sha256") != expected_source_artifact_sha256:
        raise CorpusError("historical replay source checksum differs from normalization receipt")
    scenario_input = inputs.get("scenario_matrix")
    if not isinstance(scenario_input, dict) or scenario_input.get("sha256") != _sha256_file(
        scenario_file
    ):
        raise CorpusError("historical replay scenario input checksum differs from provenance")


def _validate_native_replay_execution(episode: Mapping[str, Any]) -> None:
    planner = episode.get("algorithm_metadata")
    planner_identity = planner.get("planner_kinematics") if isinstance(planner, dict) else None
    if (
        not isinstance(planner_identity, dict)
        or planner_identity.get("execution_mode") != "native"
        or planner.get("status") != "ok"
        or episode.get("integrity", {}).get("effective_view", {}).get("degraded") is not False
    ):
        raise CorpusError("historical replay is not a native, non-degraded planner execution")


def _historical_replay_projection(episode: Mapping[str, Any]) -> dict[str, Any]:
    outcome = episode.get("outcome")
    metrics = episode.get("metrics")
    if not isinstance(outcome, dict) or not isinstance(metrics, dict):
        raise CorpusError("historical replay outcome or metrics are missing")
    fields = ("collision_event", "route_complete", "timeout_event")
    if any(not isinstance(outcome.get(key), bool) for key in fields):
        raise CorpusError("historical replay outcome flags must be explicit booleans")
    selected_metrics = {
        key: _require_finite_metric(metrics, key)
        for key in (
            "success",
            "collisions",
            "total_collision_count",
            "ped_collision_count",
            "near_misses",
            "min_clearance",
        )
    }
    termination = str(episode.get("termination_reason") or "")
    contradictions = outcome_contradictions(
        termination_reason=termination,
        outcome=outcome,
        metrics=metrics,
    )
    if contradictions:
        raise CorpusError(
            "historical replay record is internally contradictory: " + "; ".join(contradictions)
        )
    if termination != "collision" or episode.get("status") != "collision":
        raise CorpusError("historical replay did not terminate with the archived collision")
    return {
        "scenario_id": episode.get("scenario_id"),
        "seed": episode.get("seed"),
        "planner_id": episode.get("algo"),
        "planner_config_identity": episode.get("algorithm_metadata", {}).get("config_hash"),
        "source_revision": episode.get("git_hash"),
        "termination_reason": termination,
        "outcome": {key: outcome[key] for key in fields},
        "metrics": selected_metrics,
    }


def _validate_case_record(case: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if case.get("schema_version") != CASE_SCHEMA_VERSION:
        errors.append("unsupported case schema")
    for field in ("case_id", "effective_scenario_sha256", "scenario_id"):
        if not isinstance(case.get(field), str) or not case[field]:
            errors.append(f"{field} is missing")
    errors.extend(_validate_case_execution_contract(case))
    errors.extend(_validate_case_discovery(case))
    errors.extend(_validate_case_inputs(case))
    return errors


def _validate_case_execution_contract(case: Mapping[str, Any]) -> list[str]:
    errors = []
    if case.get("structural_validation", {}).get("status") != "valid":
        errors.append("structural validation is not valid")
    if not isinstance(case.get("scenario_seed"), int) or isinstance(
        case.get("scenario_seed"), bool
    ):
        errors.append("scenario seed must be an integer")
    admissibility = case.get("admissibility")
    if not isinstance(admissibility, dict) or admissibility.get("verdict") not in {
        "admissible_feasibility_unknown",
        "empirically_feasible",
        "planner_specific_failure",
    }:
        errors.append("admissibility verdict must be explicit and non-excluding")
    if (
        not isinstance(case.get("replay_receipt"), dict)
        or case["replay_receipt"].get("verification_status") != "repeated_current_revision_match"
    ):
        errors.append("current-revision replay verification is missing")
    if case.get("replay_receipt", {}).get("target_and_replay_revision_match") is not True:
        errors.append("target and replay revisions do not match")
    target = case.get("target_planner")
    if (
        not isinstance(target, dict)
        or not isinstance(target.get("planner_id"), str)
        or not target.get("planner_id")
        or not isinstance(target.get("config_identity"), str)
        or not target.get("config_identity")
    ):
        errors.append("target planner/configuration identity is incomplete")
    return errors


def _validate_case_discovery(case: Mapping[str, Any]) -> list[str]:
    errors = []
    discovery = case.get("discovery")
    if (
        not isinstance(discovery, dict)
        or not isinstance(discovery.get("candidate_parameters"), dict)
        or not isinstance(discovery.get("objective"), dict)
        or not isinstance(discovery.get("search_source"), dict)
    ):
        errors.append("discovery provenance, candidate parameters, or objective is missing")
    return errors


def _validate_case_inputs(case: Mapping[str, Any]) -> list[str]:
    errors = []
    inputs = case.get("inputs")
    if (
        not isinstance(inputs, dict)
        or not _is_sha256(inputs.get("scenario_sha256"))
        or not _is_sha256(inputs.get("route_overrides_sha256"))
    ):
        errors.append("materialized scenario/route digest is missing")
    return errors


def _validate_evaluation(item: Mapping[str, Any]) -> list[str]:
    errors = _validate_evaluation_identity(item)
    errors.extend(_validate_evaluation_execution_metadata(item))
    evidence_status = item.get("evidence_status")
    if evidence_status not in {"complete", "failed", "partial", "missing", "unknown"}:
        errors.append("evidence_status_invalid")
    if evidence_status == "complete":
        errors.extend(_validate_complete_evaluation(item))
    else:
        errors.extend(_validate_incomplete_evaluation(item))
    return errors


def _validate_evaluation_identity(item: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required_text = (
        "case_id",
        "effective_scenario_sha256",
        "planner_id",
        "planner_config_identity",
        "source_revision",
    )
    for field in required_text:
        value = item.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field}_missing")
    if not _is_sha256(item.get("effective_scenario_sha256")):
        errors.append("effective_scenario_sha256_invalid")
    return errors


def _validate_evaluation_execution_metadata(item: Mapping[str, Any]) -> list[str]:
    errors = []
    evidence_status = item.get("evidence_status")
    if not isinstance(item.get("metrics"), dict):
        errors.append("metrics_missing")
    for field in ("execution_mode", "readiness_status", "availability_status"):
        if not isinstance(item.get(field), str) or not item[field]:
            errors.append(f"{field}_missing")
    if not isinstance(item.get("fallback_or_degraded"), bool):
        errors.extend(
            []
            if evidence_status != "complete" and item.get("fallback_or_degraded") is None
            else ["fallback_or_degraded_missing"]
        )
    if "error" not in item:
        errors.append("error_status_missing")
    elif item["error"] is not None and not isinstance(item["error"], str):
        errors.append("error_status_invalid")
    return errors


def _validate_complete_evaluation(item: Mapping[str, Any]) -> list[str]:
    errors = []
    if not _is_sha256(item.get("episode_sha256")):
        errors.append("episode_sha256_invalid")
    if (
        not isinstance(item.get("termination_reason"), str)
        or not item["termination_reason"].strip()
    ):
        errors.append("termination_reason_missing")
    outcome = item.get("outcome")
    if not isinstance(outcome, dict) or any(
        not isinstance(outcome.get(field), bool)
        for field in ("collision_event", "route_complete", "timeout_event")
    ):
        errors.append("canonical_outcome_flags_missing")
    return errors


def _validate_incomplete_evaluation(item: Mapping[str, Any]) -> list[str]:
    errors = []
    episode_hash = item.get("episode_sha256")
    if episode_hash is not None and not _is_sha256(episode_hash):
        errors.append("episode_sha256_invalid")
    termination = item.get("termination_reason")
    if termination is not None and (not isinstance(termination, str) or not termination.strip()):
        errors.append("termination_reason_invalid")
    outcome = item.get("outcome")
    if outcome is not None and (
        not isinstance(outcome, dict)
        or any(
            outcome.get(field) is not None and not isinstance(outcome.get(field), bool)
            for field in ("collision_event", "route_complete", "timeout_event")
        )
    ):
        errors.append("canonical_outcome_flags_invalid")
    if not isinstance(item.get("error"), str) or not item["error"].strip():
        errors.append("incomplete_evaluation_requires_reason")
    return errors


def _evaluation_state(item: Mapping[str, Any]) -> dict[str, Any]:
    reasons = _validate_evaluation(item)
    if reasons:
        return {"status": "unknown", "reason_codes": reasons}
    if item["evidence_status"] != "complete":
        return {
            "status": "unknown",
            "reason_codes": [f"evaluation_evidence_{item['evidence_status']}"],
        }
    reasons = _complete_evaluation_eligibility_errors(item)
    if reasons:
        return {"status": "unknown", "reason_codes": sorted(set(reasons))}
    return _complete_outcome_state(item)


def _complete_evaluation_eligibility_errors(item: Mapping[str, Any]) -> list[str]:
    reasons = []
    if item["execution_mode"] != "native":
        reasons.append("execution_mode_not_native")
    if item["readiness_status"] != "native":
        reasons.append("readiness_not_native")
    if item["availability_status"] != "available":
        reasons.append("availability_not_available")
    if item["fallback_or_degraded"] is not False:
        reasons.append("fallback_or_degraded_execution")
    if item["error"] is not None:
        reasons.append("evaluation_error")
    metrics = item["metrics"]
    for key in ("success", "collisions", "total_collision_count"):
        if key == "success" and isinstance(metrics.get(key), bool):
            continue
        if _finite_number(metrics.get(key)) is None:
            reasons.append(f"metric_{key}_missing_or_invalid")
    reasons.extend(
        outcome_contradictions(
            termination_reason=item["termination_reason"],
            outcome=item["outcome"],
            metrics=metrics,
        )
    )
    return reasons


def _complete_outcome_state(item: Mapping[str, Any]) -> dict[str, Any]:
    outcome = item["outcome"]
    metrics = item["metrics"]
    if (
        outcome["route_complete"]
        and not outcome["collision_event"]
        and not outcome["timeout_event"]
    ):
        if item["termination_reason"] != "success" or float(metrics["success"]) <= 0.0:
            return {"status": "unknown", "reason_codes": ["success_termination_or_metric_mismatch"]}
        return {"status": "solved", "reason_codes": []}
    return {"status": "unsolved", "reason_codes": []}


def _scenario_input_paths(case: Mapping[str, Any], corpus_root: Path) -> dict[str, Path]:
    inputs = case.get("inputs")
    if not isinstance(inputs, dict):
        raise CorpusError("case has no materialized input paths")
    paths = {}
    for key in ("scenario_path", "route_overrides_path"):
        relative = inputs.get(key)
        if not isinstance(relative, str) or not relative:
            raise CorpusError(f"case input reference missing: {key}")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise CorpusError(f"unsafe corpus input path: {relative}")
        path = (corpus_root / Path(*pure.parts)).resolve()
        try:
            path.relative_to(corpus_root.resolve())
        except ValueError as exc:
            raise CorpusError(f"corpus input path escapes root: {relative}") from exc
        if not path.is_file() or _sha256_file(path) != inputs.get(
            "scenario_sha256" if key == "scenario_path" else "route_overrides_sha256"
        ):
            raise CorpusError(f"corpus input missing or checksum mismatch: {relative}")
        paths["scenario" if key == "scenario_path" else "route"] = path
    scenario_document = _load_yaml_object(paths["scenario"], "scenario")
    scenario_rows = scenario_document.get("scenarios")
    if (
        not isinstance(scenario_rows, list)
        or len(scenario_rows) != 1
        or not isinstance(scenario_rows[0], dict)
    ):
        raise CorpusError("corpus case scenario input must contain exactly one scenario")
    scenario = scenario_rows[0]
    if _scenario_seed(scenario) != case.get("scenario_seed"):
        raise CorpusError("corpus case seed differs from its materialized scenario")
    route_payload = _load_yaml_object(paths["route"], "route overrides")
    if compute_effective_scenario_hash(scenario, route_payload) != case.get(
        "effective_scenario_sha256"
    ):
        raise CorpusError("corpus effective-scenario identity differs from its stored inputs")
    return paths


def _case_input_paths(case: Mapping[str, Any], corpus_root: Path) -> dict[str, Path]:
    return _scenario_input_paths(case, corpus_root)


def _validate_scenario_structure(scenario_path: Path) -> list[str]:
    """Run the repository's canonical scenario loader/schema/asset validation."""
    report = validate_scenario_payload(str(scenario_path))
    if report.get("valid") is True and report.get("num_scenarios") == 1:
        return []
    errors = report.get("errors")
    return [
        str(error.get("code") or error.get("message") or "scenario_invalid")
        for error in errors
        if isinstance(error, dict)
    ] or [str(report.get("status") or "scenario_invalid")]


def _record_attempt(
    corpus: dict[str, Any],
    *,
    source_kind: str,
    source_id: str,
    decision: str,
    blockers: list[str],
    candidate_identity: str | None,
    duplicate_case_id: str | None = None,
    near_duplicate_report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    attempt = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "source_kind": source_kind,
        "source_id": source_id,
        "decision": decision,
        "blockers": sorted(set(blockers)),
        "candidate_identity": candidate_identity,
        "duplicate_case_id": duplicate_case_id,
        "near_duplicate_report": dict(near_duplicate_report),
    }
    attempt["attempt_id"] = hashlib.sha256(_stable_json(attempt).encode("utf-8")).hexdigest()
    corpus["admission_attempts"] = _append_unique(
        corpus, "admission_attempts", attempt, key="attempt_id"
    )["admission_attempts"]
    return corpus, {**attempt, "case_id": None}


def _near_duplicate_report(
    case: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    mechanism = case.get("mechanism_group")
    matches = [
        item["case_id"]
        for item in cases
        if isinstance(mechanism, dict) and item.get("mechanism_group") == mechanism
    ]
    if matches:
        return {
            "status": "mechanism_group_matches_reported",
            "method": "same archived mechanism cluster key; report-only, never merged",
            "case_ids": sorted(matches),
        }
    return _unassessed_near_duplicates()


def _unassessed_near_duplicates() -> dict[str, Any]:
    return {
        "status": "not_assessed",
        "method": None,
        "reason": "no calibrated cross-case near-duplicate metric is available; no case is merged",
        "case_ids": [],
    }


def _merge_source_evidence(existing: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    evidence = existing.setdefault("supporting_source_evidence", [])
    key = _stable_json(incoming)
    if all(_stable_json(item) != key for item in evidence):
        evidence.append(dict(incoming))
        evidence.sort(key=_stable_json)


def _append_unique(
    corpus: dict[str, Any], collection: str, item: Mapping[str, Any], *, key: str
) -> dict[str, Any]:
    values = corpus.setdefault(collection, [])
    if not any(existing.get(key) == item.get(key) for existing in values):
        values.append(dict(item))
        values.sort(key=lambda value: str(value.get(key, "")))
    return corpus


def _materialize_case_artifacts(
    case: dict[str, Any], source_files: Mapping[str, Any], corpus_root: Path
) -> None:
    """Atomically copy a verified case's replay inputs and source evidence into corpus custody."""
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id.startswith("case-"):
        raise CorpusError("case has no safe stable ID for artifact materialization")
    cases_dir = corpus_root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    final_dir = cases_dir / case_id
    if final_dir.exists():
        raise CorpusError(f"case artifact path already exists without a corpus record: {case_id}")
    staging = Path(tempfile.mkdtemp(prefix=f".{case_id}.", dir=cases_dir))
    promoted = False
    try:
        _copy_case_inputs(source_files, staging, corpus_root)
        _copy_case_source_evidence(source_files, staging, corpus_root)
        _copy_historical_source_snapshots(source_files, staging, corpus_root)
        staging.replace(final_dir)
        promoted = True
        case["inputs"]["scenario_path"] = f"cases/{case_id}/inputs/scenario.yaml"
        case["inputs"]["route_overrides_path"] = f"cases/{case_id}/inputs/route_overrides.yaml"
        case["source_evidence"]["corpus_files"] = _case_file_inventory(final_dir, corpus_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if promoted:
            shutil.rmtree(final_dir, ignore_errors=True)
        raise


def _copy_case_inputs(source_files: Mapping[str, Any], staging: Path, corpus_root: Path) -> None:
    for key, relative in {
        "scenario": "inputs/scenario.yaml",
        "route": "inputs/route_overrides.yaml",
    }.items():
        source = source_files.get(key)
        if not isinstance(source, Path):
            raise CorpusError(f"case source artifact is missing: {key}")
        _copy_artifact(source, staging / relative, corpus_root)


def _copy_case_source_evidence(
    source_files: Mapping[str, Any], staging: Path, corpus_root: Path
) -> None:
    destinations = {
        "historical_packet": "historical_issue_1501_failure_0002.json",
        "replay_validation": "replay_validation.json",
        "path_normalization": "path_normalization.json",
        "source_archive": "issue_1501_archive.json",
        "source_run_report": "issue_1501_adversarial_smoke_run.md",
        "replay_1": "replay_1.jsonl",
        "replay_1_provenance": "replay_1.provenance.json",
        "replay_2": "replay_2.jsonl",
        "replay_2_provenance": "replay_2.provenance.json",
    }
    for key, filename in destinations.items():
        source = source_files.get(key)
        if not isinstance(source, Path):
            raise CorpusError(f"case source evidence is missing: {key}")
        _copy_artifact(source, staging / "source_evidence" / filename, corpus_root)


def _copy_historical_source_snapshots(
    source_files: Mapping[str, Any], staging: Path, corpus_root: Path
) -> None:
    snapshots = source_files.get("historical_sources")
    if not isinstance(snapshots, Mapping) or not snapshots:
        raise CorpusError("historical source snapshots are missing")
    for name, content in sorted(snapshots.items()):
        safe_name = (
            isinstance(name, str)
            and PurePosixPath(name).name == name
            and name not in {"", ".", ".."}
        )
        if not safe_name or not isinstance(content, bytes):
            raise CorpusError("historical source snapshot has an unsafe name or value")
        _copy_artifact(
            content,
            staging / "source_evidence" / "historical_sources" / name,
            corpus_root,
        )


def _copy_artifact(source: Path | bytes, destination: Path, corpus_root: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(source, bytes):
        destination.write_bytes(source)
    else:
        shutil.copyfile(source, destination)
    return destination.resolve().relative_to(corpus_root.resolve()).as_posix()


def _case_file_inventory(case_dir: Path, corpus_root: Path) -> list[dict[str, str]]:
    return [
        {
            "path": item.resolve().relative_to(corpus_root.resolve()).as_posix(),
            "sha256": _sha256_file(item),
        }
        for item in sorted(path for path in case_dir.rglob("*") if path.is_file())
    ]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusError(f"could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorpusError(f"JSON root is not an object: {path}")
    return value


def _read_single_jsonl_record(path: Path) -> dict[str, Any]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError) as exc:
        raise CorpusError(f"could not read replay JSONL {path}: {exc}") from exc
    if len(lines) != 1:
        raise CorpusError(f"replay JSONL must contain exactly one record: {path}")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise CorpusError(f"replay JSONL contains malformed JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorpusError(f"replay JSONL record must be an object: {path}")
    return value


def _load_yaml_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CorpusError(f"could not load {label} YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorpusError(f"{label} YAML root must be an object: {path}")
    return value


def _scenario_seed(scenario: Mapping[str, Any]) -> int:
    seeds = scenario.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 1:
        raise CorpusError("scenario must bind exactly one seed")
    seed = seeds[0]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CorpusError("scenario seed must be an integer")
    return seed


def _require_finite_metric(metrics: Mapping[str, Any], key: str) -> float | bool:
    if key == "success" and isinstance(metrics.get(key), bool):
        return metrics[key]
    value = _finite_number(metrics.get(key))
    if value is None:
        raise CorpusError(f"selected replay metric is missing or non-finite: {key}")
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _stable_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )
