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
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.adversarial.bundle import compute_effective_scenario_hash
from robot_sf.benchmark.algorithm_metadata import canonical_algorithm_name
from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker
from robot_sf.benchmark.termination_reason import (
    TERMINATION_REASONS,
    outcome_contradictions,
    status_from_termination_reason,
)
from robot_sf.cli_scenarios import validate_scenario_payload

CORPUS_SCHEMA_VERSION = "adversarial-counterexample-corpus.v1"
CASE_SCHEMA_VERSION = "adversarial-counterexample.v1"
ATTEMPT_SCHEMA_VERSION = "adversarial-counterexample-admission-attempt.v1"
EVALUATION_SCHEMA_VERSION = "adversarial-counterexample-planner-evaluation.v1"
EVALUATION_REPLAY_RECEIPT_SCHEMA_VERSION = "adversarial-planner-replay-receipt.v1"
SLICE_SCHEMA_VERSION = "adversarial-counterexample-slice.v1"
ISSUE_9645_SUMMARY_SCHEMA = "issue_9645_bounded_pilot_summary.v1"
ISSUE_9645_REPLAY_SCHEMA = "issue_9645_replay_validation_collection.v1"
ISSUE_9645_BUNDLE_SCHEMA = "evidence_bundle.v1"
ISSUE_9656_SUMMARY_SCHEMA = "benchmark-hard-case-slice.v1"
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
        "historical_candidates": [],
        "historical_candidate_imports": [],
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
    _validate_historical_candidate_registry(corpus)


def _validate_historical_candidate_registry(corpus: Mapping[str, Any]) -> None:
    candidates = corpus.get("historical_candidates", [])
    imports = corpus.get("historical_candidate_imports", [])
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise CorpusError("historical candidate_id values must be unique")
    import_ids = [receipt["import_id"] for receipt in imports]
    if len(import_ids) != len(set(import_ids)):
        raise CorpusError("historical candidate import_id values must be unique")
    imports_by_id = {receipt["import_id"]: set(receipt["candidate_ids"]) for receipt in imports}
    known_candidate_ids = set(candidate_ids)
    source_identity_by_import: dict[str, Mapping[str, Any]] = {}
    for receipt in imports:
        source_identity = receipt["source_identity"]
        expected_import_id = hashlib.sha256(
            _stable_json(source_identity).encode("utf-8")
        ).hexdigest()
        if receipt["import_id"] != expected_import_id:
            raise CorpusError("historical candidate import ID does not bind its source identity")
        source_identity_by_import[receipt["import_id"]] = source_identity
        if receipt["candidate_count"] != len(receipt["candidate_ids"]) or not imports_by_id[
            receipt["import_id"]
        ].issubset(known_candidate_ids):
            raise CorpusError("historical candidate import receipt has absent candidate rows")
    for candidate in candidates:
        import_id = candidate["source_provenance"]["import_id"]
        if (
            import_id not in imports_by_id
            or candidate["candidate_id"] not in imports_by_id[import_id]
        ):
            raise CorpusError("historical candidate references an absent import receipt")
        identity = {
            **source_identity_by_import[import_id],
            "source_case_id": candidate["source_case_id"],
            "source_record_sha256": candidate["source_record_sha256"],
        }
        expected_candidate_id = hashlib.sha256(_stable_json(identity).encode("utf-8")).hexdigest()
        if candidate["candidate_id"] != expected_candidate_id:
            raise CorpusError(
                "historical candidate ID does not bind its source alias and row digest"
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

    stored_case = existing if existing is not None else case
    for index, observation in enumerate(observations, start=1):
        observation["case_id"] = stored_case_id
        observation["effective_scenario_sha256"] = case["effective_scenario_sha256"]
        observation["replay_receipt"] = create_planner_replay_receipt(
            observation,
            stored_case,
            artifact_path=(f"cases/{stored_case_id}/source_evidence/replay_{index}.jsonl"),
            corpus_root=root,
        )
        corpus = append_planner_evaluation(corpus, observation, corpus_root=root)

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


def import_issue9656_candidates(
    summary_path: str | Path,
    materialized_root: str | Path,
    evidence_bundle_root: str | Path,
    campaign_root: str | Path,
    corpus: dict[str, Any],
    *,
    corpus_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append #9656 historical rows as pending candidates, never admitted cases.

    The evidence bundle is verified before any corpus or artifact mutation. Each source
    alias is retained verbatim while the corpus candidate receives its own content-derived
    ID. Imported replay matrices are path-normalized into candidate custody with the
    original input digests retained separately.
    """
    validate_corpus(corpus)
    summary_file = Path(summary_path).resolve()
    materialized = Path(materialized_root).resolve()
    bundle_root = Path(evidence_bundle_root).resolve()
    campaign = Path(campaign_root).resolve()
    root = Path(corpus_root).resolve()
    summary, summary_receipts = _verify_issue9656_source_summary(summary_file, bundle_root)
    materialized_manifest_path = materialized / "manifest.json"
    materialized_manifest = _read_json_object(materialized_manifest_path)
    source_rows = _validate_issue9656_materialization(
        summary, materialized, materialized_manifest, campaign
    )
    materialized_manifest_sha = _sha256_file(materialized_manifest_path)

    import_identity = {
        "source_issue": 9656,
        "source_row_binding": "verified_episode_file_and_line_sha256",
        "summary_sha256": summary_receipts["summary_sha256"],
        "evidence_bundle_manifest_sha256": summary_receipts["manifest_sha256"],
        "evidence_checksums_sha256": summary_receipts["checksums_sha256"],
        "materialized_manifest_sha256": materialized_manifest_sha,
    }
    import_id = hashlib.sha256(_stable_json(import_identity).encode("utf-8")).hexdigest()
    existing_import = next(
        (
            item
            for item in corpus.get("historical_candidate_imports", [])
            if item.get("import_id") == import_id
        ),
        None,
    )
    if existing_import is not None:
        if existing_import.get("source_identity") != import_identity:
            raise CorpusError("historical candidate import ID conflicts with stored provenance")
        expected_candidate_ids = sorted(
            hashlib.sha256(
                _stable_json(
                    {
                        **import_identity,
                        "source_case_id": source_case["case_id"],
                        "source_record_sha256": source_case["source_record"]["record_sha256"],
                    }
                ).encode("utf-8")
            ).hexdigest()
            for source_case, _materialized_case, *_artifacts in source_rows
        )
        if existing_import.get("candidate_ids") != expected_candidate_ids:
            raise CorpusError("duplicate #9656 import does not contain every source alias")
        expected_candidate_set = set(expected_candidate_ids)
        existing_candidates = [
            item
            for item in corpus.get("historical_candidates", [])
            if item["candidate_id"] in expected_candidate_set
        ]
        _verify_issue9656_duplicate_materialized_cases(source_rows, existing_candidates)
        _verify_existing_issue9656_candidate_artifacts(
            root,
            existing_import,
            existing_candidates,
        )
        return corpus, {**existing_import, "decision": "duplicate"}

    candidates: list[dict[str, Any]] = []
    staging_root = root / ".historical_candidate_import_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{import_id}.", dir=staging_root))
    promoted_dirs: list[Path] = []
    try:
        import_artifacts = staging / "historical_candidate_imports" / import_id
        import_artifacts.mkdir(parents=True)
        for payload_receipt in summary_receipts["payload_files"]:
            relative = payload_receipt["path"]
            source_payload_file = _safe_materialized_path(
                bundle_root / "payload", relative, label="evidence payload file"
            )
            _copy_verified_source_file(
                source_payload_file,
                import_artifacts / "payload" / Path(*PurePosixPath(relative).parts),
                expected_sha256=payload_receipt["sha256"],
            )
        _copy_verified_source_file(
            bundle_root / "evidence_bundle_manifest.json",
            import_artifacts / "evidence_bundle_manifest.json",
            expected_sha256=summary_receipts["manifest_sha256"],
        )
        _copy_verified_source_file(
            bundle_root / "checksums.sha256",
            import_artifacts / "checksums.sha256",
            expected_sha256=summary_receipts["checksums_sha256"],
        )
        _copy_verified_source_file(
            materialized_manifest_path,
            import_artifacts / "materialized_manifest.json",
            expected_sha256=materialized_manifest_sha,
        )

        candidates = [
            _stage_issue9656_candidate(
                source_row,
                staging=staging,
                import_id=import_id,
                import_identity=import_identity,
                summary_receipts=summary_receipts,
                materialized_manifest_sha256=materialized_manifest_sha,
            )
            for source_row in source_rows
        ]

        import_record = _issue9656_candidate_import_record(
            summary,
            import_id=import_id,
            source_identity=import_identity,
            summary_receipts=summary_receipts,
            materialized_manifest_sha256=materialized_manifest_sha,
            candidates=candidates,
        )
        import_record["artifact_paths"] = {
            "payload_root": f"historical_candidate_imports/{import_id}/payload",
            "evidence_bundle_manifest": f"historical_candidate_imports/{import_id}/evidence_bundle_manifest.json",
            "checksums": f"historical_candidate_imports/{import_id}/checksums.sha256",
            "materialized_manifest": f"historical_candidate_imports/{import_id}/materialized_manifest.json",
        }

        prospective = dict(corpus)
        prospective["historical_candidates"] = sorted(
            [*corpus.get("historical_candidates", []), *candidates],
            key=lambda item: item["candidate_id"],
        )
        prospective["historical_candidate_imports"] = sorted(
            [*corpus.get("historical_candidate_imports", []), import_record],
            key=lambda item: item["import_id"],
        )
        validate_corpus(prospective)
        _promote_issue9656_candidate_artifacts(
            staging,
            root,
            import_id=import_id,
            candidate_ids=[candidate["candidate_id"] for candidate in candidates],
            promoted_dirs=promoted_dirs,
        )
        corpus["historical_candidates"] = prospective["historical_candidates"]
        corpus["historical_candidate_imports"] = prospective["historical_candidate_imports"]
        return corpus, {**import_record, "decision": "imported"}
    except BaseException:
        for promoted in reversed(promoted_dirs):
            shutil.rmtree(promoted, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _stage_issue9656_candidate(
    source_row: tuple[dict[str, Any], dict[str, Any], Path, Path, Path, dict[str, Any]],
    *,
    staging: Path,
    import_id: str,
    import_identity: Mapping[str, Any],
    summary_receipts: Mapping[str, Any],
    materialized_manifest_sha256: str,
) -> dict[str, Any]:
    source_case, materialized_case, source_case_path, source_matrix, source_config, source_map = (
        source_row
    )
    source_record = source_case["source_record"]
    candidate_identity = {
        **import_identity,
        "source_case_id": source_case["case_id"],
        "source_record_sha256": source_record["record_sha256"],
    }
    candidate_id = hashlib.sha256(_stable_json(candidate_identity).encode("utf-8")).hexdigest()
    candidate_dir = staging / "historical_candidates" / candidate_id
    (candidate_dir / "source").mkdir(parents=True)
    (candidate_dir / "replay_input").mkdir()
    (candidate_dir / "maps").mkdir()

    source_case_sha = _sha256_file(source_case_path)
    if source_case_sha != source_map.get("_materialized_case_sha256"):
        raise CorpusError("#9656 materialized case changed after source validation")
    source_matrix_sha = _sha256_file(source_matrix)
    source_config_sha = _sha256_file(source_config)
    _copy_verified_source_file(
        source_case_path, candidate_dir / "source_case.json", expected_sha256=source_case_sha
    )
    _copy_verified_source_file(
        source_matrix,
        candidate_dir / "source" / "replay_matrix.yaml",
        expected_sha256=source_case["replay_input"]["scenario_matrix_sha256"],
    )
    _copy_verified_source_file(
        source_config,
        candidate_dir / "source" / "planner_config.yaml",
        expected_sha256=source_case["replay_input"]["planner_config_sha256"],
    )
    normalized_matrix, map_receipt = _normalize_issue9656_replay_matrix(
        source_matrix, candidate_dir, source_map
    )
    matrix_path = candidate_dir / "replay_input" / "replay_matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(normalized_matrix, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
    config_path = candidate_dir / "replay_input" / "planner_config.yaml"
    _copy_verified_source_file(
        source_config,
        config_path,
        expected_sha256=source_case["replay_input"]["planner_config_sha256"],
    )
    return _issue9656_candidate_record(
        source_case,
        materialized_case,
        {
            "candidate_id": candidate_id,
            "import_id": import_id,
            "summary_receipts": summary_receipts,
            "materialized_manifest_sha256": materialized_manifest_sha256,
            "source_case_sha256": source_case_sha,
            "source_matrix_sha256": source_matrix_sha,
            "source_config_sha256": source_config_sha,
            "normalized_matrix_sha256": _sha256_file(matrix_path),
            "normalized_config_sha256": _sha256_file(config_path),
            "map_receipt": map_receipt,
            "source_binding": source_map["_source_binding"],
        },
    )


def _issue9656_candidate_record(
    source_case: Mapping[str, Any],
    materialized_case: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    source_record = source_case["source_record"]
    source = materialized_case.get("source", {})
    summary_receipts = context["summary_receipts"]
    source_binding = context["source_binding"]
    replay_status = source_case["replay"]["status"]
    candidate_status = {
        "not_attempted": "pending_exact_replay",
        "unavailable_model_artifact": "blocked_unavailable_model_artifact",
        "mismatch_different_revision": "blocked_replay_revision_mismatch",
    }[replay_status]
    if source_binding["status"] != "verified":
        candidate_status = "blocked_source_provenance_mismatch"
    return {
        "schema_version": "adversarial-historical-candidate.v1",
        "candidate_id": context["candidate_id"],
        "source_issue": 9656,
        "source_case_id": source_case["case_id"],
        "source_record_sha256": source_record["record_sha256"],
        "source_provenance": {
            "import_id": context["import_id"],
            "source_row_binding": "verified_episode_file_and_line_sha256",
            "summary_sha256": summary_receipts["summary_sha256"],
            "materialized_manifest_sha256": context["materialized_manifest_sha256"],
            "source_bundle_sha256": source.get("bundle_sha256"),
            "campaign_id": source.get("campaign_id"),
            "campaign_source_revision": source.get("campaign_source_revision"),
            "row_git_hash": source.get("row_git_hash"),
            "summary_source_revision": source_binding["summary_source_revision"],
            "episode_git_hash": source_binding["episode_git_hash"],
            "source_identity_binding_status": source_binding["status"],
            "source_identity_binding_issues": source_binding["issues"],
            "episode_file": source_record["episode_file"],
            "episode_file_sha256": source_record["episode_file_sha256"],
            "line_number": source_record["line_number"],
            "materialized_planner_config_hash": source.get("planner_config_hash"),
            "episode_planner_config_hash": source_binding["episode_planner_config_hash"],
            "episode_scenario_algo_config_hash": source_binding[
                "episode_scenario_algo_config_hash"
            ],
            "raw_planner_alias": source_case["planner_key"],
            "episode_canonical_algorithm": source_binding["episode_canonical_algorithm"],
            "materialized_canonical_algorithm": source_binding["materialized_canonical_algorithm"],
            "source_case_file_sha256": context["source_case_sha256"],
            "source_replay_matrix_sha256": context["source_matrix_sha256"],
            "source_planner_config_sha256": context["source_config_sha256"],
        },
        "candidate_status": candidate_status,
        "source_replay_status": replay_status,
        "source_replay": dict(source_case["replay"]),
        "scenario_id": source_case["scenario_id"],
        "scenario_family": source_case["scenario_family"],
        "scenario_seed": source_case["seed"],
        "benchmark_eligible": source_case["benchmark_eligible"],
        "target_planner": {
            "planner_id": source_case["planner_key"],
            "canonical_algorithm": source_binding["canonical_algorithm"],
            "config_hash": (
                source_binding["episode_planner_config_hash"]
                if source_binding["status"] == "verified"
                else None
            ),
        },
        "criticality": dict(source_case["criticality"]),
        "replay_inputs": {
            "source_matrix_sha256": context["source_matrix_sha256"],
            "normalized_matrix_sha256": context["normalized_matrix_sha256"],
            "source_planner_config_sha256": context["source_config_sha256"],
            "normalized_planner_config_sha256": context["normalized_config_sha256"],
            "map_assets": context["map_receipt"],
        },
        "artifact_paths": {
            "source_case": f"historical_candidates/{context['candidate_id']}/source_case.json",
            "source_matrix": f"historical_candidates/{context['candidate_id']}/source/replay_matrix.yaml",
            "source_planner_config": f"historical_candidates/{context['candidate_id']}/source/planner_config.yaml",
            "replay_matrix": f"historical_candidates/{context['candidate_id']}/replay_input/replay_matrix.yaml",
            "planner_config": f"historical_candidates/{context['candidate_id']}/replay_input/planner_config.yaml",
            "import_summary": f"historical_candidate_imports/{context['import_id']}/payload/summary.json",
            "import_materialized_manifest": f"historical_candidate_imports/{context['import_id']}/materialized_manifest.json",
        },
    }


def _verify_issue9656_source_summary(
    summary_file: Path, bundle_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_summary = (bundle_root / "payload" / "summary.json").resolve()
    if summary_file != expected_summary:
        raise CorpusError("#9656 summary must be the payload/summary.json in its evidence bundle")
    manifest_path = bundle_root / "evidence_bundle_manifest.json"
    checksum_path = bundle_root / "checksums.sha256"
    bundle_manifest = _read_json_object(manifest_path)
    if bundle_manifest.get("schema_version") != ISSUE_9645_BUNDLE_SCHEMA:
        raise CorpusError("#9656 evidence bundle manifest schema is invalid")
    entries = bundle_manifest.get("files")
    totals = bundle_manifest.get("totals")
    if not isinstance(entries, list) or not isinstance(totals, dict):
        raise CorpusError("#9656 evidence bundle inventory is incomplete")
    files_by_path = _validated_bundle_file_map(entries)
    total_bytes = sum(item["size_bytes"] for item in files_by_path.values())
    if totals.get("file_count") != len(files_by_path) or totals.get("total_bytes") != total_bytes:
        raise CorpusError("#9656 evidence bundle totals disagree with its file inventory")
    checksums = _parse_sha256_receipt(checksum_path, label="#9656 bundle checksums")
    expected_checksums = {
        f"payload/{relative}": record["sha256"] for relative, record in files_by_path.items()
    }
    if checksums != expected_checksums:
        raise CorpusError("#9656 checksum sidecar disagrees with the evidence manifest")
    for relative, record in files_by_path.items():
        _verify_bundle_payload_file(bundle_root / "payload", relative, record)
    summary = _read_json_object(summary_file)
    if summary.get("schema_version") != ISSUE_9656_SUMMARY_SCHEMA:
        raise CorpusError("unsupported #9656 hard-case summary schema")
    summary_entry = files_by_path.get("summary.json")
    if summary_entry is None:
        raise CorpusError("#9656 evidence bundle does not bind summary.json")
    return summary, {
        "summary_sha256": summary_entry["sha256"],
        "manifest_sha256": _sha256_file(manifest_path),
        "checksums_sha256": _sha256_file(checksum_path),
        "payload_files": [
            {"path": relative, "sha256": item["sha256"]}
            for relative, item in sorted(files_by_path.items())
        ],
    }


def _validate_issue9656_campaign_source(
    summary: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, str]:
    source = summary.get("source")
    materialized_source = manifest.get("source")
    required = (
        "source_revision",
        "source_campaign_id",
        "bundle_sha256",
        "matrix_path",
        "matrix_sha256",
    )
    if not isinstance(source, dict) or not isinstance(materialized_source, dict):
        raise CorpusError("#9656 source summary or materialized manifest lacks source identity")
    revision = source.get("source_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or not isinstance(source.get("source_campaign_id"), str)
        or not source["source_campaign_id"].strip()
        or not _is_sha256(source.get("bundle_sha256"))
        or not _safe_bundle_relative_path(source.get("matrix_path"))
        or not _is_sha256(source.get("matrix_sha256"))
    ):
        raise CorpusError("#9656 source summary has invalid campaign identity fields")
    if any(materialized_source.get(key) != source.get(key) for key in required):
        raise CorpusError("#9656 materialized manifest source identity differs from summary")
    historical_matrix = _issue9656_historical_git_blob(revision, source["matrix_path"])
    if hashlib.sha256(historical_matrix).hexdigest() != source["matrix_sha256"]:
        raise CorpusError("#9656 source matrix digest does not match its campaign revision")
    return {key: source[key] for key in required}


def _issue9656_historical_git_blob(revision: str, relative_path: str) -> bytes:
    """Read one path from a pinned local Git commit, without consulting the worktree file."""
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or not _safe_bundle_relative_path(relative_path)
    ):
        raise CorpusError("#9656 historical Git blob identity is invalid")
    try:
        result = subprocess.run(
            ["git", "show", f"{revision}:{relative_path}"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CorpusError(
            f"#9656 historical Git blob is unavailable: {revision}:{relative_path}"
        ) from exc
    return result.stdout


def _validate_issue9656_materialization(
    summary: Mapping[str, Any],
    materialized: Path,
    manifest: Mapping[str, Any],
    campaign_root: Path,
) -> list[tuple[dict[str, Any], dict[str, Any], Path, Path, Path, dict[str, Any]]]:
    if manifest.get("schema_version") != ISSUE_9656_SUMMARY_SCHEMA:
        raise CorpusError("#9656 materialized manifest schema differs from the source summary")
    summary_source = _validate_issue9656_campaign_source(summary, manifest)
    summary_cases = summary.get("cases")
    manifest_cases = manifest.get("cases")
    selection = summary.get("selection")
    if not isinstance(summary_cases, list) or not isinstance(manifest_cases, list):
        raise CorpusError("#9656 summary or materialized manifest has no case rows")
    if not isinstance(selection, dict) or selection.get("case_count") != len(summary_cases):
        raise CorpusError("#9656 selection count does not match its candidate rows")
    rows_by_id, manifest_by_id = _issue9656_rows_by_alias(summary_cases, manifest_cases, selection)
    replay = _issue9656_replay_summary(summary)
    status_counts: Counter[str] = Counter()
    anomalies: Counter[str] = Counter()
    rows = []
    for case_id, summary_case in rows_by_id.items():
        manifest_case = manifest_by_id[case_id]
        status = _validate_issue9656_summary_manifest_row(summary_case, manifest_case)
        status_counts[status] += 1
        anomalies.update(summary_case["criticality"]["anomalies"])
        rows.append(
            _validate_issue9656_materialized_case(
                summary_case, manifest_case, materialized, campaign_root, summary_source
            )
        )
    if dict(status_counts) != replay["status_counts"]:
        raise CorpusError("#9656 row replay statuses do not reconcile with the summary totals")
    if dict(anomalies) != summary.get("criticality_anomaly_counts"):
        raise CorpusError("#9656 row criticality anomalies do not reconcile with summary totals")
    _validate_issue9656_setup_accounting(summary)
    return rows


def _issue9656_rows_by_alias(
    summary_cases: list[Any], manifest_cases: list[Any], selection: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if any(not isinstance(case, dict) for case in summary_cases + manifest_cases):
        raise CorpusError("#9656 case lists contain a malformed row")
    summary_ids = [case["case_id"] for case in summary_cases]
    manifest_ids = [case["case_id"] for case in manifest_cases]
    if (
        any(not isinstance(case_id, str) for case_id in summary_ids + manifest_ids)
        or len(summary_ids) != len(set(summary_ids))
        or summary_ids != selection.get("case_ids")
        or set(summary_ids) != set(manifest_ids)
        or len(manifest_ids) != len(set(manifest_ids))
    ):
        raise CorpusError("#9656 stable source case aliases disagree across the manifests")
    if not all(_is_issue9656_source_alias(case_id) for case_id in summary_ids):
        raise CorpusError("#9656 source aliases must retain their case- plus 16-hex form")
    return (
        {case["case_id"]: case for case in summary_cases},
        {case["case_id"]: case for case in manifest_cases},
    )


def _issue9656_replay_summary(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    replay = summary.get("replay")
    if not isinstance(replay, dict) or not isinstance(replay.get("status_counts"), dict):
        raise CorpusError("#9656 replay status accounting is absent")
    return replay


def _is_issue9656_source_alias(value: str) -> bool:
    return (
        len(value) == 21
        and value.startswith("case-")
        and all(character in "0123456789abcdef" for character in value[5:])
    )


def _validate_issue9656_summary_manifest_row(
    summary_case: Mapping[str, Any], manifest_case: Mapping[str, Any]
) -> str:
    projected_manifest_case = {
        key: value for key, value in manifest_case.items() if key not in {"case_file", "replay"}
    }
    projected_summary_case = {key: value for key, value in summary_case.items() if key != "replay"}
    replay_row = summary_case.get("replay")
    manifest_replay = manifest_case.get("replay")
    binding_keys = ("attempted", "status", "source_revision", "replay_revision", "comparison")
    if (
        projected_manifest_case != projected_summary_case
        or not isinstance(replay_row, dict)
        or not isinstance(manifest_replay, dict)
        or any(manifest_replay.get(key) != replay_row.get(key) for key in binding_keys)
    ):
        raise CorpusError("#9656 materializer changed source summary row")
    status = replay_row.get("status")
    supported_statuses = {
        "not_attempted",
        "unavailable_model_artifact",
        "mismatch_different_revision",
    }
    if (
        not isinstance(status, str)
        or status not in supported_statuses
        or replay_row.get("attempted") is not (status == "mismatch_different_revision")
    ):
        raise CorpusError("#9656 replay status or attempt flag is unsupported")
    criticality = summary_case.get("criticality")
    source_record = summary_case.get("source_record")
    replay_input = summary_case.get("replay_input")
    if (
        not isinstance(criticality, dict)
        or not isinstance(criticality.get("anomalies"), list)
        or not isinstance(source_record, dict)
        or not _is_sha256(source_record.get("record_sha256"))
        or not isinstance(replay_input, dict)
        or replay_input.get("status") != "materialized"
    ):
        raise CorpusError("#9656 candidate row lacks required source evidence")
    return status


def _validate_issue9656_materialized_case(
    summary_case: dict[str, Any],
    manifest_case: dict[str, Any],
    materialized: Path,
    campaign_root: Path,
    summary_source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path, dict[str, Any]]:
    case_id = summary_case["case_id"]
    replay_input = summary_case["replay_input"]
    case_file = _safe_materialized_path(
        materialized, manifest_case.get("case_file"), label="case file"
    )
    materialized_case = _read_json_object(case_file)
    if not _issue9656_case_document_matches(summary_case, manifest_case, materialized_case):
        raise CorpusError(f"#9656 materialized case does not bind summary row {case_id}")
    source_row = _verify_issue9656_source_row(campaign_root, summary_case, materialized_case)
    source_binding = _issue9656_source_identity_binding(
        summary_case, materialized_case, source_row, summary_source
    )
    matrix_path = _safe_materialized_path(
        case_file.parent, replay_input.get("scenario_matrix_path"), label="scenario matrix"
    )
    config_path = _safe_materialized_path(
        case_file.parent, replay_input.get("planner_config_path"), label="planner config"
    )
    matrix_sha = replay_input.get("scenario_matrix_sha256")
    config_sha = replay_input.get("planner_config_sha256")
    if _sha256_file(matrix_path) != matrix_sha or _sha256_file(config_path) != config_sha:
        raise CorpusError(f"#9656 materialized replay input digest differs for {case_id}")
    map_info = _issue9656_map_asset(matrix_path, summary_case, summary_source["source_revision"])
    map_info["_source_binding"] = source_binding
    map_info["_materialized_case_sha256"] = _sha256_file(case_file)
    return summary_case, materialized_case, case_file, matrix_path, config_path, map_info


def _issue9656_case_document_matches(
    summary_case: Mapping[str, Any],
    manifest_case: Mapping[str, Any],
    materialized_case: Mapping[str, Any],
) -> bool:
    source_record = summary_case["source_record"]
    source = materialized_case.get("source")
    planner = materialized_case.get("planner")
    scenario = materialized_case.get("scenario")
    if not all(isinstance(item, dict) for item in (source, planner, scenario)):
        return False
    return (
        materialized_case.get("schema_version") == "benchmark-hard-case.v1"
        and materialized_case.get("case_id") == summary_case.get("case_id")
        and all(source.get(key) == value for key, value in source_record.items())
        and materialized_case.get("criticality") == summary_case.get("criticality")
        and materialized_case.get("replay") == manifest_case.get("replay")
        and materialized_case.get("replay_input") == summary_case.get("replay_input")
        and planner.get("key") == summary_case.get("planner_key")
        and scenario.get("scenario_id") == summary_case.get("scenario_id")
        and scenario.get("seed") == summary_case.get("seed")
    )


def _verify_issue9656_source_row(
    campaign_root: Path,
    summary_case: Mapping[str, Any],
    materialized_case: Mapping[str, Any],
) -> dict[str, Any]:
    source_ref = summary_case["source_record"]
    episode_path = _safe_materialized_path(
        campaign_root, source_ref.get("episode_file"), label="source episode file"
    )
    if _sha256_file(episode_path) != source_ref.get("episode_file_sha256"):
        raise CorpusError(f"#9656 source episode file digest differs for {summary_case['case_id']}")
    line_number = source_ref.get("line_number")
    if not isinstance(line_number, int) or isinstance(line_number, bool) or line_number < 1:
        raise CorpusError(
            f"#9656 source episode line number is invalid for {summary_case['case_id']}"
        )
    lines = episode_path.read_bytes().splitlines(keepends=True)
    if line_number > len(lines):
        raise CorpusError(f"#9656 source episode row is absent for {summary_case['case_id']}")
    raw_line = lines[line_number - 1]
    if hashlib.sha256(raw_line).hexdigest() != source_ref.get("record_sha256"):
        raise CorpusError(f"#9656 source episode row digest differs for {summary_case['case_id']}")
    try:
        row = json.loads(raw_line)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusError(
            f"#9656 source episode row is malformed for {summary_case['case_id']}"
        ) from exc
    if (
        not isinstance(row, dict)
        or row != materialized_case.get("source_record")
        or row.get("scenario_id") != summary_case.get("scenario_id")
        or row.get("seed") != summary_case.get("seed")
        or row.get("algo") != summary_case.get("planner_key")
    ):
        raise CorpusError("#9656 materialized source row differs from campaign evidence")
    return row


def _issue9656_source_identity_binding(
    summary_case: Mapping[str, Any],
    materialized_case: Mapping[str, Any],
    source_row: Mapping[str, Any],
    summary_source: Mapping[str, Any],
) -> dict[str, Any]:
    source = materialized_case.get("source")
    planner = materialized_case.get("planner")
    if not isinstance(source, dict) or not isinstance(planner, dict):
        raise CorpusError("#9656 materialized source identity is malformed")
    metadata = source_row.get("algorithm_metadata")
    planner_metadata = planner.get("algorithm_metadata")
    scenario_params = source_row.get("scenario_params")
    if not isinstance(metadata, dict) or not isinstance(planner_metadata, dict):
        metadata = {}
        planner_metadata = {}
    if not isinstance(scenario_params, dict):
        scenario_params = {}
    summary_revision = summary_source["source_revision"]
    episode_git_hash = source_row.get("git_hash")
    # The materializer defines planner_config_hash from algorithm_metadata.config_hash;
    # scenario_params.algo_config_hash is retained separately because the source can differ.
    episode_config_hash = metadata.get("config_hash")
    canonical_algorithm = canonical_algorithm_name(str(summary_case["planner_key"]))
    expected = {
        "campaign_source_revision": summary_revision,
        "row_git_hash": episode_git_hash,
        "episode_git_hash": summary_revision,
        "planner_config_hash": episode_config_hash,
        "materialized_planner_config_hash": episode_config_hash,
        "campaign_id": summary_source["source_campaign_id"],
        "bundle_sha256": summary_source["bundle_sha256"],
        "scenario_matrix": summary_source["matrix_path"],
        "scenario_matrix_sha256": summary_source["matrix_sha256"],
        "planner_key": summary_case["planner_key"],
        "episode_planner_key": summary_case["planner_key"],
        "materialized_planner_key": summary_case["planner_key"],
        "episode_canonical_algorithm": canonical_algorithm,
        "materialized_canonical_algorithm": canonical_algorithm,
    }
    actual = {
        "campaign_source_revision": source.get("campaign_source_revision"),
        "row_git_hash": source.get("row_git_hash"),
        "episode_git_hash": episode_git_hash,
        "episode_planner_config_hash": episode_config_hash,
        "planner_config_hash": source.get("planner_config_hash"),
        "materialized_planner_config_hash": planner_metadata.get("config_hash"),
        "campaign_id": source.get("campaign_id"),
        "bundle_sha256": source.get("bundle_sha256"),
        "scenario_matrix": source.get("scenario_matrix"),
        "scenario_matrix_sha256": source.get("scenario_matrix_sha256"),
        "planner_key": source.get("planner_key"),
        "episode_planner_key": source_row.get("algo"),
        "materialized_planner_key": planner.get("key"),
        "episode_canonical_algorithm": metadata.get("canonical_algorithm"),
        "materialized_canonical_algorithm": planner_metadata.get("canonical_algorithm"),
    }
    issues = sorted(field for field, value in expected.items() if actual[field] != value)
    for field in ("episode_git_hash", "episode_planner_config_hash"):
        value = actual[field]
        if not isinstance(value, str) or not value.strip():
            issues = sorted({*issues, field})
    return {
        "status": "verified" if not issues else "blocked",
        "issues": issues,
        "summary_source_revision": summary_revision,
        "episode_git_hash": episode_git_hash,
        "episode_planner_config_hash": episode_config_hash,
        "episode_scenario_algo_config_hash": scenario_params.get("algo_config_hash"),
        "canonical_algorithm": canonical_algorithm,
        "episode_canonical_algorithm": metadata.get("canonical_algorithm"),
        "materialized_canonical_algorithm": planner_metadata.get("canonical_algorithm"),
    }


def _validate_issue9656_setup_accounting(summary: Mapping[str, Any]) -> None:
    replay_budget = summary.get("replay_budget_accounting")
    failed_attempts = summary.get("failed_replay_attempts")
    if (
        not isinstance(replay_budget, dict)
        or not isinstance(failed_attempts, dict)
        or not isinstance(failed_attempts.get("attempts"), list)
        or not isinstance(replay_budget.get("failed_setup_jobs"), int)
        or isinstance(replay_budget.get("failed_setup_jobs"), bool)
        or replay_budget["failed_setup_jobs"] < 0
    ):
        raise CorpusError("#9656 failed replay setup accounting is absent or malformed")
    if any(
        not isinstance(attempt, dict)
        or not isinstance(attempt.get("return_code"), int)
        or isinstance(attempt.get("return_code"), bool)
        or attempt.get("return_code") == 0
        or attempt.get("replay_row_count") != 0
        for attempt in failed_attempts["attempts"]
    ):
        raise CorpusError("#9656 setup failures must remain explicit zero-row failed attempts")


def _safe_materialized_path(root: Path, relative: Any, *, label: str) -> Path:
    if not _safe_bundle_relative_path(relative):
        raise CorpusError(f"#9656 {label} path is unsafe")
    resolved = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise CorpusError(f"#9656 {label} path escapes materialized root") from exc
    if not resolved.is_file():
        raise CorpusError(f"#9656 {label} is missing: {relative}")
    return resolved


def _issue9656_map_asset(
    matrix_path: Path, summary_case: Mapping[str, Any], campaign_source_revision: str
) -> dict[str, Any]:
    map_path, relative = _issue9656_map_path(matrix_path, summary_case)
    if not map_path.is_file():
        raise CorpusError(f"#9656 current map file is unavailable: {relative.as_posix()}")
    try:
        current_bytes = map_path.read_bytes()
    except OSError as exc:
        raise CorpusError(f"#9656 current map file is unavailable: {relative.as_posix()}") from exc
    historical_bytes = _issue9656_historical_git_blob(campaign_source_revision, relative.as_posix())
    if current_bytes != historical_bytes:
        raise CorpusError(f"#9656 map differs from campaign source revision: {relative.as_posix()}")
    return {
        "source_path": relative.as_posix(),
        "source_sha256": hashlib.sha256(historical_bytes).hexdigest(),
        "source_revision": campaign_source_revision,
        "_source_bytes": historical_bytes,
        "_repo_relative": relative.as_posix(),
    }


def _issue9656_map_path(matrix_path: Path, summary_case: Mapping[str, Any]) -> tuple[Path, Path]:
    try:
        matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CorpusError(f"#9656 replay matrix could not be parsed: {exc}") from exc
    scenarios = matrix.get("scenarios") if isinstance(matrix, dict) else None
    if not isinstance(scenarios, list) or len(scenarios) != 1 or not isinstance(scenarios[0], dict):
        raise CorpusError(
            f"#9656 replay matrix must contain one scenario for {summary_case['case_id']}"
        )
    scenario = scenarios[0]
    if (
        scenario.get("id") != summary_case.get("scenario_id")
        or scenario.get("seeds") != [summary_case.get("seed")]
        or scenario.get("algo") != summary_case.get("planner_key")
    ):
        raise CorpusError(f"#9656 replay matrix identity differs for {summary_case['case_id']}")
    map_value = scenario.get("map_file")
    if not isinstance(map_value, str) or not map_value.strip():
        raise CorpusError(f"#9656 replay matrix has no map file for {summary_case['case_id']}")
    map_path = (matrix_path.parent / map_value).resolve()
    repo_root = next(
        (ancestor for ancestor in map_path.parents if (ancestor / "maps" / "svg_maps").is_dir()),
        None,
    )
    if repo_root is None:
        raise CorpusError("#9656 materialized input is not below a repository containing maps")
    try:
        relative = map_path.relative_to(repo_root)
    except ValueError as exc:
        raise CorpusError(
            "#9656 candidate map path must resolve inside its source repository"
        ) from exc
    if (
        not relative.parts
        or relative.parts[0] != "maps"
        or not relative.as_posix().startswith("maps/svg_maps/")
    ):
        raise CorpusError("#9656 candidate map path must resolve to a repository map file")
    return map_path, relative


def _normalize_issue9656_replay_matrix(
    source_matrix: Path, candidate_dir: Path, map_info: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    try:
        matrix = yaml.safe_load(source_matrix.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CorpusError(f"#9656 source replay matrix could not be parsed: {exc}") from exc
    if not isinstance(matrix, dict) or not isinstance(matrix.get("scenarios"), list):
        raise CorpusError("#9656 source replay matrix must be an object with scenarios")
    scenario = matrix["scenarios"][0]
    repo_relative = str(map_info["_repo_relative"])
    source_map = map_info.get("_source_bytes")
    if not isinstance(source_map, bytes):
        raise CorpusError("#9656 historical map bytes are unavailable")
    map_output = candidate_dir / Path(*PurePosixPath(repo_relative).parts)
    map_output.parent.mkdir(parents=True, exist_ok=True)
    map_output.write_bytes(source_map)
    if _sha256_file(map_output) != map_info["source_sha256"]:
        raise CorpusError("#9656 copied historical map digest differs")
    scenario["map_file"] = (Path("..") / Path(*PurePosixPath(repo_relative).parts)).as_posix()
    matrix["map_search_paths"] = ["../maps/svg_maps"]
    map_receipt = [
        {
            "source_path": map_info["source_path"],
            "source_revision": map_info["source_revision"],
            "source_sha256": map_info["source_sha256"],
            "stored_path": f"historical_candidates/{candidate_dir.name}/{repo_relative}",
            "stored_sha256": _sha256_file(map_output),
        }
    ]
    return matrix, map_receipt


def _issue9656_candidate_import_record(
    summary: Mapping[str, Any],
    *,
    import_id: str,
    source_identity: Mapping[str, Any],
    summary_receipts: Mapping[str, Any],
    materialized_manifest_sha256: str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    replay_counts = Counter(candidate["source_replay_status"] for candidate in candidates)
    candidate_status_counts = Counter(candidate["candidate_status"] for candidate in candidates)
    return {
        "schema_version": "adversarial-historical-candidate-import.v1",
        "import_id": import_id,
        "source_issue": 9656,
        "source_identity": dict(source_identity),
        "source_files": [
            *[
                {
                    "path": f"payload/{item['path']}",
                    "stored_path": f"historical_candidate_imports/{import_id}/payload/{item['path']}",
                    "sha256": item["sha256"],
                }
                for item in summary_receipts["payload_files"]
            ],
            {
                "path": "evidence_bundle_manifest.json",
                "stored_path": f"historical_candidate_imports/{import_id}/evidence_bundle_manifest.json",
                "sha256": summary_receipts["manifest_sha256"],
            },
            {
                "path": "checksums.sha256",
                "stored_path": f"historical_candidate_imports/{import_id}/checksums.sha256",
                "sha256": summary_receipts["checksums_sha256"],
            },
            {
                "path": "materialized_manifest.json",
                "stored_path": f"historical_candidate_imports/{import_id}/materialized_manifest.json",
                "sha256": materialized_manifest_sha256,
            },
        ],
        "candidate_ids": sorted(candidate["candidate_id"] for candidate in candidates),
        "candidate_count": len(candidates),
        "source_rows_verified": len(candidates),
        "candidate_status_counts": dict(sorted(candidate_status_counts.items())),
        "source_replay_status_counts": dict(sorted(replay_counts.items())),
        "criticality_anomaly_counts": dict(
            sorted((summary.get("criticality_anomaly_counts") or {}).items())
        ),
        "failed_replay_attempts": summary["failed_replay_attempts"],
        "failed_setup_jobs": summary["replay_budget_accounting"]["failed_setup_jobs"],
        "evidence_tier": summary.get("evidence_tier"),
        "claim_boundary": summary.get("claim_boundary"),
    }


def _copy_verified_source_file(source: Path, destination: Path, *, expected_sha256: str) -> None:
    if (
        not source.is_file()
        or not _is_sha256(expected_sha256)
        or _sha256_file(source) != expected_sha256
    ):
        raise CorpusError(f"source artifact digest differs: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _sha256_file(destination) != expected_sha256:
        raise CorpusError(f"copied artifact digest differs: {destination}")


def _verify_existing_issue9656_candidate_artifacts(
    corpus_root: Path,
    import_record: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    if len(candidates) != import_record.get("candidate_count"):
        raise CorpusError("duplicate #9656 import is missing candidate registry rows")
    for source_file in import_record.get("source_files", []):
        if not isinstance(source_file, dict):
            raise CorpusError("duplicate #9656 import contains a malformed source receipt")
        _verify_corpus_artifact(
            corpus_root, source_file.get("stored_path"), source_file.get("sha256")
        )
    for candidate in candidates:
        paths = candidate.get("artifact_paths", {})
        provenance = candidate.get("source_provenance", {})
        inputs = candidate.get("replay_inputs", {})
        for field, digest in (
            ("source_case", provenance.get("source_case_file_sha256")),
            ("source_matrix", inputs.get("source_matrix_sha256")),
            ("source_planner_config", inputs.get("source_planner_config_sha256")),
            ("replay_matrix", inputs.get("normalized_matrix_sha256")),
            ("planner_config", inputs.get("normalized_planner_config_sha256")),
        ):
            _verify_corpus_artifact(corpus_root, paths.get(field), digest)
        for map_asset in inputs.get("map_assets", []):
            if not isinstance(map_asset, dict):
                raise CorpusError("duplicate #9656 import contains a malformed map receipt")
            _verify_corpus_artifact(
                corpus_root,
                map_asset.get("stored_path"),
                map_asset.get("stored_sha256"),
            )


def _verify_issue9656_duplicate_materialized_cases(
    source_rows: Sequence[tuple[Any, ...]], candidates: Sequence[Mapping[str, Any]]
) -> None:
    candidates_by_alias = {candidate["source_case_id"]: candidate for candidate in candidates}
    if len(candidates_by_alias) != len(source_rows):
        raise CorpusError("duplicate #9656 import has missing materialized-case receipts")
    for source_row in source_rows:
        source_case, _materialized_case, case_path, *_artifacts = source_row
        candidate = candidates_by_alias.get(source_case["case_id"])
        stored_sha256 = (
            candidate.get("source_provenance", {}).get("source_case_file_sha256")
            if candidate is not None
            else None
        )
        if not _is_sha256(stored_sha256) or _sha256_file(case_path) != stored_sha256:
            raise CorpusError(
                "duplicate #9656 import materialized case differs from its verified source"
            )


def _verify_corpus_artifact(root: Path, relative: Any, expected_sha256: Any) -> None:
    if not _safe_bundle_relative_path(relative) or not _is_sha256(expected_sha256):
        raise CorpusError("stored #9656 artifact receipt contains an unsafe path or digest")
    source = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        source.relative_to(root.resolve())
    except ValueError as exc:
        raise CorpusError("stored #9656 artifact path escapes corpus root") from exc
    if not source.is_file() or _sha256_file(source) != expected_sha256:
        raise CorpusError(f"stored #9656 artifact digest differs: {relative}")


def _promote_issue9656_candidate_artifacts(
    staging: Path,
    root: Path,
    *,
    import_id: str,
    candidate_ids: Sequence[str],
    promoted_dirs: list[Path],
) -> None:
    source_import = staging / "historical_candidate_imports" / import_id
    source_candidates = staging / "historical_candidates"
    final_import = root / "historical_candidate_imports" / import_id
    final_candidates_root = root / "historical_candidates"
    final_import.parent.mkdir(parents=True, exist_ok=True)
    final_candidates_root.mkdir(parents=True, exist_ok=True)
    destinations = [
        (source_candidates / candidate_id, final_candidates_root / candidate_id)
        for candidate_id in candidate_ids
    ]
    destinations.append((source_import, final_import))
    if any(destination.exists() for _source, destination in destinations):
        raise CorpusError(
            "#9656 candidate artifact destination already exists without its import receipt"
        )
    try:
        for source, destination in destinations:
            source.replace(destination)
            promoted_dirs.append(destination)
    except BaseException:
        for promoted in reversed(promoted_dirs):
            shutil.rmtree(promoted, ignore_errors=True)
        promoted_dirs.clear()
        raise


def create_planner_replay_receipt(
    observation: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    artifact_path: str,
    corpus_root: str | Path,
) -> dict[str, Any]:
    """Build a receipt only when one stored episode artifact matches the observation."""
    item = dict(observation)
    root = Path(corpus_root).resolve()
    artifact = _resolve_corpus_artifact(artifact_path, root)
    episode_sha256 = _sha256_file(artifact)
    if item.get("episode_sha256") is None:
        item["episode_sha256"] = episode_sha256
    record = _read_single_jsonl_record(artifact)
    inputs = case.get("inputs")
    inputs = inputs if isinstance(inputs, dict) else {}
    receipt = {
        "schema_version": EVALUATION_REPLAY_RECEIPT_SCHEMA_VERSION,
        "verification_status": "artifact_projection_match",
        "artifact_path": PurePosixPath(artifact_path).as_posix(),
        "artifact_sha256": episode_sha256,
        "case_id": item.get("case_id"),
        "effective_scenario_sha256": item.get("effective_scenario_sha256"),
        "scenario_id": case.get("scenario_id"),
        "scenario_seed": case.get("scenario_seed"),
        "scenario_input_sha256": inputs.get("scenario_sha256"),
        "route_overrides_sha256": inputs.get("route_overrides_sha256"),
        "planner_id": item.get("planner_id"),
        "planner_config_identity": item.get("planner_config_identity"),
        "source_revision": item.get("source_revision"),
        "episode_sha256": episode_sha256,
        "outcome": item.get("outcome"),
        "termination_reason": item.get("termination_reason"),
        "metrics": item.get("metrics"),
        "selected_event_identity": _selected_event_identity(record),
        "execution_mode": item.get("execution_mode"),
        "readiness_status": item.get("readiness_status"),
        "availability_status": item.get("availability_status"),
        "fallback_or_degraded": item.get("fallback_or_degraded"),
    }
    errors = _validate_replay_artifact(item, case, receipt, root)
    if errors:
        raise CorpusError("replay receipt rejected: " + "; ".join(errors))
    return receipt


def append_planner_evaluation(
    corpus: dict[str, Any],
    observation: Mapping[str, Any],
    *,
    corpus_root: str | Path | None = None,
) -> dict[str, Any]:
    """Append a planner result only when its case, planner, and input binding are explicit."""
    validate_corpus(corpus)
    item = dict(observation)
    if item.get("evidence_status") == "complete" and not isinstance(
        item.get("replay_receipt"), dict
    ):
        raise CorpusError("planner evaluation rejected: replay_receipt_missing")
    errors = _validate_evaluation(item)
    if errors:
        raise CorpusError("planner evaluation rejected: " + "; ".join(errors))
    case = next((entry for entry in corpus["cases"] if entry["case_id"] == item["case_id"]), None)
    if case is None:
        raise CorpusError(f"unknown case_id: {item['case_id']}")
    if item["effective_scenario_sha256"] != case["effective_scenario_sha256"]:
        raise CorpusError("planner evaluation input hash does not match the corpus case")
    if item.get("evidence_status") == "complete":
        if corpus_root is None:
            raise CorpusError("complete planner evaluations require a replay artifact corpus_root")
        replay_errors = _validate_replay_artifact(
            item, case, item["replay_receipt"], Path(corpus_root).resolve()
        )
        if replay_errors:
            raise CorpusError(
                "planner evaluation replay evidence rejected: " + "; ".join(replay_errors)
            )
    identity = {
        key: item.get(key)
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
            "replay_receipt",
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
    corpus: Mapping[str, Any],
    *,
    planner_id: str,
    planner_config_identity: str,
    corpus_root: str | Path | None = None,
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
        states = [
            _evaluation_state(
                row,
                case=case,
                corpus_root=Path(corpus_root).resolve() if corpus_root is not None else None,
            )
            for row in rows
        ]
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
            source_scenario = dict(scenarios[0])
            route_payload = _load_yaml_object(route_path, "route overrides")
            route_relative = f"routes/{case['case_id']}.yaml"
            config_relative = f"planner_configs/{case['case_id']}.yaml"
            scenario, identity_mapping = _map_slice_scenario_identity(
                case, source_scenario, route_payload, route_relative
            )
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
                    "identity_mapping": identity_mapping,
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
        exported_matrix = _load_yaml_object(matrix_path, "exported replay matrix")
        exported_rows = exported_matrix.get("scenarios")
        if not isinstance(exported_rows, list) or len(exported_rows) != len(manifest_cases):
            raise CorpusError("exported replay matrix does not preserve the case rows")
        for scenario, manifest_case in zip(exported_rows, manifest_cases, strict=True):
            route_ref = scenario.get("route_overrides_file")
            route_doc = _load_yaml_object(staging / str(route_ref), "exported route overrides")
            actual_hash = compute_effective_scenario_hash(scenario, route_doc)
            if (
                actual_hash
                != manifest_case["identity_mapping"]["exported_effective_scenario_sha256"]
            ):
                raise CorpusError(
                    f"{manifest_case['case_id']}: exported scenario identity mapping is invalid"
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


def _map_slice_scenario_identity(
    case: Mapping[str, Any],
    source_scenario: Mapping[str, Any],
    route_payload: Mapping[str, Any],
    route_relative: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_hash = compute_effective_scenario_hash(source_scenario, route_payload)
    if source_hash != case["effective_scenario_sha256"]:
        raise CorpusError(f"{case['case_id']}: source scenario identity changed")
    scenario = dict(source_scenario)
    source_route = scenario.get("route_overrides_file")
    scenario["route_overrides_file"] = route_relative
    scenario["seeds"] = [int(case["scenario_seed"])]
    exported_hash = compute_effective_scenario_hash(scenario, route_payload)
    mapping = {
        "schema_version": "adversarial-slice-identity-mapping.v1",
        "verification_status": "source_and_export_hashes_verified",
        "normalization_fields": ["route_overrides_file"],
        "source_effective_scenario_sha256": source_hash,
        "exported_effective_scenario_sha256": exported_hash,
        "source_route_overrides_file": source_route,
        "exported_route_overrides_file": route_relative,
    }
    return scenario, mapping


def _verify_issue9645_pilot(payload: Path) -> dict[str, Any]:
    summary = _read_json_object(payload / "summary.json")
    metadata = _read_json_object(payload / "run_metadata.json")
    row_status = _read_json_object(payload / "row_status.json")
    bundle_receipts = _verify_issue9645_bundle(
        payload,
        source_revision=str(metadata.get("experiment_source_commit") or ""),
        source_file_hashes=metadata.get("source_file_sha256"),
        required_payload_paths=_pilot_accounting_paths(payload),
    )
    _validate_pilot_summary(summary, metadata)
    _validate_pilot_design(metadata)
    manifests_by_identity = _verify_pilot_manifests(payload, metadata)
    _verify_pilot_candidate_rows(payload, row_status)
    return _pilot_search_run(summary, metadata, manifests_by_identity, payload, bundle_receipts)


def _pilot_accounting_paths(payload: Path) -> list[str]:
    source_manifests = sorted((payload / "source_manifests").glob("*.json"))
    return [
        "summary.json",
        "run_metadata.json",
        "row_status.json",
        "candidate_evaluations.csv",
        "source_hashes.sha256",
        "report_provenance.json",
        "inputs/crossing_ttc.yaml",
        "inputs/issue_9645_pilot_space.v1.yaml",
        *[path.relative_to(payload).as_posix() for path in source_manifests],
    ]


def _verify_issue9645_bundle(
    payload: Path,
    *,
    source_revision: str,
    source_file_hashes: Any,
    required_payload_paths: Sequence[str],
) -> list[dict[str, str]]:
    """Verify the bundle's outer receipt and hashes for every consumed payload file."""
    bundle_root = payload.parent
    manifest_path, checksums_path, files_by_path = _read_issue9645_bundle_inventory(
        bundle_root, source_revision
    )
    _verify_issue9645_payload_receipts(
        payload,
        source_revision=source_revision,
        source_file_hashes=source_file_hashes,
        required_payload_paths=required_payload_paths,
        files_by_path=files_by_path,
    )
    return [
        {
            "source_path": "evidence_bundle_manifest.json",
            "path": "packet_receipts/evidence_bundle_manifest.json",
            "sha256": _sha256_file(manifest_path),
        },
        {
            "source_path": "checksums.sha256",
            "path": "packet_receipts/checksums.sha256",
            "sha256": _sha256_file(checksums_path),
        },
    ]


def _read_issue9645_bundle_inventory(
    bundle_root: Path, source_revision: str
) -> tuple[Path, Path, dict[str, dict[str, Any]]]:
    manifest_path = bundle_root / "evidence_bundle_manifest.json"
    checksums_path = bundle_root / "checksums.sha256"
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema_version") != ISSUE_9645_BUNDLE_SCHEMA:
        raise CorpusError("#9645 evidence bundle manifest schema is invalid")
    if (
        manifest.get("bundle_name") != "issue_9645_bounded_falsification_2026-09-24"
        or manifest.get("commit") != source_revision
    ):
        raise CorpusError("#9645 evidence bundle identity differs from the source run")
    bundle_files = manifest.get("files")
    totals = manifest.get("totals")
    if not isinstance(bundle_files, list) or not isinstance(totals, dict):
        raise CorpusError("#9645 evidence bundle file inventory is incomplete")
    files_by_path = _validated_bundle_file_map(bundle_files)
    total_bytes = sum(item["size_bytes"] for item in files_by_path.values())
    if totals.get("file_count") != len(files_by_path) or totals.get("total_bytes") != total_bytes:
        raise CorpusError("#9645 evidence bundle totals disagree with its file inventory")

    checksums = _parse_sha256_receipt(checksums_path, label="bundle checksums")
    expected_checksums = {
        f"payload/{relative}": item["sha256"] for relative, item in files_by_path.items()
    }
    if checksums != expected_checksums:
        raise CorpusError("#9645 bundle checksum file disagrees with its manifest")
    return manifest_path, checksums_path, files_by_path


def _verify_issue9645_payload_receipts(
    payload: Path,
    *,
    source_revision: str,
    source_file_hashes: Any,
    required_payload_paths: Sequence[str],
    files_by_path: Mapping[str, Mapping[str, Any]],
) -> None:
    source_hashes = _parse_sha256_receipt(payload / "source_hashes.sha256", label="source hashes")
    if not isinstance(source_file_hashes, dict) or source_hashes != source_file_hashes:
        raise CorpusError("#9645 source hashes disagree with run metadata")
    for relative in required_payload_paths:
        record = files_by_path.get(relative)
        if record is None:
            raise CorpusError(f"#9645 consumed file is absent from the bundle manifest: {relative}")
        _verify_bundle_payload_file(payload, relative, record)

    report = _read_json_object(payload / "report_provenance.json")
    if (
        report.get("schema_version") != "issue_9645_report_build_provenance.v1"
        or report.get("source_search_commit") != source_revision
    ):
        raise CorpusError("#9645 report provenance does not bind the source search revision")
    comparison_record = files_by_path.get("pilot_comparison.json")
    if comparison_record is None:
        raise CorpusError("#9645 pilot comparison is absent from the bundle manifest")
    _verify_bundle_payload_file(payload, "pilot_comparison.json", comparison_record)
    if report.get("comparison_sha256") != comparison_record["sha256"]:
        raise CorpusError("#9645 report provenance does not bind the pilot comparison")


def _validated_bundle_file_map(entries: list[Any]) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise CorpusError("#9645 bundle file inventory contains a malformed entry")
        relative = entry.get("path")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if (
            not _safe_bundle_relative_path(relative)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not _is_sha256(digest)
            or relative in files
        ):
            raise CorpusError("#9645 bundle file inventory contains invalid identity or checksum")
        files[relative] = {"size_bytes": size, "sha256": digest}
    if not files:
        raise CorpusError("#9645 bundle file inventory is empty")
    return files


def _safe_bundle_relative_path(value: Any) -> bool:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
    ):
        return False
    path = PurePosixPath(value)
    return (
        value not in {"", "."}
        and not path.is_absolute()
        and path.as_posix() == value
        and ".." not in path.parts
    )


def _parse_sha256_receipt(path: Path, *, label: str) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CorpusError(f"could not read #9645 {label}: {exc}") from exc
    records: dict[str, str] = {}
    for line in lines:
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not _is_sha256(fields[0]):
            raise CorpusError(f"#9645 {label} contains a malformed checksum line")
        relative = fields[1].strip()
        if not _safe_bundle_relative_path(relative) or relative in records:
            raise CorpusError(f"#9645 {label} contains an unsafe or duplicate path")
        records[relative] = fields[0]
    if not records:
        raise CorpusError(f"#9645 {label} is empty")
    return records


def _verify_bundle_payload_file(payload: Path, relative: str, record: Mapping[str, Any]) -> None:
    if not _safe_bundle_relative_path(relative):
        raise CorpusError("#9645 consumed payload path is unsafe")
    source = (payload / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        source.relative_to(payload.resolve())
    except ValueError as exc:
        raise CorpusError("#9645 consumed payload path escapes the packet") from exc
    if (
        not source.is_file()
        or source.stat().st_size != record.get("size_bytes")
        or _sha256_file(source) != record.get("sha256")
    ):
        raise CorpusError(f"#9645 consumed payload checksum differs: {relative}")


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
    bundle_receipts: list[dict[str, str]],
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
        "bundle_receipts": bundle_receipts,
    }


def _copy_pilot_evidence(payload: Path) -> list[dict[str, str]]:
    """Return exact checksums for the minimal #9645 pilot accounting packet."""
    relative_paths = [
        "summary.json",
        "run_metadata.json",
        "candidate_evaluations.csv",
        "row_status.json",
        "source_hashes.sha256",
        "report_provenance.json",
        "pilot_comparison.json",
        "inputs/crossing_ttc.yaml",
        "inputs/issue_9645_pilot_space.v1.yaml",
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
    persisted_paths = _persist_pilot_payload_files(payload, evidence_dir, corpus_root, pilot)
    pilot["source_files"] = [
        {
            "path": persisted_paths[source["path"]],
            "sha256": source["sha256"],
        }
        for source in pilot["source_files"]
    ]
    pilot["manifest_files"] = [
        {
            **manifest,
            "path": persisted_paths[manifest["path"]],
        }
        for manifest in pilot["manifest_files"]
    ]
    pilot["bundle_receipts"] = _persist_pilot_bundle_receipts(
        payload, evidence_dir, corpus_root, pilot["bundle_receipts"]
    )
    pilot["evidence_bundle_root"] = "evidence/issue_9645_pilot"


def _persist_pilot_payload_files(
    payload: Path,
    evidence_dir: Path,
    corpus_root: Path,
    pilot: Mapping[str, Any],
) -> dict[str, str]:
    expected_hashes = {item["path"]: item["sha256"] for item in pilot["source_files"]}
    expected_hashes.update({item["path"]: item["sha256"] for item in pilot["manifest_files"]})
    persisted: dict[str, str] = {}
    for relative, expected in sorted(expected_hashes.items()):
        source = payload / relative
        destination = evidence_dir / relative
        _copy_verified_file(source, destination, expected, f"pilot evidence {relative}")
        persisted[relative] = destination.resolve().relative_to(corpus_root.resolve()).as_posix()
    return persisted


def _persist_pilot_bundle_receipts(
    payload: Path,
    evidence_dir: Path,
    corpus_root: Path,
    receipts: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    persisted = []
    for receipt in receipts:
        source_path = receipt["source_path"]
        relative = receipt["path"]
        source = payload.parent / source_path
        destination = evidence_dir / relative
        _copy_verified_file(source, destination, receipt["sha256"], f"bundle receipt {source_path}")
        persisted.append(
            {
                "source_path": source_path,
                "path": destination.resolve().relative_to(corpus_root.resolve()).as_posix(),
                "sha256": receipt["sha256"],
            }
        )
    return persisted


def _copy_verified_file(source: Path, destination: Path, expected_sha256: str, label: str) -> None:
    if not source.is_file() or _sha256_file(source) != expected_sha256:
        raise CorpusError(f"{label} does not match its verified source checksum")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != expected_sha256:
            raise CorpusError(f"corpus path conflicts with {label}")
    else:
        shutil.copyfile(source, destination)
    if _sha256_file(destination) != expected_sha256:
        raise CorpusError(f"corpus copy changed {label}")


def _build_issue9645_historical_case(
    payload: Path,
) -> tuple[dict[str, Any], dict[str, Path], list[dict[str, Any]]]:
    metadata = _read_json_object(payload / "run_metadata.json")
    _verify_issue9645_bundle(
        payload,
        source_revision=str(metadata.get("experiment_source_commit") or ""),
        source_file_hashes=metadata.get("source_file_sha256"),
        required_payload_paths=[
            "historical_issue_1501_failure_0002.json",
            "historical_issue_1501_failure_0002/scenario.yaml",
            "historical_issue_1501_failure_0002/route_overrides.yaml",
            "historical_issue_1501_failure_0002/replay_1.jsonl",
            "historical_issue_1501_failure_0002/replay_1.provenance.json",
            "historical_issue_1501_failure_0002/replay_2.jsonl",
            "historical_issue_1501_failure_0002/replay_2.provenance.json",
            "path_normalization.json",
            "replay_validation.json",
        ],
    )
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


def _validate_replay_artifact(
    item: Mapping[str, Any],
    case: Mapping[str, Any],
    receipt: Mapping[str, Any],
    corpus_root: Path,
) -> list[str]:
    """Verify a stored one-episode artifact against its receipt, case, and evaluation."""
    errors = _evaluation_receipt_binding_errors(item, case, receipt)
    try:
        _scenario_input_paths(case, corpus_root)
    except (CorpusError, OSError, ValueError, TypeError) as exc:
        errors.append(f"case_input_binding_invalid:{exc}")
    try:
        artifact = _resolve_corpus_artifact(receipt.get("artifact_path"), corpus_root)
    except CorpusError as exc:
        errors.append(f"replay_artifact_path_invalid:{exc}")
        return errors
    artifact_sha256 = _sha256_file(artifact) if artifact.is_file() else None
    if artifact_sha256 is None:
        errors.append("replay_artifact_missing")
        return errors
    if artifact_sha256 != receipt.get("artifact_sha256"):
        errors.append("replay_artifact_checksum_mismatch")
        return errors
    if artifact_sha256 != item.get("episode_sha256"):
        errors.append("episode_sha256_does_not_match_replay_artifact")
        return errors
    try:
        record = _read_single_jsonl_record(artifact)
        event_identity = _selected_event_identity(record)
    except CorpusError as exc:
        errors.append(f"replay_artifact_invalid:{exc}")
        return errors
    errors.extend(_validate_replay_record_projection(item, case, receipt, record, event_identity))
    errors.extend(_validate_replay_execution_evidence(item, record))
    return errors


def _validate_replay_record_projection(
    item: Mapping[str, Any],
    case: Mapping[str, Any],
    receipt: Mapping[str, Any],
    record: Mapping[str, Any],
    event_identity: Mapping[str, Any],
) -> list[str]:
    errors = []
    if record.get("scenario_id") != case.get("scenario_id"):
        errors.append("replay_artifact_scenario_id_mismatch")
    if isinstance(record.get("seed"), bool) or record.get("seed") != case.get("scenario_seed"):
        errors.append("replay_artifact_scenario_seed_mismatch")
    metadata = record.get("algorithm_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if record.get("algo") != item.get("planner_id"):
        errors.append("replay_artifact_planner_id_mismatch")
    if metadata.get("config_hash") != item.get("planner_config_identity"):
        errors.append("replay_artifact_planner_config_mismatch")
    if record.get("git_hash") != item.get("source_revision"):
        errors.append("replay_artifact_source_revision_mismatch")
    if not _is_full_git_revision(item.get("source_revision")):
        errors.append("replay_artifact_source_revision_invalid")

    artifact_outcome = record.get("outcome")
    if not isinstance(artifact_outcome, dict) or any(
        artifact_outcome.get(key) != item.get("outcome", {}).get(key)
        for key in ("collision_event", "route_complete", "timeout_event")
    ):
        errors.append("replay_artifact_outcome_mismatch")
    if record.get("termination_reason") != item.get("termination_reason"):
        errors.append("replay_artifact_termination_reason_mismatch")
    errors.extend(_replay_episode_status_projection_errors(record))
    artifact_metrics = record.get("metrics")
    if not isinstance(artifact_metrics, dict) or any(
        artifact_metrics.get(key) != value for key, value in item.get("metrics", {}).items()
    ):
        errors.append("replay_artifact_selected_metrics_mismatch")
    errors.extend(_validate_replay_event_projection(item, case, receipt, event_identity))
    return errors


def _replay_episode_status_projection_errors(record: Mapping[str, Any]) -> list[str]:
    termination_reason = record.get("termination_reason")
    if not isinstance(termination_reason, str) or termination_reason not in TERMINATION_REASONS:
        return ["replay_artifact_termination_reason_unsupported"]
    if record.get("status") != status_from_termination_reason(termination_reason):
        return ["replay_artifact_status_termination_mismatch"]
    return []


def _validate_replay_event_projection(
    item: Mapping[str, Any],
    case: Mapping[str, Any],
    receipt: Mapping[str, Any],
    event_identity: Mapping[str, Any],
) -> list[str]:
    errors = []
    if event_identity != receipt.get("selected_event_identity"):
        errors.append("replay_artifact_event_identity_mismatch")
    expected_event_outcome = {
        "collision": item.get("outcome", {}).get("collision_event"),
        "goal_reached": item.get("outcome", {}).get("route_complete"),
        "timeout": item.get("outcome", {}).get("timeout_event"),
    }
    if event_identity.get("exact_events") != expected_event_outcome:
        errors.append("replay_artifact_event_outcome_mismatch")
    if event_identity.get("scenario_id") != case.get("scenario_id") or event_identity.get(
        "seed"
    ) != case.get("scenario_seed"):
        errors.append("replay_artifact_event_case_identity_mismatch")
    if event_identity.get("planner_id") != item.get("planner_id") or event_identity.get(
        "source_revision"
    ) != item.get("source_revision"):
        errors.append("replay_artifact_event_planner_identity_mismatch")
    return errors


def _validate_replay_execution_evidence(
    item: Mapping[str, Any], record: Mapping[str, Any]
) -> list[str]:
    errors = _replay_episode_validity_errors(record)
    metadata = record.get("algorithm_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    kinematics = metadata.get("planner_kinematics")
    kinematics = kinematics if isinstance(kinematics, dict) else {}
    actual_mode = kinematics.get("execution_mode")
    if actual_mode != item.get("execution_mode"):
        errors.append("replay_artifact_execution_mode_mismatch")
    expected_readiness = {
        "native": "native",
        "adapter": "adapter",
        "mixed": "adapter",
    }.get(actual_mode)
    if expected_readiness is None:
        errors.append("replay_artifact_execution_mode_unsupported")
    else:
        recorded_readiness = record.get("readiness_status") or metadata.get(
            "readiness_status", metadata.get("benchmark_readiness_status")
        )
        if recorded_readiness is not None and recorded_readiness != expected_readiness:
            errors.append("replay_artifact_readiness_mode_mismatch")
        if item.get("readiness_status") != expected_readiness:
            errors.append("replay_artifact_readiness_not_benchmark_capable")

    benchmark_availability = record.get("benchmark_availability")
    benchmark_availability = (
        benchmark_availability if isinstance(benchmark_availability, dict) else {}
    )
    recorded_availability = (
        record.get("availability_status")
        or benchmark_availability.get("availability_status")
        or metadata.get("availability_status")
    )
    if recorded_availability is not None and recorded_availability != "available":
        errors.append("replay_artifact_availability_not_available")
    elif metadata.get("status") not in {"ok", "available"}:
        errors.append("replay_artifact_planner_status_not_available")
    if item.get("availability_status") != "available":
        errors.append("replay_artifact_evaluation_availability_not_available")

    actual_fallback_or_degraded = _replay_fallback_marker(record, metadata)
    if actual_fallback_or_degraded is None:
        errors.append("replay_artifact_fallback_status_missing")
    elif item.get("fallback_or_degraded") is not actual_fallback_or_degraded:
        errors.append("replay_artifact_fallback_status_mismatch")
    return errors


def _replay_episode_validity_errors(record: Mapping[str, Any]) -> list[str]:
    if _replay_episode_is_invalid(record):
        return ["replay_artifact_invalid_run"]
    return []


def _replay_episode_is_invalid(record: Mapping[str, Any]) -> bool:
    status = record.get("status")
    if not isinstance(status, str):
        return True
    event_ledger = record.get("event_ledger")
    event_ledger = event_ledger if isinstance(event_ledger, dict) else {}
    exact_events = event_ledger.get("exact_events")
    exact_events = exact_events if isinstance(exact_events, dict) else {}
    return (
        status in {"invalid", "error"}
        or record.get("termination_reason") == "error"
        or exact_events.get("invalid_run") is not False
    )


def _replay_fallback_marker(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool | None:
    integrity = record.get("integrity")
    integrity = integrity if isinstance(integrity, dict) else {}
    effective_view = integrity.get("effective_view", {})
    effective_view = effective_view if isinstance(effective_view, dict) else {}
    runtime_metadata = {
        key: value
        for key, value in {
            "status": metadata.get("status"),
            "readiness_status": metadata.get("readiness_status"),
            "availability_status": metadata.get("availability_status"),
            "fallback_or_degraded": metadata.get("fallback_or_degraded"),
            "planner_runtime": metadata.get("planner_runtime"),
        }.items()
        if value is not None
    }
    execution_fields = {
        "algorithm_metadata": runtime_metadata,
        "integrity": {"effective_view": effective_view},
    }
    for key in (
        "benchmark_availability",
        "readiness_status",
        "availability_status",
        "fallback_or_degraded",
    ):
        if record.get(key) is not None:
            execution_fields[key] = record[key]
    if runtime_fallback_or_degraded_marker(execution_fields) is not None:
        return True
    explicit_marker = record.get("fallback_or_degraded", metadata.get("fallback_or_degraded"))
    degraded = effective_view.get("degraded")
    if isinstance(degraded, bool):
        return degraded
    if isinstance(explicit_marker, bool):
        return explicit_marker
    return None


def _evaluation_receipt_binding_errors(
    item: Mapping[str, Any], case: Mapping[str, Any], receipt: Mapping[str, Any]
) -> list[str]:
    if not isinstance(receipt, Mapping):
        return ["replay_receipt_missing"]
    errors = []
    if receipt.get("schema_version") != EVALUATION_REPLAY_RECEIPT_SCHEMA_VERSION:
        errors.append("replay_receipt_schema_invalid")
    if receipt.get("verification_status") != "artifact_projection_match":
        errors.append("replay_receipt_verification_status_invalid")
    errors.extend(_validate_replay_receipt_bindings(item, case, receipt))
    errors.extend(_validate_replay_receipt_artifact_fields(item, receipt))
    errors.extend(_validate_replay_receipt_case_fields(receipt))
    return errors


def _validate_replay_receipt_bindings(
    item: Mapping[str, Any], case: Mapping[str, Any], receipt: Mapping[str, Any]
) -> list[str]:
    inputs = case.get("inputs")
    inputs = inputs if isinstance(inputs, dict) else {}
    expected = {
        "case_id": item.get("case_id"),
        "effective_scenario_sha256": item.get("effective_scenario_sha256"),
        "scenario_id": case.get("scenario_id"),
        "scenario_seed": case.get("scenario_seed"),
        "scenario_input_sha256": inputs.get("scenario_sha256"),
        "route_overrides_sha256": inputs.get("route_overrides_sha256"),
        "planner_id": item.get("planner_id"),
        "planner_config_identity": item.get("planner_config_identity"),
        "source_revision": item.get("source_revision"),
        "episode_sha256": item.get("episode_sha256"),
        "outcome": item.get("outcome"),
        "termination_reason": item.get("termination_reason"),
        "metrics": item.get("metrics"),
        "execution_mode": item.get("execution_mode"),
        "readiness_status": item.get("readiness_status"),
        "availability_status": item.get("availability_status"),
        "fallback_or_degraded": item.get("fallback_or_degraded"),
    }
    errors = []
    for field, value in expected.items():
        if receipt.get(field) != value:
            errors.append(f"replay_receipt_{field}_mismatch")
    return errors


def _validate_replay_receipt_artifact_fields(
    item: Mapping[str, Any], receipt: Mapping[str, Any]
) -> list[str]:
    errors = []
    if not _is_sha256(receipt.get("artifact_sha256")):
        errors.append("replay_receipt_artifact_sha256_invalid")
    if receipt.get("artifact_sha256") != item.get("episode_sha256"):
        errors.append("replay_receipt_episode_sha256_mismatch")
    for field in ("scenario_input_sha256", "route_overrides_sha256"):
        if not _is_sha256(receipt.get(field)):
            errors.append(f"replay_receipt_{field}_invalid")
    if not isinstance(receipt.get("artifact_path"), str) or not receipt["artifact_path"]:
        errors.append("replay_receipt_artifact_path_missing")
    return errors


def _validate_replay_receipt_case_fields(receipt: Mapping[str, Any]) -> list[str]:
    errors = []
    selected_event_identity = receipt.get("selected_event_identity")
    if not isinstance(selected_event_identity, dict):
        errors.append("replay_receipt_selected_event_identity_missing")
    else:
        episode_status = selected_event_identity.get("episode_status")
        if not isinstance(episode_status, str) or not episode_status.strip():
            errors.append("replay_receipt_episode_status_missing")
        if not isinstance(selected_event_identity.get("invalid_run"), bool):
            errors.append("replay_receipt_invalid_run_missing")
    if not isinstance(receipt.get("scenario_id"), str) or not receipt["scenario_id"].strip():
        errors.append("replay_receipt_scenario_id_missing")
    if not isinstance(receipt.get("scenario_seed"), int) or isinstance(
        receipt.get("scenario_seed"), bool
    ):
        errors.append("replay_receipt_scenario_seed_invalid")
    return errors


def _selected_event_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    ledger = record.get("event_ledger")
    if not isinstance(ledger, dict):
        raise CorpusError("replay episode event ledger is missing")
    exact_events = ledger.get("exact_events")
    event_fields = ("collision", "goal_reached", "timeout")
    if not isinstance(exact_events, dict) or any(
        not isinstance(exact_events.get(key), bool) for key in event_fields
    ):
        raise CorpusError("replay episode exact event identities are incomplete")
    if not isinstance(exact_events.get("invalid_run"), bool):
        raise CorpusError("replay episode invalid-run event identity is missing")
    episode_status = record.get("status")
    if not isinstance(episode_status, str) or not episode_status.strip():
        raise CorpusError("replay episode status is missing")
    return {
        "schema_version": ledger.get("schema_version"),
        "scenario_id": ledger.get("scenario_id"),
        "seed": ledger.get("seed"),
        "planner_id": ledger.get("planner"),
        "source_revision": ledger.get("software_commit"),
        "episode_status": episode_status,
        "invalid_run": exact_events["invalid_run"],
        "exact_events": {key: exact_events[key] for key in event_fields},
    }


def _resolve_corpus_artifact(path_value: Any, corpus_root: Path) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise CorpusError("replay artifact path must be a non-empty relative path")
    relative = PurePosixPath(path_value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise CorpusError(f"unsafe replay artifact path: {path_value}")
    path = (corpus_root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(corpus_root.resolve())
    except ValueError as exc:
        raise CorpusError(f"replay artifact path escapes corpus root: {path_value}") from exc
    if not path.is_file():
        raise CorpusError(f"replay artifact is missing: {path_value}")
    return path


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
    if not _is_full_git_revision(item.get("source_revision")):
        errors.append("source_revision_invalid")
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
    receipt = item.get("replay_receipt")
    if isinstance(receipt, dict):
        errors.extend(
            _evaluation_receipt_binding_errors(
                item,
                {
                    "scenario_id": receipt.get("scenario_id"),
                    "scenario_seed": receipt.get("scenario_seed"),
                    "inputs": {
                        "scenario_sha256": receipt.get("scenario_input_sha256"),
                        "route_overrides_sha256": receipt.get("route_overrides_sha256"),
                    },
                },
                receipt,
            )
        )
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


def _evaluation_state(
    item: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    corpus_root: Path | None,
) -> dict[str, Any]:
    reasons = _validate_evaluation(item)
    if reasons:
        return {"status": "unknown", "reason_codes": reasons}
    if item["evidence_status"] != "complete":
        return {
            "status": "unknown",
            "reason_codes": [f"evaluation_evidence_{item['evidence_status']}"],
        }
    if not isinstance(item.get("replay_receipt"), Mapping):
        return {"status": "unknown", "reason_codes": ["replay_receipt_missing"]}
    if corpus_root is None:
        return {"status": "unknown", "reason_codes": ["replay_artifact_root_required"]}
    replay_errors = _validate_replay_artifact(item, case, item.get("replay_receipt"), corpus_root)
    if replay_errors:
        return {"status": "unknown", "reason_codes": sorted(set(replay_errors))}
    reasons = _complete_evaluation_eligibility_errors(item)
    if reasons:
        return {"status": "unknown", "reason_codes": sorted(set(reasons))}
    return _complete_outcome_state(item)


def _complete_evaluation_eligibility_errors(item: Mapping[str, Any]) -> list[str]:
    reasons = []
    expected_readiness = {
        "native": "native",
        "adapter": "adapter",
        "mixed": "adapter",
    }.get(item["execution_mode"])
    if expected_readiness is None:
        reasons.append("execution_mode_not_benchmark_capable")
    elif item["readiness_status"] != expected_readiness:
        reasons.append("readiness_not_benchmark_capable")
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


def _is_full_git_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )
