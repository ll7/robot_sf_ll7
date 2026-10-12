"""Append-only observed-window corpus using the existing persistence replay gate.

Admission verifies an observed critical case, not perturbation robustness or
mathematical feasibility. Unknown feasibility is retained. No simulation runs
here, and evidence is never replaced on exact duplicate admission.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.benchmark.scenario_generation.persistence_gate import (
    ScenarioPersistenceValidationError,
    UnsupportedWindowAdmissionError,
    resolve_bound_window_artifact,
    verify_window_replay_receipt,
)

CASE_SCHEMA = "counterexample-case.v1"
CORPUS_SCHEMA = "counterexample-corpus.v1"
_SCHEMA = Path(__file__).resolve().parents[1] / "benchmark/schemas/counterexample_case.v1.json"


def _digest(value: Any) -> str:
    """Return the canonical content identity."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_case(record: dict[str, Any]) -> None:
    """Reject malformed records and inconsistent immutable case identities."""
    Draft202012Validator(json.loads(_SCHEMA.read_text())).validate(record)
    if record["case_id"] != "case-" + _digest(record["identity"]):
        raise ScenarioPersistenceValidationError("corpus case identity mismatch")
    receipt = record["window_receipt"]
    identity = record["identity"]
    if (
        identity["source_commit"] != receipt["source_commit"]
        or identity["source_config_hash"] != receipt["source_config_hash"]
        or identity["model_digest"] != receipt["planner"]["model"]["digest"]
        or identity["planner"] != receipt["planner"]["id"]
        or identity["seed"] != receipt["seed"]
        or identity["window"] != receipt["window"]
        or identity["map_sha256"] != receipt["bindings"]["map"]["sha256"]
        or identity["scenario_sha256"] != receipt["bindings"]["scenario"]["sha256"]
        or identity["source_window_trace_sha256"]
        != receipt["parity_proof"]["source_window_trace_sha256"]
    ):
        raise ScenarioPersistenceValidationError("corpus case/receipt identity mismatch")


def _atomic_json(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    """Commit a complete file, refusing to replace an immutable case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read historical immutable records without certifying current admissibility.

    This remains available for forensic inspection. Admission and replay export
    must establish authority independently; a historical status is not proof.
    """
    records = []
    index = corpus_dir / "corpus.json"
    if index.exists() and json.loads(index.read_text()).get("schema_version") != CORPUS_SCHEMA:
        raise ScenarioPersistenceValidationError("unsupported corpus index schema")
    for path in sorted((corpus_dir / "cases").glob("case-*.json")):
        if path.name.endswith(".review.json"):
            continue
        record = json.loads(path.read_text())
        validate_case(record)
        if path.stem != record["case_id"]:
            raise ScenarioPersistenceValidationError("corpus case filename mismatch")
        records.append(record)
    return records


def _write_index(corpus_dir: Path) -> dict[str, Any]:
    """Rebuild the deterministic index, including after interrupted admission."""
    index = {
        "schema_version": CORPUS_SCHEMA,
        "claim_boundary": "observed simulator regression cases only",
        "cases": [
            {
                "case_id": r["case_id"],
                "path": f"cases/{r['case_id']}.json",
                "canonical_record_sha256": _digest(r),
            }
            for r in read_corpus(corpus_dir)
        ],
    }
    path = corpus_dir / "corpus.json"
    if path.exists() and json.loads(path.read_text()) == index:
        return index
    _atomic_json(path, index)
    return index


def admit_window(
    candidate_path: Path, receipt_path: Path, *, corpus_dir: Path, repo_root: Path
) -> dict[str, Any]:
    """Admit a hash-bound window only after recomputing both replay proofs.

    Returns:
        Admission status, stable case identity and deterministic near-duplicates.
    """
    candidate = json.loads(candidate_path.read_text())
    if candidate.get("schema_version") != "counterexample-candidate.v1":
        raise ScenarioPersistenceValidationError("unsupported corpus candidate schema")
    if hashlib.sha256(receipt_path.read_bytes()).hexdigest() != candidate.get(
        "window_receipt_sha256"
    ):
        raise ScenarioPersistenceValidationError(
            "corpus window parity receipt digest missing/mismatched"
        )
    verification = verify_window_replay_receipt(receipt_path, repo_root=repo_root)
    receipt = verification["receipt"]
    feasibility = candidate.get("feasibility", {})
    if feasibility.get("status") not in {"unknown", "admissible"} or not feasibility.get("reason"):
        raise ScenarioPersistenceValidationError(
            "corpus feasibility/admissibility verdict required"
        )
    discovery = candidate.get("discovery", {})
    if (
        discovery.get("kind") != "known_failure_positive_control"
        or discovery.get("round") != 0
        or discovery.get("objective") != "near_miss"
        or not discovery.get("parameters")
        or not discovery.get("budget")
    ):
        raise ScenarioPersistenceValidationError("corpus discovery provenance incomplete")
    parameters = discovery["parameters"]
    budget = discovery["budget"]
    if (
        parameters.get("seed") != receipt["seed"]
        or parameters.get("scenario") != verification["source_scenario_id"]
        or parameters.get("window_start_s") != receipt["window"]["start_s"]
        or parameters.get("window_end_s") != receipt["window"]["end_s"]
        or budget.get("window_transitions") != receipt["window"]["steps"]
        or budget.get("source_episode_steps") != verification["original_budget_steps"]
        or budget.get("new_search_evaluations") != 0
    ):
        raise ScenarioPersistenceValidationError("corpus discovery parameters/budget mismatch")
    manifest_path = resolve_bound_window_artifact(repo_root.resolve(), discovery["source_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("source_commit") != receipt["source_commit"]
        or manifest["planner"]["run_config_hash"] != receipt["source_config_hash"]
        or manifest["planner"]["analytic_implementation_sha256"]
        != receipt["planner"]["model"]["digest"]
        or manifest["seed_policy"]["episode_seeds"] != [receipt["seed"]]
    ):
        raise ScenarioPersistenceValidationError("corpus discovery/source provenance mismatch")
    identity = {
        "source_commit": receipt["source_commit"],
        "source_config_hash": receipt["source_config_hash"],
        "planner": receipt["planner"]["id"],
        "model_digest": receipt["planner"]["model"]["digest"],
        "seed": receipt["seed"],
        "window": receipt["window"],
        "map_sha256": receipt["bindings"]["map"]["sha256"],
        "scenario_sha256": receipt["bindings"]["scenario"]["sha256"],
        "source_window_trace_sha256": receipt["parity_proof"]["source_window_trace_sha256"],
    }
    case_id = "case-" + _digest(identity)
    record = {
        "schema_version": CASE_SCHEMA,
        "admission_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "case_id": case_id,
        "identity": identity,
        "admission_status": "admitted",
        "claim_boundary": "observed simulator regression case only",
        "benchmark_evidence": False,
        "feasibility": feasibility,
        "discovery": discovery,
        "window_receipt": receipt,
        "replay_verification": {k: v for k, v in verification.items() if k != "receipt"},
        "perturbation_persistence": {
            "status": "unknown",
            "reason": "independent grid not executed",
        },
        "planner_status": {receipt["planner"]["id"]: "observed_critical"},
    }
    validate_case(record)
    # Lock only the short local commit/index section; all expensive verification
    # happens before it. Cases are the authority if a process dies before index.
    import fcntl  # noqa: PLC0415 - repository admission is a POSIX local-file operation

    corpus_dir.mkdir(parents=True, exist_ok=True)
    with (corpus_dir / ".admission.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        prior = read_corpus(corpus_dir)
        duplicate = next((r for r in prior if r["case_id"] == case_id), None)
        near = sorted(
            r["case_id"]
            for r in prior
            if r["case_id"] != case_id
            and r["identity"]["map_sha256"] == identity["map_sha256"]
            and r["identity"]["seed"] == identity["seed"]
            and r["identity"]["window"]["start_s"] < identity["window"]["end_s"]
            and identity["window"]["start_s"] < r["identity"]["window"]["end_s"]
        )
        if duplicate is None:
            _atomic_json(corpus_dir / "cases" / f"{case_id}.json", record, exclusive=True)
        _write_index(corpus_dir)
    return {
        "schema_version": "counterexample-admission.v1",
        "case_id": case_id,
        "status": "already_admitted" if duplicate else "admitted",
        "near_duplicates": near,
    }


def export_replay_slice(corpus_dir: Path) -> dict[str, Any]:
    """Refuse legacy admissions; an old recorded pass cannot authorize replay export."""
    records = read_corpus(corpus_dir)
    if records:
        raise UnsupportedWindowAdmissionError(
            "historical corpus cases lack checkpoint-owned admission authority; "
            "replay export is disabled pending a complete capture/reconstruction contract"
        )
    return {
        "schema_version": "counterexample-replay-slice.v1",
        "cases": [
            {
                "case_id": r["case_id"],
                "scenario": r["window_receipt"]["bindings"]["scenario"],
                "planner": r["window_receipt"]["planner"],
                "seed": r["identity"]["seed"],
                "window": r["identity"]["window"],
                "runner_options": {
                    "horizon": r["replay_verification"]["original_budget_steps"],
                    "dt": r["identity"]["window"]["dt_s"],
                    "algo": r["identity"]["planner"],
                    "algo_config_path": r["window_receipt"]["bindings"]["planner_config"]["path"],
                    "record_forces": True,
                    "record_simulation_step_trace": True,
                    "benchmark_profile": "experimental",
                    "workers": 1,
                    "resume": False,
                },
            }
            for r in records
        ],
    }
