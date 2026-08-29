#!/usr/bin/env python3
"""Run the frozen matched-compute production-observed canary (issue #7893).

Executes exactly one canary per arm (open-loop Social Force scenario search
and reactive residual search) through the real repository seams, without
injected evaluators, test fixtures, controller-only snapshots, fallback, or
degraded execution, and emits complete production-observed candidate and
simulator-step provenance.

This is a bounded local execution-and-evidence-capture slice.  It does not
authorize a comparison campaign, SLURM submission, optimizer expansion,
planner ranking, or claim promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping

import yaml

from robot_sf.adversarial.config import SearchConfig
from robot_sf.adversarial.matched_compute import (
    MatchedComputeRuntimeTrace,
    open_loop_runtime_trace_from_result,
)
from robot_sf.adversarial.search import run_adversarial_search
from robot_sf.benchmark.schema_validator import load_schema
from robot_sf.benchmark.termination_reason import outcome_contradictions
from robot_sf.evidence.writers import write_json
from robot_sf.ped_npc.residual_adversary import BoundedResidualAdversary
from robot_sf.ped_npc.residual_search import FiniteGridSearchPolicy

CANDIDATE_RECORD_SCHEMA = "matched_compute_candidate_record.v1"
CANDIDATE_STATUSES = frozenset(
    {"accepted", "rejected", "failed", "invalid", "fallback", "unavailable"}
)
OBSERVED_STEP_SOURCES = frozenset({"observed_episode_record", "observed_simulator"})
RECEIPT_SCHEMA = "matched_compute_production_canary.v1"
FROZEN_INPUTS_SCHEMA = "matched_compute_frozen_inputs.v1"
DEFAULT_PACKET_PATH = Path("configs/adversarial/issue_6921_matched_compute_packet.yaml")
DEFAULT_INPUT_DIGESTS_PATH = Path(
    "docs/context/evidence/issue_7893_matched_compute_production_canary/input_digests.json"
)
ISSUE_OUTPUT_SCOPE = Path("output/matched_compute_canary")
EPISODE_SCHEMA_PATH = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")
_CERTIFICATION_STATUS_KEYS = frozenset({"availability", "execution", "mode", "readiness", "status"})
_RUNTIME_TRACE_FIELDS = tuple(field.name for field in fields(MatchedComputeRuntimeTrace))
_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")


def _digest_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_text(text: str) -> str:
    """Return the SHA-256 digest of a text payload."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_sha256_hex(value: Any) -> bool:
    """Return whether a value is a lowercase hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX_DIGITS for character in value)
    )


def _repository_root() -> Path:
    """Return the repository root containing this checked-in entry point."""
    return Path(__file__).resolve().parents[2]


def _lexical_repository_path(path: Path, repository_root: Path) -> Path:
    """Return an absolute normalized path without following symlinks."""
    candidate = path if path.is_absolute() else repository_root / path
    return Path(os.path.abspath(candidate))


def _symlink_component(path: Path, repository_root: Path) -> Path | None:
    """Return the first symlink component from the repository to ``path``."""
    try:
        relative = path.relative_to(repository_root)
    except ValueError:
        return None
    current = repository_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return current
    return None


def _absolute_symlink_component(path: Path) -> Path | None:
    """Return the first symlink component in an absolute lexical path."""
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            return current
    return None


def _output_directory_member_problem(member: Path) -> str | None:
    """Return a safety problem for one pre-existing output directory member."""
    if member.is_symlink():
        return f"output directory contains symlink member: {member}"
    if not member.is_dir():
        return f"output directory contains non-directory member: {member}"
    return None


def _output_file_member_problem(member: Path) -> str | None:
    """Return a safety problem for one pre-existing output file member."""
    if member.is_symlink():
        return f"output directory contains symlink member: {member}"
    if not member.is_file():
        return f"output directory contains non-regular member: {member}"
    if member.stat().st_nlink != 1:
        return f"output directory contains hard-linked file: {member}"
    return None


def _existing_output_tree_problem(output_dir: Path) -> str | None:
    """Reject unsafe pre-existing members that an arm could overwrite."""
    if not output_dir.exists():
        return None
    try:
        for current_root, directory_names, file_names in os.walk(output_dir, followlinks=False):
            current = Path(current_root)
            for name in directory_names:
                if problem := _output_directory_member_problem(current / name):
                    return problem
            for name in file_names:
                if problem := _output_file_member_problem(current / name):
                    return problem
    except OSError as exc:
        return f"cannot inspect existing output directory: {exc}"
    return None


def _git_destination_check(
    repository_root: Path, relative_path: Path, *, command: str
) -> subprocess.CompletedProcess[str]:
    """Run one read-only Git destination query or fail closed."""
    args = (
        ["git", "ls-files", "-z", "--", relative_path.as_posix()]
        if command == "tracked"
        else ["git", "check-ignore", "-q", "--no-index", "--", relative_path.as_posix()]
    )
    try:
        return subprocess.run(
            args,
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ValueError(f"cannot verify {command} destination state: {exc}") from exc


def _validate_destination_scope(
    label: str, destination: Path, repository_root: Path, allowed_scope: Path
) -> None:
    """Require one lexical and resolved destination to stay in the issue scope."""
    try:
        destination.relative_to(allowed_scope)
    except ValueError as exc:
        raise ValueError(
            f"{label} is outside the issue-scoped {ISSUE_OUTPUT_SCOPE.as_posix()} tree"
        ) from exc
    symlink = _symlink_component(destination, repository_root)
    if symlink is not None:
        raise ValueError(f"{label} contains symlink component: {symlink}")
    try:
        destination.resolve(strict=False).relative_to(allowed_scope.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the issue-scoped output tree") from exc


def _validate_destination_git_state(label: str, destination: Path, repository_root: Path) -> None:
    """Require one destination to be untracked and ignored."""
    relative = destination.relative_to(repository_root)
    tracked = _git_destination_check(repository_root, relative, command="tracked")
    if tracked.returncode != 0:
        raise ValueError(f"cannot verify tracked state for {label}")
    if tracked.stdout:
        raise ValueError(f"{label} contains or names a tracked path")
    ignored = _git_destination_check(repository_root, relative, command="ignored")
    if ignored.returncode not in {0, 1}:
        raise ValueError(f"cannot verify ignored state for {label}")
    if ignored.returncode != 0:
        raise ValueError(f"{label} is not ignored by repository policy")


def _validate_execution_destinations(
    output_dir: Path, receipt_path: Path, repository_root: Path
) -> tuple[Path, Path]:
    """Admit only ignored, untracked destinations in the issue-scoped output tree."""
    root = repository_root.resolve()
    allowed_scope = _lexical_repository_path(ISSUE_OUTPUT_SCOPE, root)
    destinations = {
        "output directory": _lexical_repository_path(output_dir, root),
        "receipt": _lexical_repository_path(receipt_path, root),
    }
    for label, destination in destinations.items():
        _validate_destination_scope(label, destination, root, allowed_scope)
        _validate_destination_git_state(label, destination, root)

    resolved_output = destinations["output directory"]
    resolved_receipt = destinations["receipt"]
    if resolved_output.exists() and not resolved_output.is_dir():
        raise ValueError("output directory must be a directory when it exists")
    if resolved_receipt.exists() and not resolved_receipt.is_file():
        raise ValueError("receipt must be a regular file when it exists")
    if resolved_receipt.exists() and resolved_receipt.stat().st_nlink != 1:
        raise ValueError("receipt must not be a hard-linked file")
    output_tree_problem = _existing_output_tree_problem(resolved_output)
    if output_tree_problem is not None:
        raise ValueError(output_tree_problem)
    if resolved_receipt == resolved_output or resolved_output.is_relative_to(resolved_receipt):
        raise ValueError("receipt must not be the output directory or its ancestor")
    if not resolved_output.is_relative_to(resolved_receipt.parent):
        raise ValueError("receipt parent must contain the execution output artifact bundle")
    return resolved_output, resolved_receipt


def _resolve_repository_file(path: Path, repository_root: Path, *, label: str) -> tuple[str, Path]:
    """Resolve a repository-relative file while rejecting unsafe references."""
    root = repository_root.resolve()
    if path.is_absolute():
        candidate = path
    else:
        candidate = root / path
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the repository: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    return relative.as_posix(), resolved


def _resolve_yaml_reference(
    raw_reference: str, current_file: Path, repository_root: Path
) -> tuple[str, Path]:
    """Resolve a YAML reference using repository or declaring-file semantics."""
    reference = Path(raw_reference)
    if reference.is_absolute():
        return _resolve_repository_file(reference, repository_root, label="YAML reference")

    candidates: list[tuple[str, Path]] = []
    for base in (repository_root, current_file.parent):
        try:
            resolved = _resolve_repository_file(
                base / reference, repository_root, label="YAML reference"
            )
        except ValueError:
            continue
        if resolved not in candidates:
            candidates.append(resolved)
    if not candidates:
        raise ValueError(
            f"YAML reference from {current_file} is missing or unsafe: {raw_reference}"
        )
    if len(candidates) > 1:
        raise ValueError(f"YAML reference from {current_file} is ambiguous: {raw_reference}")
    return candidates[0]


def _yaml_file_values(value: Any) -> list[str]:
    """Collect YAML file references from a decoded configuration value."""
    if isinstance(value, dict):
        values: list[str] = []
        for nested in value.values():
            values.extend(_yaml_file_values(nested))
        return values
    if isinstance(value, list):
        values = []
        for nested in value:
            values.extend(_yaml_file_values(nested))
        return values
    if isinstance(value, str) and Path(value).suffix.lower() in {".yaml", ".yml"}:
        return [value]
    return []


def _discover_frozen_input_files(packet_path: Path, repository_root: Path) -> dict[str, Path]:
    """Discover the packet and every YAML file it references, recursively."""
    packet_relative, packet_file = _resolve_repository_file(
        packet_path, repository_root, label="packet"
    )
    discovered: dict[str, Path] = {packet_relative: packet_file}
    pending = [packet_file]
    while pending:
        current = pending.pop()
        try:
            payload = yaml.safe_load(current.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise ValueError(f"cannot parse frozen YAML input {current}") from exc
        for raw_reference in _yaml_file_values(payload):
            relative, resolved = _resolve_yaml_reference(raw_reference, current, repository_root)
            if relative not in discovered:
                discovered[relative] = resolved
                pending.append(resolved)
    return dict(sorted(discovered.items()))


def _load_frozen_input_expectations(
    path: Path, repository_root: Path, packet_relative: str
) -> dict[str, str]:
    """Load and validate the checked-in expected digest map."""
    if not path.is_absolute():
        path = repository_root / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read frozen input digest lock: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != FROZEN_INPUTS_SCHEMA:
        raise ValueError(f"frozen input digest lock must use {FROZEN_INPUTS_SCHEMA}")
    if payload.get("packet") != packet_relative:
        raise ValueError("frozen input digest lock packet does not match the requested packet")
    raw_files = payload.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise ValueError("frozen input digest lock must contain a non-empty files mapping")
    expectations: dict[str, str] = {}
    for raw_path, raw_digest in raw_files.items():
        if not isinstance(raw_path, str) or not isinstance(raw_digest, str):
            raise ValueError("frozen input digest lock entries must be string pairs")
        relative, _ = _resolve_repository_file(
            Path(raw_path), repository_root, label="frozen input digest lock entry"
        )
        if relative != raw_path:
            raise ValueError("frozen input digest lock paths must be repository-relative")
        if len(raw_digest) != 64 or any(char not in "0123456789abcdef" for char in raw_digest):
            raise ValueError(f"invalid SHA-256 digest in frozen input lock: {raw_path}")
        expectations[relative] = raw_digest
    return dict(sorted(expectations.items()))


def _verify_frozen_inputs(
    packet_path: Path,
    *,
    repository_root: Path | None = None,
    digest_lock_path: Path | None = None,
    expected_digests: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Verify packet and recursively referenced YAML bytes before execution."""
    root = (repository_root or _repository_root()).resolve()
    packet_relative, _ = _resolve_repository_file(packet_path, root, label="packet")
    discovered = _discover_frozen_input_files(packet_path, root)
    expected = (
        dict(expected_digests)
        if expected_digests is not None
        else _load_frozen_input_expectations(
            digest_lock_path or root / DEFAULT_INPUT_DIGESTS_PATH, root, packet_relative
        )
    )
    expected = dict(sorted(expected.items()))
    discovered_paths = set(discovered)
    expected_paths = set(expected)
    if discovered_paths != expected_paths:
        missing = sorted(discovered_paths - expected_paths)
        stale = sorted(expected_paths - discovered_paths)
        raise ValueError(
            "frozen input digest lock does not match packet references: "
            f"missing expectations={missing}, stale expectations={stale}"
        )
    observed = {relative: _digest_file(path) for relative, path in discovered.items()}
    mismatches = [
        f"{relative}: expected {expected[relative]}, observed {observed[relative]}"
        for relative in sorted(observed)
        if observed[relative] != expected[relative]
    ]
    if mismatches:
        raise ValueError("frozen input digest mismatch: " + "; ".join(mismatches))
    return observed


@dataclass(frozen=True)
class CandidateRecord:
    """Versioned per-candidate production-observed record."""

    schema: str = CANDIDATE_RECORD_SCHEMA
    packet_digest: str = ""
    arm: str = ""
    candidate_identity: str = ""
    scenario_template: str = ""
    scenario_seed: int = 0
    search_seed: int = 0
    macro_action_index: int | None = None
    native_seam: str = ""
    policy_identity: str = ""
    objective_identity: str = ""
    repository_commit: str = ""
    status: str = "unavailable"
    simulator_steps: int | None = None
    simulator_steps_source: str = "unavailable"
    episode_identity: str = ""
    episode_digest: str = ""
    objective_value: float | None = None
    fallback_flags: tuple[str, ...] = ()
    degraded_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class _ReceiptArmValidation:
    """Shared packet and filesystem context for one receipt arm validation."""

    packet_digest: str
    commit: str
    blocked_before_execution: bool
    frozen_budget: int
    expected: dict[str, Any] | None
    repository_root: Path
    episode_artifact_root: Path


def load_packet(path: Path) -> dict[str, Any]:
    """Load and validate the frozen matched-compute packet."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "matched_compute_packet.v2"
    ):
        raise ValueError("packet must be matched_compute_packet.v2")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"open_loop", "reactive"}:
        raise ValueError("packet must declare open_loop and reactive arms")
    return payload


def _budget_reconcile(records: list[CandidateRecord], frozen_budget: int) -> list[str]:
    """Return violations when candidate records do not reconcile to the budget."""
    counts: dict[str, int] = {}
    identities: list[str] = []
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
        identities.append(record.candidate_identity)
    total = sum(counts.values())
    problems: list[str] = []
    if total != frozen_budget:
        problems.append(f"candidate records {total} != frozen budget {frozen_budget}")
    missing_identity_count = sum(not identity.strip() for identity in identities)
    if missing_identity_count:
        problems.append(
            f"candidate records contain {missing_identity_count} missing identity value(s)"
        )
    duplicate_identities = sorted(
        {identity for identity in identities if identity and identities.count(identity) > 1}
    )
    if duplicate_identities:
        problems.append(
            "duplicate candidate identities: "
            + ", ".join(repr(identity) for identity in duplicate_identities)
        )
    for status in counts:
        if status not in CANDIDATE_STATUSES:
            problems.append(f"unknown candidate status {status!r}")
    for status in ("fallback", "unavailable"):
        if counts.get(status, 0):
            problems.append(
                f"arm inadmissible as production_observed: {counts[status]} {status} record(s)"
            )
    return problems


def _aggregate_reconcile(
    records: list[CandidateRecord],
    trace: MatchedComputeRuntimeTrace,
    *,
    expected_simulator_steps: int | None = None,
) -> list[str]:
    """Reconcile the companion records with the aggregate trace without changing v1 semantics."""
    problems: list[str] = []
    candidate_budget = getattr(trace, "candidate_budget", None)
    if candidate_budget is not None and trace.candidate_evaluations > candidate_budget:
        problems.append(
            "trace candidate_evaluations "
            f"{trace.candidate_evaluations} exceeds candidate_budget {candidate_budget}"
        )
    accepted = sum(1 for r in records if r.status == "accepted")
    rejected_nonfailed = sum(1 for r in records if r.status == "rejected")
    failed = sum(1 for r in records if r.status == "failed")
    invalid = sum(1 for r in records if r.status == "invalid")
    if accepted != trace.accepted:
        problems.append(f"accepted {accepted} != trace.accepted {trace.accepted}")
    if rejected_nonfailed + failed != trace.rejected:
        problems.append(
            f"rejected+failed {rejected_nonfailed + failed} != trace.rejected {trace.rejected}"
        )
    if invalid != trace.invalid:
        problems.append(f"invalid {invalid} != trace.invalid {trace.invalid}")
    well_typed_steps = [
        record.simulator_steps
        for record in records
        if isinstance(record.simulator_steps, int) and not isinstance(record.simulator_steps, bool)
    ]
    if len(well_typed_steps) == len(records):
        aggregate_steps = sum(well_typed_steps)
        if trace.simulator_physics_steps != aggregate_steps:
            problems.append(
                f"trace simulator_physics_steps {trace.simulator_physics_steps} != "
                f"candidate record simulator steps {aggregate_steps}"
            )
    if expected_simulator_steps is not None:
        for index, record in enumerate(records):
            if record.simulator_steps != expected_simulator_steps:
                problems.append(
                    f"record[{index}] simulator_steps {record.simulator_steps} != "
                    f"frozen simulator steps {expected_simulator_steps}"
                )
    return problems


def _manifest_candidate_identity(entry: dict[str, Any], index: int) -> str:
    """Return the stable execution identity represented by a native manifest entry."""
    bundle_path = entry.get("bundle_path")
    if bundle_path:
        bundle_name = Path(str(bundle_path)).name
        if bundle_name:
            return bundle_name
    return f"candidate_{index:04d}"


def _validated_open_loop_manifest(
    manifest: Any,
    config: SearchConfig,
    *,
    frozen_scenario_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate native manifest lineage before emitting candidate evidence."""
    if not isinstance(manifest, dict):
        raise ValueError("native open-loop manifest must be a mapping")
    if manifest.get("schema_version") != "adversarial-search-manifest.v1":
        raise ValueError("native open-loop manifest schema_version mismatch")
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, dict) or manifest_config != config.to_json():
        raise ValueError("native open-loop manifest config does not match the frozen packet run")
    raw_candidates = manifest.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("native open-loop manifest candidates must be a list")
    candidates: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(raw_candidates):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"native open-loop manifest candidates[{index}] must be a mapping")
        candidate = raw_entry.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError(
                f"native open-loop manifest candidates[{index}].candidate must be a mapping"
            )
        scenario_seed = candidate.get("scenario_seed")
        if (
            isinstance(scenario_seed, bool)
            or not isinstance(scenario_seed, int)
            or scenario_seed != frozen_scenario_seed
        ):
            raise ValueError(
                f"native open-loop manifest candidates[{index}].scenario_seed "
                "does not match the frozen packet"
            )
        candidates.append(raw_entry)
    return manifest_config, candidates


def _run_open_loop(
    packet: dict[str, Any],
    output_dir: Path,
    commit: str,
    packet_digest: str,
    *,
    repository_root: Path | None = None,
) -> tuple[list[CandidateRecord], MatchedComputeRuntimeTrace]:
    """Run the open-loop arm through the real production seams."""
    arm = packet["arms"]["open_loop"]
    binding = arm["runner_binding"]
    scenario = packet["scenario"]
    budget = int(binding["budget"])
    seed = int(binding["search_seed"])
    scenario_template = Path(scenario["template"])
    search_space = Path(scenario["search_space"])
    if repository_root is not None:
        _, scenario_template = _resolve_repository_file(
            scenario_template, repository_root, label="scenario template"
        )
        _, search_space = _resolve_repository_file(
            search_space, repository_root, label="scenario search space"
        )

    config = SearchConfig.from_files(
        policy=binding["policy"],
        scenario_template=scenario_template,
        search_space=search_space,
        objective=binding["objective"],
        output_dir=output_dir / "open_loop",
        budget=budget,
        seed=seed,
        horizon=int(binding["horizon_steps"]),
        dt=float(binding["dt_s"]),
    )
    result = run_adversarial_search(config)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest_config, manifest_candidates = _validated_open_loop_manifest(
        manifest,
        config,
        frozen_scenario_seed=int(scenario["template_seed"]),
    )
    trace = open_loop_runtime_trace_from_result(
        config,
        result,
        macro_actions=int(packet["budget"]["macro_actions_per_episode"]),
    )

    records: list[CandidateRecord] = []
    for index, entry in enumerate(manifest_candidates):
        status = _manifest_status(entry)
        steps = _manifest_steps(entry)
        candidate = entry["candidate"]
        records.append(
            CandidateRecord(
                packet_digest=packet_digest,
                arm="open_loop",
                candidate_identity=_manifest_candidate_identity(entry, index),
                scenario_template=scenario["template_id"],
                scenario_seed=int(candidate["scenario_seed"]),
                search_seed=int(manifest_config["seed"]),
                native_seam=(
                    "robot_sf.adversarial.search.run_adversarial_search"
                    "(default production evaluator)"
                ),
                policy_identity=str(manifest_config["policy"]),
                objective_identity=str(manifest_config["objective"]),
                repository_commit=commit,
                status=status,
                simulator_steps=steps,
                simulator_steps_source="observed_episode_record" if steps else "unavailable",
                episode_identity=str(entry.get("episode_record_path", "")),
                episode_digest=_digest_file(Path(entry["episode_record_path"]))
                if entry.get("episode_record_path") and Path(entry["episode_record_path"]).exists()
                else "",
                objective_value=_manifest_objective(entry, status=status),
                degraded_reason=(
                    str(entry.get("error") or "episode record unavailable") if steps is None else ""
                ),
            )
        )
    return records, trace


def _failure_attribution_status(entry: dict[str, Any]) -> str | None:
    """Return a non-success status attributed by the candidate failure details."""
    attribution = entry.get("failure_attribution")
    if isinstance(attribution, dict):
        details = attribution.get("details")
        if isinstance(details, dict):
            attribution_values = {
                value.replace("-", "_") for value in _certification_status_values(details)
            }
            if attribution_values & {"fallback", "degraded"}:
                return "fallback"
            if attribution_values & {"unavailable", "not_available"}:
                return "unavailable"
            if attribution_values & {"failed", "partial_failure"}:
                return "failed"
    return None


def _manifest_status(entry: dict[str, Any]) -> str:
    """Map a manifest candidate entry to the disjoint status vocabulary."""
    attribution_status = _failure_attribution_status(entry)
    if attribution_status is not None:
        return attribution_status
    certification = entry.get("certification_status", "")
    status_values = tuple(_certification_status_values(certification))
    if isinstance(certification, str):
        status_values = (*status_values, certification.strip().lower())
    if "unavailable" in status_values or "not_available" in status_values:
        return "unavailable"
    if "fallback" in status_values or "degraded" in status_values:
        return "fallback"
    cert = str(certification)
    if "invalid" in cert.lower() or "validation" in cert.lower():
        return "invalid"
    if entry.get("error") or "fail" in cert.lower():
        return "failed"
    if entry.get("bundle_path") is None:
        return "failed"
    return "accepted"


def _certification_status_values(value: Any) -> list[str]:
    """Collect structured status/mode values from nested certification metadata."""
    values: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = str(key).strip().lower()
            if (
                normalized_key in _CERTIFICATION_STATUS_KEYS
                or normalized_key.endswith("_status")
                or normalized_key.endswith("_mode")
            ) and isinstance(nested, str):
                values.append(nested.strip().lower())
            values.extend(_certification_status_values(nested))
    elif isinstance(value, list):
        for nested in value:
            values.extend(_certification_status_values(nested))
    return values


def _manifest_steps(entry: dict[str, Any]) -> int | None:
    """Return observed simulator steps from the episode record when available."""
    record_path = entry.get("episode_record_path")
    if not record_path:
        return None
    path = Path(record_path)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(record, dict):
        steps = record.get("steps") or record.get("simulator_steps") or record.get("total_steps")
        if steps is not None:
            try:
                return int(steps)
            except (TypeError, ValueError):
                return None
    return None


def _objective_value_problem(value: Any, *, status: str, prefix: str) -> str | None:
    """Return why an objective is inadmissible for one candidate status."""
    if value is None:
        return (
            f"{prefix}.objective_value is required for accepted candidate"
            if status == "accepted"
            else None
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{prefix}.objective_value is not numeric"
    if not math.isfinite(float(value)):
        return f"{prefix}.objective_value is not finite"
    return None


def _manifest_objective(entry: dict[str, Any], *, status: str) -> float | None:
    """Return a finite manifest objective, rejecting invalid accepted evidence."""
    value = entry.get("objective_value")
    problem = _objective_value_problem(
        value,
        status=status,
        prefix="native open-loop manifest candidate",
    )
    if problem is not None:
        raise ValueError(problem)
    return float(value) if value is not None else None


def _run_reactive(
    packet: dict[str, Any], output_dir: Path, commit: str, packet_digest: str
) -> tuple[list[CandidateRecord], MatchedComputeRuntimeTrace | None]:
    """Build unavailable records for an isolated diagnostic of the residual seams.

    This helper is intentionally not used by the production canary entry point:
    a real environment-episode driver is required before any arm execution can
    start.  Keeping this isolated helper makes its unavailable status explicit
    for validator fixtures without promoting controller-only counters to
    production evidence.
    """
    arm = packet["arms"]["reactive"]
    residual = arm["residual_search"]
    scenario = packet["scenario"]
    macro_actions = int(packet["budget"]["macro_actions_per_episode"])
    candidates_per_macro = int(packet["budget"]["candidates_per_macro_action_per_arm"])

    records: list[CandidateRecord] = []
    # Preflight: the real residual seams must be importable and constructible.
    if not callable(FiniteGridSearchPolicy) or not callable(BoundedResidualAdversary):
        raise ValueError("reactive native seams must resolve to callable classes")

    for macro_index in range(macro_actions):
        for candidate_index in range(candidates_per_macro):
            records.append(
                CandidateRecord(
                    packet_digest=packet_digest,
                    arm="reactive",
                    candidate_identity=f"macro_{macro_index:02d}_cand_{candidate_index:02d}",
                    scenario_template=scenario["template_id"],
                    scenario_seed=int(scenario["template_seed"]),
                    search_seed=int(residual["seed"]),
                    macro_action_index=macro_index,
                    native_seam=(
                        "robot_sf.ped_npc.residual_search.FiniteGridSearchPolicy+"
                        "robot_sf.ped_npc.residual_adversary.BoundedResidualAdversary"
                    ),
                    policy_identity="finite_grid_search_v1",
                    objective_identity=residual["objective_proxy"],
                    repository_commit=commit,
                    status="unavailable",
                    simulator_steps=None,
                    simulator_steps_source="unavailable",
                    degraded_reason=(
                        "reactive production integration seam requires a real environment "
                        "episode; not fabricated from controller snapshots"
                    ),
                )
            )
    return records, None


def _reactive_production_preflight_problem(packet: dict[str, Any]) -> str | None:
    """Return the exact missing hook that blocks reactive production execution.

    The repository exposes the residual policy and adversary classes and the
    simulator's internal residual wiring, but the frozen canary packet has no
    production runner that drives a real environment episode while exposing one
    record per residual proposal and observed simulator-step provenance.  That
    boundary cannot be inferred from a controller snapshot or aggregate counter.
    """
    del packet
    if not callable(FiniteGridSearchPolicy) or not callable(BoundedResidualAdversary):
        return "reactive native policy/adversary seam is not callable"
    return (
        "reactive production canary blocked before arm execution: no native environment-episode "
        "driver exposes per-residual candidate evaluations and observed simulator-step provenance; "
        "FiniteGridSearchPolicy/BoundedResidualAdversary resolution alone is insufficient"
    )


def _resolve_episode_artifact(identity: Any, repository_root: Path | None = None) -> Path | None:
    """Resolve an episode identity to an existing regular, non-symlink artifact."""
    if not isinstance(identity, str) or not identity or identity != identity.strip():
        return None
    try:
        path = Path(identity)
        root = (repository_root or _repository_root()).resolve()
        if not path.is_absolute():
            path = root / path
        if _absolute_symlink_component(path) is not None:
            return None
        resolved = path.resolve()
        return resolved if resolved.is_file() else None
    except (OSError, RuntimeError, ValueError):
        return None


@lru_cache(maxsize=1)
def _episode_schema() -> dict[str, Any]:
    """Load the repository's canonical episode schema once."""
    return load_schema(_repository_root() / EPISODE_SCHEMA_PATH)


@lru_cache(maxsize=1)
def _episode_validator() -> Draft202012Validator:
    """Compile the canonical episode schema once for bounded receipt validation."""
    return Draft202012Validator(_episode_schema())


def _observed_episode_steps(payload: dict[str, Any]) -> int:
    """Derive simulator steps from the canonical episode-record representation."""
    raw_steps = payload.get("steps")
    if isinstance(raw_steps, list):
        return len(raw_steps)
    if isinstance(raw_steps, bool) or not isinstance(raw_steps, int) or raw_steps < 0:
        raise ValueError("episode record steps must be a non-negative integer or list")
    return raw_steps


@lru_cache(maxsize=1024)
def _canonical_episode_observation(
    observed_digest: str, artifact_bytes: bytes
) -> tuple[dict[str, Any], int]:
    """Parse the exact immutable byte buffer whose digest was observed."""
    if hashlib.sha256(artifact_bytes).hexdigest() != observed_digest:
        raise ValueError("episode artifact buffer does not match its observed digest")
    try:
        artifact_text = artifact_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"episode artifact is not UTF-8: {exc}") from exc
    nonempty_lines = [line for line in artifact_text.splitlines() if line.strip()]
    if len(nonempty_lines) != 1:
        raise ValueError("episode artifact must contain exactly one JSONL record")
    try:
        payload = json.loads(nonempty_lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("episode artifact record is malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("episode artifact record must be a JSON object")
    try:
        _episode_validator().validate(payload)
    except JsonSchemaValidationError as exc:
        raise ValueError(f"episode artifact violates the canonical schema: {exc.message}") from exc
    semantic_contradictions = outcome_contradictions(
        termination_reason=str(payload.get("termination_reason", "")),
        outcome=payload.get("outcome", {}),
        metrics=payload.get("metrics"),
    )
    if semantic_contradictions:
        raise ValueError(
            "episode artifact violates canonical outcome semantics: "
            + "; ".join(semantic_contradictions)
        )
    contradictions = payload.get("integrity", {}).get("contradictions", [])
    if contradictions:
        raise ValueError("episode artifact declares integrity contradictions")
    # The cache key contains the verified digest and immutable bytes, never a
    # mutable path that would need to be reopened for parsing.
    return payload, _observed_episode_steps(payload)


def _episode_artifact_binding_problems(
    record: CandidateRecord,
    prefix: str,
    artifact: Path,
    episode_artifact_root: Path | None = None,
) -> list[str]:
    """Return path-contract problems for one resolved episode artifact."""
    problems: list[str] = []
    if episode_artifact_root is not None:
        try:
            artifact.relative_to(episode_artifact_root.resolve())
        except ValueError:
            problems.append(f"{prefix}.episode_identity is outside the receipt's artifact bundle")
    if artifact.name != "episode_records.jsonl":
        problems.append(f"{prefix}.episode_identity is not a canonical episode_records.jsonl")
    if artifact.parent.name != record.candidate_identity:
        problems.append(f"{prefix}.episode_identity is not bound to its candidate identity")
    return problems


def _episode_payload_binding_problems(
    record: CandidateRecord,
    prefix: str,
    payload: dict[str, Any],
    observed_steps: int,
) -> list[str]:
    """Return candidate identity mismatches derived from canonical record bytes."""
    problems: list[str] = []
    if observed_steps != record.simulator_steps:
        problems.append(f"{prefix}.simulator_steps does not match canonical episode artifact")
    if payload.get("seed") != record.scenario_seed:
        problems.append(f"{prefix}.scenario_seed does not match canonical episode artifact")
    if payload.get("scenario_id") != record.scenario_template:
        problems.append(f"{prefix}.scenario_template does not match canonical episode artifact")
    return problems


def _episode_observation_problems(
    record: CandidateRecord, prefix: str, artifact: Path
) -> list[str]:
    """Return digest, schema, and byte-derived identity problems for one artifact."""
    try:
        artifact_bytes = artifact.read_bytes()
    except OSError as exc:
        return [f"{prefix}.episode_identity cannot be read: {exc}"]
    observed_digest = hashlib.sha256(artifact_bytes).hexdigest()
    if observed_digest != record.episode_digest:
        return [f"{prefix}.episode_digest does not match referenced episode artifact"]
    try:
        payload, observed_steps = _canonical_episode_observation(observed_digest, artifact_bytes)
    except ValueError as exc:
        return [f"{prefix}.episode_identity is not authoritative: {exc}"]
    return _episode_payload_binding_problems(record, prefix, payload, observed_steps)


def _episode_provenance_problems(
    record: CandidateRecord,
    prefix: str,
    repository_root: Path | None = None,
    episode_artifact_root: Path | None = None,
) -> list[str]:
    """Return problems with one candidate-bound canonical episode artifact."""
    problems: list[str] = []
    digest_is_valid = _is_sha256_hex(record.episode_digest)
    if not digest_is_valid:
        problems.append(f"{prefix}.episode_digest is not a lowercase SHA-256 digest")
    artifact = _resolve_episode_artifact(record.episode_identity, repository_root)
    if artifact is None:
        problems.append(
            f"{prefix}.episode_identity does not reference a regular non-symlink artifact file"
        )
        return problems
    problems.extend(
        _episode_artifact_binding_problems(record, prefix, artifact, episode_artifact_root)
    )
    if digest_is_valid:
        problems.extend(_episode_observation_problems(record, prefix, artifact))
    return problems


def _episode_collection_problems(
    records: list[CandidateRecord], arm_name: str, repository_root: Path | None
) -> list[str]:
    """Reject artifact or byte reuse across candidate observations in one arm."""
    observed = [
        record for record in records if record.simulator_steps_source in OBSERVED_STEP_SOURCES
    ]
    artifact_identities = [
        resolved.as_posix()
        for record in observed
        if (resolved := _resolve_episode_artifact(record.episode_identity, repository_root))
        is not None
    ]
    digests = [
        record.episode_digest for record in observed if _is_sha256_hex(record.episode_digest)
    ]
    problems: list[str] = []
    if len(artifact_identities) != len(set(artifact_identities)):
        problems.append(f"{arm_name} records reuse an episode artifact across candidates")
    if len(digests) != len(set(digests)):
        problems.append(f"{arm_name} records reuse episode artifact bytes across candidates")
    return problems


def _cross_arm_episode_reuse_problems(
    open_loop_records: list[CandidateRecord],
    reactive_records: list[CandidateRecord],
    repository_root: Path | None,
) -> list[str]:
    """Reject source paths or bytes reused between the two comparison arms."""

    def _artifact_identities(records: list[CandidateRecord]) -> set[str]:
        return {
            resolved.as_posix()
            for record in records
            if (resolved := _resolve_episode_artifact(record.episode_identity, repository_root))
            is not None
        }

    open_paths = _artifact_identities(open_loop_records)
    reactive_paths = _artifact_identities(reactive_records)
    open_digests = {
        record.episode_digest
        for record in open_loop_records
        if _is_sha256_hex(record.episode_digest)
    }
    reactive_digests = {
        record.episode_digest
        for record in reactive_records
        if _is_sha256_hex(record.episode_digest)
    }
    problems: list[str] = []
    if open_paths & reactive_paths:
        problems.append("receipt arms reuse an episode artifact across candidates")
    if open_digests & reactive_digests:
        problems.append("receipt arms reuse episode artifact bytes across candidates")
    return problems


def _record_step_problems(
    record: CandidateRecord,
    prefix: str,
    repository_root: Path | None = None,
    expected_simulator_steps: int | None = None,
    episode_artifact_root: Path | None = None,
) -> list[str]:
    """Return malformed simulator-step provenance for one record."""
    problems: list[str] = []
    if record.simulator_steps is not None:
        if isinstance(record.simulator_steps, bool) or not isinstance(record.simulator_steps, int):
            problems.append(f"{prefix}.simulator_steps is not an integer")
        elif record.simulator_steps < 0:
            problems.append(f"{prefix}.simulator_steps is negative")
        else:
            mismatch = _expected_step_mismatch(record, prefix, expected_simulator_steps)
            if mismatch:
                problems.append(mismatch)

    problems.extend(
        _record_step_source_problems(record, prefix, repository_root, episode_artifact_root)
    )
    return problems


def _record_step_source_problems(
    record: CandidateRecord,
    prefix: str,
    repository_root: Path | None,
    episode_artifact_root: Path | None,
) -> list[str]:
    """Return provenance problems for the declared simulator-step source."""
    source = str(record.simulator_steps_source or "").strip()
    if source not in OBSERVED_STEP_SOURCES and source != "unavailable":
        return [f"{prefix}.simulator_steps_source is synthetic or unsupported: {source!r}"]
    if source in OBSERVED_STEP_SOURCES:
        problems: list[str] = []
        if record.simulator_steps is None:
            problems.append(f"{prefix} claims observed simulator steps without a step count")
        if not str(record.episode_identity or "").strip():
            problems.append(f"{prefix} claims observed steps without an episode identity")
        if not str(record.episode_digest or "").strip():
            problems.append(f"{prefix} claims observed steps without an episode digest")
        else:
            problems.extend(
                _episode_provenance_problems(record, prefix, repository_root, episode_artifact_root)
            )
        return problems
    if not str(record.degraded_reason or "").strip():
        return [f"{prefix} has unavailable simulator steps without an explicit reason"]
    return []


def _expected_step_mismatch(
    record: CandidateRecord, prefix: str, expected_simulator_steps: int | None
) -> str | None:
    """Return a packet step mismatch for a well-typed candidate record."""
    if expected_simulator_steps is None or record.simulator_steps == expected_simulator_steps:
        return None
    return (
        f"{prefix}.simulator_steps {record.simulator_steps} != frozen simulator steps "
        f"{expected_simulator_steps}"
    )


def _record_scalar_problems(
    record: CandidateRecord,
    prefix: str,
    repository_root: Path | None = None,
    expected_simulator_steps: int | None = None,
    episode_artifact_root: Path | None = None,
) -> list[str]:
    """Return malformed numeric or observed-episode fields for one record."""
    problems: list[str] = []
    objective_problem = _objective_value_problem(
        record.objective_value,
        status=record.status,
        prefix=prefix,
    )
    if objective_problem is not None:
        problems.append(objective_problem)
    problems.extend(
        _record_step_problems(
            record,
            prefix,
            repository_root,
            expected_simulator_steps,
            episode_artifact_root,
        )
    )
    return problems


def _packet_arm_expectations(packet: dict[str, Any], arm_name: str) -> dict[str, Any]:
    """Return candidate-record identity fields frozen by one packet arm."""
    try:
        scenario = packet["scenario"]
        arm = packet["arms"][arm_name]
        binding = arm["runner_binding"]
        macro_actions = int(packet["budget"]["macro_actions_per_episode"])
        candidates_per_macro = int(packet["budget"]["candidates_per_macro_action_per_arm"])
        simulator_steps = int(packet["simulation"]["total_sim_steps"])
        if arm_name == "open_loop":
            return {
                "scenario_template": str(scenario["template_id"]),
                "scenario_seed": int(scenario["template_seed"]),
                "search_seed": int(binding["search_seed"]),
                "policy_identity": str(binding["policy"]),
                "objective_identity": str(binding["objective"]),
                "native_seam_fragment": str(binding["runner"]),
                "adapter": "adversarial_search_production_candidate",
                "native_path": str(binding["runner"]),
                "macro_actions": macro_actions,
                "candidates_per_macro": candidates_per_macro,
                "simulator_steps": simulator_steps,
            }
        residual = arm["residual_search"]
        return {
            "scenario_template": str(scenario["template_id"]),
            "scenario_seed": int(scenario["template_seed"]),
            "search_seed": int(residual["seed"]),
            "policy_identity": str(residual["algorithm_name"]),
            "objective_identity": str(residual["objective_proxy"]),
            "native_seam_fragment": str(binding["search_policy"]),
            "adapter": "finite_grid_residual_adversary",
            "native_path": (f"{binding['search_policy']}+{binding['controller']}"),
            "macro_actions": macro_actions,
            "candidates_per_macro": candidates_per_macro,
            "simulator_steps": simulator_steps,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"packet is missing {arm_name} candidate identity fields") from exc


def _frozen_arm_budgets(packet: dict[str, Any]) -> dict[str, int]:
    """Return the packet's per-episode candidate budget for each arm."""
    try:
        budget = packet["budget"]
        shared_budget = int(budget["candidates_per_arm_per_episode"])
        macro_candidates = int(budget["candidates_per_macro_action_per_arm"])
        macro_actions = int(budget["macro_actions_per_episode"])
        open_loop_budget = int(packet["arms"]["open_loop"]["runner_binding"]["budget"])
        reactive_budget = (
            int(packet["arms"]["reactive"]["residual_search"]["max_candidates"]) * macro_actions
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("packet is missing per-arm candidate budget fields") from exc
    if shared_budget < 0 or macro_candidates < 0 or macro_actions < 0:
        raise ValueError("packet candidate budgets must be non-negative")
    if open_loop_budget != shared_budget:
        raise ValueError("packet open-loop budget does not match candidates_per_arm_per_episode")
    if reactive_budget != shared_budget or reactive_budget != macro_candidates * macro_actions:
        raise ValueError("packet reactive budget does not match candidates_per_arm_per_episode")
    return {"open_loop": shared_budget, "reactive": shared_budget}


def _record_packet_identity_problems(
    record: CandidateRecord, prefix: str, expected: dict[str, Any]
) -> list[str]:
    """Return frozen packet identity mismatches for one record."""
    problems: list[str] = []
    for field in ("scenario_template", "scenario_seed", "search_seed"):
        if getattr(record, field) != expected[field]:
            problems.append(
                f"{prefix}.{field}={getattr(record, field)!r} does not match frozen packet"
            )
    if record.policy_identity != expected["policy_identity"]:
        problems.append(f"{prefix}.policy_identity does not match frozen packet")
    if record.objective_identity != expected["objective_identity"]:
        problems.append(f"{prefix}.objective_identity does not match frozen packet")
    if expected.get("native_seam_fragment") not in record.native_seam:
        problems.append(f"{prefix}.native_seam does not match frozen packet")
    if record.arm == "reactive":
        macro_actions = expected["macro_actions"]
        if record.macro_action_index is None or not 0 <= record.macro_action_index < macro_actions:
            problems.append(f"{prefix}.macro_action_index does not match frozen packet")
    return problems


def _record_integrity_problems(
    records: list[CandidateRecord],
    *,
    arm_name: str,
    packet_digest: str,
    commit: str,
    expected: dict[str, Any] | None = None,
    expected_simulator_steps: int | None = None,
    repository_root: Path | None = None,
    episode_artifact_root: Path | None = None,
) -> list[str]:
    """Return identity and provenance mismatches in one arm's candidate records."""
    problems: list[str] = []
    for index, record in enumerate(records):
        prefix = f"{arm_name} records[{index}]"
        if record.schema != CANDIDATE_RECORD_SCHEMA:
            problems.append(f"{prefix}.schema does not match {CANDIDATE_RECORD_SCHEMA}")
        if record.arm != arm_name:
            problems.append(f"{prefix}.arm={record.arm!r} does not match arm name")
        if record.packet_digest != packet_digest:
            problems.append(f"{prefix}.packet_digest does not match receipt packet_digest")
        if record.repository_commit != commit:
            problems.append(f"{prefix}.repository_commit does not match receipt repository_commit")
        if expected is not None:
            problems.extend(_record_packet_identity_problems(record, prefix, expected))
        problems.extend(
            _record_scalar_problems(
                record,
                prefix,
                repository_root,
                expected_simulator_steps=expected_simulator_steps,
                episode_artifact_root=episode_artifact_root,
            )
        )
    problems.extend(_episode_collection_problems(records, arm_name, repository_root))
    if records and expected is not None and arm_name == "reactive":
        expected_macro_indices = [
            index
            for index in range(expected["macro_actions"])
            for _ in range(expected["candidates_per_macro"])
        ]
        actual_macro_indices = [record.macro_action_index for record in records]
        if sorted(actual_macro_indices) != sorted(expected_macro_indices):
            problems.append(
                f"{arm_name} macro_action_index values do not reconcile with frozen "
                "macro-action budget"
            )
    return problems


def _arm_evidence_status(
    records: list[CandidateRecord],
    *,
    repository_root: Path | None = None,
    episode_artifact_root: Path | None = None,
) -> str:
    """Return ``production_observed`` only when every record is native and observed."""
    if not records:
        return "not_production_observed"
    if any(
        record.status in {"fallback", "unavailable"}
        or record.fallback_flags
        or record.degraded_reason
        or bool(
            _record_scalar_problems(
                record,
                "record",
                repository_root,
                episode_artifact_root=episode_artifact_root,
            )
        )
        for record in records
    ):
        return "not_production_observed"
    if _episode_collection_problems(records, records[0].arm, repository_root):
        return "not_production_observed"
    return "production_observed"


def _runtime_trace_is_production_observed(
    trace: MatchedComputeRuntimeTrace | None,
    *,
    arm_name: str | None = None,
    expected: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether a runtime trace satisfies positive-evidence admission."""
    return bool(
        trace is not None
        and trace.status == "native"
        and trace.execution_mode == "native"
        and trace.evidence_status == "production_observed"
        and trace.simulator_steps_source in {"observed_episode_record", "observed_simulator"}
        and trace.simulator_physics_steps is not None
        and not trace.fallback
        and not trace.degraded
        and (arm_name is None or trace.arm == arm_name)
        and (
            expected is None
            or (
                trace.adapter == expected["adapter"]
                and trace.native_path == expected["native_path"]
                and trace.scenario_seed == expected["scenario_seed"]
                and trace.search_seed == expected["search_seed"]
                and trace.macro_actions == expected["macro_actions"]
            )
        )
    )


def _runtime_trace_admission_problems(
    runtime_traces: Mapping[str, MatchedComputeRuntimeTrace | None],
    expected: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """Return positive-receipt violations in the per-arm runtime traces."""
    problems: list[str] = []
    for arm_name in ("open_loop", "reactive"):
        trace = runtime_traces.get(arm_name)
        if trace is None:
            problems.append(f"production-observed receipt is missing {arm_name} runtime trace")
        elif not _runtime_trace_is_production_observed(
            trace, arm_name=arm_name, expected=(expected or {}).get(arm_name)
        ):
            problems.append(
                f"production-observed receipt {arm_name} runtime trace must be "
                "native production_observed evidence"
            )
    return problems


def _write_receipt(
    *,
    packet_digest: str,
    commit: str,
    open_loop_records: list[CandidateRecord],
    reactive_records: list[CandidateRecord],
    problems: list[str],
    output: Path,
    input_digests: Mapping[str, str] | None = None,
    runtime_traces: Mapping[str, MatchedComputeRuntimeTrace | None] | None = None,
) -> None:
    """Write the versioned canary receipt."""
    receipt_problems = list(problems)
    episode_artifact_root = output.resolve().parent
    arm_statuses = {
        "open_loop": _arm_evidence_status(
            open_loop_records, episode_artifact_root=episode_artifact_root
        ),
        "reactive": _arm_evidence_status(
            reactive_records, episode_artifact_root=episode_artifact_root
        ),
    }
    serialized_runtime_traces = {
        arm_name: trace.to_dict()
        for arm_name, trace in (runtime_traces or {}).items()
        if trace is not None
    }
    runtime_trace_map = runtime_traces or {}
    receipt_problems.extend(
        _cross_arm_episode_reuse_problems(open_loop_records, reactive_records, repository_root=None)
    )
    if all(status == "production_observed" for status in arm_statuses.values()):
        receipt_problems.extend(_runtime_trace_admission_problems(runtime_trace_map))
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "repository_commit": commit,
        "packet_digest": packet_digest,
        "input_digests": dict(sorted((input_digests or {}).items())),
        "runtime_traces": serialized_runtime_traces,
        "arms": {
            "open_loop": {
                "records": [record.as_dict() for record in open_loop_records],
                "evidence_status": arm_statuses["open_loop"],
            },
            "reactive": {
                "records": [record.as_dict() for record in reactive_records],
                "evidence_status": arm_statuses["reactive"],
            },
        },
        "problems": receipt_problems,
        "evidence_status": (
            "production_observed"
            if not receipt_problems
            and arm_statuses["open_loop"] == "production_observed"
            and arm_statuses["reactive"] == "production_observed"
            and not _runtime_trace_admission_problems(runtime_trace_map)
            else "blocked"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, receipt)


def _parse_receipt_arms(
    receipt: dict[str, Any], problems: list[str]
) -> dict[str, tuple[dict[str, Any], list[CandidateRecord]]]:
    """Parse candidate records from both receipt arms while collecting errors."""
    arms = receipt.get("arms")
    if not isinstance(arms, dict):
        problems.append("receipt arms must be a mapping")
        arms = {}
    parsed_arms: dict[str, tuple[dict[str, Any], list[CandidateRecord]]] = {}
    for arm_name in ("open_loop", "reactive"):
        arm = arms.get(arm_name)
        if not isinstance(arm, dict) or not isinstance(arm.get("records"), list):
            problems.append(f"receipt {arm_name} arm records are missing or malformed")
            parsed_arms[arm_name] = ({}, [])
            continue
        records: list[CandidateRecord] = []
        for index, row in enumerate(arm["records"]):
            if not isinstance(row, dict):
                problems.append(f"{arm_name} records[{index}] is not a mapping")
                continue
            try:
                records.append(CandidateRecord(**row))
            except (TypeError, ValueError) as exc:
                problems.append(f"{arm_name} records[{index}] is malformed: {exc}")
        parsed_arms[arm_name] = (arm, records)
    return parsed_arms


def _parse_runtime_trace(
    raw_trace: Any,
    arm_name: str,
    problems: list[str],
) -> MatchedComputeRuntimeTrace | None:
    """Parse one emitted ``matched_compute_trace.v1`` payload."""
    if not isinstance(raw_trace, dict):
        problems.append(f"receipt {arm_name} runtime trace is malformed")
        return None
    trace_payload = {
        field_name: raw_trace[field_name]
        for field_name in _RUNTIME_TRACE_FIELDS
        if field_name in raw_trace
    }
    if "simulator_physics_steps" not in trace_payload and "simulator_steps" in raw_trace:
        trace_payload["simulator_physics_steps"] = raw_trace["simulator_steps"]
    try:
        return MatchedComputeRuntimeTrace(**trace_payload)
    except (TypeError, ValueError) as exc:
        problems.append(f"receipt {arm_name} runtime trace is malformed: {exc}")
        return None


def _runtime_trace_budget_problems(
    trace: MatchedComputeRuntimeTrace,
    arm_name: str,
    expected_budget: int | None,
) -> list[str]:
    """Return trace budget mismatches against the packet and local trace contract."""
    problems: list[str] = []
    if expected_budget is not None and trace.candidate_budget != expected_budget:
        problems.append(
            f"{arm_name} runtime trace candidate_budget {trace.candidate_budget} "
            f"does not match frozen budget {expected_budget}"
        )
    if trace.candidate_evaluations > trace.candidate_budget:
        problems.append(
            f"{arm_name} runtime trace candidate_evaluations "
            f"{trace.candidate_evaluations} exceeds candidate_budget "
            f"{trace.candidate_budget}"
        )
    return problems


def _parse_runtime_traces(
    receipt: dict[str, Any],
    parsed_arms: dict[str, tuple[dict[str, Any], list[CandidateRecord]]],
    problems: list[str],
    *,
    expected_budgets: Mapping[str, int] | None = None,
    expected_identities: Mapping[str, Mapping[str, Any]] | None = None,
    expected_simulator_steps: int | None = None,
) -> dict[str, MatchedComputeRuntimeTrace]:
    """Parse and reconcile emitted ``matched_compute_trace.v1`` arm traces."""
    raw_traces = receipt.get("runtime_traces")
    requires_traces = receipt.get("evidence_status") == "production_observed" or any(
        records for _, records in parsed_arms.values()
    )
    if not isinstance(raw_traces, dict):
        if requires_traces:
            problems.append("production-observed receipt runtime_traces are missing or malformed")
        return {}

    parsed_traces: dict[str, MatchedComputeRuntimeTrace] = {}
    for arm_name in ("open_loop", "reactive"):
        raw_trace = raw_traces.get(arm_name)
        if raw_trace is None:
            if requires_traces:
                problems.append(f"production-observed receipt is missing {arm_name} runtime trace")
            continue
        trace = _parse_runtime_trace(raw_trace, arm_name, problems)
        if trace is None:
            continue
        if receipt.get("evidence_status") == "production_observed" and not (
            _runtime_trace_is_production_observed(
                trace,
                arm_name=arm_name,
                expected=(expected_identities or {}).get(arm_name),
            )
        ):
            problems.append(
                f"production-observed receipt {arm_name} runtime trace must be "
                "native production_observed evidence"
            )
        if trace.arm != arm_name:
            problems.append(
                f"receipt {arm_name} runtime trace arm {trace.arm!r} does not match arm name"
            )
        problems.extend(
            _runtime_trace_budget_problems(trace, arm_name, (expected_budgets or {}).get(arm_name))
        )
        records = parsed_arms[arm_name][1]
        if trace.candidate_evaluations != len(records):
            problems.append(
                f"{arm_name} runtime trace candidate_evaluations "
                f"{trace.candidate_evaluations} != record count {len(records)}"
            )
        problems.extend(
            _aggregate_reconcile(records, trace, expected_simulator_steps=expected_simulator_steps)
        )
        parsed_traces[arm_name] = trace
    return parsed_traces


def _receipt_arm_problems(
    arm_name: str,
    arm: dict[str, Any],
    records: list[CandidateRecord],
    *,
    validation: _ReceiptArmValidation,
) -> list[str]:
    """Validate one receipt arm's accounting and evidence status."""
    problems: list[str] = []
    if validation.blocked_before_execution and records:
        problems.append("blocked-before-execution receipt must not contain arm records")
    if not (validation.blocked_before_execution and not records):
        problems.extend(_budget_reconcile(records, validation.frozen_budget))
    problems.extend(
        _record_integrity_problems(
            records,
            arm_name=arm_name,
            packet_digest=validation.packet_digest,
            commit=validation.commit,
            expected=validation.expected,
            expected_simulator_steps=(validation.expected or {}).get("simulator_steps"),
            repository_root=validation.repository_root,
            episode_artifact_root=validation.episode_artifact_root,
        )
    )
    expected_arm_status = _arm_evidence_status(
        records,
        repository_root=validation.repository_root,
        episode_artifact_root=validation.episode_artifact_root,
    )
    if arm.get("evidence_status") != expected_arm_status:
        problems.append(
            f"{arm_name} evidence_status {arm.get('evidence_status')!r} "
            f"does not match {expected_arm_status!r}"
        )
    for status in ("fallback", "unavailable"):
        if any(record.status == status for record in records):
            problems.append(f"{arm_name} arm has {status} record(s)")
    return problems


def _receipt_status_problems(
    receipt: dict[str, Any],
    parsed_arms: dict[str, tuple[dict[str, Any], list[CandidateRecord]]],
    problems: list[str],
    parsed_traces: dict[str, MatchedComputeRuntimeTrace] | None = None,
    repository_root: Path | None = None,
    episode_artifact_root: Path | None = None,
) -> list[str]:
    """Validate aggregate receipt evidence status against both arms."""
    traces = parsed_traces or {}
    expected_receipt_status = (
        "production_observed"
        if all(
            _arm_evidence_status(
                parsed_arms[name][1],
                repository_root=repository_root,
                episode_artifact_root=episode_artifact_root,
            )
            == "production_observed"
            for name in ("open_loop", "reactive")
        )
        and all(
            _runtime_trace_is_production_observed(traces.get(name))
            for name in ("open_loop", "reactive")
        )
        and not problems
        else "blocked"
    )
    status_problems: list[str] = []
    if receipt.get("evidence_status") != expected_receipt_status:
        status_problems.append(
            f"receipt evidence_status {receipt.get('evidence_status')!r} "
            f"does not match {expected_receipt_status!r}"
        )
    if receipt.get("evidence_status") == "blocked" and not receipt.get("problems"):
        status_problems.append("blocked receipt must record at least one problem")
    return status_problems


def _receipt_input_digest_problems(
    receipt: dict[str, Any], packet_digest: str, packet_path: Path | None
) -> list[str]:
    """Return receipt input-digest mismatches against the frozen input lock."""
    raw_input_digests = receipt.get("input_digests", {})
    if not isinstance(raw_input_digests, dict):
        return ["receipt input_digests must be a mapping"]
    if not raw_input_digests:
        arms = receipt.get("arms")
        has_records = isinstance(arms, dict) and any(
            isinstance(arm, dict) and arm.get("records") for arm in arms.values()
        )
        if receipt.get("evidence_status") == "production_observed" or has_records:
            return ["production-observed receipt input_digests must be non-empty"]
        return []
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw_input_digests.items()
    ):
        return ["receipt input_digests entries must be string pairs"]
    if packet_path is None:
        return []
    try:
        observed_digests = _verify_frozen_inputs(packet_path)
        packet_relative, _ = _resolve_repository_file(
            packet_path, _repository_root(), label="packet"
        )
    except ValueError as exc:
        return [str(exc)]
    problems: list[str] = []
    normalized_input_digests = dict(raw_input_digests)
    if normalized_input_digests != observed_digests:
        problems.append("receipt input_digests do not match current frozen inputs")
    if normalized_input_digests.get(packet_relative) != packet_digest:
        problems.append("receipt packet_digest does not match input_digests packet entry")
    return problems


def _receipt_repository_commit_problems(commit: str, packet_path: Path | None) -> list[str]:
    """Require a checked receipt to name a commit present in the current source checkout."""
    if packet_path is None:
        return []
    repository_root = _repository_root()
    try:
        object_check = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return [f"cannot verify receipt repository_commit: {exc}"]
    if object_check.returncode != 0:
        return ["receipt repository_commit is not present in the current source checkout"]

    try:
        ancestry_check = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return [f"cannot verify receipt repository_commit ancestry: {exc}"]
    if ancestry_check.returncode != 0:
        return ["receipt repository_commit is not an ancestor of the current source checkout"]
    return []


def _check_receipt(path: Path, *, packet_path: Path | None = None) -> int:
    """Validate an existing receipt deterministically."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("receipt schema mismatch")
    raw_problems = receipt.get("problems", [])
    problems = (
        [str(problem) for problem in raw_problems]
        if isinstance(raw_problems, list)
        else ["receipt problems must be a list"]
    )
    packet_digest = str(receipt.get("packet_digest", ""))
    commit = str(receipt.get("repository_commit", ""))
    frozen_arm_budgets: dict[str, int] | None = None
    frozen_arm_expectations: dict[str, dict[str, Any]] | None = None
    frozen_simulator_steps: int | None = None
    if packet_path is not None:
        packet = load_packet(packet_path)
        frozen_arm_budgets = _frozen_arm_budgets(packet)
        frozen_simulator_steps = int(packet["simulation"]["total_sim_steps"])
        frozen_arm_expectations = {
            arm_name: _packet_arm_expectations(packet, arm_name)
            for arm_name in ("open_loop", "reactive")
        }
    if len(packet_digest) != 64 or any(char not in "0123456789abcdef" for char in packet_digest):
        problems.append("receipt packet_digest must be 64 lowercase hexadecimal characters")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        problems.append("receipt repository_commit must be 40 lowercase hexadecimal characters")
    elif packet_path is not None:
        problems.extend(_receipt_repository_commit_problems(commit, packet_path))
    problems.extend(_receipt_input_digest_problems(receipt, packet_digest, packet_path))
    parsed_arms = _parse_receipt_arms(receipt, problems)
    repository_root = _repository_root()
    episode_artifact_root = path.resolve().parent
    blocked_before_execution = any(
        "blocked before arm execution" in problem for problem in problems
    )
    for arm_name, (arm, records) in parsed_arms.items():
        problems.extend(
            _receipt_arm_problems(
                arm_name,
                arm,
                records,
                validation=_ReceiptArmValidation(
                    packet_digest=packet_digest,
                    commit=commit,
                    blocked_before_execution=blocked_before_execution,
                    frozen_budget=(frozen_arm_budgets or {}).get(arm_name, 90),
                    expected=(frozen_arm_expectations or {}).get(arm_name),
                    repository_root=repository_root,
                    episode_artifact_root=episode_artifact_root,
                ),
            )
        )
    problems.extend(
        _cross_arm_episode_reuse_problems(
            parsed_arms["open_loop"][1],
            parsed_arms["reactive"][1],
            repository_root,
        )
    )
    parsed_traces = _parse_runtime_traces(
        receipt,
        parsed_arms,
        problems,
        expected_budgets=frozen_arm_budgets,
        expected_identities=frozen_arm_expectations,
        expected_simulator_steps=frozen_simulator_steps,
    )
    problems.extend(
        _receipt_status_problems(
            receipt,
            parsed_arms,
            problems,
            parsed_traces,
            repository_root=repository_root,
            episode_artifact_root=episode_artifact_root,
        )
    )
    if problems:
        print(json.dumps(receipt, indent=2, sort_keys=True))
        for problem in problems:
            print(f"check failed: {problem}")
        return 1
    print("receipt check passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the canary CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        default=DEFAULT_PACKET_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/matched_compute_canary"))
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="validate an existing receipt")
    parser.add_argument("--commit", default=None, help="repository commit (default: HEAD)")
    args = parser.parse_args(argv)

    if args.check:
        if args.receipt is None:
            parser.error("--check requires --receipt")
        return _check_receipt(args.receipt, packet_path=args.packet)

    repository_root = _repository_root()
    requested_receipt_path = args.receipt or args.output_dir / "receipt.json"
    output_dir, receipt_path = _validate_execution_destinations(
        args.output_dir, requested_receipt_path, repository_root
    )
    commit = (
        args.commit
        or subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    _, packet_path = _resolve_repository_file(args.packet, repository_root, label="packet")
    packet = load_packet(packet_path)
    input_digests = _verify_frozen_inputs(packet_path, repository_root=repository_root)
    packet_digest = input_digests[packet_path.resolve().relative_to(repository_root).as_posix()]

    problems: list[str] = []
    open_loop_trace: MatchedComputeRuntimeTrace | None = None
    reactive_trace: MatchedComputeRuntimeTrace | None = None
    reactive_preflight_problem = _reactive_production_preflight_problem(packet)
    if reactive_preflight_problem is not None:
        # A failed arm gate must not leave a partial comparison behind.  In
        # particular, do not run the open-loop arm before the reactive seam is
        # known to be production-observable.
        problems.append(reactive_preflight_problem)
        open_loop_records: list[CandidateRecord] = []
        reactive_records: list[CandidateRecord] = []
    else:
        open_loop_records, open_loop_trace = _run_open_loop(
            packet,
            output_dir,
            commit,
            packet_digest,
            repository_root=repository_root,
        )
        reactive_records, reactive_trace = _run_reactive(packet, output_dir, commit, packet_digest)
        problems.extend(_budget_reconcile(open_loop_records, 90))
        problems.extend(_budget_reconcile(reactive_records, 90))
        if open_loop_trace is None:
            problems.append("open-loop runtime trace is missing")
        else:
            problems.extend(
                _aggregate_reconcile(
                    open_loop_records,
                    open_loop_trace,
                    expected_simulator_steps=int(packet["simulation"]["total_sim_steps"]),
                )
            )
        if reactive_trace is None:
            problems.append("reactive runtime trace is missing")
        else:
            problems.extend(
                _aggregate_reconcile(
                    reactive_records,
                    reactive_trace,
                    expected_simulator_steps=int(packet["simulation"]["total_sim_steps"]),
                )
            )

    runtime_expectations = {
        arm_name: _packet_arm_expectations(packet, arm_name)
        for arm_name in ("open_loop", "reactive")
    }
    runtime_trace_map = {"open_loop": open_loop_trace, "reactive": reactive_trace}
    if all(
        _arm_evidence_status(records, episode_artifact_root=receipt_path.resolve().parent)
        == "production_observed"
        for records in (open_loop_records, reactive_records)
    ):
        problems.extend(_runtime_trace_admission_problems(runtime_trace_map, runtime_expectations))

    _write_receipt(
        packet_digest=packet_digest,
        commit=commit,
        open_loop_records=open_loop_records,
        reactive_records=reactive_records,
        problems=problems,
        output=receipt_path,
        input_digests=input_digests,
        runtime_traces={
            "open_loop": open_loop_trace,
            "reactive": reactive_trace,
        },
    )
    if problems:
        for problem in problems:
            print(f"blocked: {problem}")
        return 1
    print(f"canary receipt written: {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
