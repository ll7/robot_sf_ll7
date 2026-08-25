"""Deterministic, credential-safe doctor for benchmark-data release admission."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready._preflight import _resolved_seed_inventory
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.checkpoint_staging_receipt import (
    CheckpointStagingReceiptError,
    validate_checkpoint_staging_receipt,
)
from robot_sf.benchmark.release_acceptance import FULL_RELEASE_EXPECTED_EPISODE_CELLS
from robot_sf.benchmark.release_protocol import (
    RELEASE_MANIFEST_SCHEMA_VERSION_V0_2,
    load_release_manifest,
    validate_release_manifest,
)
from robot_sf.benchmark.zenodo_publisher import build_session, read_token_file

# These are the two repository-wide security and correctness workflows that
# must have evaluated the immutable source commit before publication.  The
# aggregate CI workflow already owns its internal required jobs; CodeQL is a
# separate workflow and therefore cannot be inferred from the aggregate run.
REQUIRED_CI_WORKFLOWS = ("CI", "CodeQL")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"(?:<[^<>]+>|pending[-_]|\b(?:tbd|todo|unset|unknown)\b)", re.I)
_GH_NOT_FOUND_RE = re.compile(
    r"\b(?:release|tag)\s+(?:not[ -]?found|does not exist)\b",
    re.IGNORECASE,
)
_HARDCODED_ROBOT_SF_PATH_RE = re.compile(
    r"(?<![\w/])/(?:[^\s/\\\"'<>]+/)*robot_sf_ll7(?:[/\\][^\s\\\"'<>]*)?",
    re.IGNORECASE,
)

# Resource values are part of the immutable private launch packet and its
# matching queue row.  They must not be hard-coded here: the release route may
# be LiCCA/epyc-gpu or another admitted site such as imech192/l40s.
_RESOURCE_FIELDS = (
    "cluster",
    "partition",
    "route_id",
    "cpus",
    "gpus",
    "gpu_type",
    "mem_gb",
    "wall_clock",
)
_QUEUE_RESOURCE_FIELDS = ("cluster", "partition", "route_id", "cpus", "gpus", "mem_gb")
_RESOURCE_STRING_FIELDS = {"cluster", "partition", "route_id", "gpu_type", "wall_clock", "qos"}

_REQUIRED_PACKET_HASH_FIELDS = (
    "release_manifest_sha256",
    "canonical_config_sha256",
    "scenario_matrix_sha256",
    "checkpoint_receipt_sha256",
    "runtime_smoke_receipt_sha256",
    "public_entrypoint_sha256",
    "private_wrapper_sha256",
    "startup_sentinel_sha256",
    "admission_helper_sha256",
)
_REQUIRED_PACKET_TRACE_FIELDS = {
    "job_id",
    "queue_id",
    "campaign",
    "submission_id",
    "submission_attempt",
    "public_commit",
    "config",
    "packet",
    "packet_sha256",
    "checkpoint_receipt_path",
    "checkpoint_receipt_sha256",
    "runtime_smoke_receipt_path",
    "runtime_smoke_receipt_sha256",
    "release_label",
}
_REQUIRED_PACKET_INPUT_NAMES = (
    "release_manifest",
    "canonical_campaign_config",
    "scenario_matrix",
    "public_single_node_entrypoint",
    "checkpoint_staging_receipt",
    "runtime_smoke_receipt",
    "private_wrapper",
    "release_runner",
)
_PUBLIC_PACKET_INPUT_NAMES = (
    "release_manifest",
    "canonical_campaign_config",
    "scenario_matrix",
    "public_single_node_entrypoint",
    "runtime_smoke_receipt",
    "release_runner",
)


@dataclass(frozen=True)
class ReleaseDoctorCheck:
    """One sanitized release-admission check result."""

    name: str
    status: str
    summary: str


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a bounded diagnostic command without echoing its environment.

    Returns:
        Completed subprocess result.
    """
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)


def _git_check(repo: Path, expected_sha: str) -> ReleaseDoctorCheck:
    """Check clean exact-HEAD state.

    Returns:
        Sanitized check result.
    """
    head = _run(["git", "rev-parse", "HEAD"], repo)
    status = _run(["git", "status", "--porcelain"], repo)
    if head.returncode or status.returncode:
        return ReleaseDoctorCheck("git", "fail", "Git state could not be inspected")
    problems = []
    if head.stdout.strip() != expected_sha:
        problems.append("HEAD differs from expected source SHA")
    if status.stdout.strip():
        problems.append("worktree is dirty")
    return ReleaseDoctorCheck(
        "git", "pass" if not problems else "fail", "; ".join(problems) or "clean exact source SHA"
    )


def _evaluate_workflow_runs(
    name: str, matching: list[dict[str, Any]]
) -> tuple[str | None, str | None]:
    """Evaluate runs for one required workflow and return (supported_detail, non_green_detail).

    Returns:
        tuple of (supported_detail, non_green_detail), where exactly one is non-None.
    """
    success_runs = [
        run
        for run in matching
        if str(run.get("status", "")).lower() == "completed"
        and str(run.get("conclusion", "")).lower() == "success"
    ]
    if success_runs:
        run_ids = [str(r.get("databaseId")) for r in success_runs if r.get("databaseId")]
        detail = f"{name} (run {run_ids[0]})" if run_ids else name
        return detail, None

    in_prog_runs = [run for run in matching if str(run.get("status", "")).lower() != "completed"]
    failing_runs = [
        run
        for run in matching
        if str(run.get("status", "")).lower() == "completed"
        and str(run.get("conclusion", "")).lower() == "failure"
    ]
    cancelled_runs = [
        run
        for run in matching
        if str(run.get("status", "")).lower() == "completed"
        and str(run.get("conclusion", "")).lower() == "cancelled"
    ]

    run_ids = [str(r.get("databaseId")) for r in matching if r.get("databaseId")]
    id_suffix = f" (run {', '.join(run_ids)})" if run_ids else ""

    if in_prog_runs:
        return None, f"{name} pending{id_suffix}"
    if failing_runs:
        return None, f"{name} failed{id_suffix}"
    if cancelled_runs:
        return None, f"{name} cancelled{id_suffix}"
    return None, f"{name}{id_suffix}"


def _parse_gh_runs(
    stdout: str, expected_sha: str, required_workflows: tuple[str, ...]
) -> dict[str, list[dict[str, Any]]]:
    """Group exact-SHA workflow runs by required workflow name.

    Returns:
        Mapping of workflow name to list of matching exact-SHA runs.
    """
    try:
        raw_runs = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        raw_runs = []
    if not isinstance(raw_runs, list):
        raw_runs = []
    by_workflow: dict[str, list[dict[str, Any]]] = {name: [] for name in required_workflows}
    for run in raw_runs:
        if not isinstance(run, dict) or run.get("headSha") != expected_sha:
            continue
        workflow_name = str(
            run.get("workflowName") or run.get("name") or run.get("workflow") or ""
        ).strip()
        if workflow_name in by_workflow:
            by_workflow[workflow_name].append(run)
    return by_workflow


def _ci_check(
    repo: Path,
    expected_sha: str,
    required_workflows: tuple[str, ...] = REQUIRED_CI_WORKFLOWS,
) -> ReleaseDoctorCheck:
    """Require every required workflow to be completed green for the SHA.

    Returns:
        Sanitized check result.
    """
    result = _run(
        [
            "gh",
            "run",
            "list",
            "--repo",
            "ll7/robot_sf_ll7",
            "--commit",
            expected_sha,
            "--limit",
            "100",
            "--json",
            "databaseId,headSha,status,conclusion,workflowName,name",
        ],
        repo,
    )
    if result.returncode:
        return ReleaseDoctorCheck(
            "ci", "fail", "exact-source required workflow state is unavailable"
        )
    by_workflow = _parse_gh_runs(result.stdout, expected_sha, required_workflows)
    missing: list[str] = []
    non_green: list[str] = []
    supported_details: list[str] = []

    for name, matching in by_workflow.items():
        if not matching:
            missing.append(name)
            continue
        supported, non_green_reason = _evaluate_workflow_runs(name, matching)
        if supported:
            supported_details.append(supported)
        elif non_green_reason:
            non_green.append(non_green_reason)

    problems: list[str] = []
    if missing:
        problems.append("missing " + ", ".join(missing))
    if non_green:
        problems.append("not completed green: " + ", ".join(non_green))
    green = not problems
    if green:
        detail_str = f": {', '.join(supported_details)}" if supported_details else ""
        summary_msg = f"all exact-source required workflows are green{detail_str}"
    else:
        summary_msg = "; ".join(problems)

    return ReleaseDoctorCheck(
        "ci",
        "pass" if green else "fail",
        summary_msg,
    )


def _tag_check(repo: Path, tag: str) -> ReleaseDoctorCheck:
    """Require the planned tag to be unused.

    Returns:
        Sanitized check result.
    """
    local = _run(["git", "show-ref", "--verify", "--quiet", f"refs/tags/{tag}"], repo)
    remote_ref = _run(
        ["git", "ls-remote", "--exit-code", "--refs", "origin", f"refs/tags/{tag}"],
        repo,
    )
    remote = _run(["gh", "release", "view", tag, "--repo", "ll7/robot_sf_ll7"], repo)
    local_missing = local.returncode == 1 and not local.stdout and not local.stderr
    if local.returncode not in {0, 1} or (local.returncode == 1 and not local_missing):
        return ReleaseDoctorCheck("tag_collision", "fail", "local tag state is unavailable")

    remote_missing = remote_ref.returncode == 2 and not remote_ref.stdout and not remote_ref.stderr
    if remote_ref.returncode not in {0, 2} or (remote_ref.returncode == 2 and not remote_missing):
        return ReleaseDoctorCheck("tag_collision", "fail", "remote tag state is unavailable")

    release_missing = (
        remote.returncode != 0
        and _GH_NOT_FOUND_RE.search(f"{remote.stdout}\n{remote.stderr}") is not None
    )
    if remote.returncode not in {0, 1} or (remote.returncode == 1 and not release_missing):
        return ReleaseDoctorCheck("tag_collision", "fail", "GitHub release state is unavailable")

    collision = local.returncode == 0 or remote_ref.returncode == 0 or remote.returncode == 0
    return ReleaseDoctorCheck(
        "tag_collision",
        "fail" if collision else "pass",
        "planned tag already exists" if collision else "planned tag is unused",
    )


def _manifest_check(
    manifest_path: Path, expected_cells: int
) -> tuple[ReleaseDoctorCheck, Any, Any]:
    """Validate pinned hashes and exact matrix cardinality.

    Returns:
        Check result, loaded manifest, and loaded campaign config.
    """
    try:
        manifest = load_release_manifest(manifest_path)
        cfg = load_campaign_config(manifest.canonical_campaign_config_path)
        validation = validate_release_manifest(manifest, campaign_config=cfg)
        scenarios = _load_campaign_scenarios(cfg)
        seeds = _resolved_seed_inventory(scenarios)
        planners = [planner for planner in cfg.planners if planner.enabled]
        cells = len(scenarios) * len(seeds) * len(planners)
        problems = list(validation["problems"])
        manifest_cells = getattr(manifest, "expected_episode_cells", None)
        if manifest.schema_version == RELEASE_MANIFEST_SCHEMA_VERSION_V0_2:
            if manifest_cells != FULL_RELEASE_EXPECTED_EPISODE_CELLS:
                problems.append(
                    "v0.2 benchmark-data manifest must require "
                    f"{FULL_RELEASE_EXPECTED_EPISODE_CELLS} cells"
                )
            if manifest_cells != expected_cells:
                problems.append(
                    f"doctor cardinality {expected_cells} does not match manifest-required "
                    f"{manifest_cells} cells"
                )
        if cells != expected_cells:
            problems.append(f"matrix has {cells} cells, expected {expected_cells}")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return (
            ReleaseDoctorCheck("manifest", "fail", "manifest or matrix could not be validated"),
            None,
            None,
        )
    return (
        ReleaseDoctorCheck(
            "manifest",
            "pass" if not problems else "fail",
            f"hashes valid; exact {cells}-cell matrix" if not problems else "; ".join(problems),
        ),
        manifest,
        cfg,
    )


def _checkpoint_check(
    cfg: Any,
    manifest: Any,
    receipt: Path | None,
    *,
    repo_root: Path | None = None,
    checkpoint_path_map: Any = None,
) -> ReleaseDoctorCheck:
    """Validate exact staged-checkpoint admission.

    Returns:
        Sanitized check result.
    """
    if cfg is None or manifest is None or receipt is None:
        return ReleaseDoctorCheck("checkpoints", "fail", "staged-checkpoint receipt is missing")
    try:
        mapping_kwargs = (
            {
                "checkpoint_path_map": checkpoint_path_map,
                "repo_root": repo_root,
            }
            if checkpoint_path_map
            else {}
        )
        payload = validate_checkpoint_staging_receipt(
            cfg,
            receipt,
            campaign_config_path=manifest.canonical_campaign_config_path,
            **mapping_kwargs,
        )
    except CheckpointStagingReceiptError as exc:
        return ReleaseDoctorCheck("checkpoints", "fail", str(exc))
    return ReleaseDoctorCheck(
        "checkpoints", "pass", f"{len(payload['arms'])} checkpoint references staged and verified"
    )


def _release_identity_check(manifest: Any, expected_base_sha: str, tag: str) -> ReleaseDoctorCheck:
    """Require the final v0.2 manifest to bind the exact source and tag.

    Returns:
        Sanitized check result.
    """
    problems = []
    if (
        manifest is None
        or getattr(manifest, "schema_version", None) != RELEASE_MANIFEST_SCHEMA_VERSION_V0_2
    ):
        problems.append("final manifest is not v0.2")
    if manifest is None or getattr(manifest, "latest_main_base_commit", None) != expected_base_sha:
        problems.append("manifest latest-main base commit does not match")
    if manifest is None or getattr(manifest, "release_tag", None) != tag:
        problems.append("manifest release tag does not match")
    return ReleaseDoctorCheck(
        "release_identity",
        "pass" if not problems else "fail",
        "; ".join(problems) or "v0.2 source/tag identity frozen",
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    """Load a JSON/YAML mapping.

    Returns:
        Parsed mapping.
    """
    payload = (
        json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    )
    if not isinstance(payload, dict):
        raise ValueError("expected mapping")
    return payload


def _load_queue_rows(path: Path) -> list[dict[str, Any]]:
    """Load queue rows without importing private-ops code.

    Returns:
        Queue rows represented as mappings.

    Raises:
        ValueError: If the queue payload has no row collection.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows: Any = payload
    if isinstance(payload, dict):
        for key in ("queues", "rows", "queue"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("queue file must contain a list of mapping rows")
    return rows


def _sha256(path: Path) -> str:
    """Return a file digest without exposing file contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_concrete(value: Any) -> bool:
    """Return whether a packet value is neither empty nor a freeze placeholder."""
    text = str(value or "").strip()
    return bool(text) and _PLACEHOLDER_RE.search(text) is None


def _strict_int(value: Any) -> int | None:
    """Parse a non-fractional integer without Python coercion surprises.

    Returns:
        An integer value, or ``None`` for booleans, floats, fractional strings,
        and other non-integral values.
    """
    if isinstance(value, bool) or isinstance(value, float):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        return int(value.strip())
    return None


def _resource_values_match(field: str, actual: Any, expected: Any) -> bool:
    """Compare two resource values using the packet's typed contract.

    Returns:
        ``True`` when the values are equal under the field's type semantics.
    """
    if field in _RESOURCE_STRING_FIELDS:
        return str(actual or "").strip().lower() == str(expected or "").strip().lower()
    actual_int = _strict_int(actual)
    expected_int = _strict_int(expected)
    return actual_int is not None and expected_int is not None and actual_int == expected_int


def _wall_clock_seconds(value: Any) -> int | None:
    """Convert a ``HH:MM:SS`` wall-clock value to seconds.

    Returns:
        The number of seconds, or ``None`` for an invalid value.
    """
    parts = str(value or "").strip().split(":")
    if len(parts) != 3:
        return None
    parsed = [_strict_int(part) for part in parts]
    if any(part is None for part in parsed):
        return None
    hours, minutes, seconds = parsed
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return None
    return hours * 3600 + minutes * 60 + seconds


def _validate_identity_hashes(packet: dict[str, Any]) -> list[str]:
    """Validate required packet identity hash syntax.

    Returns:
        Sanitized identity hash problems.
    """
    problems: list[str] = []
    identity = packet.get("identity")
    if not isinstance(identity, dict):
        return ["launch packet identity is missing"]
    for field in _REQUIRED_PACKET_HASH_FIELDS:
        value = identity.get(field)
        if not _SHA256_RE.fullmatch(str(value or "")):
            problems.append(f"identity.{field} is not a concrete SHA-256")
    return problems


def _validate_packet_input_hashes(packet: dict[str, Any], expected_sha: str | None) -> list[str]:
    """Validate packet input hash syntax without reading secrets.

    Returns:
        Sanitized input hash problems.
    """
    problems = _validate_identity_hashes(packet)
    inputs = packet.get("inputs")
    if not isinstance(inputs, dict):
        problems.append("launch packet inputs are missing")
        return problems
    for input_name in _REQUIRED_PACKET_INPUT_NAMES:
        item = inputs.get(input_name)
        if not isinstance(item, dict):
            problems.append(f"inputs.{input_name} is missing")
            continue
        if not _is_concrete(item.get("path")):
            problems.append(f"inputs.{input_name}.path is not concrete")
        if not _SHA256_RE.fullmatch(str(item.get("sha256") or "")):
            problems.append(f"inputs.{input_name}.sha256 is not a concrete SHA-256")
    source = inputs.get("source")
    if not isinstance(source, dict) or not _COMMIT_SHA_RE.fullmatch(
        str(source.get("public_commit") or "")
    ):
        problems.append("inputs.source.public_commit is not a concrete commit SHA")
    elif expected_sha is not None and source.get("public_commit") != expected_sha:
        problems.append("inputs.source.public_commit does not match expected source SHA")
    return problems


def _validate_packet_hash_binding(packet: dict[str, Any]) -> list[str]:
    """Validate that packet input digests are bound to identity digests.

    Returns:
        Sanitized hash-binding problems.
    """
    inputs = packet.get("inputs")
    identity_hashes = packet.get("identity")
    if not isinstance(inputs, dict) or not isinstance(identity_hashes, dict):
        return []
    problems: list[str] = []
    input_to_identity = {
        "release_manifest": "release_manifest_sha256",
        "canonical_campaign_config": "canonical_config_sha256",
        "scenario_matrix": "scenario_matrix_sha256",
        "checkpoint_staging_receipt": "checkpoint_receipt_sha256",
        "runtime_smoke_receipt": "runtime_smoke_receipt_sha256",
        "public_single_node_entrypoint": "public_entrypoint_sha256",
        "private_wrapper": "private_wrapper_sha256",
    }
    for input_name, identity_name in input_to_identity.items():
        item = inputs.get(input_name)
        if isinstance(item, dict) and item.get("sha256") != identity_hashes.get(identity_name):
            problems.append(f"{input_name} hash is not bound to packet identity")
    return problems


def _validate_packet_hashes(packet: dict[str, Any], expected_sha: str | None = None) -> list[str]:
    """Validate packet identity and input hashes without reading secrets.

    Returns:
        Sanitized hash and placeholder problems.
    """
    return _validate_packet_input_hashes(packet, expected_sha) + _validate_packet_hash_binding(
        packet
    )


def _validate_packet_state(packet: dict[str, Any]) -> list[str]:
    """Validate final private packet state and admission status.

    Returns:
        Sanitized state problems.
    """
    problems: list[str] = []
    if packet.get("schema") != "robot-sf-launch-packet.v1":
        problems.append("launch packet schema is not v1")
    if packet.get("state") not in {"admitted", "ready", "queued"}:
        problems.append("launch packet is not in a dispatchable state")
    admission = packet.get("admission")
    if isinstance(admission, dict):
        if admission.get("status") not in {"admitted", "ready"}:
            problems.append("launch packet admission status is not admitted")
        elif admission.get("dispatchable") is not True:
            problems.append("launch packet admission is not dispatchable")
    elif packet.get("status") != "admitted_frozen":
        # The private queue's frozen packet contract uses a top-level status
        # instead of duplicating an admission submapping.  Accept only that
        # exact frozen status; an absent or arbitrary status remains blocked.
        problems.append("launch packet admission status is not admitted")
    if packet.get("dispatchable") is not True:
        problems.append("launch packet is not dispatchable")

    return problems


def _validate_packet_identity(
    packet: dict[str, Any],
    expected_sha: str,
    *,
    expected_tag: str | None,
    expected_campaign_id: str | None,
) -> list[str]:
    """Validate final private packet source, tag, and campaign identity.

    Returns:
        Sanitized identity problems.
    """
    problems: list[str] = []
    identity = packet.get("identity")
    if not isinstance(identity, dict) or identity.get("public_source_commit") != expected_sha:
        problems.append("launch packet source SHA does not match")
    packet_contract = packet.get("execution_contract")
    if expected_tag is not None and (
        not isinstance(packet_contract, dict) or packet_contract.get("release_tag") != expected_tag
    ):
        problems.append("launch packet release tag does not match")
    if expected_campaign_id is not None and packet.get("campaign_id") != expected_campaign_id:
        problems.append("launch packet campaign ID does not match")
    if packet.get("campaign_id") != packet.get("campaign"):
        problems.append("launch packet campaign identity is inconsistent")
    problems.extend(_validate_packet_hashes(packet, expected_sha))
    return problems


def _validate_packet_resource_contract(contract: dict[str, Any]) -> list[str]:
    """Validate resource completeness and internal route/time consistency.

    Returns:
        Sanitized resource-contract problems.
    """
    problems: list[str] = []
    for field in _RESOURCE_FIELDS:
        actual = contract.get(field)
        if field in _RESOURCE_STRING_FIELDS:
            valid = _is_concrete(actual)
        else:
            parsed = _strict_int(actual)
            valid = parsed is not None and parsed > 0
        if not valid:
            problems.append(f"launch packet resource contract is invalid: {field}")
    if _is_concrete(contract.get("cluster")) and _is_concrete(contract.get("partition")):
        expected_route = f"{contract['cluster']}:{contract['partition']}"
        if not _resource_values_match("route_id", contract.get("route_id"), expected_route):
            problems.append("launch packet route_id does not match cluster and partition")
    wall_clock_seconds = _wall_clock_seconds(contract.get("wall_clock"))
    if wall_clock_seconds is None:
        problems.append("launch packet wall_clock is not a valid HH:MM:SS value")
    elif wall_clock_seconds <= 0:
        problems.append("launch packet wall_clock must be positive")
    elif "wall_clock_seconds" in contract and not _resource_values_match(
        "wall_clock_seconds", contract.get("wall_clock_seconds"), wall_clock_seconds
    ):
        problems.append("launch packet wall_clock_seconds does not match wall_clock")
    return problems


def _validate_packet_execution_contract(packet: dict[str, Any]) -> list[str]:
    """Validate the frozen route and startup-source contract.

    Resource values are validated for completeness and internal consistency
    here.  Final admission compares them with the exact matching private queue
    row in :func:`_validate_packet_queue`.

    Returns:
        Sanitized route and startup-contract problems.
    """
    problems: list[str] = []
    contract = packet.get("execution_contract")
    if not isinstance(contract, dict):
        problems.append("launch packet execution contract is missing")
        contract = {}
    problems.extend(_validate_packet_resource_contract(contract))
    if contract.get("resources_exact") is not True:
        problems.append("launch packet resources_exact is not true")
    if not _is_concrete(contract.get("release_label")):
        problems.append("launch packet release_label is not concrete")
    if not isinstance(contract.get("force_cpu"), bool):
        problems.append("launch packet force_cpu is not boolean")
    if contract.get("startup_sentinel_required") is not True:
        problems.append("startup sentinel is not required")
    if "$SLURM_STARTUP_SENTINEL" not in str(contract.get("startup_prefix") or ""):
        problems.append("startup sentinel is not sourced before launch")
    problems.extend(_validate_runtime_smoke_contract(packet, contract))
    return problems


def _validate_runtime_smoke_contract(packet: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    """Validate the fresh exact-source smoke hand-off contract.

    Returns:
        Sanitized smoke-contract problems.
    """
    problems: list[str] = []
    smoke_max_age = _strict_int(contract.get("runtime_smoke_receipt_max_age_hours"))
    smoke_max_age_matches = smoke_max_age == 24
    if not smoke_max_age_matches:
        problems.append("runtime smoke receipt freshness contract is not 24 hours")
    inputs = packet.get("inputs")
    entrypoint = inputs.get("public_single_node_entrypoint") if isinstance(inputs, dict) else None
    if not isinstance(entrypoint, dict) or entrypoint.get("interface_arity") != 5:
        problems.append("public release entrypoint does not require five arguments")
    elif entrypoint.get("fifth_argument") != "exact_source_runtime_smoke_result":
        problems.append("public release entrypoint does not require exact-source runtime smoke")
    return problems


def _validate_packet_file_hashes(packet: dict[str, Any], repo: Path | None) -> list[str]:
    """Recompute every declared public-input hash from the release checkout.

    Returns:
        Sanitized file-hash problems.  Final admission has no implicit
        remote-only input mode: the private packet schema does not declare one,
        so every public input must be present in the exact checkout.
    """
    if repo is None:
        return ["public-input checkout is required for final packet admission"]
    inputs = packet.get("inputs")
    if not isinstance(inputs, dict):
        return ["launch packet inputs are missing"]
    repo_root = repo.resolve()
    problems: list[str] = []
    for input_name in _PUBLIC_PACKET_INPUT_NAMES:
        item = inputs.get(input_name)
        if not isinstance(item, dict):
            problems.append(f"inputs.{input_name} is missing")
            continue
        raw_path = str(item.get("path") or "")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(repo_root)
        except (OSError, RuntimeError, ValueError):
            problems.append(f"inputs.{input_name} is outside the public checkout")
            continue
        if not resolved.is_file():
            problems.append(f"inputs.{input_name} file is missing from the public checkout")
            continue
        expected = str(item.get("sha256") or "").lower()
        try:
            actual = _sha256(resolved).lower()
        except OSError:
            problems.append(f"inputs.{input_name} file could not be read from the public checkout")
            continue
        if actual != expected:
            problems.append(f"inputs.{input_name} hash does not match checkout")
    return problems


def _validate_packet_traceability(packet: dict[str, Any]) -> list[str]:
    """Validate startup and admission sentinel traceability requirements.

    Returns:
        Sanitized traceability problems.
    """
    problems: list[str] = []
    traceability = packet.get("sentinel_traceability")
    if not isinstance(traceability, dict) or traceability.get("required") is not True:
        problems.append("startup traceability is not required")
    else:
        required_fields = set(traceability.get("required_identity_fields") or [])
        missing_fields = sorted(_REQUIRED_PACKET_TRACE_FIELDS - required_fields)
        if missing_fields:
            problems.append("startup traceability omits required identity fields")
        if not _is_concrete(traceability.get("source")):
            problems.append("startup sentinel source is missing")
        if not _is_concrete(traceability.get("helper")):
            problems.append("startup admission helper is missing")
        if not _is_concrete(traceability.get("startup_receipt")):
            problems.append("startup receipt contract is missing")
        if not _is_concrete(traceability.get("admission_trace")):
            problems.append("admission trace contract is missing")
    return problems


def _parse_release_exports(  # noqa: C901, PLR0912
    submit_args: str,
) -> tuple[dict[str, list[str]], list[str]]:
    """Parse exact ``RELEASE_*`` assignments from Slurm ``--export`` flags.

    Queue rows store the arguments passed through ``submit_and_record.sh``.
    They may use raw scheduler options or wrap them as ``--sbatch-arg``
    options, and Slurm's export value is itself a comma-delimited list.  A
    substring search over the raw text would allow a key such as
    ``XRELEASE_CAMPAIGN_ID`` to satisfy an exact identity check, while a
    first-match parser would allow a later duplicate to change the effective
    value.  Keep every exact key visible so callers can reject both cases.

    Returns:
        A mapping from exact release environment keys to all exported values,
        plus sanitized parser problems.  Values are never included in a
        problem message.
    """
    try:
        tokens = shlex.split(submit_args)
    except ValueError:
        return {}, ["private queue submit_args quoting is invalid"]

    values: dict[str, list[str]] = {}
    problems: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--sbatch-arg":
            index += 1
            if index >= len(tokens) or not tokens[index].startswith("--"):
                problems.append("private queue --sbatch-arg is missing an option")
                continue
            token = tokens[index]
        elif token.startswith("--sbatch-arg="):
            token = token.split("=", 1)[1]

        if not token.startswith("--"):
            index += 1
            continue
        option_value = token[2:]
        if "=" in option_value:
            option, value = option_value.split("=", 1)
        else:
            option, value = option_value, None
        if option != "export":
            index += 1
            continue
        if value is None:
            next_index = index + 1
            if next_index >= len(tokens) or tokens[next_index].startswith("--"):
                problems.append("private queue --export is missing a value")
                index += 1
                continue
            value = tokens[next_index]
            index = next_index

        for assignment in value.split(","):
            if assignment in {"ALL", "NONE"} or not assignment:
                continue
            if "=" not in assignment:
                if assignment.startswith("RELEASE_"):
                    problems.append("private queue RELEASE export is missing a value")
                continue
            field, field_value = assignment.split("=", 1)
            if not field.startswith("RELEASE_"):
                continue
            if not re.fullmatch(r"RELEASE_[A-Za-z0-9_]+", field):
                problems.append("private queue RELEASE export key is invalid")
                continue
            values.setdefault(field, []).append(field_value)
        index += 1

    for field, matching in values.items():
        if len(matching) > 1:
            problems.append(f"private queue {field} is duplicated")
    return values, problems


def _release_export_value(
    values: dict[str, list[str]], field: str, problems: list[str]
) -> str | None:
    """Return one exact release export value, rejecting missing/duplicate keys."""
    matching = values.get(field, [])
    if not matching:
        problems.append(f"private queue {field} is missing")
        return None
    if len(matching) != 1:
        # The parser already records a sanitized duplicate diagnostic.  Do not
        # select a value from an ambiguous assignment list.
        return None
    return matching[0]


_SCHEDULER_OPTIONS = (
    "partition",
    "gres",
    "cpus-per-task",
    "mem",
    "time",
    "qos",
)


def _parse_scheduler_args(  # noqa: C901
    submit_args: str,
) -> tuple[dict[str, list[str]], list[str]]:
    """Parse scheduler flags from raw or ``--sbatch-arg`` submit arguments.

    Returns:
        A mapping of recognized option names to values and sanitized parser
        problems.  Duplicate options remain visible for fail-closed checks.
    """
    try:
        tokens = shlex.split(submit_args)
    except ValueError:
        return {}, ["scheduler submit_args quoting is invalid"]
    values: dict[str, list[str]] = {}
    problems: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--sbatch-arg":
            index += 1
            if index >= len(tokens) or not tokens[index].startswith("--"):
                problems.append("scheduler --sbatch-arg is missing an option")
                continue
            token = tokens[index]
        elif token.startswith("--sbatch-arg="):
            token = token.split("=", 1)[1]
        if not token.startswith("--"):
            index += 1
            continue
        option_value = token[2:]
        if "=" in option_value:
            option, value = option_value.split("=", 1)
        else:
            option, value = option_value, None
        if option not in _SCHEDULER_OPTIONS:
            index += 1
            continue
        if value is None:
            next_index = index + 1
            if next_index >= len(tokens) or tokens[next_index].startswith("--"):
                problems.append(f"scheduler --{option} is missing a value")
                index += 1
                continue
            value = tokens[next_index]
            index = next_index
        values.setdefault(option, []).append(value)
        index += 1
    return values, problems


def _scheduler_option_value(
    values: dict[str, list[str]], option: str, problems: list[str]
) -> str | None:
    """Return one scheduler option value, rejecting missing or duplicate flags."""
    matching = values.get(option, [])
    if not matching:
        problems.append(f"scheduler --{option} is missing")
        return None
    if len(matching) != 1:
        problems.append(f"scheduler --{option} is duplicated")
        return None
    return matching[0]


def _scheduler_memory_mib(value: str) -> int | None:
    """Normalize an integer Slurm memory value to mebibytes.

    Returns:
        The normalized mebibytes, or ``None`` for malformed/non-integral input.
    """
    match = re.fullmatch(r"([0-9]+)([kmgt]?i?b?)?", value.strip(), re.IGNORECASE)
    if match is None:
        return None
    amount = _strict_int(match.group(1))
    if amount is None:
        return None
    suffix = (match.group(2) or "m").lower()
    suffix = suffix.removesuffix("ib").removesuffix("b")
    if suffix == "k":
        return amount // 1024 if amount % 1024 == 0 else None
    if suffix == "m":
        return amount
    if suffix == "g":
        return amount * 1024
    if suffix == "t":
        return amount * 1024 * 1024
    return None


def _scheduler_time_seconds(value: str) -> int | None:
    """Normalize Slurm ``D-HH:MM:SS`` or ``HH:MM:SS`` time values.

    Returns:
        The normalized seconds, or ``None`` for malformed/non-integral input.
    """
    raw = value.strip()
    days = 0
    if "-" in raw:
        day_text, raw = raw.split("-", 1)
        days = _strict_int(day_text)
        if days is None:
            return None
    parts = raw.split(":")
    parsed = [_strict_int(part) for part in parts]
    if any(part is None for part in parsed) or len(parsed) not in {2, 3}:
        return None
    if len(parsed) == 2:
        hours, minutes, seconds = 0, parsed[0], parsed[1]
    else:
        hours, minutes, seconds = parsed
    if hours is None or minutes is None or seconds is None:
        return None
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return None
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return total_seconds if total_seconds > 0 else None


def _scheduler_gres(value: str) -> tuple[int, str | None] | None:
    """Parse one GPU GRES entry into count and optional type.

    A composite GRES value (e.g. ``gpu:l40s:1,mps:1``) is rejected because the
    effective submitted job would carry scheduler resources the frozen packet
    did not declare; extra entries cannot be silently discarded.

    Returns:
        A ``(count, type)`` pair, or ``None`` for malformed or composite GRES input.
    """
    entries = [entry for entry in value.split(",") if entry.lower().startswith("gpu:")]
    if len(entries) != 1 or len(value.split(",")) != 1:
        return None
    parts = entries[0].split(":")
    if len(parts) == 2:
        gpu_type, count_text = None, parts[1]
    elif len(parts) == 3:
        gpu_type, count_text = parts[1], parts[2]
    else:
        return None
    count = _strict_int(count_text)
    if count is None or count <= 0 or (gpu_type is not None and not _is_concrete(gpu_type)):
        return None
    return count, gpu_type


def _validate_scheduler_submit_args(  # noqa: C901
    submit_args: str, packet: dict[str, Any]
) -> list[str]:
    """Validate scheduler flags against the frozen packet execution contract.

    Returns:
        Sanitized scheduler-contract problems.
    """
    contract = packet.get("execution_contract")
    if not isinstance(contract, dict):
        return ["launch packet execution contract is missing"]
    values, problems = _parse_scheduler_args(submit_args)
    partition = _scheduler_option_value(values, "partition", problems)
    if partition is not None and not _resource_values_match(
        "partition", partition, contract.get("partition")
    ):
        problems.append("scheduler --partition does not match packet")
    gres = _scheduler_option_value(values, "gres", problems)
    if gres is not None:
        parsed_gres = _scheduler_gres(gres)
        expected_gpus = _strict_int(contract.get("gpus"))
        if parsed_gres is None or expected_gpus is None:
            problems.append("scheduler --gres is invalid")
        else:
            gpu_count, gpu_type = parsed_gres
            if gpu_count != expected_gpus:
                problems.append("scheduler --gres GPU count does not match packet")
            expected_gpu_type = contract.get("gpu_type")
            if (
                _is_concrete(expected_gpu_type)
                and gpu_type is not None
                and not _resource_values_match("gpu_type", gpu_type, expected_gpu_type)
            ):
                problems.append("scheduler --gres GPU type does not match packet")
    cpus = _scheduler_option_value(values, "cpus-per-task", problems)
    if cpus is not None and not _resource_values_match("cpus", cpus, contract.get("cpus")):
        problems.append("scheduler --cpus-per-task does not match packet")
    memory = _scheduler_option_value(values, "mem", problems)
    expected_mem_gb = _strict_int(contract.get("mem_gb"))
    if memory is not None and (
        expected_mem_gb is None or _scheduler_memory_mib(memory) != expected_mem_gb * 1024
    ):
        problems.append("scheduler --mem does not match packet")
    scheduler_time = _scheduler_option_value(values, "time", problems)
    expected_time = contract.get("wall_clock_seconds")
    if expected_time is None:
        expected_time = _wall_clock_seconds(contract.get("wall_clock"))
    if scheduler_time is not None and (
        expected_time is None or _scheduler_time_seconds(scheduler_time) != expected_time
    ):
        problems.append("scheduler --time does not match packet")
    expected_qos = contract.get("qos")
    if expected_qos is None or not _is_concrete(expected_qos):
        if values.get("qos"):
            problems.append("scheduler --qos is undeclared by the packet")
    else:
        qos = _scheduler_option_value(values, "qos", problems)
        if qos is not None and not _resource_values_match("qos", qos, expected_qos):
            problems.append("scheduler --qos does not match packet")
    return problems


def _has_exact_scheduler_time(submit_args: str, packet: dict[str, Any]) -> bool:
    """Return whether submit args provide exact packet-bound scheduler time."""
    contract = packet.get("execution_contract")
    if not isinstance(contract, dict):
        return False
    values, problems = _parse_scheduler_args(submit_args)
    if problems or len(values.get("time", [])) != 1:
        return False
    expected = contract.get("wall_clock_seconds")
    if expected is None:
        expected = _wall_clock_seconds(contract.get("wall_clock"))
    actual = _scheduler_time_seconds(values["time"][0])
    return expected is not None and actual == expected


def _validate_queue_identity_args(
    exports: dict[str, list[str]], packet: dict[str, Any], problems: list[str]
) -> list[str]:
    """Validate queue environment values against packet identity.

    Returns:
        Sanitized queue identity problems.
    """
    identity = packet.get("identity")
    inputs = packet.get("inputs")
    if isinstance(identity, dict) and isinstance(inputs, dict):
        queue_to_identity = {
            "RELEASE_MANIFEST_SHA256": "release_manifest_sha256",
            "RELEASE_CONFIG_SHA256": "canonical_config_sha256",
            "RELEASE_SCENARIO_SHA256": "scenario_matrix_sha256",
            "RELEASE_CHECKPOINT_RECEIPT_SHA256": "checkpoint_receipt_sha256",
            "RELEASE_RUNTIME_SMOKE_RECEIPT_SHA256": "runtime_smoke_receipt_sha256",
            "RELEASE_PUBLIC_SCRIPT_SHA256": "public_entrypoint_sha256",
        }
        for queue_field, identity_field in queue_to_identity.items():
            if _release_export_value(exports, queue_field, problems) != identity.get(
                identity_field
            ):
                problems.append(f"private queue {queue_field} is not bound to packet identity")
        input_paths = {
            "RELEASE_MANIFEST_PATH": "release_manifest",
            "RELEASE_SCENARIO_PATH": "scenario_matrix",
            "RELEASE_CHECKPOINT_RECEIPT_PATH": "checkpoint_staging_receipt",
            "RELEASE_RUNTIME_SMOKE_RECEIPT_PATH": "runtime_smoke_receipt",
        }
        for queue_field, input_name in input_paths.items():
            item = inputs.get(input_name)
            expected_path = item.get("path") if isinstance(item, dict) else None
            if _release_export_value(exports, queue_field, problems) != expected_path:
                problems.append(f"private queue {queue_field} is not bound to packet inputs")
    contract = packet.get("execution_contract")
    if not isinstance(contract, dict):
        problems.append("launch packet execution contract is missing")
        contract = {}
    expected_identity = {
        "RELEASE_CAMPAIGN_ID": packet.get("campaign_id"),
        "RELEASE_LABEL": contract.get("release_label"),
        "RELEASE_EXPECTED_CPUS": contract.get("cpus"),
        "RELEASE_EXPECTED_GPUS": contract.get("gpus"),
        "RELEASE_EXPECTED_MEM_GB": contract.get("mem_gb"),
        "RELEASE_EXPECTED_WALLTIME": contract.get("wall_clock"),
        "RELEASE_FORCE_CPU": "1" if contract.get("force_cpu") is True else "0",
    }
    for field, expected in expected_identity.items():
        if _release_export_value(exports, field, problems) != str(expected):
            problems.append(f"private queue {field} is not bound to packet identity")
    return problems


def _validate_queue_submit_args(
    submit_args: str, packet_path: Path, packet: dict[str, Any]
) -> list[str]:
    """Validate frozen queue launch arguments and their content hashes.

    Returns:
        Sanitized submit-argument problems.
    """
    if not _is_concrete(submit_args):
        return ["private queue submit arguments are not frozen"]
    problems: list[str] = []
    exports, export_problems = _parse_release_exports(submit_args)
    problems.extend(export_problems)
    expected_packet_hash = _sha256(packet_path)
    packet_hash = _release_export_value(exports, "RELEASE_LAUNCH_PACKET_SHA256", problems)
    if packet_hash != expected_packet_hash:
        problems.append("private queue packet hash is not bound")
    packet_path_export = _release_export_value(exports, "RELEASE_LAUNCH_PACKET_PATH", problems)
    if (
        not _is_concrete(packet_path_export)
        or Path(str(packet_path_export)).name != packet_path.name
    ):
        problems.append("private queue packet path is not bound")
    for field in (
        "RELEASE_MANIFEST_SHA256",
        "RELEASE_CONFIG_SHA256",
        "RELEASE_SCENARIO_SHA256",
        "RELEASE_CHECKPOINT_RECEIPT_SHA256",
        "RELEASE_RUNTIME_SMOKE_RECEIPT_SHA256",
        "RELEASE_PUBLIC_SCRIPT_SHA256",
    ):
        value = _release_export_value(exports, field, problems)
        if value is None or not _SHA256_RE.fullmatch(value):
            problems.append(f"private queue {field} is not a concrete SHA-256")
    problems.extend(_validate_scheduler_submit_args(submit_args, packet))
    identity_problems: list[str] = []
    problems.extend(_validate_queue_identity_args(exports, packet, identity_problems))
    return problems


def _validate_queue_resources(  # noqa: C901
    row: dict[str, Any], packet: dict[str, Any]
) -> list[str]:
    """Validate queue resources against the exact packet execution contract.

    Returns:
        Sanitized resource problems.
    """
    problems: list[str] = []
    contract = packet.get("execution_contract")
    if not isinstance(contract, dict):
        return ["launch packet execution contract is missing"]
    for field in _QUEUE_RESOURCE_FIELDS:
        if field not in row:
            problems.append(f"private queue resource contract is missing: {field}")
            continue
        if not _resource_values_match(field, row[field], contract.get(field)):
            problems.append(f"private queue resource contract mismatch: {field}")
    # Queue rows encode wall-clock limits as seconds, while the launch packet
    # carries the scheduler-facing HH:MM:SS value.  Accept the explicit packet
    # seconds when present and otherwise derive them from the frozen string.
    expected_wall_clock_seconds = contract.get("wall_clock_seconds")
    if expected_wall_clock_seconds is None:
        expected_wall_clock_seconds = _wall_clock_seconds(contract.get("wall_clock"))
    if "estimated_elapsed_sec" in row:
        estimated_elapsed_sec = _strict_int(row["estimated_elapsed_sec"])
        if estimated_elapsed_sec is None or estimated_elapsed_sec <= 0:
            problems.append("private queue estimated_elapsed_sec must be positive")
        elif expected_wall_clock_seconds is None or not _resource_values_match(
            "estimated_elapsed_sec", estimated_elapsed_sec, expected_wall_clock_seconds
        ):
            problems.append("private queue resource contract mismatch: wall_clock")
    elif not _has_exact_scheduler_time(str(row.get("submit_args") or ""), packet):
        problems.append("private queue estimated_elapsed_sec or exact scheduler --time is required")
    if "gpu_type" in row and not _resource_values_match(
        "gpu_type", row["gpu_type"], contract.get("gpu_type")
    ):
        problems.append("private queue resource contract mismatch: gpu_type")
    if "wall_clock" in row and not _resource_values_match(
        "wall_clock", row["wall_clock"], contract.get("wall_clock")
    ):
        problems.append("private queue resource contract mismatch: wall_clock")
    if "qos" in row and not _resource_values_match("qos", row["qos"], contract.get("qos")):
        problems.append("private queue resource contract mismatch: qos")
    return problems


def _validate_packet_queue(  # noqa: C901
    packet: dict[str, Any], packet_path: Path, queue_path: Path | None, expected_sha: str
) -> list[str]:
    """Validate private queue identity, hashes, and packet-bound resources.

    Returns:
        Sanitized queue admission problems.
    """
    problems: list[str] = []
    if queue_path is None:
        return ["private queue path is required for final admission"]
    try:
        rows = _load_queue_rows(queue_path)
    except (OSError, ValueError, yaml.YAMLError):
        return ["private queue is missing or invalid"]
    queue_id = packet.get("queue_id")
    if not isinstance(queue_id, str) or not _is_concrete(queue_id):
        return ["launch packet queue_id is missing or not concrete"]
    if any(
        not isinstance(row.get("queue_id"), str) or not _is_concrete(row.get("queue_id"))
        for row in rows
    ):
        return ["private queue row queue_id is missing or not concrete"]
    matching = [row for row in rows if row.get("queue_id") == queue_id]
    if len(matching) != 1:
        return ["private queue does not contain exactly one packet row"]
    row = matching[0]
    if not isinstance(row.get("queue_id"), str) or not _is_concrete(row.get("queue_id")):
        return ["private queue row queue_id is missing or not concrete"]
    if row.get("campaign") != packet.get("campaign_id"):
        problems.append("private queue campaign identity does not match")
    if row.get("expected_public_commit") != expected_sha:
        problems.append("private queue source SHA does not match")
    if row.get("state") not in {"ready", "queued"}:
        problems.append("private queue row is not dispatchable")
    # Host-specific private-ops roots differ between the local and submit
    # machines.  Compare the stable packet filename while the submit-args
    # digest binds the exact bytes.
    artifact_manifest = str(row.get("artifact_manifest") or "")
    artifact_path, separator, artifact_digest = artifact_manifest.partition(" sha256:")
    if not artifact_path.endswith(packet_path.name):
        problems.append("private queue artifact manifest does not match packet")
    if not separator:
        problems.append("private queue artifact manifest is not digest-bound to the packet")
    elif (
        not _SHA256_RE.fullmatch(artifact_digest)
        or artifact_digest.lower() != _sha256(packet_path).lower()
    ):
        problems.append("private queue artifact manifest hash does not match packet")
    submit_args = str(row.get("submit_args") or "")
    problems.extend(_validate_queue_submit_args(submit_args, packet_path, packet))
    problems.extend(_validate_queue_resources(row, packet))
    return list(dict.fromkeys(problems))


def _validate_packet_private_evidence(
    packet: dict[str, Any],
    *,
    checkpoint_receipt_path: Path | None,
) -> list[str]:
    """Verify private evidence files match the digests pinned by the packet.

    The packet's identity block pins ``checkpoint_receipt_sha256``.  The doctor
    must resolve the actual private evidence from the admitted packet and fail
    closed when the file at the CLI-provided path drifts from the pinned digest,
    so a stale, moved, or contradictory receipt cannot silently satisfy final
    admission.

    Returns:
        Sanitized private-evidence drift problems.
    """
    problems: list[str] = []
    identity = packet.get("identity")
    pinned_receipt_sha = (
        str(identity.get("checkpoint_receipt_sha256") or "").lower()
        if isinstance(identity, dict)
        else ""
    )
    if not pinned_receipt_sha:
        return ["packet identity does not pin checkpoint_receipt_sha256"]
    if checkpoint_receipt_path is None or not checkpoint_receipt_path.is_file():
        return ["checkpoint receipt file is missing for packet-pinned evidence"]
    try:
        actual_sha = _sha256(checkpoint_receipt_path).lower()
    except OSError:
        return ["checkpoint receipt file could not be read for packet-pinned evidence"]
    if actual_sha != pinned_receipt_sha:
        problems.append("checkpoint receipt hash does not match packet-pinned evidence")
    return problems


def _validate_packet_contract(
    packet: dict[str, Any],
    expected_sha: str,
    *,
    expected_tag: str | None,
    expected_campaign_id: str | None,
    packet_path: Path,
    queue_path: Path | None,
    repo: Path | None,
    checkpoint_receipt: Path | None = None,
) -> list[str]:
    """Validate final private launch identity, route, hashes, and startup contract.

    Returns:
        Sanitized final-admission problems.
    """
    problems = _validate_packet_state(packet)
    problems.extend(
        _validate_packet_identity(
            packet,
            expected_sha,
            expected_tag=expected_tag,
            expected_campaign_id=expected_campaign_id,
        )
    )
    problems.extend(_validate_packet_file_hashes(packet, repo))
    problems.extend(
        _validate_packet_private_evidence(packet, checkpoint_receipt_path=checkpoint_receipt)
    )
    problems.extend(_validate_packet_execution_contract(packet))
    problems.extend(_validate_packet_traceability(packet))
    problems.extend(_validate_packet_queue(packet, packet_path, queue_path, expected_sha))
    return list(dict.fromkeys(problems))


def _cluster_check(
    packet_path: Path | None,
    expected_sha: str,
    *,
    final: bool = False,
    expected_tag: str | None = None,
    expected_campaign_id: str | None = None,
    queue_path: Path | None = None,
    repo: Path | None = None,
    checkpoint_receipt: Path | None = None,
) -> ReleaseDoctorCheck:
    """Require an admitted launch packet bound to the frozen public SHA.

    Returns:
        Sanitized check result.
    """
    if packet_path is None or not packet_path.is_file():
        return ReleaseDoctorCheck("cluster_admission", "fail", "private launch packet is missing")
    try:
        packet = _load_mapping(packet_path)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
        return ReleaseDoctorCheck("cluster_admission", "fail", "private launch packet is invalid")
    admission = packet.get("admission")
    admitted = (
        isinstance(admission, dict) and admission.get("status") in {"admitted", "ready"}
    ) or packet.get("status") == "admitted_frozen"
    dispatchable = packet.get("dispatchable") is True and (
        not isinstance(admission, dict) or admission.get("dispatchable", True) is True
    )
    identity = packet.get("identity")
    identity_source_sha = (
        identity.get("public_source_commit") if isinstance(identity, dict) else None
    )
    source_sha = packet.get("public_source_sha") or packet.get("release_sha") or identity_source_sha
    problems = []
    if not admitted:
        problems.append("launch packet is not admitted")
    if not dispatchable:
        problems.append("launch packet is not dispatchable")
    if source_sha != expected_sha:
        problems.append("launch packet source SHA does not match")
    if final:
        problems.extend(
            _validate_packet_contract(
                packet,
                expected_sha,
                expected_tag=expected_tag,
                expected_campaign_id=expected_campaign_id,
                packet_path=packet_path,
                queue_path=queue_path,
                repo=repo,
                checkpoint_receipt=checkpoint_receipt,
            )
        )
    return ReleaseDoctorCheck(
        "cluster_admission",
        "pass" if not problems else "fail",
        "; ".join(problems) or "launch packet admitted",
    )


def _disk_check(path: Path, minimum_free_gib: float) -> ReleaseDoctorCheck:
    """Check free space at the intended artifact root.

    Returns:
        Sanitized check result.
    """
    free_gib = shutil.disk_usage(path).free / (1024**3)
    passed = free_gib >= minimum_free_gib
    return ReleaseDoctorCheck(
        "disk_capacity",
        "pass" if passed else "fail",
        f"{free_gib:.1f} GiB free; requires {minimum_free_gib:.1f} GiB",
    )


def _zenodo_check(
    repo: Path,
    token_file: Path | None,
    *,
    require_hook_disabled: bool,
) -> list[ReleaseDoctorCheck]:
    """Check personal-token hygiene/auth and GitHub Zenodo-hook state.

    Returns:
        Sanitized auth and hook checks.
    """
    checks: list[ReleaseDoctorCheck] = []
    try:
        read_token_file(token_file or Path("<missing>"))
        session = build_session(token_file or Path("<missing>"))
        response = session.get("https://zenodo.org/api/deposit/depositions?size=1", timeout=30)
        response.raise_for_status()
    except (OSError, RuntimeError, ValueError):
        checks.append(ReleaseDoctorCheck("zenodo_auth", "fail", "Zenodo token/auth is unavailable"))
    else:
        checks.append(ReleaseDoctorCheck("zenodo_auth", "pass", "Zenodo token/auth is usable"))
    hooks = _run(["gh", "api", "repos/ll7/robot_sf_ll7/hooks"], repo)
    if hooks.returncode:
        checks.append(ReleaseDoctorCheck("zenodo_hook", "fail", "GitHub hook state is unavailable"))
        return checks
    try:
        payload = json.loads(hooks.stdout)
        zenodo_hooks = [
            hook
            for hook in payload
            if isinstance(hook, dict)
            and "zenodo" in str((hook.get("config") or {}).get("url", "")).lower()
        ]
        active = any(bool(hook.get("active")) for hook in zenodo_hooks)
    except (json.JSONDecodeError, AttributeError):
        checks.append(ReleaseDoctorCheck("zenodo_hook", "fail", "GitHub hook state is invalid"))
        return checks
    passed = not require_hook_disabled or (bool(zenodo_hooks) and not active)
    summary = (
        "Zenodo webhook is disabled"
        if zenodo_hooks and not active
        else "Zenodo webhook remains active"
        if active
        else "Zenodo webhook was not found"
    )
    checks.append(ReleaseDoctorCheck("zenodo_hook", "pass" if passed else "fail", summary))
    return checks


def _dissertation_check(path: Path | None) -> ReleaseDoctorCheck:
    """Check dissertation release paths.

    Returns:
        Sanitized check result.
    """
    if path is None or not path.is_dir():
        return ReleaseDoctorCheck("dissertation_paths", "fail", "dissertation worktree is missing")
    required = [
        path / "diss" / "robot_sf_release.tex",
        path / "docs" / "context" / "evidence_pins.yaml",
        path / "spine" / "evidence_release.yaml",
    ]
    missing = [item.relative_to(path).as_posix() for item in required if not item.is_file()]
    problems = []
    if missing:
        problems.append(f"missing required paths: {', '.join(missing)}")
    if _contains_hard_coded_robot_sf_path(path):
        problems.append("hard-coded Robot SF checkout paths remain")
    return ReleaseDoctorCheck(
        "dissertation_paths",
        "pass" if not problems else "fail",
        "; ".join(problems) or "release paths healthy",
    )


def _contains_hard_coded_robot_sf_path(root: Path) -> bool:
    """Detect absolute local Robot SF checkout paths without echoing file text.

    Repository URLs and relative references such as ``robot_sf_ll7/...`` are
    intentionally allowed.  Generated/build directories are skipped so a
    stale PDF or cache cannot create a false release blocker; source and
    metadata files remain covered recursively.

    Returns:
        ``True`` when an absolute local checkout path is found.
    """
    ignored_parts = {".git", ".venv", "build", "dist", "output", "__pycache__"}
    for candidate in root.rglob("*"):
        if not candidate.is_file() or ignored_parts.intersection(candidate.parts):
            continue
        try:
            if candidate.stat().st_size > 4 * 1024 * 1024:
                continue
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _HARDCODED_ROBOT_SF_PATH_RE.search(text):
            return True
    return False


def collect_release_doctor_report(  # noqa: PLR0913
    *,
    repo: Path,
    manifest_path: Path,
    expected_release_sha: str,
    expected_base_sha: str,
    tag: str,
    checkpoint_receipt: Path | None,
    private_launch_packet: Path | None,
    dissertation: Path | None,
    token_file: Path | None,
    checkpoint_path_map: Any = None,
    expected_cells: int = 20160,
    minimum_free_gib: float = 100.0,
    require_zenodo_webhook_disabled: bool = False,
    private_queue: Path | None = None,
    expected_campaign_id: str | None = None,
    final: bool = False,
    publication_mode: str | None = None,
) -> dict[str, Any]:
    """Collect every release-admission check without exposing credentials.

    Returns:
        Machine-readable doctor report.
    """
    if publication_mode is not None:
        if publication_mode not in {"pre-publication", "final"}:
            raise ValueError("publication_mode must be pre-publication or final")
        final = publication_mode == "final"
    cardinality_override = final and expected_cells != FULL_RELEASE_EXPECTED_EPISODE_CELLS
    manifest_check, manifest, cfg = _manifest_check(
        manifest_path,
        FULL_RELEASE_EXPECTED_EPISODE_CELLS if cardinality_override else expected_cells,
    )
    if cardinality_override:
        manifest_check = ReleaseDoctorCheck(
            "manifest",
            "fail",
            "final doctor cardinality is fixed to manifest-required "
            f"{FULL_RELEASE_EXPECTED_EPISODE_CELLS} cells; unsafe override rejected; "
            + manifest_check.summary,
        )
    if final or private_queue is not None or expected_campaign_id is not None:
        cluster_check = _cluster_check(
            private_launch_packet,
            expected_release_sha,
            final=final,
            expected_tag=tag,
            expected_campaign_id=expected_campaign_id,
            queue_path=private_queue,
            repo=repo,
            checkpoint_receipt=checkpoint_receipt,
        )
    else:
        # Preserve the lightweight preparation-mode contract for callers that
        # only have a draft packet and no queue row yet.
        cluster_check = _cluster_check(private_launch_packet, expected_release_sha)
    if final and checkpoint_path_map:
        checkpoint_check = ReleaseDoctorCheck(
            "checkpoints",
            "fail",
            "checkpoint path remaps are diagnostic-only and cannot satisfy final publication "
            "admission",
        )
    else:
        checkpoint_kwargs = (
            {
                "repo_root": repo,
                "checkpoint_path_map": checkpoint_path_map,
            }
            if checkpoint_path_map
            else {}
        )
        checkpoint_check = _checkpoint_check(
            cfg,
            manifest,
            checkpoint_receipt,
            **checkpoint_kwargs,
        )
    checks = [
        _git_check(repo, expected_release_sha),
        _ci_check(repo, expected_release_sha),
        _tag_check(repo, tag),
        manifest_check,
        _release_identity_check(manifest, expected_base_sha, tag),
        checkpoint_check,
        cluster_check,
        _disk_check(repo, minimum_free_gib),
        *_zenodo_check(
            repo,
            token_file,
            require_hook_disabled=require_zenodo_webhook_disabled or final,
        ),
        _dissertation_check(dissertation),
    ]
    failed = [check.name for check in checks if check.status != "pass"]
    return {
        "schema_version": "robot-sf-release-doctor.v1",
        "status": "pass" if not failed else "blocked",
        "expected_release_sha": expected_release_sha,
        "expected_base_sha": expected_base_sha,
        "release_tag": tag,
        "publication_mode": "final" if final else "pre-publication",
        "checks": [asdict(check) for check in checks],
        "failed_checks": failed,
    }


__all__ = ["ReleaseDoctorCheck", "collect_release_doctor_report"]
