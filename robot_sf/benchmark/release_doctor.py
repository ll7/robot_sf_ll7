"""Deterministic, credential-safe doctor for benchmark-data release admission."""

from __future__ import annotations

import hashlib
import json
import re
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
_HARDCODED_ROBOT_SF_PATH_RE = re.compile(
    r"(?<![\w/])/(?:[^\s/\\\"'<>]+/)*robot_sf_ll7(?:[/\\][^\s\\\"'<>]*)?",
    re.IGNORECASE,
)

_RELEASE_RESOURCES = {
    "cluster": "licca",
    "partition": "epyc-gpu",
    "route_id": "licca:epyc-gpu",
    "cpus": 36,
    "gpus": 1,
    "gpu_type": "a100",
    "mem_gb": 256,
    "wall_clock": "36:00:00",
}

_REQUIRED_PACKET_HASH_FIELDS = (
    "release_manifest_sha256",
    "canonical_config_sha256",
    "scenario_matrix_sha256",
    "checkpoint_receipt_sha256",
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
    "release_label",
}


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
            "headSha,status,conclusion,workflowName,name",
        ],
        repo,
    )
    if result.returncode:
        return ReleaseDoctorCheck(
            "ci", "fail", "exact-source required workflow state is unavailable"
        )
    try:
        runs = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        runs = []
    if not isinstance(runs, list):
        runs = []
    exact = [run for run in runs if isinstance(run, dict) and run.get("headSha") == expected_sha]
    by_workflow: dict[str, list[dict[str, Any]]] = {name: [] for name in required_workflows}
    for run in exact:
        workflow_name = str(
            run.get("workflowName") or run.get("name") or run.get("workflow") or ""
        ).strip()
        if workflow_name in by_workflow:
            by_workflow[workflow_name].append(run)

    missing = [name for name, matching in by_workflow.items() if not matching]
    non_green = [
        name
        for name, matching in by_workflow.items()
        if matching
        and any(
            str(run.get("status", "")).lower() != "completed"
            or str(run.get("conclusion", "")).lower() != "success"
            for run in matching
        )
    ]
    problems = []
    if missing:
        problems.append("missing " + ", ".join(missing))
    if non_green:
        problems.append("not completed green: " + ", ".join(non_green))
    green = not problems
    return ReleaseDoctorCheck(
        "ci",
        "pass" if green else "fail",
        "all exact-source required workflows are green" if green else "; ".join(problems),
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
    if remote_ref.returncode not in {0, 2}:
        return ReleaseDoctorCheck("tag_collision", "fail", "remote tag state is unavailable")
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


def _checkpoint_check(cfg: Any, manifest: Any, receipt: Path | None) -> ReleaseDoctorCheck:
    """Validate exact staged-checkpoint admission.

    Returns:
        Sanitized check result.
    """
    if cfg is None or manifest is None or receipt is None:
        return ReleaseDoctorCheck("checkpoints", "fail", "staged-checkpoint receipt is missing")
    try:
        payload = validate_checkpoint_staging_receipt(
            cfg,
            receipt,
            campaign_config_path=manifest.canonical_campaign_config_path,
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
    for input_name in (
        "release_manifest",
        "canonical_campaign_config",
        "scenario_matrix",
        "public_single_node_entrypoint",
        "checkpoint_staging_receipt",
        "private_wrapper",
    ):
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
    if not isinstance(admission, dict) or admission.get("status") not in {"admitted", "ready"}:
        problems.append("launch packet admission status is not admitted")
    elif admission.get("dispatchable") is not True:
        problems.append("launch packet admission is not dispatchable")
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


def _validate_packet_execution_contract(packet: dict[str, Any]) -> list[str]:
    """Validate the fixed LiCCA route and startup-source contract.

    Returns:
        Sanitized route and startup-contract problems.
    """
    problems: list[str] = []
    contract = packet.get("execution_contract")
    if not isinstance(contract, dict):
        problems.append("launch packet execution contract is missing")
        contract = {}
    for field, expected in _RELEASE_RESOURCES.items():
        actual = contract.get(field)
        if field in {"cluster", "partition", "route_id", "wall_clock"}:
            matches = str(actual or "").strip().lower() == str(expected).lower()
        elif field == "gpu_type":
            matches = str(actual or "").strip().lower() == str(expected).lower()
        else:
            try:
                matches = int(actual) == int(expected)
            except (TypeError, ValueError):
                matches = False
        if not matches:
            problems.append(f"launch packet resource contract mismatch: {field}")
    if contract.get("resources_exact") is not True:
        problems.append("launch packet resources_exact is not true")
    if contract.get("startup_sentinel_required") is not True:
        problems.append("startup sentinel is not required")
    if "$SLURM_STARTUP_SENTINEL" not in str(contract.get("startup_prefix") or ""):
        problems.append("startup sentinel is not sourced before launch")
    return problems


def _validate_packet_file_hashes(packet: dict[str, Any], repo: Path | None) -> list[str]:
    """Recompute declared public-input hashes when the checkout is available.

    Returns:
        Sanitized file-hash problems.  Missing remote-only files are left to
        the submit wrapper; they are not treated as local mismatches.
    """
    if repo is None:
        return []
    inputs = packet.get("inputs")
    if not isinstance(inputs, dict):
        return []
    problems: list[str] = []
    for input_name in (
        "release_manifest",
        "canonical_campaign_config",
        "scenario_matrix",
        "public_single_node_entrypoint",
    ):
        item = inputs.get(input_name)
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("path") or "")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = repo / candidate
        if not candidate.is_file():
            continue
        expected = str(item.get("sha256") or "").lower()
        if _sha256(candidate).lower() != expected:
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


def _submit_arg_value(submit_args: str, field: str) -> str | None:
    """Extract one comma-delimited ``--export`` value from queue arguments.

    Returns:
        The value, or ``None`` when the field is absent.
    """
    match = re.search(rf"(?:^|,){field}=([^,\s]+)", submit_args)
    return match.group(1) if match else None


def _validate_queue_identity_args(submit_args: str, packet: dict[str, Any]) -> list[str]:
    """Validate queue environment values against packet identity.

    Returns:
        Sanitized queue identity problems.
    """
    problems: list[str] = []
    identity = packet.get("identity")
    inputs = packet.get("inputs")
    if isinstance(identity, dict) and isinstance(inputs, dict):
        queue_to_identity = {
            "RELEASE_MANIFEST_SHA256": "release_manifest_sha256",
            "RELEASE_CONFIG_SHA256": "canonical_config_sha256",
            "RELEASE_SCENARIO_SHA256": "scenario_matrix_sha256",
            "RELEASE_CHECKPOINT_RECEIPT_SHA256": "checkpoint_receipt_sha256",
            "RELEASE_PUBLIC_SCRIPT_SHA256": "public_entrypoint_sha256",
        }
        for queue_field, identity_field in queue_to_identity.items():
            if _submit_arg_value(submit_args, queue_field) != identity.get(identity_field):
                problems.append(f"private queue {queue_field} is not bound to packet identity")
        input_paths = {
            "RELEASE_MANIFEST_PATH": "release_manifest",
            "RELEASE_SCENARIO_PATH": "scenario_matrix",
            "RELEASE_CHECKPOINT_RECEIPT_PATH": "checkpoint_staging_receipt",
        }
        for queue_field, input_name in input_paths.items():
            item = inputs.get(input_name)
            expected_path = item.get("path") if isinstance(item, dict) else None
            if _submit_arg_value(submit_args, queue_field) != expected_path:
                problems.append(f"private queue {queue_field} is not bound to packet inputs")
    expected_identity = {
        "RELEASE_CAMPAIGN_ID": packet.get("campaign_id"),
        "RELEASE_EXPECTED_CPUS": "36",
        "RELEASE_EXPECTED_GPUS": "1",
        "RELEASE_EXPECTED_MEM_GB": "256",
        "RELEASE_EXPECTED_WALLTIME": "36:00:00",
    }
    for field, expected in expected_identity.items():
        if _submit_arg_value(submit_args, field) != str(expected):
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
    expected_packet_hash = _sha256(packet_path)
    if f"RELEASE_LAUNCH_PACKET_SHA256={expected_packet_hash}" not in submit_args:
        problems.append("private queue packet hash is not bound")
    for field in (
        "RELEASE_MANIFEST_SHA256",
        "RELEASE_CONFIG_SHA256",
        "RELEASE_SCENARIO_SHA256",
        "RELEASE_CHECKPOINT_RECEIPT_SHA256",
        "RELEASE_PUBLIC_SCRIPT_SHA256",
    ):
        value = _submit_arg_value(submit_args, field)
        if value is None or not _SHA256_RE.fullmatch(value):
            problems.append(f"private queue {field} is not a concrete SHA-256")
    problems.extend(_validate_queue_identity_args(submit_args, packet))
    return problems


def _validate_queue_resources(row: dict[str, Any]) -> list[str]:
    """Validate queue resource values when present.

    Returns:
        Sanitized resource problems.
    """
    problems: list[str] = []
    for field, expected in _RELEASE_RESOURCES.items():
        if field not in row:
            continue
        if field in {"cluster", "partition", "route_id", "gpu_type", "wall_clock"}:
            matches = str(row[field]).lower() == str(expected).lower()
        else:
            try:
                matches = int(row[field]) == int(expected)
            except (TypeError, ValueError):
                matches = False
        if not matches:
            problems.append(f"private queue resource contract mismatch: {field}")
    if "estimated_elapsed_sec" in row:
        try:
            if int(row["estimated_elapsed_sec"]) != 129600:
                problems.append("private queue resource contract mismatch: wall_clock")
        except (TypeError, ValueError):
            problems.append("private queue resource contract mismatch: wall_clock")
    return problems


def _validate_packet_queue(
    packet: dict[str, Any], packet_path: Path, queue_path: Path | None, expected_sha: str
) -> list[str]:
    """Validate private queue identity, hashes, and fixed resource values.

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
    matching = [row for row in rows if row.get("queue_id") == queue_id]
    if len(matching) != 1:
        return ["private queue does not contain exactly one packet row"]
    row = matching[0]
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
    if not artifact_manifest.endswith(packet_path.name):
        problems.append("private queue artifact manifest does not match packet")
    submit_args = str(row.get("submit_args") or "")
    problems.extend(_validate_queue_submit_args(submit_args, packet_path, packet))
    problems.extend(_validate_queue_resources(row))
    return list(dict.fromkeys(problems))


def _validate_packet_contract(
    packet: dict[str, Any],
    expected_sha: str,
    *,
    expected_tag: str | None,
    expected_campaign_id: str | None,
    packet_path: Path,
    queue_path: Path | None,
    repo: Path | None,
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
    admitted = isinstance(admission, dict) and admission.get("status") in {"admitted", "ready"}
    dispatchable = bool(packet.get("dispatchable")) and (
        not isinstance(admission, dict) or bool(admission.get("dispatchable", True))
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
    manifest_check, manifest, cfg = _manifest_check(manifest_path, expected_cells)
    if final or private_queue is not None or expected_campaign_id is not None:
        cluster_check = _cluster_check(
            private_launch_packet,
            expected_release_sha,
            final=final,
            expected_tag=tag,
            expected_campaign_id=expected_campaign_id,
            queue_path=private_queue,
            repo=repo,
        )
    else:
        # Preserve the lightweight preparation-mode contract for callers that
        # only have a draft packet and no queue row yet.
        cluster_check = _cluster_check(private_launch_packet, expected_release_sha)
    checks = [
        _git_check(repo, expected_release_sha),
        _ci_check(repo, expected_release_sha),
        _tag_check(repo, tag),
        manifest_check,
        _release_identity_check(manifest, expected_base_sha, tag),
        _checkpoint_check(cfg, manifest, checkpoint_receipt),
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
