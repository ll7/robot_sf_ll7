"""Build and validate the public no-submit Slurm launch-manifest contract.

The camera-ready runner's ``campaign_manifest.json`` describes campaign
configuration and execution provenance.  This module owns the separate,
pre-submit intent packet consumed by ``slurm_campaign_preflight.py``.  It is
deliberately stdlib-only so the no-submit validator remains usable on a
minimal operations host.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "robot-sf-slurm-launch-manifest.v1"
EXPECTED_PLANNER_ARMS = 14
EXPECTED_SCENARIOS = 48
EXPECTED_SEEDS = 30
EXPECTED_EPISODE_CELLS = 20_160
EXPECTED_HORIZON_STEPS = 600
EXPECTED_KINEMATICS = ("differential_drive",)
FULL_COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_FORBIDDEN_KEY_RE = re.compile(r"(?:success|result|job(?:_|$))", re.IGNORECASE)


class LaunchManifestError(ValueError):
    """Raised when a launch manifest cannot be generated or validated."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize the same canonical JSON form used by release identities.

    Returns:
        Canonical UTF-8 JSON bytes terminated by a newline.
    """
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object.

    Returns:
        The decoded JSON object.

    Raises:
        LaunchManifestError: If the file is unreadable or not an object.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LaunchManifestError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise LaunchManifestError(f"JSON value must be an object: {path}")
    return value


def _has_symlink_component(path: Path) -> bool:
    """Return whether a lexical path contains a symlink component."""
    lexical = Path(os.path.abspath(os.fspath(path)))
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _resolve_file(
    value: object,
    *,
    anchor: Path,
    repository_root: Path,
    label: str,
) -> Path:
    """Resolve a repository-contained regular file without following links.

    Returns:
        The resolved regular-file path.

    Raises:
        LaunchManifestError: If the path is missing, linked, outside the
            repository, or not a regular file.
    """
    raw = str(value or "").strip()
    if not raw:
        raise LaunchManifestError(f"{label} is missing")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = anchor / candidate
    if _has_symlink_component(candidate):
        raise LaunchManifestError(f"{label} contains a symlink: {raw}")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(repository_root):
        raise LaunchManifestError(f"{label} escapes repository: {raw}")
    if not resolved.is_file():
        raise LaunchManifestError(f"{label} is not a regular file: {raw}")
    return resolved


def _resolve_directory(
    value: object,
    *,
    anchor: Path,
    repository_root: Path,
    label: str,
) -> Path:
    """Resolve a repository-contained directory without following links.

    Returns:
        The resolved directory path.

    Raises:
        LaunchManifestError: If the path is missing, linked, outside the
            repository, or not a directory.
    """
    raw = str(value or "").strip()
    if not raw:
        raise LaunchManifestError(f"{label} is missing")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = anchor / candidate
    if _has_symlink_component(candidate):
        raise LaunchManifestError(f"{label} contains a symlink: {raw}")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(repository_root):
        raise LaunchManifestError(f"{label} escapes repository: {raw}")
    if not resolved.is_dir():
        raise LaunchManifestError(f"{label} is not a directory: {raw}")
    return resolved


def _relative_path(path: Path, anchor: Path) -> str:
    """Return a stable POSIX path relative to an output manifest directory."""
    return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LaunchManifestError(f"{label} must be an object")
    return value


def _require_string(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise LaunchManifestError(f"{label} is missing")
    return text


def _require_sha(value: object, label: str) -> str:
    text = _require_string(value, label).lower()
    if SHA256_RE.fullmatch(text) is None:
        raise LaunchManifestError(f"{label} is not a SHA-256 digest")
    return text


def _require_commit(value: object, label: str) -> str:
    text = _require_string(value, label).lower()
    if FULL_COMMIT_RE.fullmatch(text) is None:
        raise LaunchManifestError(f"{label} is not an exact Git SHA")
    return text


def _require_int(value: object, label: str, expected: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LaunchManifestError(f"{label} must be an integer")
    if expected is not None and value != expected:
        raise LaunchManifestError(f"{label} must equal {expected}, got {value}")
    return value


def _manifest_record(
    *,
    role: str,
    path: Path,
    output_parent: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic checksummed path record.

    Returns:
        A role, relative path, and SHA-256 record.
    """
    record: dict[str, Any] = {
        "role": role,
        "path": _relative_path(path, output_parent),
        "sha256": sha256_file(path),
    }
    if extra:
        record.update(extra)
    return record


def _identity_input_records(  # noqa: C901
    resolved_manifest: Mapping[str, Any],
    *,
    output_parent: Path,
    repository_root: Path,
) -> list[dict[str, Any]]:
    """Extract content-addressed release inputs from a resolved identity.

    Returns:
        Sorted content-addressed input records.
    """
    records: dict[str, dict[str, Any]] = {}

    def add(role: str, raw_path: object, raw_sha: object, *, required: bool = False) -> None:
        if raw_path in (None, ""):
            if required:
                raise LaunchManifestError(f"resolved identity input is missing: {role}")
            return
        path = _resolve_file(
            raw_path,
            anchor=repository_root,
            repository_root=repository_root,
            label=f"resolved identity input {role}",
        )
        checksum = _require_sha(raw_sha, f"resolved identity input {role}.sha256")
        observed = sha256_file(path)
        if observed != checksum:
            raise LaunchManifestError(f"resolved identity input hash mismatch: {role}")
        record = _manifest_record(role=role, path=path, output_parent=output_parent)
        if record["sha256"] != checksum:
            raise LaunchManifestError(f"resolved identity input hash changed: {role}")
        prior = records.get(record["path"])
        if prior is not None and prior["sha256"] != checksum:
            raise LaunchManifestError(
                f"resolved identity input path has conflicting hashes: {role}"
            )
        records[record["path"]] = record

    scenario = _require_mapping(resolved_manifest.get("scenario"), "resolved_manifest.scenario")
    add(
        "scenario_matrix", scenario.get("matrix_path"), scenario.get("matrix_sha256"), required=True
    )

    seed_policy = _require_mapping(
        resolved_manifest.get("seed_policy"), "resolved_manifest.seed_policy"
    )
    release_contract = _require_mapping(
        resolved_manifest.get("release_contract"), "resolved_manifest.release_contract"
    )
    add(
        "seed_sets",
        seed_policy.get("seed_sets_path"),
        release_contract.get("seed_sets_sha256"),
        required=True,
    )

    for role, section_key, path_key, sha_key in (
        ("suite_policy", "release_contract", "suite_policy_path", "suite_policy_sha256"),
        (
            "route_certification",
            "release_contract",
            "route_certification_path",
            "route_certification_sha256",
        ),
        ("snqi_weights", "metrics", "snqi_weights_path", "snqi_weights_sha256"),
        ("snqi_baseline", "metrics", "snqi_baseline_path", "snqi_baseline_sha256"),
    ):
        section_value = resolved_manifest.get(section_key)
        if section_value is None:
            continue
        section = _require_mapping(section_value, f"resolved_manifest.{section_key}")
        add(role, section.get(path_key), section.get(sha_key))

    planner_section = _require_mapping(
        resolved_manifest.get("planners"), "resolved_manifest.planners"
    )
    config_identities = planner_section.get("config_identities", [])
    if not isinstance(config_identities, list):
        raise LaunchManifestError("resolved_manifest.planners.config_identities must be a list")
    for identity in config_identities:
        planner_identity = _require_mapping(identity, "planner config identity")
        planner_key = _require_string(planner_identity.get("key"), "planner config identity.key")
        add(
            f"planner_config:{planner_key}",
            planner_identity.get("path"),
            planner_identity.get("sha256"),
        )

    return [records[key] for key in sorted(records)]


def _validate_file_record(
    record: object,
    *,
    manifest_path: Path,
    repository_root: Path,
    label: str,
    blockers: list[str],
    require_role: str | None = None,
) -> Path | None:
    """Validate one path/hash record and return its resolved path.

    Returns:
        The resolved file path, or ``None`` when validation fails.
    """
    if not isinstance(record, Mapping):
        blockers.append(f"{label} must be an object")
        return None
    if require_role is not None and record.get("role") != require_role:
        blockers.append(f"{label}.role must be {require_role}")
    raw_path = str(record.get("path", "") or "").strip()
    if raw_path and Path(raw_path).is_absolute():
        blockers.append(f"{label}.path must be relative")
    try:
        path = _resolve_file(
            record.get("path"),
            anchor=manifest_path.resolve().parent,
            repository_root=repository_root,
            label=f"{label}.path",
        )
        checksum = _require_sha(record.get("sha256"), f"{label}.sha256")
    except LaunchManifestError as exc:
        blockers.append(str(exc))
        return None
    if sha256_file(path) != checksum:
        blockers.append(f"{label}.sha256 does not match file bytes")
    return path


def _validate_no_outcome_fields(value: object, *, prefix: str, blockers: list[str]) -> None:
    """Reject scheduler/result fields from a pre-submit intent packet."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if _FORBIDDEN_KEY_RE.search(key_text):
                blockers.append(f"future outcome field is forbidden: {prefix}.{key_text}")
            _validate_no_outcome_fields(child, prefix=f"{prefix}.{key_text}", blockers=blockers)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_outcome_fields(child, prefix=f"{prefix}[{index}]", blockers=blockers)


def _validate_launch_matrix(  # noqa: C901, PLR0912, PLR0915
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    repository_root: Path,
    blockers: list[str],
) -> tuple[list[str], list[int]]:
    """Validate the fixed 14×48×30 release matrix and arm cells.

    Returns:
        Planner keys and resolved seeds in their manifest order.
    """
    matrix = manifest.get("matrix")
    if not isinstance(matrix, Mapping):
        blockers.append("matrix is missing")
        return [], []

    try:
        planner_arms = _require_int(
            matrix.get("planner_arms"), "matrix.planner_arms", EXPECTED_PLANNER_ARMS
        )
        scenario_count = _require_int(
            matrix.get("scenarios"), "matrix.scenarios", EXPECTED_SCENARIOS
        )
        seed_count = _require_int(matrix.get("seeds"), "matrix.seeds", EXPECTED_SEEDS)
        expected_rows = _require_int(
            matrix.get("expected_episode_cells"),
            "matrix.expected_episode_cells",
            EXPECTED_EPISODE_CELLS,
        )
        _require_int(matrix.get("horizon_steps"), "matrix.horizon_steps", EXPECTED_HORIZON_STEPS)
    except LaunchManifestError as exc:
        blockers.append(str(exc))
        return [], []

    raw_keys = matrix.get("planner_keys")
    if not isinstance(raw_keys, list) or not raw_keys:
        blockers.append("matrix.planner_keys is missing")
        planner_keys: list[str] = []
    else:
        planner_keys = [str(value).strip() for value in raw_keys]
        if len(planner_keys) != planner_arms:
            blockers.append("matrix.planner_keys count does not match matrix.planner_arms")
        if len(set(planner_keys)) != len(planner_keys):
            blockers.append("matrix.planner_keys contains duplicates")

    raw_seeds = matrix.get("resolved_seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        blockers.append("matrix.resolved_seeds is missing")
        resolved_seeds: list[int] = []
    else:
        resolved_seeds = []
        for value in raw_seeds:
            if isinstance(value, bool) or not isinstance(value, int):
                blockers.append("matrix.resolved_seeds must contain only integers")
                continue
            resolved_seeds.append(value)
        if len(resolved_seeds) != seed_count:
            blockers.append("matrix.resolved_seeds count does not match matrix.seeds")
        if len(set(resolved_seeds)) != len(resolved_seeds):
            blockers.append("matrix.resolved_seeds contains duplicates")

    raw_kinematics = matrix.get("kinematics")
    if raw_kinematics != list(EXPECTED_KINEMATICS):
        blockers.append("matrix.kinematics must be ['differential_drive']")

    cells = manifest.get("cells")
    if not isinstance(cells, list):
        blockers.append("campaign cells are missing")
        cells = []
    expected_rows_per_arm = scenario_count * seed_count
    if expected_rows_per_arm * planner_arms != expected_rows:
        blockers.append("matrix dimensions do not multiply to matrix.expected_episode_cells")
    if len(cells) != planner_arms:
        blockers.append(f"campaign cells must contain exactly {planner_arms} planner arms")

    observed_keys: list[str] = []
    output_roots: set[Path] = set()
    for index, raw_cell in enumerate(cells):
        if not isinstance(raw_cell, Mapping):
            blockers.append(f"cells[{index}] must be an object")
            continue
        key = str(raw_cell.get("key", "") or "").strip()
        planner_key = str(raw_cell.get("planner_key", "") or "").strip()
        if not key:
            blockers.append(f"cells[{index}].key is missing")
        if key != planner_key:
            blockers.append(f"cells[{index}].key and planner_key must match")
        observed_keys.append(key)
        try:
            _require_int(
                raw_cell.get("scenario_count"),
                f"cells[{index}].scenario_count",
                scenario_count,
            )
            _require_int(raw_cell.get("seed_count"), f"cells[{index}].seed_count", seed_count)
            _require_int(
                raw_cell.get("declared_rows"),
                f"cells[{index}].declared_rows",
                expected_rows_per_arm,
            )
            _require_int(
                raw_cell.get("instantiated_rows"),
                f"cells[{index}].instantiated_rows",
                expected_rows_per_arm,
            )
        except LaunchManifestError as exc:
            blockers.append(str(exc))
        if raw_cell.get("kinematics") != EXPECTED_KINEMATICS[0]:
            blockers.append(f"cells[{index}].kinematics must be differential_drive")
        if raw_cell.get("status") != "available":
            blockers.append(f"cells[{index}].status must be available launch readiness")
        if raw_cell.get("execution_status") != "not_started":
            blockers.append(f"cells[{index}].execution_status must be not_started")
        output_root = str(raw_cell.get("output_root", "") or "").strip()
        if not output_root:
            blockers.append(f"cells[{index}].output_root is missing")
        elif Path(output_root).is_absolute():
            blockers.append(f"cells[{index}].output_root must be relative")
        elif output_root.split("/")[0] != "slurm_cells":
            blockers.append(f"cells[{index}].output_root must be under slurm_cells/")
        else:
            candidate = manifest_path.parent / output_root
            if _has_symlink_component(candidate):
                blockers.append(f"cells[{index}].output_root contains a symlink")
            normalized = candidate.resolve(strict=False)
            slurm_root = (manifest_path.parent / "slurm_cells").resolve(strict=False)
            if not normalized.is_relative_to(repository_root):
                blockers.append(f"cells[{index}].output_root escapes repository")
            elif not normalized.is_relative_to(slurm_root):
                blockers.append(f"cells[{index}].output_root escapes slurm_cells/")
            elif normalized in output_roots:
                blockers.append(f"duplicate cell output_root: {output_root}")
            output_roots.add(normalized)
        if not str(raw_cell.get("artifact_contract", "") or "").strip():
            blockers.append(f"cells[{index}].artifact_contract is missing")

    if len(set(observed_keys)) != len(observed_keys):
        blockers.append("campaign cells contain duplicate keys")
    if planner_keys and set(observed_keys) != set(planner_keys):
        missing = sorted(set(planner_keys) - set(observed_keys))
        unexpected = sorted(set(observed_keys) - set(planner_keys))
        if missing:
            blockers.append(f"campaign cells are missing planner arms: {', '.join(missing)}")
        if unexpected:
            blockers.append(
                f"campaign cells contain unexpected planner arms: {', '.join(unexpected)}"
            )
    return planner_keys, resolved_seeds


def _validate_identity_and_inputs(  # noqa: C901, PLR0912, PLR0915
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    repository_root: Path,
    actual_public_commit: str,
    blockers: list[str],
) -> tuple[str, str, list[dict[str, Any]], dict[str, Any] | None]:
    """Validate source identity, packet inputs, and preflight artifact hashes.

    Returns:
        Source commit, expected commit, input records, and the decoded identity
        payload when it was readable.
    """
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        blockers.append("source identity block is missing")
        return "", "", [], None
    try:
        source_commit = _require_commit(source.get("commit"), "source.commit")
        expected_commit = _require_commit(
            manifest.get("expected_public_commit"), "expected_public_commit"
        )
    except LaunchManifestError as exc:
        blockers.append(str(exc))
        source_commit = ""
        expected_commit = ""
    if source_commit and expected_commit and source_commit != expected_commit:
        blockers.append("source.commit does not match expected_public_commit")
    if actual_public_commit:
        try:
            actual = _require_commit(actual_public_commit, "actual public commit")
        except LaunchManifestError as exc:
            blockers.append(str(exc))
        else:
            if source_commit and actual != source_commit:
                blockers.append("source.commit does not match the actual public commit")

    identity_record = source.get("resolved_identity")
    identity_path = _validate_file_record(
        identity_record,
        manifest_path=manifest_path,
        repository_root=repository_root,
        label="source.resolved_identity",
        blockers=blockers,
        require_role="resolved_release_identity",
    )
    identity_payload: dict[str, Any] | None = None
    if identity_path is not None:
        try:
            identity_payload = _load_json_object(identity_path)
        except LaunchManifestError as exc:
            blockers.append(str(exc))
        if identity_payload is None:
            resolved_manifest = None
        else:
            resolved_manifest = identity_payload.get("resolved_manifest")
            if identity_payload.get("schema_version") != "benchmark-release-resolved-identity.v1":
                blockers.append("resolved identity schema_version is unsupported")
            if not isinstance(resolved_manifest, Mapping):
                blockers.append("resolved identity has no resolved_manifest object")
            else:
                recorded_hash = (
                    str(source.get("resolved_manifest_sha256", "") or "").strip().lower()
                )
                identity_hash = (
                    str(identity_payload.get("resolved_manifest_sha256", "") or "").strip().lower()
                )
                if SHA256_RE.fullmatch(recorded_hash) is None:
                    blockers.append("source.resolved_manifest_sha256 is missing or invalid")
                if SHA256_RE.fullmatch(identity_hash) is None:
                    blockers.append(
                        "resolved identity resolved_manifest_sha256 is missing or invalid"
                    )
                elif recorded_hash and identity_hash != recorded_hash:
                    blockers.append("resolved identity hash does not match source binding")
                elif (
                    hashlib.sha256(_canonical_json_bytes(resolved_manifest)).hexdigest()
                    != recorded_hash
                ):
                    blockers.append("source.resolved_manifest_sha256 does not match identity bytes")
                identity_source = str(identity_payload.get("source_commit", "")).strip().lower()
                if source_commit and identity_source != source_commit:
                    blockers.append("resolved identity source_commit does not match source.commit")

    packet = manifest.get("packet")
    if not isinstance(packet, Mapping):
        blockers.append("packet identity is missing")
    else:
        packet_path = _validate_file_record(
            {
                "role": "campaign_config",
                "path": packet.get("config"),
                "sha256": packet.get("sha256"),
            },
            manifest_path=manifest_path,
            repository_root=repository_root,
            label="packet",
            blockers=blockers,
            require_role="campaign_config",
        )
        if packet_path is not None and identity_payload is not None:
            resolved_manifest = identity_payload.get("resolved_manifest")
            if isinstance(resolved_manifest, Mapping):
                expected_config = str(
                    resolved_manifest.get("canonical_campaign_config", "") or ""
                ).strip()
                expected_hash = (
                    str(resolved_manifest.get("canonical_campaign_config_sha256", "") or "")
                    .strip()
                    .lower()
                )
                actual_packet_path = packet_path.relative_to(repository_root).as_posix()
                if expected_config and actual_packet_path != expected_config:
                    blockers.append("packet.config does not match resolved identity config")
                if expected_hash and sha256_file(packet_path) != expected_hash:
                    blockers.append("packet.sha256 does not match resolved identity config hash")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        blockers.append("source input records are missing")
        input_records: list[dict[str, Any]] = []
    else:
        input_records = []
        seen_paths: set[str] = set()
        for index, record in enumerate(inputs):
            _validate_file_record(
                record,
                manifest_path=manifest_path,
                repository_root=repository_root,
                label=f"inputs[{index}]",
                blockers=blockers,
            )
            if isinstance(record, Mapping):
                path_text = str(record.get("path", "") or "")
                if path_text in seen_paths:
                    blockers.append(f"inputs contains duplicate path: {path_text}")
                seen_paths.add(path_text)
                input_records.append(dict(record))

    preflight = manifest.get("preflight")
    if not isinstance(preflight, Mapping):
        blockers.append("runner preflight binding is missing")
    else:
        if preflight.get("status") != "valid":
            blockers.append("runner preflight status must be valid")
        _validate_file_record(
            preflight.get("runner_report"),
            manifest_path=manifest_path,
            repository_root=repository_root,
            label="preflight.runner_report",
            blockers=blockers,
            require_role="runner_preflight",
        )
        artifacts = preflight.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            blockers.append("preflight artifact records are missing")
        else:
            for index, record in enumerate(artifacts):
                _validate_file_record(
                    record,
                    manifest_path=manifest_path,
                    repository_root=repository_root,
                    label=f"preflight.artifacts[{index}]",
                    blockers=blockers,
                )
    return source_commit, expected_commit, input_records, identity_payload


def _validate_identity_projection(  # noqa: C901, PLR0912
    manifest: Mapping[str, Any],
    *,
    identity_payload: Mapping[str, Any] | None,
    input_records: list[dict[str, Any]],
    repository_root: Path,
    manifest_path: Path,
    blockers: list[str],
) -> None:
    """Ensure packet fields still project the bound resolved identity.

    The file hashes prove that referenced bytes were not changed.  These
    projection checks additionally prevent replacing a valid referenced file
    with another planner, seed set, or release configuration while preserving
    the packet's outer shape.
    """
    if identity_payload is None:
        return
    resolved_manifest = identity_payload.get("resolved_manifest")
    if not isinstance(resolved_manifest, Mapping):
        return

    release = manifest.get("release")
    identity_release_id = str(resolved_manifest.get("release_id", "") or "").strip()
    identity_release_tag = str(resolved_manifest.get("release_tag", "") or "").strip()
    if isinstance(release, Mapping):
        if release.get("release_id") != identity_release_id:
            blockers.append("release.release_id does not match resolved identity")
        if release.get("release_tag") != identity_release_tag:
            blockers.append("release.release_tag does not match resolved identity")

    planners = resolved_manifest.get("planners")
    matrix = resolved_manifest.get("matrix")
    kinematics = resolved_manifest.get("kinematics")
    release_contract = resolved_manifest.get("release_contract")
    manifest_matrix = manifest.get("matrix")
    if not all(
        isinstance(section, Mapping)
        for section in (planners, matrix, kinematics, release_contract, manifest_matrix)
    ):
        blockers.append("resolved identity matrix projection is incomplete")
        return
    if manifest_matrix.get("planner_keys") != planners.get("keys"):
        blockers.append("matrix.planner_keys do not match resolved identity")
    if manifest_matrix.get("resolved_seeds") != release_contract.get("resolved_seeds"):
        blockers.append("matrix.resolved_seeds do not match resolved identity")
    if manifest_matrix.get("expected_episode_cells") != matrix.get("expected_episode_cells"):
        blockers.append("matrix.expected_episode_cells does not match resolved identity")
    if manifest_matrix.get("horizon_steps") != matrix.get("horizon_steps"):
        blockers.append("matrix.horizon_steps does not match resolved identity")
    if manifest_matrix.get("kinematics") != kinematics.get("matrix"):
        blockers.append("matrix.kinematics does not match resolved identity")

    try:
        expected_inputs = _identity_input_records(
            resolved_manifest,
            output_parent=manifest_path.parent,
            repository_root=repository_root,
        )
    except LaunchManifestError as exc:
        blockers.append(f"resolved identity inputs cannot be reconstructed: {exc}")
    else:
        expected_keys = {
            (str(record.get("role")), str(record.get("path")), str(record.get("sha256")).lower())
            for record in expected_inputs
        }
        observed_keys = {
            (str(record.get("role")), str(record.get("path")), str(record.get("sha256")).lower())
            for record in input_records
        }
        if observed_keys != expected_keys:
            blockers.append("inputs do not exactly match resolved identity inputs")

    packet = manifest.get("packet")
    if not isinstance(packet, Mapping):
        return
    try:
        observed_config = _resolve_file(
            packet.get("config"),
            anchor=manifest_path.parent,
            repository_root=repository_root,
            label="packet.config",
        )
        expected_config = _resolve_file(
            resolved_manifest.get("canonical_campaign_config"),
            anchor=repository_root,
            repository_root=repository_root,
            label="resolved identity canonical_campaign_config",
        )
    except LaunchManifestError:
        return
    if observed_config != expected_config:
        blockers.append("packet.config does not resolve to resolved identity config")


def validate_launch_manifest(  # noqa: C901, PLR0912
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    repository_root: Path | None = None,
    actual_public_commit: str = "",
) -> list[str]:
    """Return fail-closed blockers for a generated launch manifest."""
    root = (repository_root or Path.cwd()).resolve()
    path = Path(manifest_path).resolve()
    blockers: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return blockers
    _validate_no_outcome_fields(manifest, prefix="manifest", blockers=blockers)
    required_literals = {
        "manifest_kind": "pre-submit-launch",
        "status": "pre_submit_intent",
        "no_submit": True,
        "execution_status": "not_started",
        "submission_status": "not_submitted",
        "evidence_status": "not-benchmark-evidence",
    }
    for key, expected in required_literals.items():
        if manifest.get(key) != expected:
            blockers.append(f"{key} must be {expected!r}")
    if not str(manifest.get("campaign_id", "") or "").strip():
        blockers.append("campaign_id is missing")
    if not path.parent.is_relative_to(root):
        blockers.append("launch manifest must be inside repository_root")
    planner_keys, resolved_seeds = _validate_launch_matrix(
        manifest,
        manifest_path=path,
        repository_root=root,
        blockers=blockers,
    )
    source_commit, _expected_commit, input_records, identity_payload = (
        _validate_identity_and_inputs(
            manifest,
            manifest_path=path,
            repository_root=root,
            actual_public_commit=actual_public_commit,
            blockers=blockers,
        )
    )

    release = manifest.get("release")
    if not isinstance(release, Mapping):
        blockers.append("release identity block is missing")
    else:
        if not str(release.get("release_id", "") or "").strip():
            blockers.append("release.release_id is missing")
        if not str(release.get("release_tag", "") or "").strip():
            blockers.append("release.release_tag is missing")

    matrix = manifest.get("matrix")
    if isinstance(matrix, Mapping):
        if planner_keys and list(matrix.get("planner_keys", [])) != planner_keys:
            blockers.append("matrix.planner_keys order is not stable")
        if resolved_seeds and list(matrix.get("resolved_seeds", [])) != resolved_seeds:
            blockers.append("matrix.resolved_seeds order is not stable")

    aggregate = manifest.get("aggregate")
    if not isinstance(aggregate, Mapping):
        blockers.append("aggregate artifact contract is missing")
    else:
        if aggregate.get("status") != "available":
            blockers.append("aggregate.status must be available launch readiness")
        if aggregate.get("execution_status") != "not_started":
            blockers.append("aggregate.execution_status must be not_started")
        if not str(aggregate.get("artifact_contract", "") or "").strip():
            blockers.append("aggregate.artifact_contract is missing")

    if source_commit and not FULL_COMMIT_RE.fullmatch(source_commit):
        blockers.append("source.commit is invalid")
    _validate_identity_projection(
        manifest,
        identity_payload=identity_payload,
        input_records=input_records,
        repository_root=root,
        manifest_path=path,
        blockers=blockers,
    )
    _validate_bound_runner_preflight(
        manifest,
        identity_payload=identity_payload,
        manifest_path=path,
        repository_root=root,
        blockers=blockers,
    )
    return list(dict.fromkeys(blockers))


def _planner_slug(planner_key: str) -> str:
    """Return the stable output-root component for one planner key."""
    slug = _SLUG_RE.sub("_", planner_key).strip("_.")
    if not slug:
        raise LaunchManifestError(f"planner key cannot produce an output-root slug: {planner_key}")
    return slug


def _validate_runner_preflight(  # noqa: C901, PLR0912, PLR0915
    runner: Mapping[str, Any],
    *,
    identity_payload: Mapping[str, Any],
    runner_path: Path,
    repository_root: Path,
) -> tuple[str, list[Path]]:
    """Validate runner preflight semantics and return bound artifact paths.

    Returns:
        The campaign ID and resolved runner artifact paths.
    """
    if runner.get("mode") != "preflight":
        raise LaunchManifestError("runner preflight report mode must be preflight")
    validation = _require_mapping(runner.get("manifest_validation"), "runner manifest_validation")
    if validation.get("status") != "valid":
        raise LaunchManifestError("runner preflight manifest_validation.status must be valid")
    if validation.get("problems") not in (None, []):
        raise LaunchManifestError("runner preflight contains manifest validation problems")
    identity_resolved = identity_payload.get("resolved_manifest")
    report_resolved = runner.get("resolved_manifest")
    if not isinstance(identity_resolved, Mapping) or report_resolved != identity_resolved:
        raise LaunchManifestError("runner preflight resolved_manifest does not match identity")
    campaign_id = _require_string(runner.get("campaign_id"), "runner campaign_id")
    _resolve_directory(
        runner.get("campaign_root"),
        anchor=runner_path.parent,
        repository_root=repository_root,
        label="runner campaign_root",
    )

    required_artifact_fields = (
        ("validate_config_path", "validate_config"),
        ("preview_scenarios_path", "preview_scenarios"),
        ("matrix_summary_json", "matrix_summary"),
    )
    contract = identity_resolved.get("release_contract")
    expected_seeds = contract.get("resolved_seeds") if isinstance(contract, Mapping) else None
    artifact_paths: list[Path] = []
    for field, role in required_artifact_fields:
        path = _resolve_file(
            runner.get(field),
            anchor=runner_path.parent,
            repository_root=repository_root,
            label=f"runner {field}",
        )
        artifact_paths.append(path)

        payload = _load_json_object(path)
        if payload.get("campaign_id") != campaign_id:
            raise LaunchManifestError(f"runner artifact {role} campaign_id does not match")
        if role == "validate_config":
            if payload.get("scenario_count") != EXPECTED_SCENARIOS:
                raise LaunchManifestError("runner validate_config scenario_count must be 48")
            if payload.get("planner_count") != EXPECTED_PLANNER_ARMS:
                raise LaunchManifestError("runner validate_config planner_count must be 14")
            if payload.get("horizon") != EXPECTED_HORIZON_STEPS:
                raise LaunchManifestError("runner validate_config horizon must be 600")
            seed_policy = payload.get("seed_policy")
            seeds = seed_policy.get("resolved_seeds") if isinstance(seed_policy, Mapping) else None
            if seeds != expected_seeds:
                raise LaunchManifestError("runner validate_config seeds do not match identity")
        elif role == "preview_scenarios":
            if payload.get("scenario_count") != EXPECTED_SCENARIOS:
                raise LaunchManifestError("runner preview scenario_count must be 48")
        else:
            rows = payload.get("rows")
            if not isinstance(rows, list) or len(rows) != EXPECTED_PLANNER_ARMS:
                raise LaunchManifestError("runner matrix_summary must contain 14 rows")
            planners = identity_resolved.get("planners")
            identity_keys = (
                set(planners.get("keys", [])) if isinstance(planners, Mapping) else set()
            )
            row_keys = {str(row.get("planner_key", "")) for row in rows if isinstance(row, Mapping)}
            if row_keys != identity_keys:
                raise LaunchManifestError(
                    "runner matrix_summary planner keys do not match identity"
                )
            for row in rows:
                if not isinstance(row, Mapping):
                    raise LaunchManifestError("runner matrix_summary rows must be objects")
                if row.get("scenario_count") != EXPECTED_SCENARIOS:
                    raise LaunchManifestError("runner matrix_summary scenario_count must be 48")
                if row.get("repeats") != EXPECTED_SEEDS:
                    raise LaunchManifestError("runner matrix_summary repeats must be 30")
                if row.get("resolved_seeds") != expected_seeds:
                    raise LaunchManifestError(
                        "runner matrix_summary resolved seeds do not match identity"
                    )
                if row.get("horizon") != EXPECTED_HORIZON_STEPS:
                    raise LaunchManifestError("runner matrix_summary horizon must be 600")
                if row.get("kinematics") != EXPECTED_KINEMATICS[0]:
                    raise LaunchManifestError(
                        "runner matrix_summary kinematics must be differential_drive"
                    )

    optional_csv = runner.get("matrix_summary_csv")
    if optional_csv:
        artifact_paths.append(
            _resolve_file(
                optional_csv,
                anchor=runner_path.parent,
                repository_root=repository_root,
                label="runner matrix_summary_csv",
            )
        )
    return campaign_id, artifact_paths


def _validate_bound_runner_preflight(  # noqa: C901
    manifest: Mapping[str, Any],
    *,
    identity_payload: Mapping[str, Any] | None,
    manifest_path: Path,
    repository_root: Path,
    blockers: list[str],
) -> None:
    """Recheck that the hashed runner report still names the bound inputs."""
    if identity_payload is None:
        return
    preflight = manifest.get("preflight")
    if not isinstance(preflight, Mapping):
        return
    runner_path = _validate_file_record(
        preflight.get("runner_report"),
        manifest_path=manifest_path,
        repository_root=repository_root,
        label="preflight.runner_report.binding",
        blockers=blockers,
        require_role="runner_preflight",
    )
    if runner_path is None:
        return
    try:
        runner = _load_json_object(runner_path)
        campaign_id, artifact_paths = _validate_runner_preflight(
            runner,
            identity_payload=identity_payload,
            runner_path=runner_path,
            repository_root=repository_root,
        )
    except (OSError, LaunchManifestError) as exc:
        blockers.append(f"bound runner preflight is invalid: {exc}")
        return
    if campaign_id != manifest.get("campaign_id"):
        blockers.append("bound runner preflight campaign_id does not match launch manifest")

    records = preflight.get("artifacts")
    if not isinstance(records, list):
        return
    if len(records) != len(artifact_paths):
        blockers.append("preflight artifact records do not match runner report artifacts")
        return
    expected_roles = (
        "validate_config",
        "preview_scenarios",
        "matrix_summary",
        "matrix_summary_csv",
    )
    for index, (record, artifact_path) in enumerate(zip(records, artifact_paths, strict=True)):
        expected_path = _relative_path(artifact_path, manifest_path.parent)
        if not isinstance(record, Mapping):
            continue
        if record.get("role") != expected_roles[index]:
            blockers.append(f"preflight.artifacts[{index}].role does not match runner report")
        if record.get("path") != expected_path:
            blockers.append(f"preflight.artifacts[{index}].path does not match runner report")
        if str(record.get("sha256", "")).lower() != sha256_file(artifact_path):
            blockers.append(f"preflight.artifacts[{index}].sha256 does not match runner report")


def generate_slurm_launch_manifest(  # noqa: C901, PLR0912, PLR0915
    *,
    resolved_identity_path: Path,
    runner_preflight_path: Path,
    output_path: Path,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Generate one deterministic pre-submit launch manifest.

    The resolved identity is verified through the release protocol before any
    packet bytes are written.  The runner report must be a successful
    ``--mode preflight`` result for the same resolved manifest.  No timestamps,
    scheduler identifiers, or execution results are copied into the output.

    Returns:
        The generated launch-manifest object.
    """
    root = (repository_root or Path.cwd()).resolve()
    identity_path = _resolve_file(
        resolved_identity_path,
        anchor=root,
        repository_root=root,
        label="resolved identity",
    )
    runner_path = _resolve_file(
        runner_preflight_path,
        anchor=root,
        repository_root=root,
        label="runner preflight report",
    )
    destination = Path(output_path).expanduser()
    if not destination.is_absolute():
        destination = root / destination
    if _has_symlink_component(destination):
        raise LaunchManifestError("launch manifest output contains a symlink")
    destination = destination.resolve(strict=False)
    if not destination.parent.is_relative_to(root):
        raise LaunchManifestError("launch manifest output escapes repository")
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Import the release protocol lazily so the no-submit validator remains
    # usable without the benchmark package's optional dependencies.
    release_protocol = importlib.import_module("robot_sf.benchmark.release_protocol")
    release_protocol.verify_resolved_release_identity(identity_path, repository_root=root)
    identity_payload = _load_json_object(identity_path)
    if identity_payload.get("schema_version") != "benchmark-release-resolved-identity.v1":
        raise LaunchManifestError("resolved identity schema_version is unsupported")
    resolved_manifest = _require_mapping(
        identity_payload.get("resolved_manifest"), "resolved_manifest"
    )
    source_commit = _require_commit(identity_payload.get("source_commit"), "identity.source_commit")
    release_id = _require_string(
        resolved_manifest.get("release_id"), "resolved_manifest.release_id"
    )
    release_tag = _require_string(
        resolved_manifest.get("release_tag"), "resolved_manifest.release_tag"
    )
    planner_section = _require_mapping(
        resolved_manifest.get("planners"), "resolved_manifest.planners"
    )
    planner_keys_raw = planner_section.get("keys")
    if not isinstance(planner_keys_raw, list) or len(planner_keys_raw) != EXPECTED_PLANNER_ARMS:
        raise LaunchManifestError("resolved identity must contain exactly 14 planner keys")
    planner_keys = [
        _require_string(value, "resolved_manifest.planners.keys entry")
        for value in planner_keys_raw
    ]
    if len(set(planner_keys)) != len(planner_keys):
        raise LaunchManifestError("resolved identity planner keys contain duplicates")

    matrix = _require_mapping(resolved_manifest.get("matrix"), "resolved_manifest.matrix")
    _require_int(
        matrix.get("expected_episode_cells"),
        "identity expected episode cells",
        EXPECTED_EPISODE_CELLS,
    )
    _require_int(matrix.get("horizon_steps"), "identity horizon", EXPECTED_HORIZON_STEPS)
    kinematics = _require_mapping(
        resolved_manifest.get("kinematics"), "resolved_manifest.kinematics"
    )
    if kinematics.get("matrix") != list(EXPECTED_KINEMATICS):
        raise LaunchManifestError("resolved identity kinematics must be differential_drive")
    release_contract = _require_mapping(
        resolved_manifest.get("release_contract"), "resolved_manifest.release_contract"
    )
    seeds_raw = release_contract.get("resolved_seeds")
    if not isinstance(seeds_raw, list) or len(seeds_raw) != EXPECTED_SEEDS:
        raise LaunchManifestError("resolved identity must contain exactly 30 resolved seeds")
    resolved_seeds = [_require_int(value, "resolved identity resolved seed") for value in seeds_raw]
    if len(set(resolved_seeds)) != len(resolved_seeds):
        raise LaunchManifestError("resolved identity resolved seeds contain duplicates")

    runner = _load_json_object(runner_path)
    campaign_id, runner_artifacts = _validate_runner_preflight(
        runner,
        identity_payload=identity_payload,
        runner_path=runner_path,
        repository_root=root,
    )

    packet_config = _resolve_file(
        resolved_manifest.get("canonical_campaign_config"),
        anchor=root,
        repository_root=root,
        label="canonical campaign config",
    )
    config_hash = _require_sha(
        resolved_manifest.get("canonical_campaign_config_sha256"),
        "canonical campaign config hash",
    )
    if sha256_file(packet_config) != config_hash:
        raise LaunchManifestError("canonical campaign config hash does not match file bytes")

    identity_hash = sha256_file(identity_path)
    resolved_manifest_hash = (
        str(identity_payload.get("resolved_manifest_sha256", "")).strip().lower()
    )
    if SHA256_RE.fullmatch(resolved_manifest_hash) is None:
        raise LaunchManifestError("identity resolved_manifest_sha256 is missing or invalid")
    if (
        hashlib.sha256(_canonical_json_bytes(resolved_manifest)).hexdigest()
        != resolved_manifest_hash
    ):
        raise LaunchManifestError("identity resolved_manifest_sha256 does not match resolved bytes")

    inputs = _identity_input_records(
        resolved_manifest,
        output_parent=destination.parent,
        repository_root=root,
    )
    slug_paths: set[str] = set()
    cells: list[dict[str, Any]] = []
    rows_per_arm = EXPECTED_SCENARIOS * EXPECTED_SEEDS
    for planner_key in planner_keys:
        slug = _planner_slug(planner_key)
        if slug in slug_paths:
            raise LaunchManifestError(f"planner keys collide in output roots: {planner_key}")
        slug_paths.add(slug)
        cells.append(
            {
                "key": planner_key,
                "planner_key": planner_key,
                "scenario_count": EXPECTED_SCENARIOS,
                "seed_count": EXPECTED_SEEDS,
                "kinematics": EXPECTED_KINEMATICS[0],
                "declared_rows": rows_per_arm,
                "instantiated_rows": rows_per_arm,
                "row_semantics": "declared launch combinations; execution not started",
                "output_root": f"slurm_cells/{slug}",
                "artifact_contract": "episodes.jsonl,run_meta.json,arm-summary.json",
                "status": "available",
                "execution_status": "not_started",
            }
        )

    packet = {
        "config": _relative_path(packet_config, destination.parent),
        "sha256": config_hash,
    }
    preflight_artifact_records = [
        _manifest_record(role=role, path=path, output_parent=destination.parent)
        for role, path in (
            ("validate_config", runner_artifacts[0]),
            ("preview_scenarios", runner_artifacts[1]),
            ("matrix_summary", runner_artifacts[2]),
        )
    ]
    if len(runner_artifacts) > 3:
        preflight_artifact_records.append(
            _manifest_record(
                role="matrix_summary_csv",
                path=runner_artifacts[3],
                output_parent=destination.parent,
            )
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": "pre-submit-launch",
        "status": "pre_submit_intent",
        "no_submit": True,
        "execution_status": "not_started",
        "submission_status": "not_submitted",
        "evidence_status": "not-benchmark-evidence",
        "claim_boundary": (
            "Pre-submit launch intent and structural completeness only; no scheduler, execution, "
            "benchmark, or publication claim."
        ),
        "campaign_id": campaign_id,
        "expected_public_commit": source_commit,
        "release": {"release_id": release_id, "release_tag": release_tag},
        "source": {
            "commit": source_commit,
            "resolved_manifest_sha256": resolved_manifest_hash,
            "resolved_identity": {
                "role": "resolved_release_identity",
                "path": _relative_path(identity_path, destination.parent),
                "sha256": identity_hash,
            },
        },
        "packet": packet,
        "matrix": {
            "planner_arms": EXPECTED_PLANNER_ARMS,
            "planner_keys": planner_keys,
            "scenarios": EXPECTED_SCENARIOS,
            "seeds": EXPECTED_SEEDS,
            "resolved_seeds": resolved_seeds,
            "expected_episode_cells": EXPECTED_EPISODE_CELLS,
            "horizon_steps": EXPECTED_HORIZON_STEPS,
            "kinematics": list(EXPECTED_KINEMATICS),
        },
        "preflight": {
            "status": "valid",
            "runner_report": _manifest_record(
                role="runner_preflight",
                path=runner_path,
                output_parent=destination.parent,
            ),
            "artifacts": preflight_artifact_records,
        },
        "inputs": inputs,
        "cells": cells,
        "aggregate": {
            "status": "available",
            "execution_status": "not_started",
            "artifact_contract": "per-arm rows,campaign summary,finalizer receipt",
        },
    }
    blockers = validate_launch_manifest(
        payload,
        manifest_path=destination,
        repository_root=root,
        actual_public_commit=source_commit,
    )
    if blockers:
        raise LaunchManifestError(
            "generated launch manifest failed validation: " + "; ".join(blockers)
        )
    destination.write_bytes(_canonical_json_bytes(payload))
    return payload


__all__ = [
    "EXPECTED_EPISODE_CELLS",
    "EXPECTED_HORIZON_STEPS",
    "EXPECTED_KINEMATICS",
    "EXPECTED_PLANNER_ARMS",
    "EXPECTED_SCENARIOS",
    "EXPECTED_SEEDS",
    "SCHEMA_VERSION",
    "LaunchManifestError",
    "generate_slurm_launch_manifest",
    "sha256_file",
    "validate_launch_manifest",
]
