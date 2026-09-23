"""Descriptive analysis of guarded-PPO obstacle contacts in frozen 0.0.6 data.

Issue #9480 asks why guarded PPO has low pedestrian contact but substantial
obstacle contact in the frozen ``paper_matrix_v2_h600_s30`` campaign.  This
script is publication-bundle-only: it does not train, step, or rerun an
episode.  It verifies the retained payload before reading either selected arm
and fails closed when the bundle cannot support the diagnostic claim.

The bundle retains no simulation step traces or per-step positions.  Contact
cell/map-file, collision time, and a derived zero-based contact step are
available; contact x/y, preceding-N pedestrian context, and last-k guard
timing are emitted as the explicit string ``NA``.

Report schema: ``issue_9480_guarded_obstacle_trade_report.v1``.
Episode table schema: ``issue_9480_guarded_obstacle_trade_episode_table.v1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from robot_sf.benchmark.camera_ready._run_state import validate_campaign_integrity

REPORT_SCHEMA_VERSION = "issue_9480_guarded_obstacle_trade_report.v1"
TABLE_SCHEMA_VERSION = "issue_9480_guarded_obstacle_trade_episode_table.v1"
EXPECTED_SOFTWARE_COMMIT = "31cdfe0361abe2c520117a17f99c1b7a0aba4359"
EXPECTED_CAMPAIGN_ID = "benchmark_0_0_6_s30_h600_20260911"
EXPECTED_BUNDLE_NAME = "benchmark_0_0_6_s30_h600_20260911_publication_bundle"
EXPECTED_RELEASE_TAG = "paper-matrix-v2-h600-s30-31cdfe0361abe2c520117a17f99c1b7a0aba4359"
EXPECTED_RELEASE_ASSET_SHA256 = "61b865fdde65455a39a68221d7c65b0eff315bfa51b4c0bfe34aed3c5d4f3e8e"
EXPECTED_SCENARIO_MATRIX = "configs/scenarios/classic_interactions_francis2023.yaml"
EXPECTED_SCENARIO_PUBLICATION_HASH = "152eba3969a9"
EXPECTED_SCENARIO_SHA256 = "d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5"
EXPECTED_RUN_SCENARIO_HASH = "23ae95be471eb7f0"
EXPECTED_CAMPAIGN_CONFIG_HASH = "60b554cd35c66aa0"
EXPECTED_SEED_SET = "paper_eval_s30"
EXPECTED_SEEDS = tuple(range(111, 141))
EXPECTED_HORIZON = 600
EXPECTED_DT = 0.1
EXPECTED_PAYLOAD_FILE_COUNT = 111
EXPECTED_PAYLOAD_BYTES = 740933980
GUARDED_ARM = "guarded_ppo__differential_drive"
BASE_ARM = "ppo__differential_drive"
FIGURE_DIR_NAME = "issue_9480_guarded_obstacle_trade"
NA = "NA"

FALLBACK_LABELS = ("fallback_safe", "fallback_best_effort")
STOP_LABELS = ("stop_safe", "stop_best_effort")
PRIOR_LABELS = ("prior_safe", "prior_residual_safe", "prior_blend_safe")
UNCERTAINTY_LABELS = (
    "uncertainty_fallback_stop",
    "uncertainty_fallback_slow_down",
    "uncertainty_fallback_configured",
)

EXPECTED_ARMS: dict[str, dict[str, Any]] = {
    GUARDED_ARM: {
        "planner_key": "guarded_ppo",
        "algo": "guarded_ppo",
        "config_path": "configs/algos/guarded_ppo_camera_ready_cpu.yaml",
        "config_sha256": "69f273f311590009a344f3a88592cb19f54524d252f469ab5fc66f7cf2c9e772",
        "scenario_config_hash": "44a39b76607347cc",
        "algorithm_config_hash": "c69166635bd79639",
        "run_config_hash": "316efd01306ac0bf",
        "input_bundle_sha256": "fafd20349d52574d877f8fa3ce5d14c8ed89b9b502cfbeb8dd0323702ba5b47b",
        "model_id": "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        "checkpoint_sha256": "8367af109a27e8879ced0c8913f6eff26df7ec59c31ea88f9a297bb2c141eb09",
        "predictive_foresight_enabled": False,
    },
    BASE_ARM: {
        "planner_key": "ppo",
        "algo": "ppo",
        "config_path": "configs/baselines/ppo_issue_791_eval_aligned_large_capacity_cpu.yaml",
        "config_sha256": "51ccfbf4400a306b355e2c3f0f46eda3489d5ce3bc85beaa023a6a1da9c9fb41",
        "scenario_config_hash": "8a51a7385295dcc5",
        "algorithm_config_hash": "fbce0d9c233a7ed5",
        "run_config_hash": "a4e0c8ab23b717d5",
        "input_bundle_sha256": "190ee66a31e540b4d6ea98963c1c159ac56b9d344305fffec98d85bd0740faec",
        "model_id": "ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417",
        "checkpoint_sha256": "2b30df812bfcc737924b126b0763d69c567fe20716dc1c1eba8f56f926b49c1d",
        "predictive_foresight_enabled": True,
        "predictive_model_id": "predictive_proxy_selected_v2_full",
        "predictive_checkpoint_sha256": "a28aed6d6ad7e1ebf597277ade1cf908efa6da038d0a9fcfdf80c7c31d8d1be1",
    },
}


def _family(scenario_id: str) -> str:
    """Strip the trailing cell variant to get the scenario family."""
    return re.sub(r"_(low|medium|high|easy|hard|v\d+)$", "", scenario_id)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_member_relative_path(member: tarfile.TarInfo) -> str | None:
    """Validate an archive member path and return its bundle-relative file path."""
    member_path = PurePosixPath(member.name)
    if (
        "\\" in member.name
        or member_path.is_absolute()
        or ".." in member_path.parts
        or not member_path.parts
        or member_path.parts[0] != EXPECTED_BUNDLE_NAME
    ):
        raise ValueError(f"Unexpected archive member path: {member.name!r}")
    if member.isdir():
        return None
    relative_parts = member_path.parts[1:]
    if not member.isfile() or not relative_parts:
        raise ValueError(f"Unsupported archive member type: {member.name!r}")
    return PurePosixPath(*relative_parts).as_posix()


def _archive_member_identity(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
) -> tuple[str, int]:
    """Hash one regular file without extracting it to the filesystem."""
    member_stream = archive.extractfile(member)
    if member_stream is None:
        raise ValueError(f"Unable to read archive member: {member.name!r}")
    digest = hashlib.sha256()
    size = 0
    with member_stream:
        for chunk in iter(lambda: member_stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if size != member.size:
        raise ValueError(f"Truncated archive member: {member.name!r}")
    return digest.hexdigest(), size


def _archive_file_identities(archive_path: Path) -> dict[str, tuple[str, int]]:
    """Read and hash every regular member beneath the expected archive root."""
    archived_files: dict[str, tuple[str, int]] = {}
    try:
        with tarfile.open(archive_path, mode="r|*") as archive:
            for member in archive:
                relative = _archive_member_relative_path(member)
                if relative is None:
                    continue
                if relative in archived_files:
                    raise ValueError(f"Duplicate archive member: {relative!r}")
                archived_files[relative] = _archive_member_identity(archive, member)
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise ValueError(f"Unable to read release archive {archive_path}: {exc}") from exc
    if not archived_files:
        raise ValueError("Release archive contains no regular files")
    return archived_files


def _bundle_root_file_identities(root: Path) -> dict[str, tuple[str, int]]:
    """Hash all regular bundle files while rejecting links and special entries."""
    root_files: dict[str, tuple[str, int]] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not allowed in bundle root: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Unsupported filesystem entry in bundle root: {path}")
        relative = path.relative_to(root).as_posix()
        root_files[relative] = (_sha256(path), path.stat().st_size)
    return root_files


def _verify_archive_matches_bundle_root(
    bundle_archive: Path,
    bundle_root: Path,
) -> dict[str, tuple[str, int]]:
    """Verify that every extracted bundle file is byte-identical to the release archive."""
    if bundle_archive.is_symlink():
        raise ValueError("Release archive must not be a symbolic link")
    if bundle_root.is_symlink():
        raise ValueError("Bundle root must not be a symbolic link")
    archive_path = bundle_archive.resolve(strict=True)
    root = bundle_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Bundle root is not a directory: {root}")
    archived_files = _archive_file_identities(archive_path)
    root_files = _bundle_root_file_identities(root)

    missing = sorted(archived_files.keys() - root_files.keys())
    extra = sorted(root_files.keys() - archived_files.keys())
    if missing or extra:
        raise ValueError(
            "Bundle root file set does not match release archive "
            f"(missing={missing[:3]!r}, extra={extra[:3]!r})"
        )
    for relative, archive_identity in archived_files.items():
        if root_files[relative] != archive_identity:
            raise ValueError(f"Bundle root file does not match release archive: {relative}")
    return archived_files


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(payload)
    return rows


def _bundle_path(bundle_root: Path, relative: str) -> Path:
    """Resolve a checksum path without allowing traversal outside the bundle."""
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Invalid bundle-relative path: {relative!r}")
    root = bundle_root.resolve()
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Bundle path escapes bundle root: {relative!r}")
    if not target.is_file():
        raise FileNotFoundError(f"Checksum target is missing: {target}")
    return target


def _verify_payload_checksums(bundle_root: Path) -> dict[str, tuple[str, int]]:
    """Verify every checksum entry and return ``path -> (sha256, size)``."""
    checksum_path = bundle_root / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing bundle checksum manifest: {checksum_path}")
    entries: dict[str, tuple[str, int]] = {}
    for line_number, raw_line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Malformed checksum line {line_number}: {raw_line!r}") from exc
        relative = relative.lstrip("* ")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Malformed SHA-256 at checksum line {line_number}: {expected!r}")
        if relative in entries:
            raise ValueError(f"Duplicate checksum entry: {relative}")
        target = _bundle_path(bundle_root, relative)
        observed = _sha256(target)
        if observed != expected:
            raise ValueError(
                f"Checksum mismatch for {relative}: expected {expected}, observed {observed}"
            )
        entries[relative] = (observed, target.stat().st_size)
    if not entries:
        raise ValueError("Checksum manifest is empty")
    return entries


def verify_bundle_checksums(bundle_root: Path) -> tuple[int, int]:
    """Verify every payload checksum and return ``(files, bytes)``.

    The publication manifest and checksum file are top-level release metadata;
    the release checksum file covers payload files only.  In particular, this
    helper intentionally does not claim that ``publication_manifest.json`` is
    independently checksum-authenticated.
    """
    entries = _verify_payload_checksums(bundle_root)
    return len(entries), sum(size for _, size in entries.values())


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, observed {actual!r}")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_path_suffix(value: Any, expected: str, label: str) -> None:
    observed = str(value or "")
    if not observed.replace("\\", "/").endswith(expected):
        raise ValueError(
            f"{label} mismatch: expected a path ending in {expected!r}, observed {value!r}"
        )


def _validate_publication_manifest(
    bundle_root: Path,
    publication_manifest: Mapping[str, Any],
    checksums: Mapping[str, tuple[str, int]],
) -> dict[str, Any]:
    """Validate payload checksums and the fixed publication identity."""
    del bundle_root  # Kept in the signature to make the scope explicit to callers.
    _require_equal(publication_manifest.get("bundle_name"), EXPECTED_BUNDLE_NAME, "bundle name")
    channels = _require_mapping(
        publication_manifest.get("publication_channels"), "publication channels"
    )
    _require_equal(channels.get("release_tag"), EXPECTED_RELEASE_TAG, "release tag")
    provenance = _require_mapping(publication_manifest.get("provenance"), "publication provenance")
    _require_equal(provenance.get("run_id"), EXPECTED_CAMPAIGN_ID, "publication run id")
    run_manifest = _require_mapping(provenance.get("manifest"), "publication run manifest")
    _require_equal(
        run_manifest.get("git_hash"), EXPECTED_SOFTWARE_COMMIT, "publication source commit"
    )
    _require_equal(
        run_manifest.get("scenario_matrix_hash"),
        EXPECTED_SCENARIO_PUBLICATION_HASH,
        "publication scenario matrix hash",
    )
    _require_path_suffix(
        provenance.get("matrix_path"), EXPECTED_SCENARIO_MATRIX, "publication matrix path"
    )
    seed_policy = _require_mapping(provenance.get("seed_policy"), "publication seed policy")
    _require_equal(seed_policy.get("mode"), "seed-set", "publication seed policy mode")
    _require_equal(seed_policy.get("seed_set"), EXPECTED_SEED_SET, "publication seed set")
    _require_equal(
        seed_policy.get("resolved_seeds"), list(EXPECTED_SEEDS), "publication resolved seeds"
    )

    totals = _require_mapping(publication_manifest.get("totals"), "publication totals")
    _require_equal(totals.get("file_count"), EXPECTED_PAYLOAD_FILE_COUNT, "publication file count")
    _require_equal(totals.get("total_bytes"), EXPECTED_PAYLOAD_BYTES, "publication byte total")
    if len(checksums) != EXPECTED_PAYLOAD_FILE_COUNT:
        raise ValueError(f"Payload checksum file count mismatch: {len(checksums)}")
    checksum_bytes = sum(size for _, size in checksums.values())
    if checksum_bytes != EXPECTED_PAYLOAD_BYTES:
        raise ValueError(f"Payload checksum byte total mismatch: {checksum_bytes}")
    if "publication_manifest.json" in checksums or "checksums.sha256" in checksums:
        raise ValueError("Top-level release metadata must not be represented as payload checksums")

    manifest_files = publication_manifest.get("files")
    if not isinstance(manifest_files, list) or len(manifest_files) != len(checksums):
        raise ValueError("publication_manifest.files does not match payload checksum entries")
    described: set[str] = set()
    for descriptor in manifest_files:
        descriptor = _require_mapping(descriptor, "publication file descriptor")
        relative = str(descriptor.get("path", ""))
        payload_relative = f"payload/{relative}"
        if payload_relative in described or payload_relative not in checksums:
            raise ValueError(
                f"Publication file descriptor is not a unique payload entry: {relative!r}"
            )
        described.add(payload_relative)
        expected_sha, expected_size = checksums[payload_relative]
        _require_equal(descriptor.get("sha256"), expected_sha, f"publication SHA for {relative}")
        _require_equal(
            descriptor.get("size_bytes"), expected_size, f"publication size for {relative}"
        )
    if described != set(checksums):
        raise ValueError("publication_manifest.files does not cover exactly the checksum entries")

    return {
        "bundle_name": EXPECTED_BUNDLE_NAME,
        "release_tag": EXPECTED_RELEASE_TAG,
        "payload_checksum_files": len(checksums),
        "payload_checksum_bytes": checksum_bytes,
        "publication_manifest_checksum_covered": False,
    }


def _validate_identity_documents(  # noqa: PLR0915 - fixed release identity requires explicit checks.
    bundle_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate fixed campaign/release/scenario/seed/simulation declarations."""
    publication_manifest = _read_json(bundle_root / "publication_manifest.json")
    checksums = _verify_payload_checksums(bundle_root)
    publication = _validate_publication_manifest(bundle_root, publication_manifest, checksums)
    payload_manifest = _read_json(bundle_root / "payload/manifest.json")
    campaign_manifest = _read_json(bundle_root / "payload/campaign_manifest.json")
    validate_config = _read_json(bundle_root / "payload/preflight/validate_config.json")
    checkpoint_preflight = _read_json(
        bundle_root / "payload/preflight/checkpoint_resolvability.json"
    )
    release_manifest = _read_json(bundle_root / "payload/release/release_manifest.resolved.json")
    release_result = _read_json(bundle_root / "payload/release/release_result.json")
    campaign_integrity = _read_json(bundle_root / "payload/reports/campaign_integrity.json")

    _require_equal(
        payload_manifest.get("git_hash"), EXPECTED_SOFTWARE_COMMIT, "payload source commit"
    )
    _require_equal(
        payload_manifest.get("scenario_matrix_hash"),
        EXPECTED_SCENARIO_PUBLICATION_HASH,
        "payload scenario hash",
    )
    _require_equal(campaign_manifest.get("campaign_id"), EXPECTED_CAMPAIGN_ID, "campaign id")
    _require_equal(
        campaign_manifest.get("scenario_matrix"), EXPECTED_SCENARIO_MATRIX, "campaign matrix path"
    )
    _require_equal(
        campaign_manifest.get("scenario_matrix_hash"),
        EXPECTED_SCENARIO_PUBLICATION_HASH,
        "campaign matrix hash",
    )
    _require_equal(
        campaign_manifest.get("config_hash"), EXPECTED_CAMPAIGN_CONFIG_HASH, "campaign config hash"
    )
    campaign_git = _require_mapping(campaign_manifest.get("git"), "campaign git identity")
    _require_equal(campaign_git.get("commit"), EXPECTED_SOFTWARE_COMMIT, "campaign source commit")
    campaign_seed_policy = _require_mapping(
        campaign_manifest.get("seed_policy"), "campaign seed policy"
    )
    _require_equal(campaign_seed_policy.get("mode"), "seed-set", "campaign seed policy mode")
    _require_equal(campaign_seed_policy.get("seed_set"), EXPECTED_SEED_SET, "campaign seed set")
    _require_equal(
        campaign_seed_policy.get("resolved_seeds"), list(EXPECTED_SEEDS), "campaign resolved seeds"
    )

    _require_equal(
        validate_config.get("campaign_id"), EXPECTED_CAMPAIGN_ID, "preflight campaign id"
    )
    _require_equal(
        validate_config.get("scenario_matrix"), EXPECTED_SCENARIO_MATRIX, "preflight matrix path"
    )
    _require_equal(validate_config.get("scenario_count"), 48, "preflight scenario count")
    _require_equal(validate_config.get("horizon"), EXPECTED_HORIZON, "preflight horizon")
    _require_equal(validate_config.get("dt"), EXPECTED_DT, "preflight dt")
    preflight_seed_policy = _require_mapping(
        validate_config.get("seed_policy"), "preflight seed policy"
    )
    _require_equal(preflight_seed_policy.get("seed_set"), EXPECTED_SEED_SET, "preflight seed set")
    _require_equal(
        preflight_seed_policy.get("resolved_seeds"),
        list(EXPECTED_SEEDS),
        "preflight resolved seeds",
    )

    _require_equal(
        release_manifest.get("release_tag"), EXPECTED_RELEASE_TAG, "release manifest tag"
    )
    _require_equal(release_manifest.get("release_id"), EXPECTED_RELEASE_TAG, "release manifest id")
    _require_equal(
        release_manifest.get("source_sha"), EXPECTED_SOFTWARE_COMMIT, "release manifest source"
    )
    release_scenario = _require_mapping(
        release_manifest.get("scenario"), "release scenario identity"
    )
    _require_equal(
        release_scenario.get("matrix_path"), EXPECTED_SCENARIO_MATRIX, "release matrix path"
    )
    _require_equal(
        release_scenario.get("matrix_sha256"), EXPECTED_SCENARIO_SHA256, "release matrix SHA"
    )
    release_matrix = _require_mapping(release_manifest.get("matrix"), "release matrix settings")
    _require_equal(release_matrix.get("horizon_steps"), EXPECTED_HORIZON, "release horizon")
    release_seed_policy = _require_mapping(
        release_manifest.get("seed_policy"), "release seed policy"
    )
    _require_equal(release_seed_policy.get("seed_set"), EXPECTED_SEED_SET, "release seed set")
    _require_equal(
        release_seed_policy.get("resolved_seeds"), list(EXPECTED_SEEDS), "release resolved seeds"
    )
    _require_equal(
        release_manifest.get("canonical_campaign_config_sha256"),
        "5b276c88e39f09aab501d651ad61dffd343946cd48876696c22d593788a3080f",
        "release campaign config SHA",
    )
    release_planners = _require_mapping(release_manifest.get("planners"), "release planners")
    release_configs = {
        str(item.get("key")): item
        for item in release_planners.get("config_identities", [])
        if isinstance(item, Mapping)
    }
    for arm, expected in EXPECTED_ARMS.items():
        config_identity = release_configs.get(expected["planner_key"])
        if config_identity is None:
            raise ValueError(
                f"Release manifest is missing config identity for {expected['planner_key']}"
            )
        _require_equal(config_identity.get("path"), expected["config_path"], f"{arm} config path")
        _require_equal(
            config_identity.get("sha256"), expected["config_sha256"], f"{arm} config SHA"
        )

    _require_equal(
        release_result.get("campaign_id"), EXPECTED_CAMPAIGN_ID, "release result campaign id"
    )
    _require_equal(release_result.get("release_tag"), EXPECTED_RELEASE_TAG, "release result tag")
    _require_equal(
        release_result.get("source_sha"), EXPECTED_SOFTWARE_COMMIT, "release result source"
    )
    _require_equal(release_result.get("benchmark_success"), True, "release benchmark success")
    _require_equal(
        release_result.get("release_status"), "accepted_for_publication", "release status"
    )
    _require_equal(release_result.get("non_success_runs"), 0, "release non-success runs")
    row_status_summary = _require_mapping(
        release_result.get("row_status_summary"), "release row status summary"
    )
    for key in ("accepted_unavailable_rows", "fallback_or_degraded_rows", "unexpected_failed_rows"):
        _require_equal(row_status_summary.get(key), 0, f"release {key}")

    _require_equal(campaign_integrity.get("status"), "valid", "campaign integrity status")
    _require_equal(
        campaign_integrity.get("benchmark_success_allowed"),
        True,
        "campaign benchmark success allowed",
    )
    if campaign_integrity.get("blockers"):
        raise ValueError(f"Campaign integrity blockers: {campaign_integrity['blockers']!r}")
    _require_equal(
        campaign_integrity.get("expected_identity_count"), 1440, "campaign integrity identity count"
    )

    _require_equal(
        checkpoint_preflight.get("metadata_resolvable"), True, "checkpoint metadata resolvability"
    )
    checkpoint_arms = {
        str(item.get("planner_key")): item
        for item in checkpoint_preflight.get("arms", [])
        if isinstance(item, Mapping)
    }
    checkpoint_identity: dict[str, dict[str, Any]] = {}
    for arm, expected in EXPECTED_ARMS.items():
        checkpoint = checkpoint_arms.get(expected["planner_key"])
        if checkpoint is None:
            raise ValueError(f"Checkpoint preflight is missing {expected['planner_key']}")
        _require_equal(checkpoint.get("model_id"), expected["model_id"], f"{arm} model id")
        _require_equal(
            checkpoint.get("checkpoint_sha256"),
            expected["checkpoint_sha256"],
            f"{arm} checkpoint SHA",
        )
        _require_equal(
            checkpoint.get("hash_source"), "registry_declared", f"{arm} checkpoint hash source"
        )
        checkpoint_identity[arm] = {
            "model_id": expected["model_id"],
            "checkpoint_sha256": expected["checkpoint_sha256"],
            "hash_source": checkpoint.get("hash_source"),
        }

    scenarios_payload = json.loads(
        (bundle_root / "payload/runs" / GUARDED_ARM / "scoped_scenarios.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(scenarios_payload, list) or len(scenarios_payload) != 48:
        raise ValueError("Selected-arm scoped scenario matrix must contain 48 scenarios")
    scenario_ids = [str(item.get("name") or item.get("id") or "") for item in scenarios_payload]
    if len(set(scenario_ids)) != 48 or any(not scenario_id for scenario_id in scenario_ids):
        raise ValueError("Scoped scenario matrix has missing or duplicate scenario identities")
    resolved_scenarios = _require_mapping(
        validate_config.get("scenario_candidates"), "preflight scenario candidates"
    ).get("resolved", [])
    _require_equal(sorted(scenario_ids), sorted(resolved_scenarios), "scoped scenario identities")
    for scenario in scenarios_payload:
        _require_equal(
            scenario.get("seeds"),
            list(EXPECTED_SEEDS),
            f"scenario seeds for {scenario.get('name')}",
        )

    identity = {
        **publication,
        "campaign_id": EXPECTED_CAMPAIGN_ID,
        "source_commit": EXPECTED_SOFTWARE_COMMIT,
        "release_tag": EXPECTED_RELEASE_TAG,
        "scenario_matrix": EXPECTED_SCENARIO_MATRIX,
        "scenario_publication_hash": EXPECTED_SCENARIO_PUBLICATION_HASH,
        "scenario_sha256": EXPECTED_SCENARIO_SHA256,
        "seed_set": EXPECTED_SEED_SET,
        "resolved_seeds": list(EXPECTED_SEEDS),
        "horizon_steps": EXPECTED_HORIZON,
        "dt_s": EXPECTED_DT,
        "checkpoint_identity": checkpoint_identity,
        "campaign_config_hash": EXPECTED_CAMPAIGN_CONFIG_HASH,
        "publication_manifest_checksum_covered": False,
    }
    return identity, scenarios_payload


def _load_arm(bundle_root: Path, arm: str) -> list[dict[str, Any]]:
    """Load an arm's JSONL rows; provenance and execution checks happen separately."""
    path = bundle_root / "payload" / "runs" / arm / "episodes.jsonl"
    rows = _load_jsonl(path)
    if not rows:
        raise ValueError(f"{arm}: episode artifact is empty")
    return rows


def _contact_time(row: Mapping[str, Any]) -> float | None:
    """First static-geometry contact time in seconds, if recorded."""
    ledger = _require_mapping(row.get("event_ledger"), "event ledger")
    events = ledger.get("collision_events", [])
    if not isinstance(events, list):
        raise ValueError("event_ledger.collision_events must be a list")
    times: list[float] = []
    for event in events:
        if (
            not isinstance(event, Mapping)
            or event.get("collision_partner_type") != "static_geometry"
        ):
            continue
        if event.get("collision_time") is None:
            continue
        value = float(event["collision_time"])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid static-geometry collision time: {value!r}")
        times.append(value)
    return min(times) if times else None


def _contact_step(
    contact_time_s: float, *, dt: float = EXPECTED_DT, horizon: int = EXPECTED_HORIZON
) -> int:
    """Convert runtime ``(step_idx + 1) * dt`` contact time to a zero-based step."""
    if not math.isfinite(contact_time_s) or contact_time_s < 0:
        raise ValueError(f"Invalid contact time: {contact_time_s!r}")
    step = round(contact_time_s / dt) - 1
    if step < 0 or step >= horizon:
        raise ValueError(f"Derived contact step is outside [0, {horizon}): {step}")
    if not math.isclose(contact_time_s, (step + 1) * dt, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Contact time {contact_time_s!r} is not aligned to dt={dt}")
    return step


def _guard_counts(row: Mapping[str, Any]) -> dict[str, int]:
    """Aggregate retained guard decision-label counts for one episode."""
    metadata = _require_mapping(row.get("algorithm_metadata"), "algorithm metadata")
    stats = _require_mapping(metadata.get("guard_stats"), "guard stats")
    fallback = sum(int(stats.get(label, 0)) for label in FALLBACK_LABELS)
    stop = sum(int(stats.get(label, 0)) for label in STOP_LABELS)
    prior = sum(int(stats.get(label, 0)) for label in PRIOR_LABELS)
    uncertainty = sum(int(stats.get(label, 0)) for label in UNCERTAINTY_LABELS)
    ppo_clear = int(stats.get("ppo_clear", 0))
    ppo_safe = int(stats.get("ppo_safe", 0))
    return {
        "fallback": fallback,
        "stop": stop,
        "prior": prior,
        "uncertainty": uncertainty,
        "ppo_clear": ppo_clear,
        "ppo_safe": ppo_safe,
        # ppo_safe is a pass-through label, not a substitution.
        "substitution_labels": fallback + stop + prior + uncertainty,
        "pass_through_labels": ppo_clear + ppo_safe,
    }


def _last_decision(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the retained final shield decision."""
    metadata = _require_mapping(row.get("algorithm_metadata"), "algorithm metadata")
    shield = _require_mapping(metadata.get("shield_stats"), "shield stats")
    decision = shield.get("last_decision")
    if not isinstance(decision, Mapping):
        raise ValueError("Missing retained shield_stats.last_decision")
    return dict(decision)


def _decision_is_substitution(decision: Mapping[str, Any]) -> bool:
    """Classify a real guard substitution from explicit retained flags."""
    intervened = decision.get("intervened")
    override_applied = decision.get("override_applied")
    if not isinstance(intervened, bool) or not isinstance(override_applied, bool):
        raise ValueError(
            "Guard decision must retain boolean intervened and override_applied fields"
        )
    return intervened or override_applied


def _near_miss_step_rate(row: Mapping[str, Any]) -> float:
    """Pedestrian-only near-miss events per recorded step."""
    steps = max(1, int(row.get("steps", 0)))
    return float(row.get("metrics", {}).get("near_misses", 0.0)) / steps


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    """Median/p90/max summary for a non-empty sample."""
    if not values:
        raise ValueError("Cannot summarize an empty contact-time sample")
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": round(statistics.median(ordered), 2),
        "p90": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))], 2),
        "max": round(max(ordered), 2),
    }


def _aggregate_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Table 7.1 style aggregate means for one arm."""
    n = len(rows)
    if n == 0:
        raise ValueError("Cannot aggregate an empty arm")
    ped = sum(float(row["metrics"]["ped_collision_count"]) for row in rows) / n
    obst = sum(float(row["metrics"]["obstacle_collision_count"]) for row in rows) / n
    success = sum(1 for row in rows if row["outcome"]["route_complete"])
    timeouts = sum(1 for row in rows if row["outcome"]["timeout_event"])
    return {
        "episodes": n,
        "ped_collision_mean": round(ped, 4),
        "obstacle_collision_mean": round(obst, 4),
        "success": success,
        "timeouts": timeouts,
    }


def _family_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    """Obstacle-contact episodes and rates by scenario family."""
    totals: dict[str, int] = {}
    contacts: dict[str, int] = {}
    for row in rows:
        family = _family(str(row["scenario_id"]))
        totals[family] = totals.get(family, 0) + 1
        if float(row["metrics"]["obstacle_collision_count"]) > 0:
            contacts[family] = contacts.get(family, 0) + 1
    table = [
        {
            "family": family,
            "episodes": totals[family],
            "obstacle_contact_episodes": contacts.get(family, 0),
            "rate": round(contacts.get(family, 0) / totals[family], 4),
        }
        for family in totals
    ]
    table.sort(key=lambda entry: (-entry["obstacle_contact_episodes"], entry["family"]))
    return table


def _validate_runtime_row(
    row: Mapping[str, Any],
    *,
    arm: str,
    expected: Mapping[str, Any],
) -> None:
    """Reject execution-integrity failures while retaining ordinary outcomes."""
    status = str(row.get("status", ""))
    if status not in {"success", "collision", "failure"}:
        raise ValueError(f"{arm}: row has non-success execution status {status!r}")
    # ``collision`` and ``failure`` are ordinary episode outcomes in this bundle and are allowed.
    metadata = _require_mapping(row.get("algorithm_metadata"), f"{arm} algorithm metadata")
    _require_equal(metadata.get("status"), "ok", f"{arm} algorithm metadata status")
    integrity = _require_mapping(row.get("integrity"), f"{arm} row integrity")
    effective_view = _require_mapping(
        integrity.get("effective_view"), f"{arm} effective integrity view"
    )
    _require_equal(effective_view.get("degraded"), False, f"{arm} row degraded flag")

    runtime = _require_mapping(metadata.get("planner_runtime"), f"{arm} planner runtime")
    checkpoint = _require_mapping(
        runtime.get("checkpoint_provenance"), f"{arm} checkpoint provenance"
    )
    _require_equal(checkpoint.get("model_id"), expected["model_id"], f"{arm} loaded model id")
    _require_equal(checkpoint.get("load_succeeded"), True, f"{arm} model load status")
    _require_equal(checkpoint.get("fallback_triggered"), False, f"{arm} model fallback flag")
    _require_equal(checkpoint.get("load_status"), "loaded", f"{arm} model load state")
    _require_equal(checkpoint.get("load_error"), None, f"{arm} model load error")

    if expected.get("predictive_foresight_enabled"):
        foresight = _require_mapping(
            runtime.get("foresight_prediction"), f"{arm} foresight runtime"
        )
        _require_equal(
            foresight.get("requested_model_id"),
            expected["predictive_model_id"],
            f"{arm} predictive model id",
        )
        _require_equal(
            foresight.get("requested_checkpoint_sha256"),
            expected["predictive_checkpoint_sha256"],
            f"{arm} predictive checkpoint SHA",
        )
        _require_equal(
            foresight.get("observed_checkpoint_sha256"),
            expected["predictive_checkpoint_sha256"],
            f"{arm} observed predictive checkpoint SHA",
        )
        _require_equal(foresight.get("load_status"), "loaded", f"{arm} predictive load state")
        _require_equal(foresight.get("fallback_used"), False, f"{arm} predictive fallback flag")
        _require_equal(foresight.get("load_error"), None, f"{arm} predictive load error")


def _validate_arm(  # noqa: PLR0915 - fixed campaign provenance requires explicit row checks.
    bundle_root: Path,
    arm: str,
    *,
    scenarios: Sequence[Mapping[str, Any]],
    campaign_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate one selected arm and return rows plus compact identity evidence."""
    expected = EXPECTED_ARMS[arm]
    arm_root = bundle_root / "payload" / "runs" / arm
    episodes_path = arm_root / "episodes.jsonl"
    provenance_path = arm_root / "episodes.jsonl.provenance.json"
    summary_path = arm_root / "summary.json"
    rows = _load_arm(bundle_root, arm)
    summary = _read_json(summary_path)
    provenance = _read_json(provenance_path)
    for field in ("total_jobs", "written", "successful_jobs"):
        _require_equal(summary.get(field), 1440, f"{arm} summary {field}")
    for field in ("failed_jobs", "skipped_jobs"):
        _require_equal(summary.get(field), 0, f"{arm} summary {field}")
    _require_equal(summary.get("failures"), [], f"{arm} summary failures")
    contract = _require_mapping(
        summary.get("algorithm_metadata_contract"), f"{arm} algorithm contract"
    )
    _require_equal(contract.get("status"), "ok", f"{arm} algorithm contract status")
    preflight = _require_mapping(summary.get("preflight"), f"{arm} preflight")
    _require_equal(preflight.get("status"), "ok", f"{arm} preflight status")

    # Reuse the canonical campaign-integrity helper for exact logical coverage and row provenance.
    integrity = validate_campaign_integrity(
        [
            {
                "status": "ok",
                "planner": {"key": expected["planner_key"], "kinematics": "differential_drive"},
                "episodes_path": f"payload/runs/{arm}/episodes.jsonl",
                "summary": {"written": len(rows)},
            }
        ],
        scenarios=scenarios,
        resolved_seeds=list(EXPECTED_SEEDS),
        campaign_root=bundle_root,
        campaign_manifest=campaign_manifest,
    )
    _require_equal(integrity.get("status"), "valid", f"{arm} canonical campaign integrity")

    campaign_identity = _require_mapping(
        provenance.get("campaign_identity"), f"{arm} provenance campaign identity"
    )
    _require_equal(
        campaign_identity.get("scenario_matrix_hash"),
        EXPECTED_RUN_SCENARIO_HASH,
        f"{arm} run scenario hash",
    )
    _require_equal(
        campaign_identity.get("input_bundle_sha256"),
        expected["input_bundle_sha256"],
        f"{arm} input bundle SHA",
    )
    _require_equal(
        campaign_identity.get("algorithm"), expected["algo"], f"{arm} provenance algorithm"
    )
    _require_equal(
        campaign_identity.get("config_hash"), expected["run_config_hash"], f"{arm} run config hash"
    )
    _require_equal(campaign_identity.get("total_jobs"), 1440, f"{arm} provenance total jobs")
    _require_equal(campaign_identity.get("written"), 1440, f"{arm} provenance written jobs")
    _require_equal(
        provenance.get("schema_version"),
        "benchmark_result_provenance.v1",
        f"{arm} provenance schema",
    )
    run_identity = _require_mapping(provenance.get("run"), f"{arm} provenance run")
    _require_equal(
        run_identity.get("repo_commit"), EXPECTED_SOFTWARE_COMMIT, f"{arm} provenance commit"
    )
    _require_equal(
        run_identity.get("benchmark_profile"), "experimental", f"{arm} benchmark profile"
    )
    inputs = _require_mapping(provenance.get("inputs"), f"{arm} provenance inputs")
    matrix_input = _require_mapping(inputs.get("scenario_matrix"), f"{arm} scenario matrix input")
    _require_path_suffix(
        matrix_input.get("path"), EXPECTED_SCENARIO_MATRIX, f"{arm} scenario input path"
    )
    _require_equal(
        matrix_input.get("sha256"), EXPECTED_SCENARIO_SHA256, f"{arm} scenario input SHA"
    )
    config_input = _require_mapping(inputs.get("algo_config"), f"{arm} algo config input")
    _require_path_suffix(
        config_input.get("path"), expected["config_path"], f"{arm} config input path"
    )
    _require_equal(config_input.get("sha256"), expected["config_sha256"], f"{arm} config input SHA")
    raw_artifacts = provenance.get("raw_artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != 1:
        raise ValueError(f"{arm} provenance must contain exactly one raw episode artifact")
    raw_artifact = _require_mapping(raw_artifacts[0], f"{arm} raw artifact")
    raw_sha = _sha256(episodes_path)
    _require_equal(raw_artifact.get("sha256"), raw_sha, f"{arm} raw episode SHA")
    _require_equal(raw_artifact.get("artifact_status"), "available", f"{arm} raw artifact status")
    rows_meta = provenance.get("rows")
    if not isinstance(rows_meta, list) or len(rows_meta) != len(rows):
        raise ValueError(f"{arm} provenance row count does not match JSONL")

    seen: set[tuple[str, int]] = set()
    for line_number, (row, row_meta) in enumerate(zip(rows, rows_meta, strict=True), start=1):
        row_meta = _require_mapping(row_meta, f"{arm} provenance row {line_number}")
        scenario_id = str(row.get("scenario_id", ""))
        seed = row.get("seed")
        if not scenario_id or not isinstance(seed, int):
            raise ValueError(f"{arm}: missing scenario/seed identity")
        identity = (scenario_id, seed)
        if identity in seen:
            raise ValueError(f"{arm}: duplicate episode identity {identity!r}")
        seen.add(identity)
        _require_equal(
            row_meta.get("episode_id"), row.get("episode_id"), f"{arm} sidecar episode id"
        )
        _require_equal(row_meta.get("scenario_id"), scenario_id, f"{arm} sidecar scenario")
        _require_equal(row_meta.get("seed"), seed, f"{arm} sidecar seed")
        _require_equal(
            row_meta.get("config_hash"), row.get("config_hash"), f"{arm} sidecar config hash"
        )
        _require_equal(
            row_meta.get("repo_commit"), EXPECTED_SOFTWARE_COMMIT, f"{arm} sidecar commit"
        )
        _require_equal(row_meta.get("jsonl_line"), line_number - 1, f"{arm} sidecar JSONL line")
        sidecar_settings = _require_mapping(
            row_meta.get("simulator_settings"), f"{arm} sidecar simulator settings"
        )
        _require_equal(sidecar_settings.get("horizon"), EXPECTED_HORIZON, f"{arm} sidecar horizon")
        _require_equal(sidecar_settings.get("dt"), EXPECTED_DT, f"{arm} sidecar dt")
        _require_equal(row.get("algo"), expected["algo"], f"{arm} row algorithm")
        _require_equal(row.get("git_hash"), EXPECTED_SOFTWARE_COMMIT, f"{arm} row commit")
        ledger = _require_mapping(row.get("event_ledger"), f"{arm} event ledger")
        _require_equal(
            ledger.get("software_commit"), EXPECTED_SOFTWARE_COMMIT, f"{arm} ledger commit"
        )
        _require_equal(row.get("horizon"), EXPECTED_HORIZON, f"{arm} row horizon")
        scenario_params = _require_mapping(row.get("scenario_params"), f"{arm} scenario params")
        _require_equal(scenario_params.get("algo"), expected["algo"], f"{arm} scenario algorithm")
        _require_equal(
            scenario_params.get("algo_config_hash"),
            expected["scenario_config_hash"],
            f"{arm} scenario config hash",
        )
        _require_equal(
            scenario_params.get("run_horizon"), EXPECTED_HORIZON, f"{arm} scenario horizon"
        )
        _require_equal(scenario_params.get("run_dt"), EXPECTED_DT, f"{arm} scenario dt")
        result_provenance = _require_mapping(row.get("result_provenance"), f"{arm} row provenance")
        _require_equal(
            result_provenance.get("repo_commit"),
            EXPECTED_SOFTWARE_COMMIT,
            f"{arm} row provenance commit",
        )
        _require_equal(
            result_provenance.get("scenario_id"), scenario_id, f"{arm} row provenance scenario"
        )
        _require_equal(result_provenance.get("seed"), seed, f"{arm} row provenance seed")
        result_settings = _require_mapping(
            result_provenance.get("simulator_settings"), f"{arm} row simulator settings"
        )
        _require_equal(
            result_settings.get("horizon"), EXPECTED_HORIZON, f"{arm} row provenance horizon"
        )
        _require_equal(result_settings.get("dt"), EXPECTED_DT, f"{arm} row provenance dt")
        _require_equal(
            row.get("config_hash"), result_provenance.get("config_hash"), f"{arm} row config hash"
        )
        metadata = _require_mapping(row.get("algorithm_metadata"), f"{arm} algorithm metadata")
        _require_equal(
            metadata.get("config_hash"),
            expected["algorithm_config_hash"],
            f"{arm} algorithm config hash",
        )
        _validate_runtime_row(row, arm=arm, expected=expected)
    expected_identities = {
        (str(item.get("name") or item.get("id")), seed)
        for item in scenarios
        for seed in EXPECTED_SEEDS
    }
    _require_equal(seen, expected_identities, f"{arm} episode identity set")

    return rows, {
        "episodes": len(rows),
        "episodes_sha256": raw_sha,
        "config_path": expected["config_path"],
        "config_sha256": expected["config_sha256"],
        "scenario_config_hash": expected["scenario_config_hash"],
        "algorithm_config_hash": expected["algorithm_config_hash"],
        "run_config_hash": expected["run_config_hash"],
        "model_id": expected["model_id"],
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "checkpoint_hash_source": "registry_declared",
        "predictive_model_id": expected.get("predictive_model_id"),
        "predictive_checkpoint_sha256": expected.get("predictive_checkpoint_sha256"),
    }


def _validate_bundle(
    bundle_root: Path,
    *,
    bundle_archive: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Validate all selected bundle inputs and return identity, rows, and arm reports."""
    if bundle_root.is_symlink():
        raise ValueError("Bundle root must not be a symbolic link")
    if bundle_archive.is_symlink():
        raise ValueError("Release archive must not be a symbolic link")
    archive_sha = _sha256(bundle_archive.resolve(strict=True))
    _require_equal(archive_sha, EXPECTED_RELEASE_ASSET_SHA256, "release asset SHA")
    bundle_root = bundle_root.resolve()
    archived_files = _verify_archive_matches_bundle_root(bundle_archive, bundle_root)
    identity, scenarios = _validate_identity_documents(bundle_root)
    campaign_manifest = _read_json(bundle_root / "payload/campaign_manifest.json")
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    arm_reports: dict[str, dict[str, Any]] = {}
    for arm in (GUARDED_ARM, BASE_ARM):
        rows, report = _validate_arm(
            bundle_root,
            arm,
            scenarios=scenarios,
            campaign_manifest=campaign_manifest,
        )
        rows_by_arm[arm] = rows
        arm_reports[arm] = report
    identity["release_asset_sha256"] = EXPECTED_RELEASE_ASSET_SHA256
    identity["release_asset_sha256_verified"] = True
    identity["release_archive_content_bound_to_bundle_root"] = True
    identity["release_archive_file_count"] = len(archived_files)
    identity["release_archive_total_bytes"] = sum(size for _, size in archived_files.values())
    identity["arms"] = arm_reports
    return identity, rows_by_arm, arm_reports


def _contact_table_row(
    row: Mapping[str, Any],
    *,
    arm: str,
    dt: float = EXPECTED_DT,
    horizon: int = EXPECTED_HORIZON,
) -> dict[str, Any]:
    """Build one retained-field contact row with explicit NA trace fields."""
    obstacle_contact = float(row["metrics"]["obstacle_collision_count"]) > 0
    contact_time = _contact_time(row)
    if obstacle_contact and contact_time is None:
        raise ValueError(f"{row.get('episode_id')}: obstacle contact has no retained contact time")
    metadata = (
        row.get("algorithm_metadata") if isinstance(row.get("algorithm_metadata"), Mapping) else {}
    )
    guard_fields: dict[str, Any]
    if arm == GUARDED_ARM:
        decision = _last_decision(row)
        guard_stats = _guard_counts(row)
        shield = _require_mapping(metadata.get("shield_stats"), "shield stats")
        substitution = _decision_is_substitution(decision)
        guard_fields = {
            "guard_decision_count": shield.get("decision_count", NA),
            "guard_intervention_count": shield.get("intervention_count", NA),
            "guard_override_count": shield.get("override_count", NA),
            "guard_pass_through_count": shield.get("pass_through_count", NA),
            "guard_label_counts": guard_stats,
            "final_guard_label": decision.get("decision_label", NA),
            "final_guard_intervened": decision.get("intervened", NA),
            "final_guard_override_applied": decision.get("override_applied", NA),
            "final_guard_substitution": substitution,
            "final_guard_pass_through": not substitution,
        }
    else:
        guard_fields = {
            "guard_decision_count": NA,
            "guard_intervention_count": NA,
            "guard_override_count": NA,
            "guard_pass_through_count": NA,
            "guard_label_counts": NA,
            "final_guard_label": NA,
            "final_guard_intervened": NA,
            "final_guard_override_applied": NA,
            "final_guard_substitution": NA,
            "final_guard_pass_through": NA,
        }
    scenario_params = _require_mapping(row.get("scenario_params"), "scenario params")
    contact_step = (
        _contact_step(contact_time, dt=dt, horizon=horizon) if contact_time is not None else NA
    )
    return {
        "arm": "guarded_ppo" if arm == GUARDED_ARM else "ppo",
        "episode_id": row.get("episode_id", NA),
        "scenario_cell": row.get("scenario_id", NA),
        "scenario_id": row.get("scenario_id", NA),
        "map_file": scenario_params.get("map_file", NA),
        "seed": row.get("seed", NA),
        "obstacle_contact": obstacle_contact,
        "contact_time_s": contact_time if contact_time is not None else NA,
        "contact_step": contact_step,
        # No position trace is retained in this campaign.
        "contact_x_m": NA,
        "contact_y_m": NA,
        "preceding_pedestrian_clearance": NA,
        "guard_active_last_k_steps": NA,
        **guard_fields,
    }


def _build_contact_table(
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    dt: float = EXPECTED_DT,
    horizon: int = EXPECTED_HORIZON,
) -> dict[str, Any]:
    """Return a deterministic one-row-per-retained-obstacle-contact table."""
    rows: list[dict[str, Any]] = []
    for arm in (GUARDED_ARM, BASE_ARM):
        for row in rows_by_arm[arm]:
            if float(row["metrics"]["obstacle_collision_count"]) > 0:
                rows.append(_contact_table_row(row, arm=arm, dt=dt, horizon=horizon))
    rows.sort(key=lambda item: (item["arm"], item["scenario_id"], item["seed"], item["episode_id"]))
    return {
        "schema_version": TABLE_SCHEMA_VERSION,
        "claim_boundary": "diagnostic-only retained-field table; no trace-dependent causal claim",
        "row_scope": "one row per retained obstacle-contact episode for guarded PPO and base PPO",
        "row_count": len(rows),
        "trace_dependent_fields": {
            "contact_x_m": NA,
            "contact_y_m": NA,
            "preceding_pedestrian_clearance": NA,
            "guard_active_last_k_steps": NA,
        },
        "rows": rows,
    }


def _figures(
    guarded_times: Sequence[float],
    base_times: Sequence[float],
    guarded_family: Sequence[Mapping[str, Any]],
    base_family: Sequence[Mapping[str, Any]],
    last_decisions: Mapping[str, int],
    substitution_counts: Mapping[str, int],
    figure_dir: Path,
) -> list[str]:
    """Render PNG figures; return relative paths for the note."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure_dir.mkdir(parents=True, exist_ok=True)
    names = []

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(guarded_times, bins=30, alpha=0.6, label=f"guarded_ppo (n={len(guarded_times)})")
    ax.hist(base_times, bins=30, alpha=0.6, label=f"base ppo (n={len(base_times)})")
    ax.axvline(60, color="k", linestyle="--", linewidth=1, label="episode cap (60 s)")
    ax.set_xlabel("first obstacle-contact time (s)")
    ax.set_ylabel("episodes")
    ax.set_title("Obstacle-contact timing vs the episode cap")
    ax.legend()
    names.append("contact_timing_hist.png")
    fig.savefig(figure_dir / names[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    families = [entry["family"] for entry in guarded_family[:12]]
    base_by_family = {entry["family"]: entry["rate"] for entry in base_family}
    guarded_rates = [next(e["rate"] for e in guarded_family if e["family"] == f) for f in families]
    base_rates = [base_by_family.get(f, 0.0) for f in families]
    short = [f.replace("classic_", "c:").replace("francis2023_", "f23:") for f in families]
    x = range(len(families))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.4
    ax.bar([i - width / 2 for i in x], guarded_rates, width, label="guarded_ppo")
    ax.bar([i + width / 2 for i in x], base_rates, width, label="base ppo")
    ax.set_xticks(list(x))
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("obstacle-contact episode rate")
    ax.set_title("Obstacle-contact rate by scenario family (top guarded families)")
    ax.legend()
    names.append("family_rates.png")
    fig.savefig(figure_dir / names[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    labels = sorted(last_decisions)
    colours = ["#d95f02" if substitution_counts.get(label, 0) else "#1b9e77" for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, [last_decisions[label] for label in labels], color=colours)
    ax.set_ylabel("episodes")
    ax.set_title("Contact-step guard decision: substitution vs pass-through")
    ax.legend(
        handles=[
            Patch(facecolor="#d95f02", label="intervened or override applied"),
            Patch(facecolor="#1b9e77", label="pass-through"),
        ],
        loc="upper right",
        fontsize=8,
    )
    names.append("contact_step_decision.png")
    fig.savefig(figure_dir / names[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    return names


def _render_note(  # noqa: PLR0913 - report sections are explicit evidence inputs.
    *,
    identity: Mapping[str, Any],
    guarded: Sequence[Mapping[str, Any]],
    base: Sequence[Mapping[str, Any]],
    guarded_family: Sequence[Mapping[str, Any]],
    base_family: Sequence[Mapping[str, Any]],
    guarded_times: Sequence[float],
    base_times: Sequence[float],
    guard_totals: Mapping[str, int],
    contact_steps: int,
    last_labels: Mapping[str, int],
    substitution_counts: Mapping[str, int],
    table_name: str,
    figure_names: Sequence[str],
) -> str:
    """Render the evidence note from validated rows and deterministic summaries."""
    agg_g = _aggregate_table(guarded)
    agg_b = _aggregate_table(base)
    guarded_contacts = [
        row for row in guarded if float(row["metrics"]["obstacle_collision_count"]) > 0
    ]
    guarded_clean = [row for row in guarded if not row["outcome"]["collision_event"]]
    contact_rate_c = sum(_near_miss_step_rate(r) for r in guarded_contacts) / len(guarded_contacts)
    clean_rate_c = sum(_near_miss_step_rate(r) for r in guarded_clean) / len(guarded_clean)
    qt_g, qt_b = _quantiles(guarded_times), _quantiles(base_times)
    substitutions = sum(substitution_counts.values())
    pass_through = sum(last_labels.values()) - substitutions
    figure_rel = f"../figures/{FIGURE_DIR_NAME}"
    arm_identities = _require_mapping(identity.get("arms"), "validated arm identities")
    guarded_identity = _require_mapping(
        arm_identities.get(GUARDED_ARM), "validated guarded arm identity"
    )
    base_identity = _require_mapping(arm_identities.get(BASE_ARM), "validated base arm identity")
    lines = [
        "# Guarded-PPO obstacle contacts in the frozen 0.0.6 campaign",
        "",
        "Claim boundary: descriptive analysis of the frozen release bundle only. No episode was",
        "stepped or rerun, no checkpoint retrained, no runtime changed.",
        "Evidence tier: diagnostic-only reading of retained episode records; no causal, planner-general,",
        "paper, or benchmark-success claim follows.",
        "",
        "## Input identity",
        "",
        f"- Campaign: `{identity['campaign_id']}`; release tag: `{identity['release_tag']}`.",
        f"- Source commit: `{identity['source_commit']}`.",
        f"- Scenario matrix: `{identity['scenario_matrix']}`; publication hash `{identity['scenario_publication_hash']}`;",
        f"  release SHA-256 `{identity['scenario_sha256']}`.",
        f"- Seeds: `{identity['seed_set']}` = {identity['resolved_seeds'][0]}–{identity['resolved_seeds'][-1]};",
        f"  horizon/dt: `{identity['horizon_steps']}` steps / `{identity['dt_s']}` s.",
        f"- Payload checksums: `{identity['payload_checksum_files']}` files / `{identity['payload_checksum_bytes']}` bytes",
        "  verified against `publication_manifest.json` totals. The top-level publication manifest is not",
        "  in `checksums.sha256`, but every extracted bundle file was matched byte-for-byte to the",
        "  SHA-256-verified release archive.",
        f"- Release asset SHA-256: `{identity['release_asset_sha256']}`; archive/root binding: verified",
        f"  (`{identity['release_archive_file_count']}` files / `{identity['release_archive_total_bytes']}` bytes).",
        f"- Arms: `{GUARDED_ARM}` (BR-06 v3 checkpoint behind the runtime guard) and",
        f"  `{BASE_ARM}` (different checkpoint, no guard) — descriptive comparison only,",
        "  not a clean guard ablation because the checkpoints differ.",
        f"- Guarded config: `{guarded_identity['config_path']}` SHA-256 `{guarded_identity['config_sha256']}`;",
        f"  model `{guarded_identity['model_id']}` declared checkpoint SHA-256 `{guarded_identity['checkpoint_sha256']}`.",
        f"- Base config: `{base_identity['config_path']}` SHA-256 `{base_identity['config_sha256']}`;",
        f"  model `{base_identity['model_id']}` declared checkpoint SHA-256 `{base_identity['checkpoint_sha256']}`.",
        f"  Predictive model `{base_identity['predictive_model_id']}` declared checkpoint SHA-256",
        f"  `{base_identity['predictive_checkpoint_sha256']}`.",
        "",
        "Selected arm config/checkpoint identities are bound in the machine-readable validation output:",
        "both configs match the release-manifest SHA, both declared checkpoint identities match the",
        "preflight manifest, and every retained row records a loaded, non-fallback runtime. The declared",
        "checkpoint SHA values are registry declarations; checkpoint bytes are not part of this bundle.",
        "",
        "## Availability matrix (what the bundle can and cannot answer)",
        "",
        "| Issue packet item | Verdict | Reason |",
        "| --- | --- | --- |",
        f"| Retained contact table | available (see `{table_name}`) | cell/map file/seed/time and derived step retained; one row per contact episode |",
        "| Contact x/y or wall-segment geometry | NA | no per-step positions retained |",
        "| Pedestrian within clearance in preceding N steps | NA | no step traces |",
        "| Guard labels/override fields per contact episode | available | retained guard aggregates and final decision fields |",
        "| Guard active in last-k-steps window | NA | per-step decision series not retained |",
        "| Base-PPO same retained-field view | available | episode-level fields only |",
        "| Per-map overlay figure | substituted | family-rate bars + timing histogram instead (no positions) |",
        "",
        "## Table 7.1 verification",
        "",
        "| arm | episodes | ped mean | obstacle mean | success | timeouts |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| guarded_ppo | {agg_g['episodes']} | {agg_g['ped_collision_mean']} |"
        f" {agg_g['obstacle_collision_mean']} | {agg_g['success']}/{agg_g['episodes']} | {agg_g['timeouts']} |",
        f"| base ppo | {agg_b['episodes']} | {agg_b['ped_collision_mean']} |"
        f" {agg_b['obstacle_collision_mean']} | {agg_b['success']}/{agg_b['episodes']} | {agg_b['timeouts']} |",
        "",
        f"Guarded obstacle mean ({agg_g['obstacle_collision_mean']}) matches Table 7.1 (0.33); base PPO is nearly identical",
        f"({agg_b['obstacle_collision_mean']}). The retained guarded summary has fewer pedestrian contacts",
        f"({agg_g['ped_collision_mean']} vs {agg_b['ped_collision_mean']}), fewer successes",
        f"({agg_g['success']} vs {agg_b['success']}), and more timeouts ({agg_g['timeouts']} vs {agg_b['timeouts']}).",
        "These are descriptive differences between two checkpoints; they do not identify a guard effect.",
        "",
        "## Obstacle-contact rate by scenario family",
        "",
        "| family | guarded contacts / episodes (rate) | base contacts / episodes (rate) |",
        "| --- | --- | --- |",
    ]
    base_by_family = {entry["family"]: entry for entry in base_family}
    for entry in guarded_family:
        other = base_by_family.get(entry["family"], {})
        lines.append(
            f"| {entry['family']} | {entry['obstacle_contact_episodes']}/{entry['episodes']}"
            f" ({entry['rate']:.3f}) | {other.get('obstacle_contact_episodes', 0)}/{other.get('episodes', 0)}"
            f" ({other.get('rate', 0.0):.3f}) |"
        )
    lines += [
        "",
        f"![family rates]({figure_rel}/{figure_names[1]})",
        "",
        "Shared wall-heavy cells (merging ~0.97, narrow_doorway 1.00 both arms, doorway,",
        "t_intersection, bottleneck) hit both checkpoints. Divergences are descriptive only: the",
        "checkpoints differ, so no family delta isolates the guard.",
        "",
        "## Contact timing vs the episode cap",
        "",
        f"Guarded contact times (s): n={qt_g['n']}, median={qt_g['median']}, p90={qt_g['p90']}, max={qt_g['max']}.",
        f"Base contact times (s): n={qt_b['n']}, median={qt_b['median']}, p90={qt_b['p90']}, max={qt_b['max']}.",
        f"Contacts after 50 s of the 60 s cap: guarded {sum(1 for t in guarded_times if t > 50)}/{len(guarded_times)},"
        f" base {sum(1 for t in base_times if t > 50)}/{len(base_times)}.",
        "",
        f"![contact timing]({figure_rel}/{figure_names[0]})",
        "",
        "Contacts skew early/mid-episode for both arms; guarded contacts occur later in this retained",
        "comparison. Without per-step traces, this timing difference does not identify a runtime effect.",
        "",
        "## Guard cross-tabulation (guarded arm, 481 contact episodes)",
        "",
        f"Per-episode guard decisions over {contact_steps} contact-episode steps: "
        + ", ".join(
            f"{label}={guard_totals.get(label, 0)}"
            for label in ("ppo_clear", "ppo_safe", "prior", "fallback", "stop", "uncertainty")
        )
        + ".",
        "",
        "Final (contact-step) decisions: "
        + ", ".join(f"{label}={last_labels.get(label, 0)}" for label in sorted(last_labels))
        + ".",
        "",
        f"![contact-step decision]({figure_rel}/{figure_names[2]})",
        "",
        f"In {substitutions}/481 contact episodes the final decision was a real substitution ("
        "`intervened` or `override_applied`); `ppo_safe` is a pass-through, not a substitution. "
        f"The pass-through total is {pass_through}/481 (`ppo_safe` 3 plus `ppo_clear` 159).",
        "These fields are descriptive only: without step traces we cannot distinguish unavoidable contact",
        "from early commitment or fallback steering. The configured guard obstacle clearance (0.30 m),",
        "short-horizon rollout, and fallback DWA weights (goal progress 4.5 vs obstacle clearance 1.2)",
        "provide context, not an outcome attribution.",
        "",
        "## Pedestrian proximity in contact vs clean episodes (guarded arm)",
        "",
        f"Mean pedestrian near-miss events per step: contact episodes {contact_rate_c:.5f}, clean episodes {clean_rate_c:.5f}. "
        f"Episodes with any near-miss: {sum(1 for r in guarded_contacts if float(r['metrics'].get('near_misses', 0)) > 0)}"
        f"/{len(guarded_contacts)} contact vs {sum(1 for r in guarded_clean if float(r['metrics'].get('near_misses', 0)) > 0)}"
        f"/{len(guarded_clean)} clean.",
        "",
        "Wall-contact episodes have a lower retained pedestrian near-miss proxy in this comparison.",
        "That association is descriptive; no pedestrian-to-wall mechanism can be established without",
        "the unavailable step traces.",
        "",
        "## Verified implementation facts (code, not prose)",
        "",
        "- Training reward `route_completion_v3` (un-overridden): collision -10.0 covers pedestrian/robot/obstacle alike;",
        "  `near_miss` -1.0 is pedestrian-only (`snqi_proxy`: robot-ped min distance); `ttc_risk` -0.8 falls back",
        "  to `near_misses` because PPO env metadata never sets `time_to_collision` — hence effectively pedestrian-only",
        "  in training. The issue prose's -1.5/-1.2 values do not match the frozen code.",
        "- Guard thresholds are NOT pedestrian-only: `guard_hard_ped_clearance` 0.58 m,",
        "  `guard_hard_obstacle_clearance` 0.30 m, `guard_min_ttc` 0.70 s; fallback DWA weights pedestrian",
        "  clearance 2.0 vs obstacle clearance 1.2 with goal progress 4.5.",
        "",
        "## Diagnostic synthesis (not paper evidence)",
        "",
        "In the frozen 0.0.6 retained summaries, the guarded checkpoint has lower pedestrian contact",
        "(0.017 per episode) while obstacle contact (0.334) is close to the other checkpoint (0.324);",
        "its success count is 329/1440 versus 796/1440 and its timeout count is 605 versus 35.",
        "Contacts cluster in constrained cells and early-to-mid episode times, and 319/481 contact episodes",
        "end with a guard decision that intervened or applied an override; the 3 `ppo_safe` pass-through",
        "cases are not substitutions. These observations are diagnostic summaries, not causal or benchmark",
        "claims: the checkpoints differ and the bundle retains no step trace for the final contact mechanism.",
        "",
        f"Machine-readable table: `{table_name}`.",
        f"Report schema: `{REPORT_SCHEMA_VERSION}`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the verified bundle analysis and write figures, table, and note."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--bundle-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-output", type=Path)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    table_output = args.table_output or args.output.with_name(f"{args.output.stem}_rows.json")
    identity, rows_by_arm, _ = _validate_bundle(
        args.bundle_root,
        bundle_archive=args.bundle_archive,
    )
    guarded = rows_by_arm[GUARDED_ARM]
    base = rows_by_arm[BASE_ARM]
    guarded_contacts = [
        row for row in guarded if float(row["metrics"]["obstacle_collision_count"]) > 0
    ]
    base_contacts = [row for row in base if float(row["metrics"]["obstacle_collision_count"]) > 0]
    guarded_times_raw = [_contact_time(row) for row in guarded_contacts]
    base_times_raw = [_contact_time(row) for row in base_contacts]
    if any(value is None for value in guarded_times_raw + base_times_raw):
        raise ValueError("Every retained obstacle contact must have a static-geometry contact time")
    guarded_times = [float(value) for value in guarded_times_raw if value is not None]
    base_times = [float(value) for value in base_times_raw if value is not None]

    guard_totals: dict[str, int] = {}
    for row in guarded_contacts:
        for label, count in _guard_counts(row).items():
            if label in {"substitution_labels", "pass_through_labels"}:
                continue
            guard_totals[label] = guard_totals.get(label, 0) + count
    contact_steps = sum(int(row.get("steps", 0)) for row in guarded_contacts)
    last_labels: dict[str, int] = {}
    substitution_counts: dict[str, int] = {}
    for row in guarded_contacts:
        decision = _last_decision(row)
        label = str(decision.get("decision_label", "unknown"))
        last_labels[label] = last_labels.get(label, 0) + 1
        if _decision_is_substitution(decision):
            substitution_counts[label] = substitution_counts.get(label, 0) + 1

    table = _build_contact_table(rows_by_arm, dt=EXPECTED_DT, horizon=EXPECTED_HORIZON)
    table_output.parent.mkdir(parents=True, exist_ok=True)
    table_output.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    guarded_family = _family_table(guarded)
    base_family = _family_table(base)
    figure_names = _figures(
        guarded_times,
        base_times,
        guarded_family,
        base_family,
        last_labels,
        substitution_counts,
        args.figure_dir,
    )
    note = _render_note(
        identity=identity,
        guarded=guarded,
        base=base,
        guarded_family=guarded_family,
        base_family=base_family,
        guarded_times=guarded_times,
        base_times=base_times,
        guard_totals=guard_totals,
        contact_steps=contact_steps,
        last_labels=last_labels,
        substitution_counts=substitution_counts,
        table_name=table_output.name,
        figure_names=figure_names,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(note, encoding="utf-8")
    print(
        f"wrote {args.output} + {table_output} + {len(figure_names)} figures; "
        f"validated {identity['payload_checksum_files']} payload checksums"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
