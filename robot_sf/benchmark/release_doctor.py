"""Deterministic, credential-safe doctor for benchmark-data release admission."""

from __future__ import annotations

import json
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


def _ci_check(repo: Path, expected_sha: str) -> ReleaseDoctorCheck:
    """Require successful exact-source CI.

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
            "--workflow",
            "CI",
            "--limit",
            "10",
            "--json",
            "headSha,status,conclusion",
        ],
        repo,
    )
    if result.returncode:
        return ReleaseDoctorCheck("ci", "fail", "exact-source CI state is unavailable")
    try:
        runs = json.loads(result.stdout)
    except json.JSONDecodeError:
        runs = []
    exact = [run for run in runs if run.get("headSha") == expected_sha]
    green = any(
        run.get("status") == "completed" and run.get("conclusion") == "success" for run in exact
    )
    return ReleaseDoctorCheck(
        "ci", "pass" if green else "fail", "exact-source CI is green" if green else "no green exact-source CI run"
    )


def _tag_check(repo: Path, tag: str) -> ReleaseDoctorCheck:
    """Require the planned tag to be unused.

    Returns:
        Sanitized check result.
    """
    local = _run(["git", "show-ref", "--verify", "--quiet", f"refs/tags/{tag}"], repo)
    remote = _run(["gh", "release", "view", tag, "--repo", "ll7/robot_sf_ll7"], repo)
    collision = local.returncode == 0 or remote.returncode == 0
    return ReleaseDoctorCheck(
        "tag_collision",
        "fail" if collision else "pass",
        "planned tag already exists" if collision else "planned tag is unused",
    )


def _manifest_check(manifest_path: Path, expected_cells: int) -> tuple[ReleaseDoctorCheck, Any, Any]:
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
    if manifest is None or getattr(manifest, "schema_version", None) != RELEASE_MANIFEST_SCHEMA_VERSION_V0_2:
        problems.append("final manifest is not v0.2")
    if manifest is None or getattr(manifest, "latest_main_base_commit", None) != expected_base_sha:
        problems.append("manifest latest-main base commit does not match")
    if manifest is None or getattr(manifest, "release_tag", None) != tag:
        problems.append("manifest release tag does not match")
    return ReleaseDoctorCheck(
        "release_identity", "pass" if not problems else "fail", "; ".join(problems) or "v0.2 source/tag identity frozen"
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    """Load a JSON/YAML mapping.

    Returns:
        Parsed mapping.
    """
    payload = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("expected mapping")
    return payload


def _cluster_check(packet_path: Path | None, expected_sha: str) -> ReleaseDoctorCheck:
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
    identity_source_sha = identity.get("public_source_commit") if isinstance(identity, dict) else None
    source_sha = packet.get("public_source_sha") or packet.get("release_sha") or identity_source_sha
    problems = []
    if not admitted:
        problems.append("launch packet is not admitted")
    if not dispatchable:
        problems.append("launch packet is not dispatchable")
    if source_sha != expected_sha:
        problems.append("launch packet source SHA does not match")
    return ReleaseDoctorCheck(
        "cluster_admission", "pass" if not problems else "fail", "; ".join(problems) or "launch packet admitted"
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
    except Exception:  # noqa: BLE001
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
        else "Zenodo webhook remains active" if active else "Zenodo webhook was not found"
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
    stale = _run(["rg", "-l", "/Users/lennart/git/robot_sf_ll7", "."], path)
    problems = []
    if missing:
        problems.append(f"missing required paths: {', '.join(missing)}")
    if stale.returncode == 0 and stale.stdout.strip():
        problems.append("hard-coded Robot SF checkout paths remain")
    return ReleaseDoctorCheck(
        "dissertation_paths", "pass" if not problems else "fail", "; ".join(problems) or "release paths healthy"
    )


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
) -> dict[str, Any]:
    """Collect every release-admission check without exposing credentials.

    Returns:
        Machine-readable doctor report.
    """
    manifest_check, manifest, cfg = _manifest_check(manifest_path, expected_cells)
    checks = [
        _git_check(repo, expected_release_sha),
        _ci_check(repo, expected_release_sha),
        _tag_check(repo, tag),
        manifest_check,
        _release_identity_check(manifest, expected_base_sha, tag),
        _checkpoint_check(cfg, manifest, checkpoint_receipt),
        _cluster_check(private_launch_packet, expected_release_sha),
        _disk_check(repo, minimum_free_gib),
        *_zenodo_check(repo, token_file, require_hook_disabled=require_zenodo_webhook_disabled),
        _dissertation_check(dissertation),
    ]
    failed = [check.name for check in checks if check.status != "pass"]
    return {
        "schema_version": "robot-sf-release-doctor.v1",
        "status": "pass" if not failed else "blocked",
        "expected_release_sha": expected_release_sha,
        "expected_base_sha": expected_base_sha,
        "release_tag": tag,
        "checks": [asdict(check) for check in checks],
        "failed_checks": failed,
    }


__all__ = ["ReleaseDoctorCheck", "collect_release_doctor_report"]
