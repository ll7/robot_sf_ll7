"""Schema loading, validation, and fail-closed preflight for the real-trajectory ingestion contract.

The contract (GitHub issue #3065) is a *bring-your-own-dataset* (BYO) staging contract: a contributor
points the tooling at trajectory data they have the rights to, and the repository tracks only a
manifest (metadata, license acknowledgment, retrieval instructions, conversion shape, checksums,
split naming, and a durable pointer). Raw external data is never committed.

Two validation layers exist:

* ``validate_manifest_structure`` enforces the JSON Schema (shape, types, required fields).
* ``run_preflight`` enforces the *semantic* contract that the schema cannot express on its own:
  license acknowledgment for BYO data, git-ignored staging locations, an explicit durable-storage
  boundary, and fail-closed benchmark eligibility that stays below claim grade until the local copy
  is checksum-validated.

This module performs no network access and stages no data; it only reads a manifest file/dict.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.errors import RobotSfError

MANIFEST_SCHEMA_ID = "robot_sf_real_trajectory_ingestion_manifest.v1"

_SCHEMA_PATH = (
    Path(__file__).resolve().parent / "schemas" / "real_trajectory_ingestion_manifest.v1.json"
)

# Git-ignored roots where raw/converted BYO trajectory data may live. ``output/`` is the canonical
# worktree artifact root (git-ignored); ``ROBOT_SF_EXTERNAL_DATA_ROOT`` is the optional external
# staging root honored by scripts/tools/manage_external_data.py.
_GITIGNORED_STAGING_PREFIXES = (
    "output/",
    "${ROBOT_SF_EXTERNAL_DATA_ROOT}",
    "$ROBOT_SF_EXTERNAL_DATA_ROOT",
)


def _resolved_staging_dir(staging_dir: str) -> Path:
    """Resolve a BYO staging directory after expanding environment variables.

    Relative paths (for example the canonical ``output/`` artifact root) are
    anchored to the repository root so validated-staging checks do not depend on
    the current working directory. Paths that still contain an unresolved
    environment variable are returned unexpanded so the caller can fail closed.

    Returns:
        Expanded local filesystem path.
    """
    expanded = Path(os.path.expandvars(staging_dir)).expanduser()
    if "$" in str(expanded) or expanded.is_absolute():
        return expanded
    return get_repository_root() / expanded


def _staging_tree_sha256(source_root: Path) -> str:
    """Return aggregate SHA-256 over every file below ``source_root``.

    Relative path, byte size, and per-file SHA-256 all contribute to the
    aggregate so renamed, truncated, or modified local raw files fail closed.
    """

    digest = hashlib.sha256()
    for file_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
        file_digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        relative_path = file_path.relative_to(source_root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(file_path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.hexdigest().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_staging_tree_report(manifest: dict[str, Any]) -> dict[str, object]:
    """Return local staging-tree availability and checksum evidence for a manifest.

    The report is read-only and fail-closed: unresolved environment variables,
    missing directories, and empty staging trees return ``available: false`` with
    a reason instead of raising. Callers can include this payload in readiness
    artifacts without touching or redistributing raw external data.
    """
    staging = manifest.get("staging", {})
    staging_dir = staging.get("staging_dir") if isinstance(staging, dict) else None
    if not isinstance(staging_dir, str):
        report = {"available": False, "reason": "manifest.staging.staging_dir missing"}
        return report

    resolved = _resolved_staging_dir(staging_dir)
    if "$" in str(resolved):
        report = {
            "available": False,
            "staging_dir": str(resolved),
            "reason": "environment variable unresolved",
        }
        return report
    if not resolved.is_dir():
        report = {
            "available": False,
            "staging_dir": str(resolved),
            "reason": "staging directory missing",
        }
        return report

    file_count = sum(1 for path in resolved.rglob("*") if path.is_file())
    if file_count == 0:
        report = {
            "available": False,
            "staging_dir": str(resolved),
            "file_count": 0,
            "reason": "staging directory empty",
        }
        return report

    return {
        "available": True,
        "staging_dir": str(resolved),
        "file_count": file_count,
        "tree_sha256": _staging_tree_sha256(resolved),
    }


def _validated_staging_issues(staging_dir: str, checksums: dict[str, Any]) -> list[PreflightIssue]:
    """Return fail-closed issues for a manifest claiming validated local staging.

    Returns:
        Blocking issues for unresolved, missing, or empty staging directories.
    """
    resolved_staging_dir = _resolved_staging_dir(staging_dir)
    issues: list[PreflightIssue] = []
    if "$" in str(resolved_staging_dir):
        issues.append(
            PreflightIssue(
                code="staging.env_unresolved_for_validated",
                message=(
                    "availability 'validated' requires staging.staging_dir to resolve to a "
                    "local directory; set ROBOT_SF_EXTERNAL_DATA_ROOT or use an output/ path."
                ),
            )
        )
    elif not resolved_staging_dir.is_dir():
        issues.append(
            PreflightIssue(
                code="staging.dir_missing_for_validated",
                message=(
                    "availability 'validated' requires staging.staging_dir to exist locally: "
                    f"{resolved_staging_dir}"
                ),
            )
        )
    elif not any(resolved_staging_dir.iterdir()):
        issues.append(
            PreflightIssue(
                code="staging.dir_empty_for_validated",
                message=(
                    "availability 'validated' requires staging.staging_dir to contain "
                    f"staged trajectory files: {resolved_staging_dir}"
                ),
            )
        )
    expected_tree_sha256 = checksums["expected_tree_sha256"]
    recorded_tree_sha256 = checksums["tree_sha256"]
    if not expected_tree_sha256:
        issues.append(
            PreflightIssue(
                code="checksums.expected_missing_for_validated",
                message=(
                    "availability 'validated' requires checksums.expected_tree_sha256 "
                    "to pin the staged-file tree."
                ),
            )
        )
    elif recorded_tree_sha256 and recorded_tree_sha256 != expected_tree_sha256:
        issues.append(
            PreflightIssue(
                code="checksums.recorded_expected_mismatch",
                message=(
                    "checksums.tree_sha256 must match checksums.expected_tree_sha256 "
                    "for validated real-trajectory staging."
                ),
            )
        )

    if issues:
        return issues

    computed_tree_sha256 = _staging_tree_sha256(resolved_staging_dir)
    if computed_tree_sha256 != recorded_tree_sha256:
        issues.append(
            PreflightIssue(
                code="checksums.staging_tree_mismatch",
                message=(
                    "availability 'validated' requires the local staging tree checksum "
                    "to match checksums.tree_sha256; computed "
                    f"{computed_tree_sha256}, expected {recorded_tree_sha256}."
                ),
            )
        )

    return issues


class ContractError(RobotSfError, RuntimeError):
    """Raised when a manifest cannot be loaded or fails structural schema validation."""


@dataclass(frozen=True)
class PreflightIssue:
    """A single semantic contract violation found by :func:`run_preflight`."""

    code: str
    message: str
    severity: str = "error"  # "error" blocks readiness; "warning" is advisory only.


@dataclass
class PreflightResult:
    """Outcome of a semantic preflight over a structurally valid manifest."""

    dataset_id: str
    availability: str
    benchmark_eligibility: str
    issues: list[PreflightIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[PreflightIssue]:
        """Return only blocking issues."""
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[PreflightIssue]:
        """Return only advisory issues."""
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when no blocking issues were found."""
        return not self.errors


def load_schema() -> dict[str, Any]:
    """Load the manifest JSON Schema bundled with this package.

    Returns:
        The parsed JSON Schema as a dict.
    """
    with _SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest from a YAML or JSON file.

    Returns:
        The parsed manifest mapping.

    Raises:
        ContractError: If the file is missing or does not parse to a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"Manifest file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)  # YAML is a JSON superset, so this also parses .json manifests.
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via malformed-input tests
        raise ContractError(f"Manifest {path} is not valid YAML/JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ContractError(f"Manifest {path} must be a mapping, got {type(data).__name__}.")
    return data


def validate_manifest_structure(
    manifest: dict[str, Any], schema: dict[str, Any] | None = None
) -> None:
    """Validate manifest shape against the JSON Schema.

    Raises:
        jsonschema.ValidationError: If the manifest violates the schema.
    """
    jsonschema.validate(instance=manifest, schema=schema or load_schema())


def _staging_is_gitignored(staging_dir: str) -> bool:
    """Return True when ``staging_dir`` resolves under a known git-ignored staging root."""
    normalized = staging_dir.strip()
    return any(normalized.startswith(prefix) for prefix in _GITIGNORED_STAGING_PREFIXES)


def run_preflight(manifest: dict[str, Any], *, validate_structure: bool = True) -> PreflightResult:
    """Run the fail-closed semantic preflight over a manifest.

    Args:
        manifest: Parsed manifest mapping.
        validate_structure: When True, run JSON Schema validation first (recommended). Set False
            only when the caller has already validated structure.

    Returns:
        A :class:`PreflightResult`. Inspect ``result.ok``/``result.errors`` to gate readiness.

    Raises:
        jsonschema.ValidationError: If ``validate_structure`` is True and the manifest is malformed.
    """
    if validate_structure:
        validate_manifest_structure(manifest)

    issues: list[PreflightIssue] = []

    license_block = manifest["license"]
    if (
        license_block["posture"] == "bring-your-own"
        and not license_block["supplier_acknowledgment"]
    ):
        issues.append(
            PreflightIssue(
                code="license.acknowledgment_missing",
                message=(
                    "bring-your-own data requires license.supplier_acknowledgment: true "
                    "(the supplier must assert they hold the rights to use this data)."
                ),
            )
        )
    if license_block["posture"] == "project-hosted":
        issues.append(
            PreflightIssue(
                code="license.project_hosted_requires_decision",
                message=(
                    "license.posture 'project-hosted' requires an explicit maintainer "
                    "redistribution decision; default to 'bring-your-own'."
                ),
                severity="warning",
            )
        )

    staging = manifest["staging"]
    if not _staging_is_gitignored(staging["staging_dir"]):
        issues.append(
            PreflightIssue(
                code="staging.not_gitignored",
                message=(
                    f"staging_dir '{staging['staging_dir']}' must resolve under a git-ignored "
                    "staging root (output/ or $ROBOT_SF_EXTERNAL_DATA_ROOT) so raw data is never "
                    "committed."
                ),
            )
        )
    # The durable-storage boundary must be explicit and must not point back at the disposable
    # output/ artifact root, which is not a durable dependency.
    durable = staging["durable_storage_target"].strip()
    if durable.startswith("output/"):
        issues.append(
            PreflightIssue(
                code="staging.durable_target_is_output",
                message=(
                    "durable_storage_target must name a durable boundary (e.g. a wandb:// artifact "
                    "URI or 'local-only-byo'); the git-ignored output/ root is not durable."
                ),
            )
        )

    availability = manifest["availability"]
    eligibility = manifest["benchmark_eligibility"]
    checksums = manifest["checksums"]

    # Fail-closed evidence gate: nothing may be marked a benchmark candidate until a local copy is
    # checksum-validated. This keeps downstream claims diagnostic-only until real data is staged.
    if eligibility == "benchmark_candidate" and availability != "validated":
        issues.append(
            PreflightIssue(
                code="eligibility.not_validated",
                message=(
                    f"benchmark_eligibility 'benchmark_candidate' requires availability 'validated', "
                    f"got availability '{availability}'."
                ),
            )
        )
    if availability == "validated" and not checksums["tree_sha256"]:
        issues.append(
            PreflightIssue(
                code="checksums.missing_for_validated",
                message=(
                    "availability 'validated' requires checksums.tree_sha256 to be set "
                    "(the aggregate hash over staged files)."
                ),
            )
        )
    if availability == "validated":
        issues.extend(_validated_staging_issues(staging["staging_dir"], checksums))

    return PreflightResult(
        dataset_id=manifest["dataset_id"],
        availability=availability,
        benchmark_eligibility=eligibility,
        issues=issues,
    )
