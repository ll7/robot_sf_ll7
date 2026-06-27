"""Staging/preflight checker for external pedestrian-prior extraction (issue #2918).

This module answers a single, narrow question for issue #2918:

    Given a *metadata-only* manifest describing how a pedestrian-prior
    extraction run would consume a license-compatible external trajectory
    dataset (ETH/UCY, SDD, or SocNavBench-derived), is the extraction plan
    complete, what is still blocked, and is the manifest allowed to assert a
    dataset-backed (calibrated) prior yet?

It deliberately does **not** ingest, download, read, or stage any external
trajectory data, stores **no** raw trajectories, and makes **no** calibrated- or
representative-prior claim. It only checks the declared extraction plan -- the
external source type, provenance fields, the prior parameters the run intends to
emit, and the authored-vs-dataset-backed separation -- against the canonical
extraction contract so a real run, once data is staged and reviewed through the
data-staging workflow, can be wired up without re-deriving the schema.

The allowed external source types map to the canonical external-data asset
registry (``scripts/tools/manage_external_data.py``): ``sdd`` and the
``socnavbench-s3dis-eth`` SocNavBench ETH assets have registered staging
contracts, while the raw ETH/UCY trajectory family has no project-held staging
contract yet (it follows the opt-in BYO-dataset path tracked by #3065/#2657).

Three extraction lifecycle states are handled, because they gate claims
differently:

* ``blocked-external-input`` -- a valid extraction *plan* awaiting
  license-compatible staged data. Every prior parameter stays ``pending``; no
  dataset-backed prior claim is permitted. This is the default for issue #2918
  today.
* ``proxy-only`` -- a schema/smoke run on a proxy fixture, kept strictly
  separate from any dataset-backed prior. Every prior parameter stays
  ``proxy-placeholder``; no provenance source and no calibrated claim are
  permitted.
* ``dataset-backed`` -- license-compatible external data has been staged and
  provenance accepted. Only then may every prior parameter be ``dataset-backed``
  and the manifest assert a dataset-backed prior (still bounded to the staged
  source -- never a representativeness claim).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.common.json_pointer import json_pointer

PEDESTRIAN_PRIOR_EXTRACTION_MANIFEST_SCHEMA_VERSION = "pedestrian_prior_extraction_manifest.v1"
PEDESTRIAN_PRIOR_EXTRACTION_MANIFEST_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "pedestrian_prior_extraction_manifest.v1.json"
)

#: Explicit boundary stamped on every report so a passing preflight check is
#: never mistaken for a calibrated or representative real-world prior.
PRIOR_EXTRACTION_EVIDENCE_BOUNDARY = "prior_extraction_plan_only_no_calibrated_prior_claim"

#: Allowed external source types for pedestrian-prior extraction, mapped to their
#: canonical external-data asset id in ``scripts/tools/manage_external_data.py``
#: (``None`` means no project-held staging contract exists yet -- the source then
#: follows the opt-in BYO-dataset path and can only ever be ``blocked-external-input``
#: or ``proxy-only`` here). The asset-id values are cross-checked against the
#: external-data registry in the test suite so this map cannot silently drift.
ALLOWED_SOURCE_TYPE_ASSET_IDS: dict[str, str | None] = {
    "eth_ucy": None,
    "sdd": "sdd",
    "socnavbench": "socnavbench-s3dis-eth",
}

#: Allowed external source types for pedestrian-prior extraction.
ALLOWED_SOURCE_TYPES: tuple[str, ...] = tuple(ALLOWED_SOURCE_TYPE_ASSET_IDS)

#: Canonical pedestrian-prior parameters the extraction run must declare a plan
#: for, drawn from the issue #2918 scope (walking speed, crossing angle, local
#: density, interaction distance, and stop/yield timing).
REQUIRED_PRIOR_PARAMETERS: tuple[str, ...] = (
    "walking_speed",
    "crossing_angle",
    "density",
    "interaction_distance",
    "stop_yield_timing",
)

#: Provenance fields required before a ``dataset-backed`` manifest may claim a
#: dataset-backed prior. ``checksum`` / ``staging_manifest`` pin the run to a
#: specific staged tree without storing raw trajectories in git.
DATASET_BACKED_REQUIRED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "source_id",
    "source_uri",
    "license",
    "citation",
    "access_date",
    "checksum",
)

#: Per-status the single ``value_status`` every declared prior parameter must
#: carry, so proxy placeholders and pending plans never conflate with
#: dataset-backed priors.
_VALUE_STATUS_BY_EXTRACTION_STATUS: dict[str, str] = {
    "blocked-external-input": "pending",
    "proxy-only": "proxy-placeholder",
    "dataset-backed": "dataset-backed",
}

CONTRACT_STATUS_READY = "ready"
CONTRACT_STATUS_BLOCKED = "blocked"
CONTRACT_STATUS_PROXY_ONLY = "proxy-only"

#: Placeholder tokens that pass the schema's non-empty check but mean a field is
#: not actually resolved yet.
_PLACEHOLDER_VALUES = frozenset({"", "tbd", "unknown", "unspecified", "n/a", "pending"})


class PedestrianPriorExtractionManifestError(ValueError):
    """Raised when a pedestrian-prior extraction manifest fails schema checks."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error from schema messages."""
        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@dataclass(frozen=True, slots=True)
class PedestrianPriorExtractionManifestReport:
    """Result of checking a pedestrian-prior extraction preflight manifest."""

    schema_version: str
    manifest_id: str
    extraction_status: str
    contract_status: str
    evidence_boundary: str
    source_type: str
    external_data_asset_id: str | None
    declared_prior_parameters: list[str]
    missing_prior_parameters: list[str]
    prior_parameter_blockers: list[str]
    provenance_blockers: list[str]
    separation_blockers: list[str]
    dataset_backed_prior_claim_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return asdict(self)

    @property
    def blockers(self) -> list[str]:
        """Return all blockers aggregated across categories, sorted."""
        return sorted(
            self.missing_prior_parameters
            + self.prior_parameter_blockers
            + self.provenance_blockers
            + self.separation_blockers
        )


@lru_cache(maxsize=1)
def load_pedestrian_prior_extraction_manifest_schema() -> dict[str, Any]:
    """Load the manifest JSON schema.

    Returns:
        Parsed JSON Schema dictionary.
    """
    return json.loads(PEDESTRIAN_PRIOR_EXTRACTION_MANIFEST_SCHEMA_FILE.read_text(encoding="utf-8"))


def load_pedestrian_prior_extraction_manifest(path: str | Path) -> dict[str, Any]:
    """Load and schema-validate a manifest from JSON or YAML.

    Returns:
        The validated manifest mapping.

    Raises:
        PedestrianPriorExtractionManifestError: when the file is missing or invalid.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise PedestrianPriorExtractionManifestError(
            ["manifest file not found"], source=manifest_path
        )
    text = manifest_path.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise PedestrianPriorExtractionManifestError(
            [f"invalid YAML/JSON: {exc}"], source=manifest_path
        ) from exc
    if not isinstance(payload, Mapping):
        raise PedestrianPriorExtractionManifestError(
            ["expected a mapping payload"], source=manifest_path
        )
    _raise_on_schema_errors(payload, source=manifest_path)
    return dict(payload)


@lru_cache(maxsize=1)
def _manifest_validator() -> Draft202012Validator:
    """Return a cached schema validator.

    Compiling the schema and resolving references is comparatively expensive, so
    the validator is built once and reused across manifests (e.g. when checking
    several manifests in a loop or in CI).
    """
    return Draft202012Validator(load_pedestrian_prior_extraction_manifest_schema())


def _raise_on_schema_errors(payload: Mapping[str, Any], *, source: str | Path | None) -> None:
    """Raise ``PedestrianPriorExtractionManifestError`` if the payload violates the schema."""
    validator = _manifest_validator()
    errors = [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]
    if errors:
        raise PedestrianPriorExtractionManifestError(errors, source=source)


def check_pedestrian_prior_extraction_manifest(
    manifest: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> PedestrianPriorExtractionManifestReport:
    """Check a pedestrian-prior extraction preflight manifest.

    Args:
        manifest: A ``pedestrian_prior_extraction_manifest.v1`` mapping.
            Schema-validated here.
        source: Optional source path for error messages.

    Returns:
        A structured preflight report. The report never asserts a calibrated or
        representative prior; ``evidence_boundary`` is always
        ``prior_extraction_plan_only_no_calibrated_prior_claim``, and
        ``dataset_backed_prior_claim_allowed`` is only True for a complete
        ``dataset-backed`` manifest with accepted provenance.

    Raises:
        PedestrianPriorExtractionManifestError: when the manifest violates the schema.
    """
    _raise_on_schema_errors(manifest, source=source)

    extraction_status = str(manifest["extraction_status"])
    source_block = manifest.get("source", {})
    source_type = str(source_block.get("type", ""))
    external_data_asset_id = ALLOWED_SOURCE_TYPE_ASSET_IDS.get(source_type)

    declared_prior_parameters = sorted(
        {str(param["name"]) for param in manifest.get("prior_parameters", [])}
    )
    missing = [p for p in REQUIRED_PRIOR_PARAMETERS if p not in declared_prior_parameters]

    prior_parameter_blockers = _prior_parameter_blockers(
        manifest.get("prior_parameters", []), extraction_status=extraction_status
    )
    provenance_blockers = _provenance_blockers(
        manifest.get("provenance"),
        extraction_status=extraction_status,
        source_type=source_type,
    )
    separation_blockers = _separation_blockers(manifest.get("authored_separation", {}))

    report = PedestrianPriorExtractionManifestReport(
        schema_version=PEDESTRIAN_PRIOR_EXTRACTION_MANIFEST_SCHEMA_VERSION,
        manifest_id=str(manifest["manifest_id"]),
        extraction_status=extraction_status,
        contract_status=CONTRACT_STATUS_BLOCKED,  # placeholder, resolved below
        evidence_boundary=PRIOR_EXTRACTION_EVIDENCE_BOUNDARY,
        source_type=source_type,
        external_data_asset_id=external_data_asset_id,
        declared_prior_parameters=declared_prior_parameters,
        missing_prior_parameters=[f"missing prior parameter: {p}" for p in missing],
        prior_parameter_blockers=prior_parameter_blockers,
        provenance_blockers=provenance_blockers,
        separation_blockers=separation_blockers,
        dataset_backed_prior_claim_allowed=False,
    )

    contract_status, claim_allowed = _resolve_contract_status(
        extraction_status=extraction_status,
        source_type=source_type,
        has_blockers=bool(report.blockers),
    )

    # Frozen dataclass: rebuild with the resolved status fields.
    return PedestrianPriorExtractionManifestReport(
        **{
            **report.to_dict(),
            "contract_status": contract_status,
            "dataset_backed_prior_claim_allowed": claim_allowed,
        }
    )


def _resolve_contract_status(
    *, extraction_status: str, source_type: str, has_blockers: bool
) -> tuple[str, bool]:
    """Map extraction lifecycle state and blocker presence onto a contract status.

    Returns:
        ``(contract_status, dataset_backed_prior_claim_allowed)``. A dataset-backed
        prior claim is only allowed for a complete ``dataset-backed`` manifest whose
        source type has a registered external-data staging contract.
    """
    if extraction_status == "dataset-backed":
        # A dataset-backed claim requires a source family with a real staging
        # contract; a BYO-only family (no registered asset) cannot fail-open into
        # a calibrated claim.
        if has_blockers or ALLOWED_SOURCE_TYPE_ASSET_IDS.get(source_type) is None:
            return CONTRACT_STATUS_BLOCKED, False
        return CONTRACT_STATUS_READY, True
    if extraction_status == "proxy-only":
        # Proxy-only is a terminal, claim-free state; blockers downgrade it to
        # blocked so a malformed proxy manifest is never read as usable.
        if has_blockers:
            return CONTRACT_STATUS_BLOCKED, False
        return CONTRACT_STATUS_PROXY_ONLY, False
    # blocked-external-input: a valid plan is still blocked until data arrives.
    return CONTRACT_STATUS_BLOCKED, False


def _prior_parameter_blockers(prior_parameters: Any, *, extraction_status: str) -> list[str]:
    """Validate the declared prior parameters against the status contract.

    Every declared parameter must carry the single ``value_status`` allowed for
    the manifest's extraction status, and parameter names must be unique under
    case-insensitive normalization so a duplicate (or case-variant) entry can
    never silently satisfy coverage with conflicting metadata.

    Returns:
        Sorted blocker strings.
    """
    expected_status = _VALUE_STATUS_BY_EXTRACTION_STATUS[extraction_status]
    blockers: list[str] = []
    seen_normalized: set[str] = set()
    for entry in prior_parameters or []:
        name = str(entry.get("name", ""))
        normalized = name.strip().lower()
        if normalized in seen_normalized:
            # Fail closed: duplicate names (including case variants) make coverage
            # and per-parameter checks ambiguous.
            blockers.append(f"prior parameter {name!r} is a duplicate name (case-insensitive)")
        else:
            seen_normalized.add(normalized)
        value_status = str(entry.get("value_status", ""))
        if value_status != expected_status:
            blockers.append(
                f"prior parameter {name!r} has value_status {value_status!r}; "
                f"extraction_status {extraction_status!r} requires {expected_status!r}"
            )
    return sorted(blockers)


def _provenance_blockers(provenance: Any, *, extraction_status: str, source_type: str) -> list[str]:
    """Return provenance blockers consistent with the extraction status.

    * ``dataset-backed`` requires the full provenance field set and a source type
      with a registered external-data staging contract.
    * ``proxy-only`` must NOT declare a dataset-backed source (no conflation).
    * ``blocked-external-input`` tolerates absent/pending provenance.

    Returns:
        Sorted blocker strings.
    """
    blockers: list[str] = []
    if extraction_status == "dataset-backed":
        if ALLOWED_SOURCE_TYPE_ASSET_IDS.get(source_type) is None:
            blockers.append(
                f"source type {source_type!r} has no registered external-data staging contract; "
                "it cannot back a dataset-backed prior (follow the opt-in BYO-dataset path)"
            )
        if not isinstance(provenance, Mapping):
            blockers.append(
                "dataset-backed manifest requires provenance fields: "
                + ", ".join(DATASET_BACKED_REQUIRED_PROVENANCE_FIELDS)
            )
            return sorted(blockers)
        for field_name in DATASET_BACKED_REQUIRED_PROVENANCE_FIELDS:
            if not _non_empty(provenance.get(field_name)):
                blockers.append(f"dataset-backed manifest missing provenance.{field_name}")
    elif extraction_status == "proxy-only" and isinstance(provenance, Mapping):
        if _non_empty(provenance.get("source_uri")) or _non_empty(provenance.get("source_id")):
            blockers.append(
                "proxy-only manifest must not declare a dataset-backed provenance source; "
                "keep authored/proxy assumptions separate from dataset-backed priors"
            )
    return sorted(blockers)


def _separation_blockers(separation: Mapping[str, Any]) -> list[str]:
    """Return blockers when authored-vs-dataset-backed separation is not enforced.

    Returns:
        Sorted blocker strings.
    """
    blockers: list[str] = []
    if str(separation.get("separation", "")).strip() != "enforced":
        blockers.append(
            "authored_separation.separation must be 'enforced' to keep authored priors "
            "separate from dataset-backed priors"
        )
    return sorted(blockers)


def _non_empty(value: Any) -> bool:
    """Return whether a provenance value is present and non-placeholder."""
    if isinstance(value, Mapping):
        return bool(value)
    return bool(value) and str(value).strip().lower() not in _PLACEHOLDER_VALUES
