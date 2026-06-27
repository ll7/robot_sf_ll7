"""Staging-contract checker for dataset-backed scenario priors (issue #3161).

Issue #3161 asks whether real-data scenario priors (Stanford Drone Dataset,
SocNavBench ETH, AMV) change the scenario space relative to the authored and
trace-derived baseline from #2919. That comparison is **blocked on external
data**: no licensed dataset is staged in the repository, and the repo never
ingests or redistributes raw trajectories.

This module fills the *local, buildable* half: a metadata-only contract that
declares, per candidate dataset, what staging would have to look like before the
#2919 comparison harness can ingest a dataset-backed prior. It checks:

* **provenance / license** -- source URL, license, license status, citation;
* **distribution fields** -- the canonical scenario-prior parameter groups a
  dataset-backed prior would expose, validated against the comparison harness's
  own parameter vocabulary so the contract cannot silently drift from #2919;
* **explicit external-data blockers** -- a ``blocked-external-input`` dataset
  must name the staging issue(s) blocking it;
* **declared-vs-live staging reconciliation** -- when a live staging probe is
  supplied (e.g. ``manage_external_data.check_asset``), a dataset declared
  ``staged`` whose files are not actually present fails closed.

It deliberately does **not** ingest any dataset, read raw trajectories, run the
comparison, or assert real-world realism. ``evidence_boundary`` on every report
is :data:`STAGING_CONTRACT_EVIDENCE_BOUNDARY`, and a dataset-backed comparison is
only permitted once at least one dataset is staged and contract-clean.

The canonical distribution-field vocabulary lives with the comparison harness
(``scripts/analysis/compare_scenario_priors_issue_2919.py`` ``PARAMETER_GROUPS``).
Callers pass it in via ``allowed_distribution_groups`` so this module stays free
of a ``scripts`` import; the CLI wires the two together.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.common.json_pointer import json_pointer

SCENARIO_PRIOR_STAGING_CONTRACT_SCHEMA_VERSION = "scenario_prior_staging_contract.v1"
SCENARIO_PRIOR_STAGING_CONTRACT_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "scenario_prior_staging_contract.v1.json"
)

#: Explicit boundary stamped on every report so a passing staging-contract check
#: is never mistaken for a staged dataset, an executed comparison, or a
#: real-world realism claim.
STAGING_CONTRACT_EVIDENCE_BOUNDARY = (
    "staging_contract_only_no_dataset_ingest_no_comparison_run_no_realism_claim"
)

# Declared per-dataset staging states (must match the JSON schema enum).
STAGING_STATUS_STAGED = "staged"
STAGING_STATUS_MISSING = "missing"
STAGING_STATUS_BLOCKED = "blocked-external-input"

# Resolved overall contract states.
CONTRACT_STATUS_READY = "ready"
CONTRACT_STATUS_BLOCKED_EXTERNAL = "blocked-external-input"
CONTRACT_STATUS_INVALID = "invalid"

#: Live staging-probe statuses that count as "the declared files are actually
#: present" (matches ``manage_external_data.check_asset`` ``status`` values).
_LIVE_AVAILABLE_STATUSES = frozenset({"available", "staged"})


class ScenarioPriorStagingContractError(ValueError):
    """Raised when a scenario-prior staging contract fails schema checks."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error from schema messages."""
        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@dataclass(frozen=True, slots=True)
class DatasetStagingReport:
    """Per-dataset result of checking one scenario-prior staging entry."""

    dataset_id: str
    asset_id: str | None
    declared_staging_status: str
    live_staging_status: str | None
    effective_staged: bool
    comparison_ready: bool
    distribution_fields: list[str]
    unknown_distribution_fields: list[str]
    blockers: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScenarioPriorStagingContractReport:
    """Aggregate result of checking a scenario-prior staging contract."""

    schema_version: str
    contract_id: str
    issue: int
    evidence_boundary: str
    contract_status: str
    dataset_backed_comparison_allowed: bool
    comparison_ready_datasets: list[str]
    datasets: list[DatasetStagingReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "issue": self.issue,
            "evidence_boundary": self.evidence_boundary,
            "contract_status": self.contract_status,
            "dataset_backed_comparison_allowed": self.dataset_backed_comparison_allowed,
            "comparison_ready_datasets": list(self.comparison_ready_datasets),
            "datasets": [dataset.to_dict() for dataset in self.datasets],
        }

    @property
    def blockers(self) -> list[str]:
        """Return all per-dataset blockers, dataset-prefixed and sorted."""
        return sorted(
            f"{dataset.dataset_id}: {blocker}"
            for dataset in self.datasets
            for blocker in dataset.blockers
        )


@lru_cache(maxsize=1)
def load_scenario_prior_staging_contract_schema() -> dict[str, Any]:
    """Load the staging-contract JSON schema.

    Returns:
        Parsed JSON Schema dictionary.
    """
    return json.loads(SCENARIO_PRIOR_STAGING_CONTRACT_SCHEMA_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _contract_validator() -> Draft202012Validator:
    """Return a cached schema validator (schema compilation is reused)."""
    return Draft202012Validator(load_scenario_prior_staging_contract_schema())


def _raise_on_schema_errors(payload: Mapping[str, Any], *, source: str | Path | None) -> None:
    """Raise :class:`ScenarioPriorStagingContractError` if the payload is invalid."""
    validator = _contract_validator()
    errors = [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]
    if errors:
        raise ScenarioPriorStagingContractError(errors, source=source)


def load_scenario_prior_staging_contract(path: str | Path) -> dict[str, Any]:
    """Load and schema-validate a staging contract from JSON or YAML.

    Returns:
        The validated contract mapping.

    Raises:
        ScenarioPriorStagingContractError: when the file is missing or invalid.
    """
    contract_path = Path(path)
    if not contract_path.is_file():
        raise ScenarioPriorStagingContractError(["contract file not found"], source=contract_path)
    text = contract_path.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ScenarioPriorStagingContractError(
            [f"invalid YAML/JSON: {exc}"], source=contract_path
        ) from exc
    if not isinstance(payload, Mapping):
        raise ScenarioPriorStagingContractError(
            ["expected a mapping payload"], source=contract_path
        )
    _raise_on_schema_errors(payload, source=contract_path)
    return dict(payload)


def check_scenario_prior_staging_contract(
    contract: Mapping[str, Any],
    *,
    allowed_distribution_groups: set[str] | None = None,
    live_staging_status: Mapping[str, str] | None = None,
    source: str | Path | None = None,
) -> ScenarioPriorStagingContractReport:
    """Check a dataset-backed scenario-prior staging contract.

    Args:
        contract: A ``scenario_prior_staging_contract.v1`` mapping. Schema-validated here.
        allowed_distribution_groups: Canonical scenario-prior parameter-group names
            the comparison harness understands (e.g.
            ``set(compare_scenario_priors_issue_2919.PARAMETER_GROUPS)``). When
            provided, any declared distribution field outside this set is a
            fail-closed blocker so the contract cannot drift from the harness.
            When ``None``, the distribution-field vocabulary check is skipped.
        live_staging_status: Optional mapping of ``asset_id`` to a live staging
            status (e.g. ``manage_external_data.check_asset(...)["status"]``).
            Used to reconcile a declared ``staged`` dataset against whether its
            files are actually present.
        source: Optional source path for error messages.

    Returns:
        A structured contract report. The report never asserts a staged dataset,
        an executed comparison, or real-world realism;
        ``dataset_backed_comparison_allowed`` is only True when at least one
        dataset is staged (declared and, if probed, live) and contract-clean.

    Raises:
        ScenarioPriorStagingContractError: when the contract violates the schema.
    """
    _raise_on_schema_errors(contract, source=source)

    dataset_reports = [
        _check_dataset(
            dataset,
            allowed_distribution_groups=allowed_distribution_groups,
            live_staging_status=live_staging_status,
        )
        for dataset in contract["datasets"]
    ]

    comparison_ready = sorted(d.dataset_id for d in dataset_reports if d.comparison_ready)
    any_blockers = any(d.blockers for d in dataset_reports)
    if any_blockers:
        contract_status = CONTRACT_STATUS_INVALID
        comparison_allowed = False
    elif comparison_ready:
        contract_status = CONTRACT_STATUS_READY
        comparison_allowed = True
    else:
        # A well-formed contract with no staged dataset is the expected state for
        # #3161 today: report blocked-external-input rather than substituting a
        # synthetic stand-in (acceptance / stop rule).
        contract_status = CONTRACT_STATUS_BLOCKED_EXTERNAL
        comparison_allowed = False

    return ScenarioPriorStagingContractReport(
        schema_version=SCENARIO_PRIOR_STAGING_CONTRACT_SCHEMA_VERSION,
        contract_id=str(contract["contract_id"]),
        issue=int(contract["issue"]),
        evidence_boundary=STAGING_CONTRACT_EVIDENCE_BOUNDARY,
        contract_status=contract_status,
        dataset_backed_comparison_allowed=comparison_allowed,
        comparison_ready_datasets=comparison_ready,
        datasets=dataset_reports,
    )


def _check_dataset(
    dataset: Mapping[str, Any],
    *,
    allowed_distribution_groups: set[str] | None,
    live_staging_status: Mapping[str, str] | None,
) -> DatasetStagingReport:
    """Check one dataset staging entry and return its report.

    Returns:
        A :class:`DatasetStagingReport`. ``comparison_ready`` is True only for a
        blocker-free dataset that is declared ``staged`` and (when probed) live
        present.
    """
    dataset_id = str(dataset["dataset_id"])
    asset_id = dataset.get("asset_id")
    declared_status = str(dataset["staging_status"])
    declared_fields = [str(value) for value in dataset["distribution_fields"]]
    blockers: list[str] = []

    unknown_fields = _unknown_distribution_fields(declared_fields, allowed_distribution_groups)
    for unknown in unknown_fields:
        blockers.append(
            f"distribution field {unknown!r} is not a canonical comparison parameter group"
        )

    # An explicit external-data blocker must name the staging issue(s) holding it.
    if declared_status == STAGING_STATUS_BLOCKED and not dataset.get("blocker_issues"):
        blockers.append(
            "blocked-external-input dataset must name at least one blocker issue in blocker_issues"
        )

    live_status = _resolve_live_status(asset_id, live_staging_status)
    effective_staged = declared_status == STAGING_STATUS_STAGED
    if effective_staged and live_status is not None and live_status not in _LIVE_AVAILABLE_STATUSES:
        # Fail closed: a contract that claims staged but whose files are absent
        # would let a dataset-backed comparison run on nothing.
        blockers.append(
            f"declared staging_status 'staged' but live probe reports {live_status!r}; "
            "stage and validate the asset before claiming staged"
        )
        effective_staged = False

    comparison_ready = effective_staged and not blockers

    return DatasetStagingReport(
        dataset_id=dataset_id,
        asset_id=str(asset_id) if asset_id is not None else None,
        declared_staging_status=declared_status,
        live_staging_status=live_status,
        effective_staged=effective_staged,
        comparison_ready=comparison_ready,
        distribution_fields=declared_fields,
        unknown_distribution_fields=unknown_fields,
        blockers=sorted(blockers),
    )


def _unknown_distribution_fields(
    declared_fields: list[str], allowed_distribution_groups: set[str] | None
) -> list[str]:
    """Return declared distribution fields outside the canonical comparison vocabulary.

    Returns:
        Sorted unknown field names, or an empty list when no vocabulary is supplied.
    """
    if allowed_distribution_groups is None:
        return []
    return sorted({f for f in declared_fields if f not in allowed_distribution_groups})


def _resolve_live_status(
    asset_id: Any, live_staging_status: Mapping[str, str] | None
) -> str | None:
    """Return the live staging status for an asset id, or ``None`` when not probed."""
    if live_staging_status is None or asset_id is None:
        return None
    status = live_staging_status.get(str(asset_id))
    return str(status) if status is not None else None
