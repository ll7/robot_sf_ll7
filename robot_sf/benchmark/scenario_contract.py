"""Typed loader for ``scenario_contract.v1`` governance payloads."""

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

CONTRACT_SCHEMA_VERSION = "scenario_contract.v1"
CERT_SCHEMA_VERSION = "scenario_cert.v1"
SCENARIO_CONTRACT_SCHEMA_FILE = Path(__file__).with_name("schemas") / "scenario_contract.v1.json"
SCENARIO_GENERATION_PROFILE_EXTENSION = "scenario_generation_profile.v1"


@dataclass(frozen=True, slots=True)
class ScenarioRef:
    """Reference from a governance contract to an authored scenario surface."""

    source: str
    scenario_name: str
    scenario_family: str


@dataclass(frozen=True, slots=True)
class OperatingDesignDomain:
    """Scenario operating-design-domain assumptions."""

    environment_type: str
    map_family: str
    density: str
    flow: str
    assumptions: list[str]


@dataclass(frozen=True, slots=True)
class OddContractReference:
    """Reference from a scenario contract to an ODD declaration."""

    source: str
    contract_id: str
    required_for_benchmark_claim: bool


@dataclass(frozen=True, slots=True)
class CountRange:
    """Inclusive actor count range."""

    minimum: int
    maximum: int


@dataclass(frozen=True, slots=True)
class ActorContract:
    """Authored actor assumption in a scenario contract."""

    id: str
    kind: str
    count: CountRange
    motion_model: str
    assumptions: list[str]


@dataclass(frozen=True, slots=True)
class InvariantContract:
    """Invariant that must hold for the authored scenario intent."""

    id: str
    scope: str
    severity: str
    description: str


@dataclass(frozen=True, slots=True)
class ObservableContract:
    """Observable metric or signal expected when the scenario is executed."""

    id: str
    metric: str
    source: str
    required: bool
    interpretation: str


@dataclass(frozen=True, slots=True)
class TerminationConditionContract:
    """Termination semantics the scenario contract expects consumers to interpret."""

    id: str
    reason: str
    source: str
    description: str


@dataclass(frozen=True, slots=True)
class ScenarioCertificationCompatibility:
    """Compatibility hook connecting intent contracts to ``scenario_cert.v1``."""

    schema_version: str
    required_before_benchmark_claim: bool
    expected_eligibility: str
    notes: str


@dataclass(frozen=True, slots=True)
class BenchmarkEligibilityHooks:
    """Benchmark-use guardrails attached to a scenario contract."""

    intended_use: str
    requires_certification: bool
    claim_boundary: str
    eligibility_hooks: list[str]


@dataclass(frozen=True, slots=True)
class ProvenanceContract:
    """Provenance for an authored scenario contract."""

    source_issue: str
    authored_by: str
    source_files: list[str]
    notes: str


@dataclass(frozen=True, slots=True)
class ScenarioContract:
    """Typed ``scenario_contract.v1`` payload."""

    schema_version: str
    id: str
    scenario_ref: ScenarioRef
    odd: OperatingDesignDomain
    actors: list[ActorContract]
    invariants: list[InvariantContract]
    observables: list[ObservableContract]
    termination_conditions: list[TerminationConditionContract]
    certification: ScenarioCertificationCompatibility
    benchmark_eligibility: BenchmarkEligibilityHooks
    provenance: ProvenanceContract
    odd_contract_ref: OddContractReference | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the contract to JSON-safe primitives.

        Returns:
            Dictionary representation suitable for JSON Schema validation.
        """

        payload = asdict(self)
        if payload["odd_contract_ref"] is None:
            payload.pop("odd_contract_ref")
        if not payload["extensions"]:
            payload.pop("extensions")
        return payload


class ScenarioContractValidationError(ValueError):
    """Raised when a scenario contract fails schema or reference validation."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error."""

        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@lru_cache(maxsize=1)
def load_scenario_contract_schema() -> dict[str, Any]:
    """Load the public ``scenario_contract.v1`` JSON schema.

    Returns:
        Parsed JSON Schema dictionary.
    """

    return json.loads(SCENARIO_CONTRACT_SCHEMA_FILE.read_text(encoding="utf-8"))


def load_scenario_contracts(path: Path) -> list[ScenarioContract]:
    """Load one or more scenario contracts from YAML or JSON.

    Accepted shapes are a single contract mapping, a list of contract mappings, or
    ``{"contracts": [...]}``.

    Returns:
        Typed scenario contracts in file order.
    """

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    payloads = _contract_payloads(raw, source=path)
    return [
        scenario_contract_from_dict(payload, source=f"{path.as_posix()}[{index}]")
        for index, payload in enumerate(payloads)
    ]


def scenario_generation_profile_extension(contract: ScenarioContract) -> dict[str, Any]:
    """Return a namespaced generation profile extension payload."""

    raw_profile = contract.extensions.get(SCENARIO_GENERATION_PROFILE_EXTENSION)
    return dict(raw_profile) if isinstance(raw_profile, Mapping) else {}


def scenario_contract_from_dict(
    payload: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> ScenarioContract:
    """Validate and convert a mapping into a typed scenario contract.

    Returns:
        Typed scenario contract.
    """

    errors = _schema_validation_errors(payload)
    errors.extend(_semantic_validation_errors(payload))
    if errors:
        raise ScenarioContractValidationError(errors, source=source)

    return _contract_from_payload(payload)


def validate_scenario_contract_references(
    contract: ScenarioContract,
    *,
    repo_root: Path = Path("."),
) -> list[str]:
    """Validate that a contract references an existing scenario YAML entry.

    Returns:
        List of reference errors. An empty list means the references resolved.
    """

    errors: list[str] = []
    scenario_path = repo_root / contract.scenario_ref.source
    if not scenario_path.exists():
        return [f"scenario_ref.source '{contract.scenario_ref.source}' does not exist"]

    from robot_sf.training.scenario_loader import load_scenarios  # noqa: PLC0415

    try:
        scenarios = load_scenarios(scenario_path)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        return [f"scenario_ref.source '{contract.scenario_ref.source}' could not be loaded: {exc}"]

    scenario_names = set(_scenario_names_from_payload(scenarios))
    if contract.scenario_ref.scenario_name not in scenario_names:
        errors.append(
            f"scenario_ref.scenario_name '{contract.scenario_ref.scenario_name}' was not found in "
            f"{contract.scenario_ref.source}",
        )
    return errors


def validate_scenario_odd_contract_reference(
    contract: ScenarioContract,
    *,
    repo_root: Path = Path("."),
) -> list[str]:
    """Validate that a scenario contract's ODD reference resolves.

    Returns:
        List of reference errors. Contracts without an ODD reference return an empty list
        so existing scenario contracts remain backward compatible.
    """

    if contract.odd_contract_ref is None:
        return []

    from robot_sf.benchmark.odd_contract import validate_odd_contract_references  # noqa: PLC0415

    return validate_odd_contract_references(
        source=contract.odd_contract_ref.source,
        contract_id=contract.odd_contract_ref.contract_id,
        repo_root=repo_root,
    )


def _contract_payloads(raw: Any, *, source: Path) -> list[Mapping[str, Any]]:
    """Normalize supported file shapes into a list of contract mappings.

    Returns:
        Contract payload mappings in file order.
    """

    if isinstance(raw, Mapping) and "contracts" in raw:
        raw_contracts = raw["contracts"]
    else:
        raw_contracts = raw

    if isinstance(raw_contracts, Mapping):
        return [raw_contracts]
    if isinstance(raw_contracts, list) and all(isinstance(item, Mapping) for item in raw_contracts):
        return raw_contracts
    raise ScenarioContractValidationError(
        ["expected a contract mapping, a list of mappings, or a top-level 'contracts' list"],
        source=source,
    )


def _schema_validation_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return sorted JSON Schema validation errors for one contract payload."""

    validator = Draft202012Validator(load_scenario_contract_schema())
    return [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]


def _semantic_validation_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return cross-field validation errors not expressible in draft-2020-12 JSON Schema."""

    errors: list[str] = []
    actors = payload.get("actors")
    if isinstance(actors, list):
        for index, actor in enumerate(actors):
            if not isinstance(actor, Mapping):
                continue
            count = actor.get("count")
            if not isinstance(count, Mapping):
                continue
            minimum = count.get("minimum")
            maximum = count.get("maximum")
            if isinstance(minimum, int) and isinstance(maximum, int) and maximum < minimum:
                errors.append(
                    f"/actors/{index}/count: maximum {maximum} is smaller than minimum {minimum}",
                )
    return errors


def _contract_from_payload(payload: Mapping[str, Any]) -> ScenarioContract:
    """Build a typed contract from a schema-valid payload.

    Returns:
        Typed scenario contract.
    """

    scenario_ref = payload["scenario_ref"]
    odd = payload["odd"]
    certification = payload["certification"]
    benchmark_eligibility = payload["benchmark_eligibility"]
    provenance = payload["provenance"]
    odd_contract_ref = payload.get("odd_contract_ref")
    return ScenarioContract(
        schema_version=str(payload["schema_version"]),
        id=str(payload["id"]),
        scenario_ref=ScenarioRef(
            source=str(scenario_ref["source"]),
            scenario_name=str(scenario_ref["scenario_name"]),
            scenario_family=str(scenario_ref["scenario_family"]),
        ),
        odd=OperatingDesignDomain(
            environment_type=str(odd["environment_type"]),
            map_family=str(odd["map_family"]),
            density=str(odd["density"]),
            flow=str(odd["flow"]),
            assumptions=list(odd["assumptions"]),
        ),
        actors=[_actor_from_payload(actor) for actor in payload["actors"]],
        invariants=[
            InvariantContract(
                id=str(invariant["id"]),
                scope=str(invariant["scope"]),
                severity=str(invariant["severity"]),
                description=str(invariant["description"]),
            )
            for invariant in payload["invariants"]
        ],
        observables=[
            ObservableContract(
                id=str(observable["id"]),
                metric=str(observable["metric"]),
                source=str(observable["source"]),
                required=bool(observable["required"]),
                interpretation=str(observable["interpretation"]),
            )
            for observable in payload["observables"]
        ],
        termination_conditions=[
            TerminationConditionContract(
                id=str(condition["id"]),
                reason=str(condition["reason"]),
                source=str(condition["source"]),
                description=str(condition["description"]),
            )
            for condition in payload["termination_conditions"]
        ],
        certification=ScenarioCertificationCompatibility(
            schema_version=str(certification["schema_version"]),
            required_before_benchmark_claim=bool(certification["required_before_benchmark_claim"]),
            expected_eligibility=str(certification["expected_eligibility"]),
            notes=str(certification["notes"]),
        ),
        benchmark_eligibility=BenchmarkEligibilityHooks(
            intended_use=str(benchmark_eligibility["intended_use"]),
            requires_certification=bool(benchmark_eligibility["requires_certification"]),
            claim_boundary=str(benchmark_eligibility["claim_boundary"]),
            eligibility_hooks=list(benchmark_eligibility["eligibility_hooks"]),
        ),
        provenance=ProvenanceContract(
            source_issue=str(provenance["source_issue"]),
            authored_by=str(provenance["authored_by"]),
            source_files=list(provenance["source_files"]),
            notes=str(provenance["notes"]),
        ),
        odd_contract_ref=(
            OddContractReference(
                source=str(odd_contract_ref["source"]),
                contract_id=str(odd_contract_ref["contract_id"]),
                required_for_benchmark_claim=bool(odd_contract_ref["required_for_benchmark_claim"]),
            )
            if isinstance(odd_contract_ref, Mapping)
            else None
        ),
        extensions=dict(payload.get("extensions", {})),
    )


def _actor_from_payload(actor: Mapping[str, Any]) -> ActorContract:
    """Build an actor contract from a schema-valid payload.

    Returns:
        Typed actor contract.
    """

    count = actor["count"]
    return ActorContract(
        id=str(actor["id"]),
        kind=str(actor["kind"]),
        count=CountRange(minimum=int(count["minimum"]), maximum=int(count["maximum"])),
        motion_model=str(actor["motion_model"]),
        assumptions=list(actor["assumptions"]),
    )


def _scenario_names_from_payload(raw: Any) -> list[str]:
    """Extract direct scenario names from a scenario YAML payload.

    Returns:
        Scenario names declared directly in the payload.
    """

    scenarios = raw.get("scenarios") if isinstance(raw, Mapping) else raw
    if not isinstance(scenarios, list):
        return []

    names: list[str] = []
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            continue
        name = scenario.get("name") or scenario.get("id") or scenario.get("scenario_id")
        if isinstance(name, str):
            names.append(name)
    return names


__all__ = [
    "CERT_SCHEMA_VERSION",
    "CONTRACT_SCHEMA_VERSION",
    "SCENARIO_CONTRACT_SCHEMA_FILE",
    "SCENARIO_GENERATION_PROFILE_EXTENSION",
    "ActorContract",
    "BenchmarkEligibilityHooks",
    "CountRange",
    "InvariantContract",
    "ObservableContract",
    "OddContractReference",
    "OperatingDesignDomain",
    "ProvenanceContract",
    "ScenarioCertificationCompatibility",
    "ScenarioContract",
    "ScenarioContractValidationError",
    "ScenarioRef",
    "TerminationConditionContract",
    "load_scenario_contract_schema",
    "load_scenario_contracts",
    "scenario_contract_from_dict",
    "scenario_generation_profile_extension",
    "validate_scenario_contract_references",
    "validate_scenario_odd_contract_reference",
]
