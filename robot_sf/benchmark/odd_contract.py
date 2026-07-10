"""Typed loader for ``odd_contract.v1`` evidence-boundary payloads."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.benchmark.observation_quality import ObservationQuality
from robot_sf.common.json_pointer import json_pointer
from robot_sf.errors import RobotSfError

ODD_CONTRACT_SCHEMA_VERSION = "odd_contract.v1"
ODD_CONTRACT_SCHEMA_FILE = Path(__file__).with_name("schemas") / "odd_contract.v1.json"


@dataclass(frozen=True, slots=True)
class OddOperatingContext:
    """Public-space operating context covered by an ODD declaration."""

    environment_types: list[str]
    map_families: list[str]
    surface_conditions: list[str]
    visibility: list[str]
    semantic_features: list[str]


@dataclass(frozen=True, slots=True)
class OddAgentEnvelope:
    """Actor and motion-model assumptions for an ODD declaration."""

    actor_types: list[str]
    pedestrian_motion_models: list[str]
    robot_kinematics: list[str]


@dataclass(frozen=True, slots=True)
class OddSpeedLimits:
    """Speed envelope used to bound benchmark and falsification claims."""

    max_robot_speed_mps: float
    max_pedestrian_speed_mps: float
    notes: str


@dataclass(frozen=True, slots=True)
class OddPedestrianDensity:
    """Pedestrian-density envelope covered by an ODD declaration."""

    density_bins: list[str]
    max_pedestrians_per_scene: int
    notes: str


@dataclass(frozen=True, slots=True)
class OddObservationEnvelope:
    """Observation and sensing assumptions attached to an ODD declaration."""

    observation_modes: list[str]
    sensor_assumptions: list[str]
    observation_quality: ObservationQuality | None = None


@dataclass(frozen=True, slots=True)
class OddClaimBoundaries:
    """Evidence-boundary language for benchmark claims using this ODD."""

    evidence_status: str
    supported_claims: list[str]
    non_claims: list[str]
    caveats: list[str]


@dataclass(frozen=True, slots=True)
class OddProvenance:
    """Provenance metadata for an ODD declaration."""

    source_issue: str
    authored_by: str
    source_files: list[str]
    notes: str


@dataclass(frozen=True, slots=True)
class OddContract:
    """Typed ``odd_contract.v1`` payload."""

    schema_version: str
    id: str
    operating_context: OddOperatingContext
    agents: OddAgentEnvelope
    speed_limits: OddSpeedLimits
    pedestrian_density: OddPedestrianDensity
    observation: OddObservationEnvelope
    exclusions: list[str]
    claim_boundaries: OddClaimBoundaries
    provenance: OddProvenance
    extensions: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the contract to JSON-safe primitives.

        Returns:
            Dictionary representation suitable for JSON Schema validation.
        """

        payload = asdict(self)
        if payload["observation"].get("observation_quality") is None:
            payload["observation"].pop("observation_quality")
        if not payload["extensions"]:
            payload.pop("extensions")
        return payload


class OddContractValidationError(RobotSfError, ValueError):
    """Raised when an ODD contract fails schema or semantic validation."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error."""

        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@lru_cache(maxsize=1)
def load_odd_contract_schema() -> dict[str, Any]:
    """Load the public ``odd_contract.v1`` JSON schema.

    Returns:
        Parsed JSON Schema dictionary.
    """

    return json.loads(ODD_CONTRACT_SCHEMA_FILE.read_text(encoding="utf-8"))


def load_odd_contracts(path: Path) -> list[OddContract]:
    """Load one or more ODD contracts from YAML or JSON.

    Accepted shapes are a single contract mapping, a list of contract mappings, or
    ``{"contracts": [...]}``.

    Returns:
        Typed ODD contracts in file order.
    """

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    payloads = _contract_payloads(raw, source=path)
    return [
        odd_contract_from_dict(payload, source=f"{path.as_posix()}[{index}]")
        for index, payload in enumerate(payloads)
    ]


def odd_contract_from_dict(
    payload: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> OddContract:
    """Validate and convert a mapping into a typed ODD contract.

    Returns:
        Typed ODD contract.
    """

    errors = _schema_validation_errors(payload)
    errors.extend(_semantic_validation_errors(payload))
    if errors:
        raise OddContractValidationError(errors, source=source)

    return _contract_from_payload(payload)


def validate_odd_contract_references(
    *,
    source: str,
    contract_id: str,
    repo_root: Path = Path("."),
) -> list[str]:
    """Validate that an ODD contract reference resolves to a known declaration.

    Returns:
        List of reference errors. An empty list means the reference resolved.
    """

    contract_path = repo_root / source
    if not contract_path.exists():
        return [f"odd_contract_ref.source '{source}' does not exist"]

    try:
        contracts = load_odd_contracts(contract_path)
    except (OddContractValidationError, OSError, yaml.YAMLError) as exc:
        return [f"odd_contract_ref.source '{source}' could not be loaded: {exc}"]

    contract_ids = {contract.id for contract in contracts}
    if contract_id not in contract_ids:
        return [f"odd_contract_ref.contract_id '{contract_id}' was not found in {source}"]
    return []


def classify_odd_claim_boundary(contract: OddContract, claim_id: str) -> str:
    """Classify whether an ODD contract supports, excludes, or does not name a claim.

    Returns:
        ``"supported"``, ``"excluded"``, or ``"unknown"``.
    """

    normalized = str(claim_id).strip()
    if normalized in set(contract.claim_boundaries.supported_claims):
        return "supported"
    if normalized in set(contract.claim_boundaries.non_claims) or normalized in set(
        contract.exclusions
    ):
        return "excluded"
    return "unknown"


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
    raise OddContractValidationError(
        ["expected a contract mapping, a list of mappings, or a top-level 'contracts' list"],
        source=source,
    )


def _schema_validation_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return sorted JSON Schema validation errors for one ODD contract payload."""

    validator = Draft202012Validator(load_odd_contract_schema())
    return [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]


def _semantic_validation_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return cross-field validation errors not expressible in the JSON Schema."""

    errors: list[str] = []
    speed_limits = payload.get("speed_limits")
    if isinstance(speed_limits, Mapping):
        for key in ("max_robot_speed_mps", "max_pedestrian_speed_mps"):
            value = speed_limits.get(key)
            if isinstance(value, int | float) and value <= 0:
                errors.append(f"/speed_limits/{key}: must be greater than 0")
    return errors


def _contract_from_payload(payload: Mapping[str, Any]) -> OddContract:
    """Build a typed ODD contract from a schema-valid payload.

    Returns:
        Typed ODD contract.
    """

    context = payload["operating_context"]
    agents = payload["agents"]
    speed_limits = payload["speed_limits"]
    density = payload["pedestrian_density"]
    observation = payload["observation"]
    boundaries = payload["claim_boundaries"]
    provenance = payload["provenance"]
    return OddContract(
        schema_version=str(payload["schema_version"]),
        id=str(payload["id"]),
        operating_context=OddOperatingContext(
            environment_types=list(context["environment_types"]),
            map_families=list(context["map_families"]),
            surface_conditions=list(context["surface_conditions"]),
            visibility=list(context["visibility"]),
            semantic_features=list(context["semantic_features"]),
        ),
        agents=OddAgentEnvelope(
            actor_types=list(agents["actor_types"]),
            pedestrian_motion_models=list(agents["pedestrian_motion_models"]),
            robot_kinematics=list(agents["robot_kinematics"]),
        ),
        speed_limits=OddSpeedLimits(
            max_robot_speed_mps=float(speed_limits["max_robot_speed_mps"]),
            max_pedestrian_speed_mps=float(speed_limits["max_pedestrian_speed_mps"]),
            notes=str(speed_limits["notes"]),
        ),
        pedestrian_density=OddPedestrianDensity(
            density_bins=list(density["density_bins"]),
            max_pedestrians_per_scene=int(density["max_pedestrians_per_scene"]),
            notes=str(density["notes"]),
        ),
        observation=OddObservationEnvelope(
            observation_modes=list(observation["observation_modes"]),
            sensor_assumptions=list(observation["sensor_assumptions"]),
            observation_quality=(
                ObservationQuality.from_dict(observation["observation_quality"])
                if "observation_quality" in observation
                else None
            ),
        ),
        exclusions=list(payload["exclusions"]),
        claim_boundaries=OddClaimBoundaries(
            evidence_status=str(boundaries["evidence_status"]),
            supported_claims=list(boundaries["supported_claims"]),
            non_claims=list(boundaries["non_claims"]),
            caveats=list(boundaries["caveats"]),
        ),
        provenance=OddProvenance(
            source_issue=str(provenance["source_issue"]),
            authored_by=str(provenance["authored_by"]),
            source_files=list(provenance["source_files"]),
            notes=str(provenance["notes"]),
        ),
        extensions=dict(payload.get("extensions", {})),
    )


__all__ = [
    "ODD_CONTRACT_SCHEMA_FILE",
    "ODD_CONTRACT_SCHEMA_VERSION",
    "ObservationQuality",
    "OddAgentEnvelope",
    "OddClaimBoundaries",
    "OddContract",
    "OddContractValidationError",
    "OddObservationEnvelope",
    "OddOperatingContext",
    "OddPedestrianDensity",
    "OddProvenance",
    "OddSpeedLimits",
    "classify_odd_claim_boundary",
    "load_odd_contract_schema",
    "load_odd_contracts",
    "odd_contract_from_dict",
    "validate_odd_contract_references",
]
